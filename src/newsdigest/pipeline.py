"""Stage orchestration.

    fetch -> dedupe -> store -> enrich -> cluster -> score -> build

Every stage is independently runnable from the CLI so a failure or a quota
problem in one does not force the others to be redone. The overriding rule is
resilience: one broken source, one failed LLM batch or one unparseable page
degrades the run's output, never aborts it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from . import clustering, enrich, render, scoring
from .config import Config
from .llm import LLMProvider, LLMQuotaError, get_provider
from .models import Article, RunStats, SourceReport, Story, utcnow
from .sources import Fetcher, adapter_for
from .store import Store

log = logging.getLogger(__name__)

# How far back clustering and story matching look.
CLUSTER_WINDOW_HOURS = 48.0
STORY_MATCH_WINDOW_HOURS = 24.0 * 7


@dataclass
class Options:
    db: Path
    out: Path
    fetch: bool = True
    do_enrich: bool = True
    do_cluster: bool = True
    build: bool = True
    enrich_limit: int | None = None
    prune: bool = True
    site_url: str = ""
    dry_run: bool = False


# --------------------------------------------------------------------------- #
# fetch
# --------------------------------------------------------------------------- #

def fetch_stage(config: Config, store: Store, stats: RunStats) -> None:
    from .dedupe import LOOKBACK_HOURS, dedupe

    sources = config.enabled_sources
    if not sources:
        log.warning("no enabled sources in config")
        return

    fingerprints = store.recent_fingerprints(LOOKBACK_HOURS)
    with Fetcher(config.user_agent) as fetcher:
        for source in sources:
            started = time.monotonic()
            report = SourceReport(name=source.name, ok=True)
            try:
                adapter = adapter_for(source, fetcher)
                articles = adapter.fetch(source)
                report.fetched = len(articles)

                known = store.known_ids([a.id for a in articles])
                keep, dropped = dedupe(
                    articles, known_ids=known, known_fingerprints=fingerprints
                )
                report.duplicates = len(dropped)
                report.new = store.insert_articles(keep)

                for article in keep:
                    fingerprints.append((article.id, "", article.source))
                log.info(
                    "%-22s %3d fetched  %3d new  %3d dupes",
                    source.name, report.fetched, report.new, report.duplicates,
                )
            except Exception as exc:  # deliberately broad: isolate the source
                report.ok = False
                report.error = f"{type(exc).__name__}: {exc}"
                log.warning("%-22s FAILED: %s", source.name, report.error)

            report.elapsed_ms = int((time.monotonic() - started) * 1000)
            stats.sources.append(report)
            stats.articles_seen += report.fetched
            stats.articles_new += report.new
            stats.duplicates += report.duplicates

    if stats.sources_failed:
        log.warning(
            "%d of %d sources failed: %s",
            len(stats.sources_failed),
            len(sources),
            ", ".join(s.name for s in stats.sources_failed),
        )


# --------------------------------------------------------------------------- #
# enrich
# --------------------------------------------------------------------------- #

def enrich_stage(
    config: Config, store: Store, provider: LLMProvider, stats: RunStats, limit: int | None
) -> None:
    report = enrich.enrich_pending(store, provider, config.llm, limit=limit)
    stats.enriched = report.enriched
    stats.enrich_cached = report.cached
    stats.enrich_failed = report.failed
    stats.llm_calls = report.calls


# --------------------------------------------------------------------------- #
# cluster
# --------------------------------------------------------------------------- #

def cluster_stage(
    config: Config, store: Store, provider: LLMProvider, stats: RunStats
) -> None:
    # Unenriched articles are included on purpose: they cluster on text alone
    # and appear with their feed description, so a quota problem degrades the
    # feed instead of emptying it.
    articles = store.recent_articles(CLUSTER_WINDOW_HOURS, enriched_only=False)
    if not articles:
        log.info("no recent articles to cluster")
        return

    clusters = clustering.cluster(articles)
    existing = store.recent_stories(STORY_MATCH_WINDOW_HOURS)
    log.info("clustered %d articles into %d stories", len(articles), len(clusters))

    quota_hit = False
    for group in clusters:
        keywords = clustering.cluster_keywords(group)
        match = clustering.match_existing_story(keywords, group, existing)
        story = clustering.build_story(group, existing=match)

        if config.llm.write_story_briefs and not quota_hit:
            if _needs_brief(store, story, group, match):
                try:
                    brief = enrich.write_brief(provider, story, group)
                except LLMQuotaError as exc:
                    log.warning("quota exhausted while writing briefs (%s)", exc)
                    quota_hit = True
                    brief = None
                if brief:
                    story.headline = brief.headline
                    story.summary = brief.summary
                    story.why_it_matters = brief.why_it_matters or story.why_it_matters
                    story.topics = brief.topics or story.topics
                    story.written_by = provider.name
            elif match is not None:
                # Keep the merged text a previous run paid for.
                story.headline = match.headline
                story.summary = match.summary
                story.why_it_matters = match.why_it_matters or story.why_it_matters
                story.written_by = match.written_by

        story.score = scoring.score_story(story, group, config.interests)
        if store.upsert_story(story):
            stats.stories_new += 1
        store.assign_story(story.article_ids, story.id)
        stats.stories_total += 1
        existing.append(story)

    stats.llm_calls = provider.calls


def _needs_brief(
    store: Store, story: Story, group: list[Article], match: Story | None
) -> bool:
    """Only pay for a brief on a new story or one that gained coverage."""
    if len({a.source for a in group}) < 2:
        return False
    if match is None or match.written_by is None:
        return True
    return store.story_article_count(story.id) != len(group)


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #

def build_stage(config: Config, store: Store, out: Path, stats: RunStats, site_url: str = "") -> dict:
    interests = config.interests
    stories = store.recent_stories(interests.max_age_hours)
    grouped = store.articles_by_story([s.id for s in stories])

    ranked: list[tuple[Story, list[Article]]] = []
    for story in stories:
        articles = grouped.get(story.id) or []
        if not articles:
            continue
        # Re-score at build time: recency has moved since clustering.
        story.score = scoring.score_story(story, articles, interests)
        ranked.append((story, articles))

    ranked.sort(key=lambda pair: pair[0].score, reverse=True)
    ranked = ranked[: interests.max_stories]
    stats.stories_published = len(ranked)

    for story, articles in ranked:
        store.upsert_story(story)

    return render.write_site(out, config, ranked, stats, site_url=site_url)


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #

def run(config: Config, options: Options) -> RunStats:
    stats = RunStats(started_at=utcnow())
    provider = get_provider(config.llm)
    log.info("llm provider: %s (%s)", provider.name, config.llm.model
             if provider.name != "none" else "offline heuristics")

    with Store(options.db) as store:
        if options.fetch:
            fetch_stage(config, store, stats)
        if options.do_enrich:
            enrich_stage(config, store, provider, stats, options.enrich_limit)
        if options.do_cluster:
            cluster_stage(config, store, provider, stats)
        if options.build and not options.dry_run:
            build_stage(config, store, options.out, stats, options.site_url)
        if options.prune and not options.dry_run:
            removed = store.prune(config.interests.retention_days)
            if removed:
                log.info("pruned %d rows past the %d-day retention window",
                         removed, config.interests.retention_days)
        if not options.dry_run:
            store.record_run(stats)

    log.info(
        "run complete: %d/%d sources ok, %d new articles, %d stories published, %d llm calls",
        stats.sources_ok, len(stats.sources), stats.articles_new,
        stats.stories_published, stats.llm_calls,
    )
    return stats
