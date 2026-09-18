"""The two pipelines, separated by cadence.

    DAILY   sources -> fetch -> normalize -> detect language -> dedupe -> SQLite
    WEEKLY  SQLite  -> embed -> cluster -> LLM -> rank -> digest -> static site

Keeping them apart is the point of the architecture: collection must be cheap and
reliable enough to run every day, while the expensive semantic work happens once
a week over a whole finished week. The daily pipeline never calls an LLM or an
embedding API.

Both are resilient by construction: one broken source, one failed batch or one
exhausted quota degrades the output rather than aborting the run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import digest as digest_module
from . import lang, render
from .config import Config
from .dedupe import LOOKBACK_HOURS, dedupe
from .embeddings import get_provider as get_embedder
from .llm import build_context, get_provider as get_llm
from .models import RunStats, SourceReport, utcnow
from .sources import Fetcher, adapter_for
from .urls import exclude_by_title, exclude_by_url
from .store import Store

log = logging.getLogger(__name__)


@dataclass
class Options:
    db: Path
    out: Path
    site_url: str = ""
    dry_run: bool = False
    prune: bool = True


# --------------------------------------------------------------------------- #
# daily collection
# --------------------------------------------------------------------------- #

def collect(config: Config, store: Store, stats: RunStats) -> None:
    """Fetch every enabled source, detect language, dedupe, store. No LLM."""
    sources = config.enabled_sources
    if not sources:
        log.warning("no enabled sources in config")
        return

    lang.configure(config.settings.supported_languages)
    store.register_sources(sources)
    fingerprints = store.recent_fingerprints(LOOKBACK_HOURS)

    with Fetcher(config.user_agent) as fetcher:
        for source in sources:
            started = time.monotonic()
            report = SourceReport(name=source.name, ok=True)
            try:
                articles = adapter_for(source, fetcher).fetch(source)
                report.fetched = len(articles)

                kept = exclude_by_url(articles, source.exclude_url_patterns)
                by_url = len(articles) - len(kept)
                kept = exclude_by_title(kept, source.exclude_title_patterns)
                report.excluded = len(articles) - len(kept)
                if report.excluded:
                    log.info(
                        "%-24s excluded %d of %d (%d by url, %d by headline)",
                        source.name, report.excluded, report.fetched,
                        by_url, report.excluded - by_url,
                    )
                articles = kept

                for article in articles:
                    article.publisher = source.publisher
                    article.source_url = source.target
                    article.language = lang.detect_article(
                        article.title,
                        article.excerpt(),
                        allowed=source.languages,
                        fallback=source.languages[0] if source.languages else None,
                    )

                known = store.known_ids([a.id for a in articles])
                keep, dropped = dedupe(
                    articles, known_ids=known, known_fingerprints=fingerprints
                )
                report.duplicates = len(dropped)
                report.new = store.insert_articles(keep)

                for article in keep:
                    fingerprints.append((article.id, "", article.source))
                    key = article.language or "unknown"
                    stats.languages[key] = stats.languages.get(key, 0) + 1

                log.info(
                    "%-24s %3d fetched  %3d new  %3d dupes  [%s]",
                    source.name, report.fetched, report.new, report.duplicates,
                    ",".join(sorted({a.language or "?" for a in keep})) or "-",
                )
            except Exception as exc:  # deliberately broad: isolate the source
                report.ok = False
                report.error = f"{type(exc).__name__}: {exc}"
                log.warning("%-24s FAILED: %s", source.name, report.error)

            report.elapsed_ms = int((time.monotonic() - started) * 1000)
            stats.sources.append(report)
            stats.articles_seen += report.fetched
            stats.articles_new += report.new
            stats.duplicates += report.duplicates
            stats.excluded += report.excluded

    if stats.sources_failed:
        log.warning(
            "%d of %d sources failed: %s",
            len(stats.sources_failed), len(sources),
            ", ".join(s.name for s in stats.sources_failed),
        )


def run_collect(config: Config, options: Options) -> RunStats:
    stats = RunStats(started_at=utcnow())
    with Store(options.db) as store:
        collect(config, store, stats)
        if options.prune and not options.dry_run:
            removed = store.prune(
                config.storage.retention_days, config.storage.embedding_retention_days
            )
            if removed:
                log.info("pruned %d rows past retention", removed)
        if not options.dry_run:
            store.record_run(stats, kind="collect")
    log.info(
        "collection complete: %d/%d sources ok, %d new articles %s",
        stats.sources_ok, len(stats.sources), stats.articles_new, stats.languages,
    )
    return stats


# --------------------------------------------------------------------------- #
# weekly processing
# --------------------------------------------------------------------------- #

def run_weekly(
    config: Config,
    options: Options,
    *,
    now: datetime | None = None,
    window: tuple[datetime, datetime] | None = None,
) -> RunStats:
    stats = RunStats(started_at=utcnow())
    llm = get_llm(config.llm)
    embedder = get_embedder(config.embeddings)
    context = build_context(config.settings, config.preferences)

    log.info(
        "llm: %s (%s) | embeddings: %s (%s, %dd) | output language: %s",
        llm.name, config.llm.model if llm.name != "none" else "offline heuristics",
        embedder.name, embedder.model, embedder.dimensions,
        config.settings.output_language,
    )

    with Store(options.db) as store:
        result = digest_module.build_digest(
            config, store, llm, embedder, context, stats,
            now=now, window=window, persist=not options.dry_run,
        )
        if not options.dry_run:
            render.write_site(
                options.out, config, store, result, site_url=options.site_url
            )
            store.record_run(stats, kind="weekly")

    log.info(
        "weekly complete: digest %s, %d stories, %d llm calls, %d embed calls",
        stats.digest_id, stats.stories_published, stats.llm_calls, stats.embed_calls,
    )
    return stats
