"""Weekly processing: turn a week of collected articles into one digest.

    window -> filter -> embed -> cluster -> pre-rank -> LLM -> rank -> persist

The pre-rank step is what keeps this affordable. Clusters are ordered using only
signals available without an LLM -- how many publishers covered it, how much
coverage there is, source weights, recency, and configured topic hints -- and
only the articles in the surviving clusters are ever sent for enrichment. That is
the difference between summarizing 30 stories and summarizing 3000 articles.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone

from . import clustering, enrich, scoring
from .config import Config
from .embeddings.base import EmbeddingProvider
from .embed import embed_articles
from .llm.base import Context, LLMProvider, LLMQuotaError
from .models import Article, Digest, RunStats, Story, utcnow
from .store import Store
from .text import normalize

log = logging.getLogger(__name__)

#: Clusters kept for LLM enrichment, as a multiple of the digest's story count.
#: Some will rank lower once the LLM has scored them, so we keep slack.
CANDIDATE_MULTIPLE = 2.0
#: Never enrich fewer clusters than this, however small max_stories is.
MIN_CANDIDATE_CLUSTERS = 10


def weekly_window(
    now: datetime | None = None, *, week_ends_on: int = 6, days: int = 7
) -> tuple[datetime, datetime]:
    """The [start, end) window for the week that has just finished.

    `week_ends_on` is the last weekday the digest covers (0 = Monday,
    6 = Sunday), so the window ends at midnight on the following day. Running on
    Monday morning with the default therefore covers the previous Monday to
    Sunday inclusive -- a whole finished week, never a partial one.
    """
    now = now or utcnow()
    end_weekday = (week_ends_on + 1) % 7
    days_back = (now.weekday() - end_weekday) % 7
    end_date = (now - timedelta(days=days_back)).date()
    end = datetime.combine(end_date, time.min, tzinfo=timezone.utc)
    return end - timedelta(days=days), end


def usable(article: Article) -> bool:
    """Drop articles that cannot carry a story."""
    return bool(article.title.strip()) and bool(article.url) and len(article.title) > 8


def select_candidates(articles: list[Article]) -> list[Article]:
    return [a for a in articles if usable(a)]


def prerank_score(articles: list[Article], config: Config) -> float:
    """Rank a cluster without any LLM output.

    Deliberately uses only what collection already gave us, because this decides
    which clusters are worth paying to enrich.
    """
    prefs = config.preferences
    publishers = {a.publisher for a in articles}
    newest = max((a.published_at or a.collected_at) for a in articles)
    age_hours = max(0.0, (utcnow() - newest).total_seconds() / 3600.0)

    corroboration = 0.0 if len(publishers) <= 1 else min(1.0, math.log(len(publishers), 4))
    size = min(1.0, math.log(len(articles) + 1, 8))
    weight = sum(a.source_weight for a in articles) / len(articles)
    recency = math.exp(-math.log(2) * age_hours / max(1.0, prefs.recency_half_life_hours))

    # Topic hints from the feeds plus the reader's keywords, both pre-LLM.
    haystack = normalize(" ".join(a.title for a in articles))
    hint_topics = {t for a in articles for t in a.source_topics}
    topic_weight = max(
        (w for t in hint_topics for name, w in prefs.topics.items()
         if name in normalize(t) or normalize(t) in name),
        default=1.0,
    )
    keyword_hits = sum(1 for keyword in prefs.keywords if keyword in haystack)
    preferred = {normalize(n) for n in prefs.preferred_sources}
    preference = prefs.preferred_source_bonus if (
        preferred and any(normalize(p) in preferred for p in publishers)
    ) else 0.0

    score = (
        1.5 * corroboration
        + 1.0 * size
        + 0.8 * recency
        + 0.6 * topic_weight
        + 0.3 * weight
        + preference
        + prefs.keyword_bonus * keyword_hits
    )
    if prefs.excluded_topics and (
        set(prefs.excluded_topics) & {normalize(t) for t in hint_topics}
        or any(term in haystack for term in prefs.excluded_topics)
    ):
        score -= prefs.excluded_penalty
    return score


@dataclass
class DigestResult:
    digest: Digest
    stories: list[tuple[Story, list[Article]]]


def build_digest(
    config: Config,
    store: Store,
    llm: LLMProvider,
    embedder: EmbeddingProvider,
    context: Context,
    stats: RunStats,
    *,
    now: datetime | None = None,
    window: tuple[datetime, datetime] | None = None,
    persist: bool = True,
) -> DigestResult:
    """Run the whole weekly pipeline.

    With `persist=False` (a dry run) no story, digest or run is written. The
    enrichment and embedding caches are still filled, since those are derived
    data and make the next real run cheaper rather than changing its output.
    """
    start, end = window or weekly_window(now, week_ends_on=config.digest.week_ends_on)
    log.info("weekly window %s .. %s", start.date(), end.date())

    articles = select_candidates(
        store.articles_in_window(start, end, languages=config.settings.supported_languages)
    )
    stats.articles_seen = len(articles)
    if not articles:
        log.warning("no articles in the window; nothing to digest")
        digest = Digest(id=Digest.week_id(end - timedelta(days=1)),
                        period_start=start, period_end=end)
        if persist:
            store.save_digest(digest)
        stats.digest_id = digest.id
        return DigestResult(digest=digest, stories=[])

    # 1. Embeddings for the whole window -- clustering needs them all.
    vectors, embed_report = embed_articles(store, embedder, articles)
    stats.embedded = embed_report.embedded
    stats.embed_cached = embed_report.cached
    stats.embed_calls = embed_report.calls

    # 2. Cluster.
    groups = clustering.cluster(
        articles,
        vectors,
        similarity_threshold=config.embeddings.similarity_threshold,
        ambiguous_threshold=config.embeddings.ambiguous_threshold,
        provider=llm if config.llm.resolve_clusters else None,
        context=context,
        max_checks=config.llm.max_cluster_checks,
        stats=stats,
    )
    stats.clusters = len(groups)
    stats.cross_language_clusters = clustering.cross_language_count(groups)
    log.info(
        "%d articles -> %d clusters (%d span more than one language)",
        len(articles), len(groups), stats.cross_language_clusters,
    )

    # 3. Pre-rank without the LLM, and keep only what is worth enriching.
    groups.sort(key=lambda g: prerank_score(g, config), reverse=True)
    keep = max(MIN_CANDIDATE_CLUSTERS, int(config.digest.max_stories * CANDIDATE_MULTIPLE))
    candidates = groups[:keep]
    log.info("enriching the top %d of %d clusters", len(candidates), len(groups))

    # 4. LLM enrichment for the candidates' articles only.
    candidate_articles = [a for group in candidates for a in group]
    enrich_report = enrich.enrich_articles(
        store, llm, config.llm, context, candidate_articles, persist=persist
    )
    stats.enriched = enrich_report.enriched
    stats.enrich_cached = enrich_report.cached
    stats.enrich_failed = enrich_report.failed

    # 5. Build stories, with a merged brief for multi-publisher ones.
    ranked: list[tuple[Story, list[Article]]] = []
    quota_hit = enrich_report.quota_exhausted
    for group in candidates:
        story = clustering.build_story(group)
        if config.llm.write_story_briefs and not quota_hit:
            try:
                brief = enrich.write_brief(llm, context, story, group)
            except LLMQuotaError as exc:
                log.warning("quota exhausted while writing briefs (%s)", exc)
                quota_hit = True
                brief = None
            if brief:
                story.headline = brief.headline
                story.summary = brief.summary
                story.why_it_matters = brief.why_it_matters or story.why_it_matters
                story.key_facts = brief.key_facts or story.key_facts
                story.topics = brief.topics or story.topics
                if brief.importance is not None:
                    story.importance = brief.importance
                if brief.relevance is not None:
                    story.relevance = brief.relevance
                story.written_by = llm.name

        story.score = scoring.score_story(story, group, config.preferences)
        ranked.append((story, group))

    stats.llm_calls = llm.calls

    # 6. Final ranking and the cut.
    ranked.sort(key=lambda pair: pair[0].score, reverse=True)
    if config.digest.min_articles > 1:
        ranked = [
            pair for pair in ranked if len(pair[1]) >= config.digest.min_articles
        ] or ranked
    ranked = ranked[: config.digest.max_stories]

    for story, group in ranked:
        if persist and store.replace_story(story, [a.id for a in group]):
            stats.stories_new += 1
        stats.stories_total += 1
    stats.stories_published = len(ranked)

    digest = Digest(
        id=Digest.week_id(end - timedelta(days=1)),
        period_start=start,
        period_end=end,
        story_ids=[story.id for story, _ in ranked],
        article_count=sum(len(group) for _, group in ranked),
        stats={
            "articles_in_window": len(articles),
            "clusters": stats.clusters,
            "cross_language_clusters": stats.cross_language_clusters,
            "embedded": stats.embedded,
            "embed_cached": stats.embed_cached,
            "enriched": stats.enriched,
            "enrich_cached": stats.enrich_cached,
            "llm_calls": stats.llm_calls,
            "llm_cluster_checks": stats.llm_checks,
            "languages": _language_counts(articles),
        },
    )
    if persist:
        store.save_digest(digest)
    stats.digest_id = digest.id
    log.info("digest %s: %d stories from %d articles",
             digest.id, len(ranked), digest.article_count)
    return DigestResult(digest=digest, stories=ranked)


def _language_counts(articles: list[Article]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for article in articles:
        key = article.language or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
