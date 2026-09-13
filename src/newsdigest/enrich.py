"""LLM enrichment stage, with the guards that keep it inside a free tier.

Four of them:
  * it runs weekly, never during daily collection;
  * only articles in candidate stories are sent, not the whole week;
  * a content-hash cache means text already processed is never sent again;
  * requests are batched, and a per-run article cap bounds a busy week.

Failure is never fatal. A quota error stops LLM work for the run, a transport
error skips one batch, and anything unenriched still reaches the digest with its
source's own description.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import LLMSettings
from .llm.base import (
    Brief,
    BriefInput,
    Context,
    EnrichInput,
    LLMError,
    LLMProvider,
    LLMQuotaError,
)
from .models import Article, Story, iso
from .store import Store
from .text import truncate

log = logging.getLogger(__name__)

#: Excerpt length sent per article.
EXCERPT_CHARS = 900
#: Give up on the provider after this many consecutive batch failures.
MAX_CONSECUTIVE_FAILURES = 2
#: Offline enrichment has no quota to protect, so the per-run cap does not apply.
OFFLINE_BUDGET = 5000


@dataclass
class EnrichReport:
    enriched: int = 0
    cached: int = 0
    failed: int = 0
    calls: int = 0
    quota_exhausted: bool = False


def _to_input(article: Article) -> EnrichInput:
    return EnrichInput(
        id=article.id,
        title=article.title,
        source=article.source,
        language=article.language,
        published=iso(article.published_at),
        excerpt=truncate(article.excerpt(), EXCERPT_CHARS),
    )


def enrich_articles(
    store: Store,
    provider: LLMProvider,
    settings: LLMSettings,
    context: Context,
    articles: list[Article],
    *,
    persist: bool = True,
) -> EnrichReport:
    """Enrich the given articles, skipping any already done.

    With `persist=False` the article rows are left alone but the cache is still
    filled, so a dry run costs API calls once and the real run that follows is
    free.
    """
    report = EnrichReport()
    pending = [a for a in articles if not a.enriched]
    if not pending:
        log.info("nothing to enrich")
        return report

    budget = OFFLINE_BUDGET if provider.name == "none" else settings.articles_per_run
    if budget <= 0:
        log.info("enrichment budget is 0, skipping")
        return report
    if len(pending) > budget:
        log.info("enrichment budget %d of %d candidates", budget, len(pending))
        pending = pending[:budget]

    cache_key = provider.cache_key(context)

    # 1. Cache pass -- free.
    hashes = {a.id: a.content_hash() for a in pending}
    cached = store.cached_enrichments(list(set(hashes.values())), cache_key)
    remaining: list[Article] = []
    for article in pending:
        hit = cached.get(hashes[article.id])
        if hit is None:
            remaining.append(article)
            continue
        hit.id = article.id
        if persist:
            store.save_enrichment(article.id, hit, f"{provider.name}:cache")
        _apply(article, hit)
        report.cached += 1

    if report.cached:
        log.info("reused %d cached enrichments", report.cached)
    if not remaining:
        return report

    # 2. Provider pass -- batched.
    log.info(
        "enriching %d articles via %s in batches of %d",
        len(remaining), provider.name, settings.batch_size,
    )
    consecutive_failures = 0
    for batch in _batches(remaining, settings.batch_size):
        try:
            results = provider.enrich([_to_input(a) for a in batch], context)
        except LLMQuotaError as exc:
            log.warning("%s quota exhausted (%s); leaving %d for next run",
                        provider.name, exc, len(batch))
            report.quota_exhausted = True
            break
        except LLMError as exc:
            consecutive_failures += 1
            report.failed += len(batch)
            log.warning("batch failed (%s)", exc)
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                log.error("%s failing repeatedly, stopping enrichment for this run",
                          provider.name)
                break
            continue

        consecutive_failures = 0
        by_id = {e.id: e for e in results}
        for article in batch:
            enrichment = by_id.get(article.id)
            if enrichment is None:
                report.failed += 1
                continue
            if persist:
                store.save_enrichment(article.id, enrichment, provider.name)
            store.cache_enrichment(article.content_hash(), cache_key, enrichment)
            _apply(article, enrichment)
            report.enriched += 1

    report.calls = provider.calls
    log.info(
        "enrichment: %d new, %d cached, %d failed, %d calls",
        report.enriched, report.cached, report.failed, report.calls,
    )
    return report


def _apply(article: Article, enrichment) -> None:
    """Mirror the stored enrichment onto the in-memory article.

    Clustering and scoring run on these objects in the same process, so without
    this they would see stale, unenriched copies.
    """
    article.summary = enrichment.summary
    article.why_it_matters = enrichment.why_it_matters
    article.key_facts = enrichment.key_facts
    article.topics = enrichment.topics
    article.entities = enrichment.entities
    article.importance = enrichment.importance
    article.relevance = enrichment.relevance
    article.content_type = enrichment.content_type


def write_brief(
    provider: LLMProvider,
    context: Context,
    story: Story,
    articles: list[Article],
) -> Brief | None:
    """Ask for a merged headline/summary for a story with several publishers.

    Single-publisher stories never get here -- they reuse the lead article's own
    enrichment, which costs nothing.
    """
    if len({a.publisher for a in articles}) < 2:
        return None
    ordered = sorted(articles, key=lambda a: -(a.importance or 0.0))[:5]
    try:
        return provider.write_brief(
            BriefInput(
                story_id=story.id,
                headlines=[a.title for a in ordered],
                sources=[a.source for a in ordered],
                languages=[a.language or "" for a in ordered],
                excerpts=[truncate(a.best_summary(), 500) for a in ordered],
            ),
            context,
        )
    except LLMQuotaError:
        raise
    except LLMError as exc:
        log.warning("brief for %s failed (%s); keeping the lead article's text",
                    story.id, exc)
        return None


def _batches(items: list, size: int):
    for start in range(0, len(items), max(1, size)):
        yield items[start : start + size]
