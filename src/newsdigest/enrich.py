"""LLM enrichment stage, with the guards that keep it inside a free tier.

Three of them:
  * a content-hash cache, so text we have already processed is never sent again;
  * batching, so N articles cost one request instead of N;
  * a per-run article cap, so a busy news day cannot drain a daily quota.

Failure is never fatal. A quota error stops LLM work for the run, a transport
error skips one batch, and anything left unenriched is simply picked up next
run -- the feed still builds from what did succeed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import LLMSettings
from .llm.base import (
    Brief,
    BriefInput,
    Enrichment,
    EnrichInput,
    LLMError,
    LLMProvider,
    LLMQuotaError,
)
from .models import Article, Story, iso
from .store import Store
from .text import truncate

log = logging.getLogger(__name__)

# Excerpt length sent per article. Enough to summarize from, small enough that a
# batch of 8 stays comfortably inside one request.
EXCERPT_CHARS = 900
# Give up on the provider after this many consecutive batch failures.
MAX_CONSECUTIVE_FAILURES = 2
# The per-run cap exists to protect an API quota. Offline enrichment has none,
# so capping it there just leaves articles showing raw feed categories instead
# of real topics.
OFFLINE_BUDGET = 2000


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
        published=iso(article.published_at),
        excerpt=truncate(article.description, EXCERPT_CHARS),
    )


def enrich_pending(
    store: Store,
    provider: LLMProvider,
    settings: LLMSettings,
    *,
    limit: int | None = None,
) -> EnrichReport:
    """Enrich stored articles that have no enrichment yet."""
    if limit is not None:
        budget = limit
    elif provider.name == "none":
        budget = max(settings.articles_per_run, OFFLINE_BUDGET)
    else:
        budget = settings.articles_per_run
    report = EnrichReport()
    if budget <= 0:
        log.info("enrichment budget is 0, skipping")
        return report

    pending = store.articles_needing_enrichment(budget)
    if not pending:
        log.info("nothing to enrich")
        return report

    # 1. Cache pass -- free.
    hashes = {article.id: article.content_hash() for article in pending}
    cached = store.cached_enrichments(list(hashes.values()))
    remaining: list[Article] = []
    for article in pending:
        hit = cached.get(hashes[article.id])
        if hit is None:
            remaining.append(article)
            continue
        hit.id = article.id
        store.save_enrichment(article.id, hit, f"{provider.name}:cache")
        report.cached += 1

    if report.cached:
        log.info("reused %d cached enrichments", report.cached)
    if not remaining:
        return report

    # 2. Provider pass -- batched.
    log.info(
        "enriching %d articles via %s in batches of %d",
        len(remaining),
        provider.name,
        settings.batch_size,
    )
    consecutive_failures = 0
    for batch in _batches(remaining, settings.batch_size):
        try:
            results = provider.enrich([_to_input(a) for a in batch])
        except LLMQuotaError as exc:
            log.warning("%s quota exhausted (%s); leaving %d for the next run",
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
                # The provider skipped it; try again next run rather than
                # writing a placeholder we would never revisit.
                report.failed += 1
                continue
            store.save_enrichment(article.id, enrichment, provider.name)
            store.cache_enrichment(article.content_hash(), provider.name, enrichment)
            report.enriched += 1

    report.calls = provider.calls
    log.info(
        "enrichment done: %d new, %d cached, %d failed, %d calls",
        report.enriched, report.cached, report.failed, report.calls,
    )
    return report


def write_brief(
    provider: LLMProvider,
    story: Story,
    articles: list[Article],
) -> Brief | None:
    """Ask the provider for a merged headline/summary for a multi-source story.

    Single-source stories never get here -- they reuse the article's own
    enrichment, which costs nothing.
    """
    if len({a.source for a in articles}) < 2:
        return None
    ordered = sorted(articles, key=lambda a: -(a.importance or 0.0))[:5]
    try:
        return provider.write_brief(
            BriefInput(
                story_id=story.id,
                headlines=[a.title for a in ordered],
                sources=[a.source for a in ordered],
                excerpts=[truncate(a.best_summary(), 500) for a in ordered],
            )
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
