"""Topic filtering (section 8).

The RSS version of this project filtered junk mostly by URL path: sport lives at
`/deportes/`, the lottery at `/loterias/`, advertorial at `/especials/`, and a
regex over the section path caught all three for free. Newsletters take that away
twice over -- the link is a tracker until it is resolved, and a newsletter's own
junk (the sponsor slot, the housekeeping block) has no path anywhere.

So filtering here is a cheap LLM pass over every article, before clustering:
topics, region, and one boolean for "is this news at all". Three properties make
it affordable and safe:

  * ONE batch call per `batch_size` articles, not one per article, and the answer
    is cached on the article's content hash, so iterating on the prompt costs
    nothing after the first run.
  * It runs at the ARTICLE level, so one sports item in a newsletter drops that
    item rather than the newsletter -- which section 8 asks for explicitly.
  * Failure keeps the article. An unclassified item is kept, a failed batch is
    kept, an exhausted quota keeps everything left. The filter has to actively
    ask for a drop, so the worst case is a noisy digest rather than an empty one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import FilterSettings, LLMSettings
from .llm.base import (
    Classification,
    ClassifyInput,
    Context,
    LLMError,
    LLMProvider,
    LLMQuotaError,
)
from .models import Article
from .regions import normalize_region
from .store import Store
from .text import normalize, truncate

log = logging.getLogger(__name__)

#: Excerpt sent per article. Shorter than enrichment's 900: this pass answers
#: "what kind of thing is this", which a headline and a sentence settle.
EXCERPT_CHARS = 400
#: Give up on the provider after this many consecutive batch failures, matching
#: `enrich.MAX_CONSECUTIVE_FAILURES`.
MAX_CONSECUTIVE_FAILURES = 2


@dataclass
class FilterReport:
    classified: int = 0
    cached: int = 0
    failed: int = 0
    calls: int = 0
    #: Dropped, by reason, so `--debug` can say which rule did it.
    dropped_not_news: list[Article] = field(default_factory=list)
    dropped_topic: list[Article] = field(default_factory=list)
    dropped_content_type: list[Article] = field(default_factory=list)
    quota_exhausted: bool = False

    @property
    def dropped(self) -> int:
        return (
            len(self.dropped_not_news)
            + len(self.dropped_topic)
            + len(self.dropped_content_type)
        )


def _to_input(article: Article) -> ClassifyInput:
    return ClassifyInput(
        id=article.id,
        title=article.title,
        language=article.language,
        excerpt=truncate(article.excerpt(), EXCERPT_CHARS),
    )


def _apply(article: Article, classification: Classification, provider: str) -> None:
    article.region = classification.region
    article.newsworthy = classification.newsworthy
    article.classified_by = provider
    if classification.topics and not article.topics:
        article.topics = list(classification.topics)
    if classification.content_type and not article.content_type:
        article.content_type = classification.content_type


def classify_articles(
    store: Store,
    llm: LLMProvider,
    settings: LLMSettings,
    filters: FilterSettings,
    context: Context,
    articles: list[Article],
    *,
    persist: bool = True,
) -> FilterReport:
    """Classify articles in place, returning what happened.

    Nothing is dropped here -- `apply_filters` does that -- because the two have
    different failure modes and it is worth being able to see the classification
    of an article that was then dropped.
    """
    report = FilterReport()
    if not articles or not filters.classify:
        return report

    budget = articles[: filters.max_items] if filters.max_items else articles
    if len(budget) < len(articles):
        log.info(
            "classifying %d of %d articles (filters.max_items)",
            len(budget), len(articles),
        )

    cache_key = llm.classify_cache_key(context)
    hashes = {a.id: a.content_hash() for a in budget}
    cached = store.cached_classifications(list(set(hashes.values())), cache_key)

    pending: list[Article] = []
    for article in budget:
        hit = cached.get(hashes[article.id])
        if hit is None:
            pending.append(article)
            continue
        _apply(article, hit, llm.name)
        report.cached += 1
        if persist:
            store.save_classification(article.id, hit, llm.name)

    failures = 0
    for batch in _chunks(pending, max(1, settings.batch_size)):
        if report.quota_exhausted or failures >= MAX_CONSECUTIVE_FAILURES:
            break
        try:
            results = llm.classify([_to_input(a) for a in batch], context)
            report.calls += 1
            failures = 0
        except LLMQuotaError as exc:
            log.warning("quota exhausted during classification (%s)", exc)
            report.quota_exhausted = True
            break
        except LLMError as exc:
            failures += 1
            report.failed += len(batch)
            log.warning(
                "classification batch failed (%s); %d/%d consecutive",
                exc, failures, MAX_CONSECUTIVE_FAILURES,
            )
            continue

        by_id = {r.id: r for r in results}
        for article in batch:
            result = by_id.get(article.id)
            if result is None:
                # Missing from the answer: keep, and try again next run.
                continue
            _apply(article, result, llm.name)
            report.classified += 1
            if persist:
                store.save_classification(article.id, result, llm.name)
                store.cache_classification(
                    hashes[article.id], cache_key, result
                )

    if failures >= MAX_CONSECUTIVE_FAILURES:
        log.warning(
            "classification abandoned after %d consecutive failures; "
            "%d articles keep their existing topics",
            failures, len(pending) - report.classified,
        )
    log.info(
        "classified %d (%d cached, %d failed) in %d calls",
        report.classified, report.cached, report.failed, report.calls,
    )
    return report


def apply_filters(
    articles: list[Article], filters: FilterSettings, report: FilterReport | None = None
) -> list[Article]:
    """Drop what does not belong in a global weekly digest.

    Three rules, all of which require a positive signal to drop:

      1. the classifier said this is not news;
      2. every topic it carries is excluded;
      3. its content type is one the config drops.

    Rule 2 requires EVERY topic to be excluded, not any. A story tagged
    `politics, sports` is a politician at a stadium opening or a state visit to a
    World Cup, and dropping it on the sports tag alone loses the political story
    -- which is the direction that cannot be noticed from the output.
    """
    report = report or FilterReport()
    excluded = {normalize(t) for t in filters.excluded_topics}
    dropped_types = {t.lower() for t in filters.drop_content_types}
    kept: list[Article] = []

    for article in articles:
        if article.newsworthy is False:
            report.dropped_not_news.append(article)
            continue

        topics = {normalize(t) for t in (article.topics or [])}
        if topics and excluded and topics <= excluded:
            report.dropped_topic.append(article)
            continue

        if dropped_types and (article.content_type or "").lower() in dropped_types:
            report.dropped_content_type.append(article)
            continue

        kept.append(article)

    if report.dropped:
        log.info(
            "filtered %d of %d articles (%d not news, %d excluded topic, %d type)",
            report.dropped, len(articles), len(report.dropped_not_news),
            len(report.dropped_topic), len(report.dropped_content_type),
        )
    return kept


def region_counts(articles: list[Article]) -> dict[str, int]:
    """region -> article count, for the run summary."""
    counts: dict[str, int] = {}
    for article in articles:
        key = normalize_region(article.region) or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]
