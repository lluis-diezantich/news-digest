"""Duplicate removal.

Deliberately narrow: this stage removes articles that are *the same article*,
and nothing else. Two outlets covering one event are not duplicates -- they are
the corroboration the digest is built on, and clustering handles them. Collapsing
them here would throw away the "sources covering this story" list.

An article is a duplicate when either holds:
  * the canonical URL matches one we already have (same id), or
  * the same source published a near-identical headline recently -- live blogs,
    re-runs and feed churn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .models import Article
from .text import similarity, title_key

log = logging.getLogger(__name__)

# Same-source headline similarity above which we call it a re-run. Measured on
# the shipped sources, same-source re-runs and live-blog updates land at
# 0.60-0.78 while genuinely different articles from one outlet top out around
# 0.24, so 0.55 sits in the gap with margin on both sides.
SAME_SOURCE_THRESHOLD = 0.55
# How far back to look for same-source re-runs.
LOOKBACK_HOURS = 72.0


@dataclass
class Dropped:
    article: Article
    reason: str


def dedupe(
    articles: list[Article],
    *,
    known_ids: set[str] | None = None,
    known_fingerprints: list[tuple[str, str, str]] | None = None,
) -> tuple[list[Article], list[Dropped]]:
    """Split a freshly fetched batch into (keep, dropped).

    `known_fingerprints` is (id, title_key, source) for stored articles inside
    the lookback window, as returned by `Store.recent_fingerprints`.
    """
    known_ids = known_ids or set()
    stored_by_source: dict[str, list[str]] = {}
    for _id, key, source in known_fingerprints or []:
        stored_by_source.setdefault(source, []).append(key)

    keep: list[Article] = []
    dropped: list[Dropped] = []
    # Prefer the earliest-published copy when a batch carries several.
    ordered = sorted(
        articles,
        key=lambda a: (a.published_at is None, a.published_at or a.collected_at),
    )
    batch_ids: set[str] = set()
    batch_by_source: dict[str, list[tuple[str, str]]] = {}

    for article in ordered:
        if not article.title or not article.canonical:
            dropped.append(Dropped(article, "incomplete"))
            continue

        if article.id in known_ids:
            dropped.append(Dropped(article, "already stored"))
            continue
        if article.id in batch_ids:
            dropped.append(Dropped(article, "duplicate url in batch"))
            continue

        key = title_key(article.title)
        if key and key in stored_by_source.get(article.source, []):
            dropped.append(Dropped(article, "same source, stored headline"))
            continue

        rerun = next(
            (
                title
                for prev_key, title in batch_by_source.get(article.source, [])
                if prev_key == key
                or similarity(article.title, title) >= SAME_SOURCE_THRESHOLD
            ),
            None,
        )
        if rerun is not None:
            dropped.append(Dropped(article, f"same source re-run of {rerun!r}"))
            continue

        batch_ids.add(article.id)
        batch_by_source.setdefault(article.source, []).append((key, article.title))
        keep.append(article)

    if dropped:
        log.debug("dedupe dropped %d of %d", len(dropped), len(articles))
    return keep, dropped
