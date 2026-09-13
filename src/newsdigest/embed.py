"""Embedding stage: vectors for the weekly window, cached by content hash.

Runs in the weekly pipeline only. Two articles with identical text share one
vector, and a re-run of the same week costs nothing, which is what keeps this
inside a free tier.

A quota error stops embedding for the run rather than failing it: clustering then
works from whatever vectors it has, plus within-language text similarity for the
rest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .embeddings.base import EmbeddingError, EmbeddingProvider, EmbeddingQuotaError
from .models import Article
from .store import Store
from .text import truncate

log = logging.getLogger(__name__)

#: Characters of each article sent to the embedder. Title plus a couple of
#: paragraphs is what distinguishes events; more mostly adds cost.
EMBED_CHARS = 700
#: Articles per provider call.
CHUNK = 50


@dataclass
class EmbedReport:
    embedded: int = 0
    cached: int = 0
    failed: int = 0
    calls: int = 0
    quota_exhausted: bool = False


def embed_articles(
    store: Store,
    provider: EmbeddingProvider,
    articles: list[Article],
) -> tuple[dict[str, np.ndarray], EmbedReport]:
    """Return {article_id: vector} for as many articles as we could embed."""
    report = EmbedReport()
    if not provider.available:
        log.warning(
            "no embedding provider (%s); clustering falls back to within-language "
            "text similarity", provider.name,
        )
        return {}, report
    if not articles:
        return {}, report

    cache_key = provider.cache_key()
    hashes = {a.id: a.content_hash() for a in articles}
    cached = store.cached_vectors(list(set(hashes.values())), cache_key)

    vectors: dict[str, np.ndarray] = {}
    pending: list[Article] = []
    for article in articles:
        hit = cached.get(hashes[article.id])
        if hit is not None and hit.size:
            vectors[article.id] = hit
            report.cached += 1
        else:
            pending.append(article)

    if report.cached:
        log.info("reused %d cached embeddings", report.cached)
    if not pending:
        return vectors, report

    log.info("embedding %d articles via %s", len(pending), provider.name)
    fresh: dict[str, np.ndarray] = {}
    for start in range(0, len(pending), CHUNK):
        batch = pending[start : start + CHUNK]
        texts = [truncate(a.embedding_text(), EMBED_CHARS) for a in batch]
        try:
            produced = provider.embed(texts)
        except EmbeddingQuotaError as exc:
            log.warning(
                "%s embedding quota exhausted (%s); %d articles left unembedded",
                provider.name, exc, len(pending) - start,
            )
            report.quota_exhausted = True
            break
        except EmbeddingError as exc:
            log.warning("embedding batch failed (%s)", exc)
            report.failed += len(batch)
            continue

        for article, vector in zip(batch, produced):
            vectors[article.id] = vector
            fresh[article.content_hash()] = vector
            report.embedded += 1

    if fresh:
        store.cache_vectors(fresh, cache_key)
    report.calls = provider.calls
    log.info(
        "embeddings: %d new, %d cached, %d failed, %d calls",
        report.embedded, report.cached, report.failed, report.calls,
    )
    return vectors, report
