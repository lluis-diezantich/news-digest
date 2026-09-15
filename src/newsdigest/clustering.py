"""Group articles covering the same event into stories, across languages.

The primary signal is embedding cosine similarity, because it is the only one
that works across languages. Measured on one headline in three languages, token
overlap scores 0.00 (en/es) and 0.06 (en/ca) against a 0.60 merge threshold --
no threshold rescues that, which is why embeddings are not optional here.

Three tiers, cheapest first:

  1. cosine >= similarity_threshold        -> same event, free
  2. ambiguous_threshold <= cosine < above -> ask the LLM, capped per run
  3. no embeddings available               -> within-language text similarity,
                                              which leaves cross-language
                                              coverage split and says so

Nothing here calls an LLM per article; only the ambiguous band does, and only up
to `max_checks` pairs, highest similarity first.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np

from .embeddings.base import cosine_matrix
from .llm.base import (
    MAX_TOPICS,
    Context,
    LLMError,
    LLMProvider,
    PairInput,
    normalize_topics,
)
from .models import Article, Story, utcnow
from .text import article_similarity, truncate

log = logging.getLogger(__name__)

#: Threshold for the no-embeddings fallback, calibrated on 37,776 real
#: within-language pairs from the configured feeds. Genuine same-event pairs that
#: text can detect at all -- syndicated copy and near-identical headlines -- score
#: 0.49-0.78; the highest scoring unrelated pair reaches 0.34. 0.45 sits in that
#: gap. Reworded coverage of one event lands at 0.30-0.33, inside the noise, and
#: is therefore NOT recoverable by text at any threshold; that is what the
#: embeddings are for.
FALLBACK_TEXT_THRESHOLD = 0.45


class _UnionFind:
    def __init__(self, keys: list[str]):
        self.parent = {k: k for k in keys}

    def find(self, key: str) -> str:
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != root:  # path compression
            self.parent[key], key = root, self.parent[key]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _groups(uf: _UnionFind, articles: list[Article]) -> list[list[Article]]:
    grouped: dict[str, list[Article]] = {}
    for article in articles:
        grouped.setdefault(uf.find(article.id), []).append(article)
    return sorted(
        grouped.values(),
        key=lambda g: (len({a.publisher for a in g}), len(g)),
        reverse=True,
    )


def cluster(
    articles: list[Article],
    vectors: dict[str, np.ndarray] | None = None,
    *,
    similarity_threshold: float = 0.82,
    ambiguous_threshold: float = 0.72,
    provider: LLMProvider | None = None,
    context: Context | None = None,
    max_checks: int = 40,
    stats: object | None = None,
) -> list[list[Article]]:
    """Partition articles into stories, most-covered first."""
    if not articles:
        return []

    ordered = sorted(
        articles, key=lambda a: (a.published_at or a.collected_at), reverse=True
    )
    usable = [a for a in ordered if vectors and a.id in vectors]

    if len(usable) < 2:
        if vectors:
            log.info("only %d of %d articles have embeddings", len(usable), len(ordered))
        return _cluster_by_text(ordered)

    uf = _UnionFind([a.id for a in ordered])
    matrix = cosine_matrix([vectors[a.id] for a in usable])

    ambiguous: list[tuple[float, Article, Article]] = []
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            score = float(matrix[i, j])
            if score >= similarity_threshold:
                uf.union(usable[i].id, usable[j].id)
            elif score >= ambiguous_threshold:
                ambiguous.append((score, usable[i], usable[j]))

    if ambiguous and provider is not None and context is not None and max_checks > 0:
        _resolve_ambiguous(uf, ambiguous, provider, context, max_checks, stats)
    elif ambiguous:
        log.debug("%d ambiguous pairs left to the embedding's own verdict", len(ambiguous))

    # Articles with no vector still deserve a home: fall back to text similarity
    # against their own language only.
    missing = [a for a in ordered if not vectors or a.id not in vectors]
    if missing:
        _attach_by_text(uf, missing, ordered)

    return _groups(uf, ordered)


def _resolve_ambiguous(
    uf: _UnionFind,
    ambiguous: list[tuple[float, Article, Article]],
    provider: LLMProvider,
    context: Context,
    max_checks: int,
    stats: object | None,
) -> None:
    """Ask the LLM about the closest calls, highest similarity first."""
    ambiguous.sort(key=lambda item: -item[0])
    pairs: list[PairInput] = []
    lookup: dict[str, tuple[Article, Article]] = {}

    for score, left, right in ambiguous:
        if len(pairs) >= max_checks:
            break
        if uf.find(left.id) == uf.find(right.id):
            continue  # already merged transitively; no need to ask
        key = f"{left.id}:{right.id}"
        lookup[key] = (left, right)
        pairs.append(
            PairInput(
                key=key,
                left_title=left.title,
                left_language=left.language,
                left_excerpt=truncate(left.best_summary(), 400),
                right_title=right.title,
                right_language=right.language,
                right_excerpt=truncate(right.best_summary(), 400),
            )
        )

    if not pairs:
        return
    log.info("asking %s to adjudicate %d borderline pairs", provider.name, len(pairs))
    try:
        verdicts = provider.same_event(pairs, context)
    except LLMError as exc:
        log.warning("cluster adjudication failed (%s); keeping embedding verdicts", exc)
        return

    merged = 0
    for key, same in verdicts.items():
        if same and key in lookup:
            left, right = lookup[key]
            uf.union(left.id, right.id)
            merged += 1
    if stats is not None:
        stats.llm_checks = getattr(stats, "llm_checks", 0) + len(pairs)
    log.info("adjudication merged %d of %d pairs", merged, len(pairs))


def _cluster_by_text(articles: list[Article]) -> list[list[Article]]:
    """Fallback with no embeddings: compare only within a language.

    Cross-language text similarity is noise (0.00-0.06 for the same event), so
    comparing across languages would only invent false merges.
    """
    log.warning(
        "clustering without embeddings: coverage of one event in different "
        "languages will stay split into separate stories"
    )
    uf = _UnionFind([a.id for a in articles])
    by_language: dict[str | None, list[Article]] = {}
    for article in articles:
        by_language.setdefault(article.language, []).append(article)

    for group in by_language.values():
        for i, left in enumerate(group):
            for right in group[i + 1 :]:
                if uf.find(left.id) == uf.find(right.id):
                    continue
                if _text_same_event(left, right):
                    uf.union(left.id, right.id)
    return _groups(uf, articles)


def _attach_by_text(uf: _UnionFind, missing: list[Article], everyone: list[Article]) -> None:
    """Place vector-less articles using text similarity within their language."""
    for article in missing:
        for other in everyone:
            if other.id == article.id or other.language != article.language:
                continue
            if uf.find(other.id) == uf.find(article.id):
                continue
            if _text_same_event(article, other):
                uf.union(other.id, article.id)
                break


def _text_same_event(a: Article, b: Article) -> bool:
    return (
        article_similarity(a.title, a.best_summary(), b.title, b.best_summary())
        >= FALLBACK_TEXT_THRESHOLD
    )


def lead_article(articles: list[Article]) -> Article:
    """The article a single-source story borrows its headline and summary from.

    Prefers reporting over opinion: a comment piece is a poor lead for a story
    several outlets covered straight.
    """
    return max(
        articles,
        key=lambda a: (
            a.content_type != "opinion",
            a.importance or 0.0,
            a.source_weight,
            a.published_at or a.collected_at,
        ),
    )


def build_story(articles: list[Article]) -> Story:
    """Assemble a Story from a cluster.

    The id is derived from the cluster's members, so re-running a week produces
    the same story ids rather than duplicates.
    """
    lead = lead_article(articles)
    seed = "|".join(sorted(a.canonical for a in articles))

    topic_counts: Counter[str] = Counter()
    entity_counts: Counter[str] = Counter()
    facts: list[str] = []
    for article in articles:
        topic_counts.update(normalize_topics(article.topics or article.source_topics))
        entity_counts.update(article.entities)
        for fact in article.key_facts:
            if fact not in facts:
                facts.append(fact)

    publishers = sorted({a.publisher for a in articles})
    languages = sorted({a.language for a in articles if a.language})

    return Story(
        id=Story.make_id(seed),
        headline=lead.title,
        summary=lead.best_summary(),
        why_it_matters=lead.why_it_matters or "",
        key_facts=facts[:4],
        topics=[t for t, _ in topic_counts.most_common(MAX_TOPICS)],
        entities=[e for e, _ in entity_counts.most_common(10)],
        # A story is as important as its most important article, nudged up when
        # several publishers independently thought it worth covering.
        importance=min(
            1.0,
            max((a.importance or 0.0) for a in articles) + 0.03 * (len(publishers) - 1),
        ),
        relevance=max((a.relevance or 0.0) for a in articles),
        article_ids=[a.id for a in articles],
        publishers=publishers,
        languages=languages,
        first_seen=min((a.published_at or a.collected_at) for a in articles),
        last_updated=utcnow(),
    )


def cross_language_count(groups: list[list[Article]]) -> int:
    return sum(1 for g in groups if len({a.language for a in g if a.language}) > 1)
