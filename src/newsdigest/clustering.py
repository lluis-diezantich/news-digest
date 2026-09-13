"""Group articles covering the same event into stories.

Two cheap deterministic signals do the work, both derived from data we already
have -- no extra LLM calls:

  1. `event_label`, the short event name the enrichment step asked the LLM for.
     Two outlets describing one event tend to produce the same label, which is
     the single strongest signal available.
  2. Text similarity over headline + summary, with a lower threshold when the
     articles also share named entities.

Signal (1) is doing most of the work, and that is not an accident. Measured on
real headlines, text similarity reliably catches near-verbatim republication
(wire copy runs 0.7+) but *cannot* separate genuinely reworded coverage of one
event ("Parliament approves the budget" vs "Budget clears its final vote", 0.10)
from two unrelated stories about the same organisation (0.07-0.09). No threshold
fixes that overlap, which is exactly why the enrichment step asks the LLM for an
`event_label`. Without a working LLM, expect reworded coverage to stay split --
degraded, but never wrong.

Clusters are then matched against recent stories by keyword overlap so a story
that gains coverage tomorrow keeps its identity instead of appearing twice.
"""

from __future__ import annotations

import logging
from collections import Counter

from .models import Article, Story, utcnow
from .text import jaccard, similarity, tokenize

log = logging.getLogger(__name__)

# Headline+summary similarity that alone implies "same event".
TEXT_THRESHOLD = 0.60
# Lower bar when the two articles also share >= ENTITY_OVERLAP entities.
TEXT_THRESHOLD_WITH_ENTITIES = 0.38
ENTITY_OVERLAP = 2
# Keyword overlap needed to attach a new cluster to an existing story.
STORY_MATCH_THRESHOLD = 0.45
# Keep comparisons cheap: only compare articles within this many hours.
WINDOW_HOURS = 48.0


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


def _text_of(article: Article) -> str:
    return f"{article.title}. {article.best_summary()}"


def _entities_of(article: Article) -> set[str]:
    return {e.lower() for e in article.entities if len(e) > 2}


def same_event(a: Article, b: Article) -> bool:
    """Whether two articles describe the same real-world event."""
    if a.event_label and a.event_label == b.event_label:
        return True

    score = similarity(_text_of(a), _text_of(b))
    if score >= TEXT_THRESHOLD:
        return True
    if score >= TEXT_THRESHOLD_WITH_ENTITIES:
        shared = _entities_of(a) & _entities_of(b)
        if len(shared) >= ENTITY_OVERLAP:
            return True
    # Distinct labels that overlap heavily still point at one event.
    if a.event_label and b.event_label:
        return similarity(a.event_label, b.event_label) >= 0.7
    return False


def cluster(articles: list[Article]) -> list[list[Article]]:
    """Partition articles into clusters, largest first."""
    if not articles:
        return []

    ordered = sorted(
        articles,
        key=lambda a: (a.published_at or a.fetched_at),
        reverse=True,
    )
    uf = _UnionFind([a.id for a in ordered])

    # O(n^2) over a 48h window is a few thousand comparisons -- fine, and much
    # easier to reason about than an approximate index.
    for i, left in enumerate(ordered):
        for right in ordered[i + 1 :]:
            if uf.find(left.id) == uf.find(right.id):
                continue
            if same_event(left, right):
                uf.union(left.id, right.id)

    groups: dict[str, list[Article]] = {}
    for article in ordered:
        groups.setdefault(uf.find(article.id), []).append(article)

    return sorted(
        groups.values(),
        key=lambda g: (len({a.source for a in g}), len(g)),
        reverse=True,
    )


def cluster_keywords(articles: list[Article], limit: int = 12) -> list[str]:
    """Stable fingerprint for a cluster, used to match it to a stored story."""
    counter: Counter[str] = Counter()
    for article in articles:
        counter.update(tokenize(article.title))
        counter.update(tokenize(article.event_label or ""))
        counter.update(e.lower() for e in article.entities)
    return [word for word, _ in counter.most_common(limit)]


def match_existing_story(
    keywords: list[str],
    articles: list[Article],
    stories: list[Story],
) -> Story | None:
    """Find the stored story this cluster is a continuation of, if any."""
    # An article already assigned to a story is the most reliable link.
    assigned = Counter(a.story_id for a in articles if a.story_id)
    by_id = {s.id: s for s in stories}
    for story_id, _ in assigned.most_common():
        if story_id in by_id:
            return by_id[story_id]

    if not keywords:
        return None
    best: tuple[float, Story | None] = (0.0, None)
    for story in stories:
        score = jaccard(set(keywords), set(story.keywords))
        if score > best[0]:
            best = (score, story)
    return best[1] if best[0] >= STORY_MATCH_THRESHOLD else None


def lead_article(articles: list[Article]) -> Article:
    """The article a single-source story borrows its headline and summary from."""
    return max(
        articles,
        key=lambda a: (
            a.importance or 0.0,
            a.source_weight,
            a.published_at or a.fetched_at,
        ),
    )


def build_story(
    articles: list[Article],
    *,
    existing: Story | None = None,
) -> Story:
    """Assemble a Story from a cluster, reusing an existing identity if given."""
    lead = lead_article(articles)
    keywords = cluster_keywords(articles)

    topic_counts: Counter[str] = Counter()
    entity_counts: Counter[str] = Counter()
    for article in articles:
        topic_counts.update(article.topics or article.source_topics)
        entity_counts.update(article.entities)

    story_id = existing.id if existing else Story.make_id(
        lead.event_label or " ".join(keywords[:6]) or lead.canonical
    )

    return Story(
        id=story_id,
        headline=lead.title,
        summary=lead.best_summary(),
        why_it_matters=lead.why_it_matters or "",
        topics=[t for t, _ in topic_counts.most_common(4)],
        entities=[e for e, _ in entity_counts.most_common(10)],
        # A story is as important as its most important article, nudged up when
        # several outlets independently thought it worth covering.
        importance=min(
            1.0,
            max((a.importance or 0.0) for a in articles)
            + 0.03 * (len({a.source for a in articles}) - 1),
        ),
        article_ids=[a.id for a in articles],
        keywords=keywords,
        first_seen=existing.first_seen if existing else utcnow(),
        last_updated=utcnow(),
        written_by=None,
    )
