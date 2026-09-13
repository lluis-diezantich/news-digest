"""Story ranking. Entirely deterministic -- the LLM only supplies `importance`.

    score = w_importance * importance
          + w_interest   * interest_match
          + w_recency    * recency_decay
          + w_corroborat * corroboration
          - mute_penalty (if muted)

Every weight lives in config/interests.yaml, so tuning the feed never means
editing code.
"""

from __future__ import annotations

import math

from .config import Interests
from .models import Article, Story, hours_since, utcnow
from .text import normalize


def _matches(needle: str, haystack: str) -> bool:
    """Substring match both ways so "ai" hits "ai policy" and vice versa."""
    return needle in haystack or haystack in needle


def interest_match(story: Story, interests: Interests) -> float:
    """Best topic weight for this story, plus a bonus per keyword hit."""
    if not interests.topics and not interests.keywords:
        return 1.0

    best = 1.0
    if interests.topics:
        weights = [
            weight
            for topic in story.topics
            for name, weight in interests.topics.items()
            if _matches(normalize(topic), name)
        ]
        best = max(weights) if weights else 1.0

    haystack = normalize(
        " ".join([story.headline, story.summary, " ".join(story.entities), " ".join(story.topics)])
    )
    hits = sum(1 for keyword in interests.keywords if keyword in haystack)
    return best + interests.keyword_bonus * hits


def is_muted(story: Story, interests: Interests) -> bool:
    if not interests.mute:
        return False
    haystack = normalize(f"{story.headline} {story.summary} {' '.join(story.topics)}")
    return any(term in haystack for term in interests.mute)


def recency(story: Story, articles: list[Article], interests: Interests) -> float:
    """Exponential decay on the newest article, in [0, 1]."""
    timestamps = [a.published_at or a.fetched_at for a in articles]
    newest = max(timestamps) if timestamps else story.last_updated
    age = hours_since(newest, now=utcnow())
    half_life = max(1.0, interests.recency_half_life_hours)
    return math.exp(-math.log(2) * age / half_life)


def corroboration(articles: list[Article]) -> float:
    """Saturating bonus for multi-source coverage, in [0, 1]."""
    distinct = len({a.source for a in articles})
    if distinct <= 1:
        return 0.0
    return min(1.0, math.log(distinct, 4))


def source_quality(articles: list[Article]) -> float:
    """Mean configured source weight, so trusted sources lift their stories."""
    if not articles:
        return 1.0
    return sum(a.source_weight for a in articles) / len(articles)


def score_story(story: Story, articles: list[Article], interests: Interests) -> float:
    value = (
        interests.weight_importance * story.importance
        + interests.weight_interest * interest_match(story, interests)
        + interests.weight_recency * recency(story, articles, interests)
        + interests.weight_corroboration * corroboration(articles)
    )
    value *= source_quality(articles)
    if is_muted(story, interests):
        value -= interests.mute_penalty
    return round(max(0.0, value), 4)
