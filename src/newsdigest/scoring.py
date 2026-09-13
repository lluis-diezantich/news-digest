"""Story ranking.

The LLM contributes two signals -- `importance` and `relevance` -- and nothing
else. Everything here is deterministic and driven by config/preferences.yaml, so
the ordering of your digest is inspectable and reproducible:

    final_score = sum(weight * signal for each term in ranking.terms)
                  - excluded_penalty (if the story hits an excluded topic)

Dropping a term from `ranking.terms` removes that signal from the formula
entirely; adding an unknown one is a config error rather than a silent no-op.
"""

from __future__ import annotations

import math
from datetime import datetime

from .config import Preferences
from .models import Article, Story, hours_since, utcnow
from .text import normalize


def _matches(needle: str, haystack: str) -> bool:
    """Substring match both ways so "ai" hits "ai policy" and vice versa."""
    return needle in haystack or haystack in needle


def interest(story: Story, prefs: Preferences) -> float:
    """Best matching topic weight, plus a bonus per keyword hit."""
    if not prefs.topics and not prefs.keywords:
        return 1.0

    best = 1.0
    if prefs.topics:
        weights = [
            weight
            for topic in story.topics
            for name, weight in prefs.topics.items()
            if _matches(normalize(topic), name)
        ]
        best = max(weights) if weights else 1.0

    haystack = normalize(
        " ".join([story.headline, story.summary, " ".join(story.entities),
                  " ".join(story.topics)])
    )
    hits = sum(1 for keyword in prefs.keywords if keyword in haystack)
    return best + prefs.keyword_bonus * hits


def recency(
    story: Story, articles: list[Article], prefs: Preferences,
    *, now: datetime | None = None,
) -> float:
    """Exponential decay on the newest article, in [0, 1].

    Over a weekly window the half-life should be long -- with the default 72h,
    Monday's news still scores about 0.16 on Sunday rather than vanishing.
    """
    timestamps = [a.published_at or a.collected_at for a in articles]
    newest = max(timestamps) if timestamps else story.last_updated
    half_life = max(1.0, prefs.recency_half_life_hours)
    return math.exp(-math.log(2) * hours_since(newest, now=now or utcnow()) / half_life)


def corroboration(articles: list[Article]) -> float:
    """Saturating bonus for multi-publisher coverage, in [0, 1].

    Counts publishers, not feeds. Seven El País section feeds covering one story
    are one outlet's view of it, and treating them as seven would let a single
    publisher dominate the digest.
    """
    distinct = len({a.publisher for a in articles})
    return 0.0 if distinct <= 1 else min(1.0, math.log(distinct, 4))


def story_size(articles: list[Article]) -> float:
    """How much coverage there is, in [0, 1]. Saturates quickly."""
    return min(1.0, math.log(len(articles) + 1, 8))


def source_preference(articles: list[Article], prefs: Preferences) -> float:
    """Configured source weights, plus a bonus for explicitly preferred outlets."""
    if not articles:
        return 1.0
    mean_weight = sum(a.source_weight for a in articles) / len(articles)
    preferred = {normalize(name) for name in prefs.preferred_sources}
    if preferred and any(
        normalize(a.publisher) in preferred or normalize(a.source) in preferred
        for a in articles
    ):
        mean_weight += prefs.preferred_source_bonus
    return mean_weight


def is_excluded(story: Story, prefs: Preferences) -> bool:
    if not prefs.excluded_topics:
        return False
    excluded = set(prefs.excluded_topics)
    if excluded & {normalize(t) for t in story.topics}:
        return True
    haystack = normalize(f"{story.headline} {story.summary}")
    return any(term in haystack for term in excluded)


def signals(
    story: Story, articles: list[Article], prefs: Preferences,
    *, now: datetime | None = None,
) -> dict[str, float]:
    """Every ranking signal, whether or not the config uses it.

    Returned as a dict so the digest can show why a story ranked where it did.
    """
    return {
        "importance": story.importance,
        "relevance": story.relevance,
        "interest": interest(story, prefs),
        "recency": recency(story, articles, prefs, now=now),
        "corroboration": corroboration(articles),
        "source_preference": source_preference(articles, prefs),
        "story_size": story_size(articles),
    }


def explain(
    story: Story, articles: list[Article], prefs: Preferences,
    *, now: datetime | None = None,
) -> dict[str, float]:
    """Per-term contributions plus the total they add up to.

    This is the primitive; `score_story` reads `total` from it. Computing the
    score twice -- once for the number and once for the explanation -- let the two
    disagree, both from rounding and because `recency` reads the clock. Deriving
    the total from the same rounded terms it displays means the breakdown shown
    to a reader provably sums to the score used for ranking.
    """
    computed = signals(story, articles, prefs, now=now)
    breakdown = {
        name: round(weight * computed[name], 4)
        for name, weight in prefs.ranking.items()
    }
    if is_excluded(story, prefs):
        breakdown["excluded_penalty"] = -prefs.excluded_penalty
    breakdown["total"] = round(max(0.0, sum(breakdown.values())), 4)
    return breakdown


def score_story(
    story: Story, articles: list[Article], prefs: Preferences,
    *, now: datetime | None = None,
) -> float:
    return explain(story, articles, prefs, now=now)["total"]
