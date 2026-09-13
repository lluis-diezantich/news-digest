"""Offline provider: no API key, no network, no cost.

Used when LLM_PROVIDER=none (or no key is configured) so the whole pipeline can
be developed, tested and demoed end to end. The summaries are extractive rather
than written, which is exactly why the real providers exist -- but the feed still
builds, and every other stage behaves identically.
"""

from __future__ import annotations

import re
from functools import lru_cache

from ..text import capitalized_phrases, normalize, sentences, tokenize, truncate
from .base import Brief, BriefInput, Enrichment, EnrichInput, LLMProvider

# Cheap topic classification: first matching keyword set wins its topic.
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ai": ("ai", "artificial intelligence", "llm", "openai", "anthropic", "model", "chatbot"),
    "technology": ("software", "chip", "app", "startup", "google", "apple", "microsoft",
                   "hardware", "cyber", "data", "internet", "semiconductor"),
    "science": ("research", "study", "scientists", "nasa", "space", "physics", "genome"),
    "health": ("health", "disease", "vaccine", "hospital", "outbreak", "drug", "patients"),
    "climate": ("climate", "emissions", "warming", "drought", "wildfire", "flood", "carbon"),
    "economics": ("inflation", "economy", "market", "stocks", "gdp", "tariff", "trade",
                  "central bank", "interest rate", "recession"),
    "politics": ("election", "parliament", "senate", "president", "minister", "vote",
                 "coalition", "law", "bill", "court"),
    "world": ("war", "border", "ceasefire", "sanctions", "un", "nato", "refugee", "strike"),
    "sports": ("match", "league", "cup", "tournament", "coach", "olympic", "goal"),
    "culture": ("film", "album", "novel", "museum", "festival", "artist"),
}

# Words that suggest a genuinely consequential story.
HIGH_SIGNAL = (
    "killed", "dead", "war", "ceasefire", "invasion", "earthquake", "collapse",
    "resigns", "elected", "ruling", "sanctions", "recall", "outbreak", "breach",
    "record", "ban", "acquires", "bankruptcy", "verdict", "indicted",
)


@lru_cache(maxsize=None)
def _phrase_pattern(*phrases: str) -> re.Pattern[str]:
    """Word-boundary matcher.

    Substring matching is wrong here and quietly so: a bare "ai" hits
    "Ukr(ai)nian", "s(ai)d" and "tr(ai)n", which tagged half of one live run's
    world news as AI coverage.
    """
    # The (?:...) group is load-bearing: alternation binds looser than the
    # lookarounds, so without it the lookbehind would guard only the first
    # alternative and the lookahead only the last -- which let "air force"
    # match the keyword "ai".
    alternatives = "|".join(re.escape(p) for p in phrases)
    return re.compile(rf"(?<![a-z0-9])(?:{alternatives})(?![a-z0-9])")


TOPIC_PATTERNS = {
    topic: _phrase_pattern(*keywords) for topic, keywords in TOPIC_KEYWORDS.items()
}
HIGH_SIGNAL_PATTERN = _phrase_pattern(*HIGH_SIGNAL)


class HeuristicProvider(LLMProvider):
    """Deterministic stand-in for a real LLM. Same interface, no calls."""

    name = "none"

    def enrich(self, items: list[EnrichInput]) -> list[Enrichment]:
        return [self._one(item) for item in items]

    def _one(self, item: EnrichInput) -> Enrichment:
        text = f"{item.title}. {item.excerpt}"
        parts = sentences(item.excerpt)[:2]
        summary = truncate(" ".join(parts) or item.title, 280)

        haystack = normalize(text)
        topics = [
            topic for topic, pattern in TOPIC_PATTERNS.items() if pattern.search(haystack)
        ][:4]

        entities = capitalized_phrases(f"{item.title}. {item.excerpt}", limit=6)

        signal = len(set(HIGH_SIGNAL_PATTERN.findall(haystack)))
        importance = min(0.85, 0.35 + 0.12 * signal)

        # Sorted content tokens make the label stable across outlets covering
        # the same event, which is what the clustering stage keys on.
        label_tokens = sorted(set(tokenize(item.title)))[:5]

        return Enrichment(
            id=item.id,
            summary=summary,
            why_it_matters="",
            topics=topics or ["general"],
            entities=entities,
            importance=importance,
            event_label=" ".join(label_tokens),
        ).clamp()

    def write_brief(self, item: BriefInput) -> Brief | None:
        # Nothing to synthesize without a model; the pipeline falls back to the
        # highest-importance article's own headline and summary.
        return None
