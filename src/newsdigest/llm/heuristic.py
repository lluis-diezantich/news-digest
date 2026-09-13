"""Offline provider: no API key, no network, no cost.

Used when LLM_PROVIDER=none (or no key is configured) and by the tests, so the
whole two-pipeline architecture can be exercised end to end for free. Summaries
are extractive rather than written, which is exactly why the real providers
exist -- but every other stage behaves identically.

It cannot translate, so with `output_language: en` and a Spanish source the
summary stays Spanish. That is a visible, documented degradation rather than a
silent one.
"""

from __future__ import annotations

import re
from functools import lru_cache

from ..text import capitalized_phrases, normalize, sentences, truncate
from .base import (
    Brief,
    BriefInput,
    Context,
    Enrichment,
    EnrichInput,
    LLMProvider,
    PairInput,
)

# Cheap multilingual topic classification. Keywords are given in all three
# supported languages so a Catalan article is not simply untagged.
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ai": ("ai", "artificial intelligence", "llm", "openai", "anthropic", "chatbot",
           "inteligencia artificial", "intel·ligència artificial", "ia"),
    "technology": ("software", "chip", "app", "startup", "google", "apple", "microsoft",
                   "hardware", "cyber", "internet", "semiconductor", "tecnología",
                   "tecnologia", "programari", "xip"),
    "science": ("research", "study", "scientists", "nasa", "space", "physics", "genome",
                "investigación", "estudio", "científicos", "recerca", "científics"),
    "health": ("health", "disease", "vaccine", "hospital", "outbreak", "drug", "patients",
               "salud", "enfermedad", "vacuna", "sanidad", "salut", "malaltia", "vacuna"),
    "climate": ("climate", "emissions", "warming", "drought", "wildfire", "flood", "carbon",
                "clima", "emisiones", "sequía", "incendio", "inundación",
                "sequera", "incendi", "inundació", "canvi climàtic"),
    "economics": ("inflation", "economy", "market", "stocks", "gdp", "tariff", "trade",
                  "central bank", "interest rate", "recession", "inflación", "economía",
                  "mercado", "aranceles", "economia", "mercat", "atur", "paro"),
    "politics": ("election", "parliament", "senate", "president", "minister", "vote",
                 "coalition", "law", "bill", "court", "elecciones", "parlamento",
                 "gobierno", "ministro", "ley", "eleccions", "govern", "parlament",
                 "conseller", "llei", "generalitat"),
    "world": ("war", "border", "ceasefire", "sanctions", "nato", "refugee", "strike",
              "guerra", "frontera", "alto el fuego", "sanciones", "refugiado",
              "frontera", "sancions", "refugiat", "otan"),
    "sports": ("match", "league", "cup", "tournament", "coach", "olympic", "goal",
               "partido", "liga", "copa", "entrenador", "gol", "partit", "lliga",
               "barça", "madrid"),
    "culture": ("film", "album", "novel", "museum", "festival", "artist",
                "película", "novela", "museo", "artista", "pel·lícula", "novel·la"),
}

HIGH_SIGNAL = (
    "killed", "dead", "war", "ceasefire", "invasion", "earthquake", "collapse",
    "resigns", "elected", "ruling", "sanctions", "recall", "outbreak", "breach",
    "record", "ban", "acquires", "bankruptcy", "verdict", "indicted",
    "muertos", "muerto", "guerra", "terremoto", "dimite", "elegido", "sentencia",
    "morts", "mort", "terratrèmol", "dimiteix", "elegit", "sentència",
)


@lru_cache(maxsize=None)
def _phrase_pattern(*phrases: str) -> re.Pattern[str]:
    """Word-boundary matcher over `normalize`d text.

    Three things here are load-bearing, each having been a real bug:

    * Word boundaries, not substrings. A bare "ai" hits "Ukr(ai)nian", "s(ai)d"
      and "tr(ai)n", which once tagged a drone attack as AI coverage.
    * The (?:...) group. Alternation binds looser than the lookarounds, so
      without it the lookbehind guards only the first alternative and the
      lookahead only the last -- letting "air force" match "ai".
    * Normalizing the keywords themselves. Haystacks pass through `normalize`,
      which strips accents, so a literal "intel\u00b7lig\u00e8ncia artificial" could
      never match the "intel\u00b7ligencia artificial" it becomes.
    """
    cleaned = [normalize(phrase) for phrase in phrases if normalize(phrase)]
    if not cleaned:
        return re.compile(r"(?!)")  # matches nothing
    alternatives = "|".join(re.escape(phrase) for phrase in cleaned)
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)")


TOPIC_PATTERNS = {
    topic: _phrase_pattern(*keywords) for topic, keywords in TOPIC_KEYWORDS.items()
}
HIGH_SIGNAL_PATTERN = _phrase_pattern(*HIGH_SIGNAL)


class HeuristicProvider(LLMProvider):
    """Deterministic stand-in for a real LLM. Same interface, no calls."""

    name = "none"
    model = "heuristic"

    def enrich(self, items: list[EnrichInput], context: Context) -> list[Enrichment]:
        return [self._one(item, context) for item in items]

    def _one(self, item: EnrichInput, context: Context) -> Enrichment:
        text = f"{item.title}. {item.excerpt}"
        haystack = normalize(text)

        parts = sentences(item.excerpt)[:2]
        summary = truncate(" ".join(parts) or item.title, 280)

        topics = [
            topic for topic, pattern in TOPIC_PATTERNS.items() if pattern.search(haystack)
        ][:4]

        signal = len(set(HIGH_SIGNAL_PATTERN.findall(haystack)))
        importance = min(0.85, 0.35 + 0.12 * signal)

        # Relevance: how many of the reader's interests this appears to touch.
        # Matched on word boundaries for the same reason as topics -- a plain
        # substring test for the interest "ai" matches "Ukraine".
        interests = tuple(context.interests)
        hits = len(set(_phrase_pattern(*interests).findall(haystack))) if interests else 0
        hits += sum(1 for topic in topics if topic in {normalize(i) for i in interests})
        relevance = min(0.9, 0.3 + 0.2 * hits)

        excluded = tuple(context.excluded_topics)
        if excluded and (
            _phrase_pattern(*excluded).search(haystack)
            # A story classified into an excluded topic counts even when the
            # word itself never appears -- "sports" is absent from every
            # Catalan football headline.
            or {normalize(x) for x in excluded} & set(topics)
        ):
            relevance = min(relevance, 0.15)

        return Enrichment(
            id=item.id,
            summary=summary,
            why_it_matters="",
            key_facts=[],
            topics=topics or ["general"],
            entities=capitalized_phrases(text, limit=6),
            importance=importance,
            relevance=relevance,
            content_type=None,
        ).clamp()

    def write_brief(self, item: BriefInput, context: Context) -> Brief | None:
        # Nothing to synthesize without a model; the caller falls back to the
        # highest-importance article's own headline and summary.
        return None

    def same_event(self, pairs: list[PairInput], context: Context) -> dict[str, bool]:
        # No opinion offered, so the embedding's own decision stands.
        return {}
