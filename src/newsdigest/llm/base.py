"""Provider-agnostic LLM contract.

Everything the pipeline needs from an LLM is behind this interface, so swapping
providers means adding one module and one registry entry.

The LLM is asked for language understanding only -- summarize, classify, extract,
rate, and adjudicate a borderline cluster. It never decides the final ranking;
`scoring.py` does that from a formula you control.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

#: Article kinds the model may return.
CONTENT_TYPES = ("reporting", "analysis", "opinion", "other")

LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "ca": "Catalan",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "da": "Danish",
}


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get((code or "").lower(), code or "English")


class LLMError(RuntimeError):
    """Any provider failure. The pipeline degrades instead of crashing."""


class LLMQuotaError(LLMError):
    """Rate limited or out of quota -- stop asking for the rest of this run."""


@dataclass
class Context:
    """What the provider needs to know about the reader, once per run."""

    output_language: str = "en"
    #: Topics the reader cares about, highest weight first. Drives `relevance`.
    interests: list[str] = field(default_factory=list)
    excluded_topics: list[str] = field(default_factory=list)

    def cache_key_part(self) -> str:
        """Enrichment depends on these, so they belong in the cache key."""
        return f"{self.output_language}|{','.join(self.interests)}"


@dataclass
class EnrichInput:
    """What we send about one article: metadata plus a short excerpt."""

    id: str
    title: str
    source: str
    language: str | None
    published: str | None
    excerpt: str


@dataclass
class Enrichment:
    """What we expect back for one article."""

    id: str
    summary: str
    why_it_matters: str = ""
    key_facts: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    importance: float = 0.5
    relevance: float = 0.5
    content_type: str | None = None

    def clamp(self) -> "Enrichment":
        self.importance = min(1.0, max(0.0, float(self.importance or 0.0)))
        self.relevance = min(1.0, max(0.0, float(self.relevance or 0.0)))
        self.topics = [t.strip().lower() for t in self.topics if str(t).strip()][:6]
        self.entities = [e.strip() for e in self.entities if str(e).strip()][:10]
        self.key_facts = [f.strip() for f in self.key_facts if str(f).strip()][:5]
        kind = (self.content_type or "").strip().lower()
        self.content_type = kind if kind in CONTENT_TYPES else None
        return self


@dataclass
class BriefInput:
    """A cluster to write one merged headline and summary for."""

    story_id: str
    headlines: list[str]
    sources: list[str]
    languages: list[str]
    excerpts: list[str]


@dataclass
class Brief:
    headline: str
    summary: str
    why_it_matters: str = ""
    key_facts: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    importance: float | None = None
    relevance: float | None = None


@dataclass
class PairInput:
    """Two articles whose embedding similarity was too close to call."""

    key: str
    left_title: str
    left_language: str | None
    left_excerpt: str
    right_title: str
    right_language: str | None
    right_excerpt: str


class LLMProvider(ABC):
    """Implement these three methods to add a provider."""

    name: str = ""
    model: str = ""

    def __init__(self) -> None:
        self.calls = 0

    def cache_key(self, context: Context) -> str:
        return f"{self.name}:{self.model}:{context.cache_key_part()}"

    @abstractmethod
    def enrich(self, items: list[EnrichInput], context: Context) -> list[Enrichment]:
        """Summarize/classify a batch. One Enrichment per input `id`.

        Partial results are fine -- missing ids are retried next run. Raise
        LLMQuotaError to stop LLM work for this run.
        """

    @abstractmethod
    def write_brief(self, item: BriefInput, context: Context) -> Brief | None:
        """Merge a multi-source cluster into one headline and summary."""

    @abstractmethod
    def same_event(self, pairs: list[PairInput], context: Context) -> dict[str, bool]:
        """Adjudicate borderline clusters. Maps each pair `key` to a verdict.

        Keys may be omitted when the model is unsure; the caller then keeps the
        embedding's own decision.
        """


def enrich_system_prompt(context: Context) -> str:
    interests = ", ".join(context.interests) or "general news"
    excluded = ", ".join(context.excluded_topics) or "none"
    return f"""\
You are a news desk editor building one reader's personal weekly digest.

Articles arrive in English, Spanish or Catalan. Read them in their original
language. Write EVERY piece of output in {language_name(context.output_language)},
whatever language the article was written in. Never translate the article's
title -- that is preserved separately and shown as published.

For each article you get title, source, language, publication time and a short
excerpt. That excerpt is all you get: never invent detail beyond it, and never
speculate about what the rest of the article says.

Return for every article:
- summary: 1-2 neutral sentences, max 45 words, in your own words.
- why_it_matters: one short clause on the consequence or stakes. Empty string if
  the excerpt does not support one.
- key_facts: up to 3 short factual statements the excerpt actually contains
  (numbers, dates, names, decisions). No inference.
- topics: 2-4 broad lowercase topics.
- entities: the people, organizations, places and products the article is about.
  Canonical names, at most 6. Keep names in their original form.
- importance: 0.0-1.0, how consequential to a general audience. 0.9+ major world
  event, 0.6 significant national or industry news, 0.3 routine, 0.1 trivia. Be
  sparing with high scores.
- relevance: 0.0-1.0, how well this matches THIS reader's interests, which are:
  {interests}. Score low for topics they exclude: {excluded}. Relevance is
  independent of importance -- a major story outside their interests is high
  importance and low relevance.
- content_type: one of reporting, analysis, opinion, other.

Return one object per input article, keeping the given id. No commentary."""


def brief_system_prompt(context: Context) -> str:
    return f"""\
Several outlets covered one event, possibly in different languages. Write the
single digest entry for it, in {language_name(context.output_language)}.

- headline: neutral, factual, max 12 words. Not a copy of any one outlet's
  headline, and not clickbait.
- summary: 2-3 sentences synthesizing what the coverage agrees on. If sources
  conflict on a material fact, say so plainly rather than picking a side.
- why_it_matters: one sentence on the consequence.
- key_facts: up to 4 short factual statements supported by the excerpts.
- topics: 2-4 broad lowercase topics.
- importance: 0.0-1.0 for a general audience.
- relevance: 0.0-1.0 for a reader interested in {', '.join(context.interests) or 'general news'}.

Use only what the excerpts support. Do not add background you were not given.
Where outlets in different languages describe the same fact, state it once."""


SAME_EVENT_SYSTEM_PROMPT = """\
You decide whether two news articles report THE SAME underlying real-world
event. They may be in different languages; that is irrelevant to the judgement.

Same event: both describe one occurrence, decision, announcement or incident,
even with different framing, detail or language.
Not the same event: same topic, organisation or people but different
occurrences; a follow-up or reaction piece about a different development; a
round-up covering many events.

For each numbered pair return its key and a boolean `same_event`. If you
genuinely cannot tell from the excerpts, omit that pair from your answer."""
