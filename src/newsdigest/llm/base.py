"""Provider-agnostic LLM contract.

Everything the pipeline needs from an LLM is behind this interface, so swapping
providers means adding one module and one registry entry -- no changes to
fetching, deduping, clustering, scoring or rendering.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class LLMError(RuntimeError):
    """Any provider failure. The pipeline degrades instead of crashing."""


class LLMQuotaError(LLMError):
    """Rate limited or out of quota -- stop asking for the rest of this run."""


@dataclass
class EnrichInput:
    """What we send about one article. Metadata plus a short excerpt only."""

    id: str
    title: str
    source: str
    published: str | None
    excerpt: str


@dataclass
class Enrichment:
    """What we expect back for one article."""

    id: str
    summary: str
    why_it_matters: str = ""
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    importance: float = 0.5
    # Short lowercase label for the underlying event, e.g. "eu ai act vote".
    # Used as a clustering signal so two outlets describing one event agree.
    event_label: str = ""

    def clamp(self) -> "Enrichment":
        self.importance = min(1.0, max(0.0, float(self.importance or 0.0)))
        self.topics = [t.strip().lower() for t in self.topics if str(t).strip()][:6]
        self.entities = [e.strip() for e in self.entities if str(e).strip()][:10]
        self.event_label = (self.event_label or "").strip().lower()[:80]
        return self


@dataclass
class BriefInput:
    """A cluster to write a merged headline and summary for."""

    story_id: str
    headlines: list[str]
    sources: list[str]
    excerpts: list[str]


@dataclass
class Brief:
    headline: str
    summary: str
    why_it_matters: str = ""
    topics: list[str] = field(default_factory=list)


class LLMProvider(ABC):
    """Implement these two methods to add a provider."""

    name: str = ""

    def __init__(self) -> None:
        self.calls = 0

    @abstractmethod
    def enrich(self, items: list[EnrichInput]) -> list[Enrichment]:
        """Summarize/classify a batch. Return one Enrichment per input `id`.

        Partial results are acceptable -- missing ids are retried next run.
        Raise LLMQuotaError to tell the pipeline to stop calling this run.
        """

    @abstractmethod
    def write_brief(self, item: BriefInput) -> Brief | None:
        """Merge a multi-source cluster into one headline + summary."""


ENRICH_SYSTEM_PROMPT = """\
You are a news desk editor building one reader's personal digest.

For each numbered article you receive title, source, publication time and a
short excerpt. That excerpt is all you get -- never invent detail that is not
in it, and never speculate about what the full article says.

For every article return:
- summary: 1-2 neutral sentences, max 45 words, in your own words. Do not copy
  phrasing from the excerpt beyond unavoidable names and terms.
- why_it_matters: one short clause on the consequence or stakes. Empty string if
  the excerpt genuinely does not support one.
- topics: 2-4 broad lowercase topics (e.g. "technology", "ai", "climate",
  "politics", "economics", "science", "health", "sports", "culture").
- entities: the people, organizations, places and products the article is
  actually about. Canonical names, at most 6.
- importance: 0.0-1.0 for how consequential this is to a general audience.
  0.9+ major world event, 0.6 significant national/industry news, 0.3 routine
  coverage, 0.1 trivia. Be sparing with high scores.
- event_label: a 3-6 word lowercase label naming the underlying real-world
  event, chosen so two outlets covering the SAME event produce the SAME label.
  Use the durable nouns (actors, place, action), no dates, no outlet framing.

Return one object per input article, keeping the given id. No extra commentary.
"""

BRIEF_SYSTEM_PROMPT = """\
Several outlets covered one event. Write the digest entry for it.

- headline: neutral, factual, max 12 words. Not a copy of any single outlet's
  headline and not clickbait.
- summary: 2-3 sentences synthesizing what all the coverage agrees on. If the
  sources conflict on a material fact, say so plainly rather than picking one.
- why_it_matters: one sentence on the consequence.
- topics: 2-4 broad lowercase topics.

Use only what the excerpts support. Do not add background you were not given.
"""
