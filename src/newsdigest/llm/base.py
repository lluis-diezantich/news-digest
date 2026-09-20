"""Provider-agnostic LLM contract.

Everything the pipeline needs from an LLM is behind this interface, so swapping
providers means adding one module and one registry entry.

The LLM is asked for language understanding only -- classify, summarize, extract,
rate, and adjudicate a borderline cluster. It never decides the final ranking;
`scoring.py` does that from a formula you control.

Four jobs, in the order the pipeline uses them:

  classify     cheap triage over every article: topics, region, is-this-news
  same_event   adjudicate a borderline cluster
  write_brief  merge one cluster into the entry a reader sees
  enrich       per-article detail, for ranking and the entity lists

The prompts for all four live in `prompts/`, not here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .. import prompts
from ..regions import REGIONS, normalize_region
from ..text import normalize

#: Article kinds the model may return.
CONTENT_TYPES = ("reporting", "analysis", "opinion", "other")

#: The controlled tag vocabulary. Asked for in the prompts and enforced on the
#: way back in, because "2-4 broad lowercase topics" with no list produced a
#: long tail nobody can filter by: `ai` alongside `artificial intelligence`,
#: `politics` alongside `spanish politics` and `us politics`, and 27 tags used
#: exactly once, most of them place names that belong in `entities`.
#:
#: Deliberately short and flat. These are filter chips on one page, not a
#: taxonomy -- a vocabulary large enough to be precise is too large to be a
#: useful filter. It is also the offline provider's contract (`heuristic.py`
#: keyword-matches exactly these), so both paths tag alike.
TOPICS = (
    "ai",
    "technology",
    "science",
    "health",
    "climate",
    "economics",
    "politics",
    "world",
    "sports",
    "culture",
    # Present so `excluded_topics: [celebrity]` has something to match. Closing
    # the vocabulary silently disarmed that: a topic that can never be emitted
    # can never be excluded, leaving only the weak headline-substring half --
    # and "celebrity" appears in no celebrity headline. Kept distinct from
    # `culture` on purpose, so excluding gossip does not exclude the arts.
    "celebrity",
)
_TOPIC_SET = frozenset(TOPICS)

#: Off-vocabulary labels worth keeping rather than dropping. Every entry here
#: was actually emitted by a model or declared by a feed. Anything absent is
#: dropped by `normalize_topics` -- including country and city names, and the
#: offline provider's old `general` placeholder.
TOPIC_ALIASES = {
    "artificial intelligence": "ai", "machine learning": "ai", "llm": "ai",
    "generative ai": "ai", "ia": "ai", "intelligencia artificial": "ai",
    "tech": "technology", "technology regulation": "technology",
    "software": "technology", "internet": "technology", "cyber": "technology",
    "cybersecurity": "technology", "semiconductors": "technology",
    "telecoms": "technology", "tecnologia": "technology",
    "research": "science", "space": "science", "physics": "science",
    "biology": "science", "ciencia": "science", "recerca": "science",
    "healthcare": "health", "medicine": "health", "public health": "health",
    "salud": "health", "salut": "health",
    "environment": "climate", "energy": "climate", "weather": "climate",
    "disaster": "climate", "natural disaster": "climate", "clima": "climate",
    "economy": "economics", "business": "economics", "trade": "economics",
    "markets": "economics", "finance": "economics", "inflation": "economics",
    "labour": "economics", "labor": "economics", "housing": "economics",
    "economia": "economics",
    "election": "politics", "elections": "politics", "polling": "politics",
    "voting rights": "politics", "us politics": "politics",
    "uk politics": "politics", "spanish politics": "politics",
    "catalan politics": "politics", "government": "politics",
    "judiciary": "politics", "law": "politics", "courts": "politics",
    "corruption": "politics", "constitutional reform": "politics",
    "self-determination": "politics", "political alliance": "politics",
    "referendum": "politics", "parliament": "politics", "policy": "politics",
    "geopolitics": "world", "diplomacy": "world", "international": "world",
    "international relations": "world", "international affairs": "world",
    "foreign policy": "world", "migration": "world", "immigration": "world",
    "human rights": "world", "war": "world", "conflict": "world",
    "military": "world", "defence": "world", "defense": "world",
    "security": "world", "terrorism": "world", "europe": "world",
    "football": "sports", "soccer": "sports", "tennis": "sports",
    "cycling": "sports", "basketball": "sports", "olympics": "sports",
    "motorsport": "sports", "formula 1": "sports", "futbol": "sports",
    "art": "culture", "arts": "culture", "film": "culture", "cinema": "culture",
    "music": "culture", "literature": "culture", "media": "culture",
    "entertainment": "culture", "cultura": "culture",
    "gossip": "celebrity", "celebrities": "celebrity", "showbiz": "celebrity",
    "famosos": "celebrity", "corazon": "celebrity",
}

#: Tags per article or story. Four is what the prompts ask for and what the
#: story card has room for.
MAX_TOPICS = 4


def normalize_topics(values: list[str] | None, limit: int = MAX_TOPICS) -> list[str]:
    """Map free-text labels onto `TOPICS`, dropping anything unrecognized.

    The dropping is the point: an unmapped label is either a synonym of one
    already here (splitting a filter in two) or an entity masquerading as a
    topic. Both are worse than no tag -- a story with no usable tag still falls
    back to its feed-declared topics in `clustering.py`.

    Runs through `text.normalize`, so an accented `economia` and a stray
    `Economics ` both land on the same key.
    """
    out: list[str] = []
    for value in values or []:
        key = normalize(str(value))
        if not key:
            continue
        topic = key if key in _TOPIC_SET else TOPIC_ALIASES.get(key)
        if topic and topic not in out:
            out.append(topic)
    return out[:limit]

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
        self.topics = normalize_topics(self.topics)
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
    #: Where the outlets differ, one sentence each. Section 15: these are kept
    #: apart from `summary` so a disagreement cannot be silently resolved into
    #: the digest's own voice.
    disagreements: list[str] = field(default_factory=list)
    region: str = ""
    importance: float | None = None
    relevance: float | None = None

    def clamp(self) -> "Brief":
        """A brief's topics overwrite the whole story's, so they matter most."""
        self.topics = normalize_topics(self.topics)
        self.region = normalize_region(self.region)
        self.disagreements = [
            d.strip() for d in self.disagreements if str(d).strip()
        ][:2]
        return self


@dataclass
class ClassifyInput:
    """One article for the cheap triage pass."""

    id: str
    title: str
    language: str | None
    excerpt: str


@dataclass
class Classification:
    """Triage verdict for one article.

    `newsworthy` defaults to True, and that default is load-bearing: this pass
    decides what is dropped before clustering, so a model that omits the field,
    or a batch that fails outright, must not silently empty the digest. Filtering
    is something the classifier has to actively ask for.
    """

    id: str
    topics: list[str] = field(default_factory=list)
    region: str = ""
    newsworthy: bool = True
    content_type: str | None = None

    def clamp(self) -> "Classification":
        self.topics = normalize_topics(self.topics)
        self.region = normalize_region(self.region)
        kind = (self.content_type or "").strip().lower()
        self.content_type = kind if kind in CONTENT_TYPES else None
        return self


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
    """Implement the three abstract methods to add a provider.

    `classify` is deliberately NOT abstract. Its default offers no opinion, which
    the pipeline reads as "keep everything" -- so a new provider that cannot
    triage degrades to no topic filtering rather than to an empty digest, and can
    be added without implementing four methods at once.
    """

    name: str = ""
    model: str = ""

    def __init__(self) -> None:
        self.calls = 0

    def cache_key(self, context: Context) -> str:
        return f"{self.name}:{self.model}:{context.cache_key_part()}"

    def classify_cache_key(self, context: Context) -> str:
        """Separate from `cache_key` so a prompt change to one pass does not
        invalidate the other's cache -- they are different questions about the
        same text and cost very different amounts to re-answer."""
        return f"{self.name}:{self.model}:classify:{context.cache_key_part()}"

    def classify(
        self, items: list["ClassifyInput"], context: Context
    ) -> list["Classification"]:
        """Triage a batch. One Classification per input `id`, or none at all."""
        return []

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


def _topic_list() -> str:
    return ", ".join(TOPICS)


def _region_list() -> str:
    return ", ".join(REGIONS)


def enrich_system_prompt(context: Context) -> str:
    return prompts.render(
        "enrich_article",
        output_language=language_name(context.output_language),
        topic_list=_topic_list(),
        interests=", ".join(context.interests) or "general news",
        excluded=", ".join(context.excluded_topics) or "none",
    )


def classify_system_prompt(context: Context) -> str:
    return prompts.render(
        "classify_story",
        topic_list=_topic_list(),
        region_list=_region_list(),
    )


def brief_system_prompt(context: Context) -> str:
    return prompts.render(
        "write_brief",
        output_language=language_name(context.output_language),
        topic_list=_topic_list(),
        region_list=_region_list(),
        interests=", ".join(context.interests) or "general news",
    )


def same_event_system_prompt(context: Context | None = None) -> str:
    return prompts.render("cluster_stories")
