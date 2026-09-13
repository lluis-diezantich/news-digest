"""The common schema every source normalizes into, plus stories and digests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

from dateutil import parser as date_parser

from .urls import canonical_url, domain

#: What the LLM may classify an article as.
CONTENT_TYPES = ("reporting", "analysis", "opinion", "other")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def iso(value: datetime | None) -> str | None:
    return to_utc(value).isoformat(timespec="seconds") if value else None


def parse_date(value: Any) -> datetime | None:
    """Best-effort date parsing across the many shapes feeds use."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return to_utc(value)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    try:
        return to_utc(date_parser.parse(str(value)))
    except (ValueError, OverflowError, TypeError):
        return None


def hours_since(value: datetime | None, *, now: datetime | None = None) -> float:
    if value is None:
        return float("inf")
    return max(0.0, ((now or utcnow()) - to_utc(value)).total_seconds() / 3600.0)


@dataclass
class Article:
    """One normalized article. `id` is derived, so sources never invent one."""

    title: str
    source: str
    url: str
    published_at: datetime | None = None
    author: str | None = None
    #: Feed summary, excerpt only -- never a full article body.
    description: str = ""
    #: Longer excerpt for scraped pages (opening paragraphs), also capped.
    content: str = ""
    #: Language code, filled in during collection. None means undetermined.
    language: str | None = None
    #: Outlet this feed belongs to; several feeds may share one.
    publisher: str = ""
    #: The feed or index URL this came from.
    source_url: str = ""
    source_topics: list[str] = field(default_factory=list)
    source_weight: float = 1.0
    collected_at: datetime = field(default_factory=utcnow)

    # --- derived ---
    canonical: str = ""
    id: str = ""

    # --- LLM enrichment ---
    summary: str | None = None
    why_it_matters: str | None = None
    key_facts: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    importance: float | None = None
    relevance: float | None = None
    content_type: str | None = None
    enriched_by: str | None = None
    enriched_at: datetime | None = None

    story_id: str | None = None

    def __post_init__(self) -> None:
        self.title = (self.title or "").strip()
        self.canonical = canonical_url(self.url)
        if not self.publisher:
            self.publisher = self.source
        if not self.id:
            self.id = hashlib.sha1(self.canonical.encode("utf-8")).hexdigest()[:16]

    @property
    def enriched(self) -> bool:
        return self.summary is not None

    @property
    def site(self) -> str:
        return domain(self.url)

    def excerpt(self) -> str:
        """The text we send to the LLM and the embedder."""
        return self.content if len(self.content) > len(self.description) else self.description

    def content_hash(self) -> str:
        """Keys the LLM and embedding caches: same text in, no new API call."""
        payload = f"{self.title}\x00{self.excerpt()}".encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:16]

    def best_summary(self) -> str:
        return self.summary or self.excerpt()

    def embedding_text(self) -> str:
        """What gets embedded. Title carries most of the signal; the excerpt
        disambiguates. Deliberately excludes source and date so the same event
        from two outlets lands in the same place."""
        return f"{self.title}\n\n{self.excerpt()}".strip()

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "publisher": self.publisher,
            "url": self.url,
            "language": self.language,
            "published_at": iso(self.published_at),
            "author": self.author,
            "summary": self.best_summary(),
            "content_type": self.content_type,
        }


@dataclass
class Story:
    """A cluster of articles covering the same event, possibly across languages."""

    id: str
    headline: str
    summary: str = ""
    why_it_matters: str = ""
    key_facts: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    importance: float = 0.0
    relevance: float = 0.0
    score: float = 0.0
    article_ids: list[str] = field(default_factory=list)
    #: Distinct publishers and languages covering this story.
    publishers: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    first_seen: datetime = field(default_factory=utcnow)
    last_updated: datetime = field(default_factory=utcnow)
    #: Provider that wrote the merged headline/summary, or None if a single
    #: article's own enrichment was reused.
    written_by: str | None = None

    @staticmethod
    def make_id(seed: str) -> str:
        return "s" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:15]

    def to_json(self, articles: list[Article]) -> dict[str, Any]:
        ordered = sorted(
            articles, key=lambda a: (a.published_at or a.collected_at), reverse=True
        )
        published = [a.published_at for a in ordered if a.published_at]
        return {
            "id": self.id,
            "headline": self.headline,
            "summary": self.summary,
            "why_it_matters": self.why_it_matters,
            "key_facts": self.key_facts,
            "topics": self.topics,
            "entities": self.entities[:8],
            "importance": round(self.importance, 3),
            "relevance": round(self.relevance, 3),
            "score": round(self.score, 3),
            "publishers": sorted({a.publisher for a in ordered}),
            "publisher_count": len({a.publisher for a in ordered}),
            "languages": sorted({a.language for a in ordered if a.language}),
            "article_count": len(ordered),
            "first_published": iso(min(published)) if published else None,
            "last_published": iso(max(published)) if published else None,
            "articles": [a.to_json() for a in ordered],
        }


@dataclass
class Digest:
    """One week's published digest."""

    id: str                      # e.g. "2026-W37"
    period_start: datetime
    period_end: datetime
    story_ids: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=utcnow)
    article_count: int = 0
    stats: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def week_id(when: datetime | date) -> str:
        """ISO week key, which sorts chronologically and is unambiguous."""
        year, week, _ = (when.date() if isinstance(when, datetime) else when).isocalendar()
        return f"{year}-W{week:02d}"

    @staticmethod
    def window(end: datetime, *, days: int = 7) -> tuple[datetime, datetime]:
        return end - timedelta(days=days), end

    @property
    def label(self) -> str:
        start, end = to_utc(self.period_start), to_utc(self.period_end)
        if start.month == end.month:
            return f"{start.day}–{end.day} {end:%B %Y}"
        return f"{start:%-d %b} – {end:%-d %b %Y}"

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "period_start": iso(self.period_start),
            "period_end": iso(self.period_end),
            "generated_at": iso(self.generated_at),
            "story_count": len(self.story_ids),
            "article_count": self.article_count,
        }


@dataclass
class SourceReport:
    """Per-source outcome for one collection run."""

    name: str
    ok: bool
    fetched: int = 0
    new: int = 0
    duplicates: int = 0
    error: str | None = None
    elapsed_ms: int = 0


@dataclass
class RunStats:
    started_at: datetime = field(default_factory=utcnow)
    sources: list[SourceReport] = field(default_factory=list)
    articles_seen: int = 0
    articles_new: int = 0
    duplicates: int = 0
    languages: dict[str, int] = field(default_factory=dict)
    embedded: int = 0
    embed_cached: int = 0
    embed_calls: int = 0
    clusters: int = 0
    cross_language_clusters: int = 0
    llm_checks: int = 0
    enriched: int = 0
    enrich_cached: int = 0
    enrich_failed: int = 0
    llm_calls: int = 0
    stories_total: int = 0
    stories_new: int = 0
    stories_published: int = 0
    digest_id: str | None = None

    @property
    def sources_ok(self) -> int:
        return sum(1 for s in self.sources if s.ok)

    @property
    def sources_failed(self) -> list[SourceReport]:
        return [s for s in self.sources if not s.ok]

    def to_json(self) -> dict[str, Any]:
        return {
            "started_at": iso(self.started_at),
            "finished_at": iso(utcnow()),
            "sources_ok": self.sources_ok,
            "sources_failed": [
                {"name": s.name, "error": s.error} for s in self.sources_failed
            ],
            "articles_seen": self.articles_seen,
            "articles_new": self.articles_new,
            "duplicates": self.duplicates,
            "languages": self.languages,
            "embedded": self.embedded,
            "embed_cached": self.embed_cached,
            "embed_calls": self.embed_calls,
            "clusters": self.clusters,
            "cross_language_clusters": self.cross_language_clusters,
            "llm_checks": self.llm_checks,
            "enriched": self.enriched,
            "enrich_cached": self.enrich_cached,
            "enrich_failed": self.enrich_failed,
            "llm_calls": self.llm_calls,
            "stories_total": self.stories_total,
            "stories_new": self.stories_new,
            "stories_published": self.stories_published,
            "digest_id": self.digest_id,
        }
