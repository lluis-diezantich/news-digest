"""The common schema every source normalizes into, plus the clustered story."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from dateutil import parser as date_parser

from .urls import canonical_url, domain


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
    now = now or utcnow()
    return max(0.0, (now - to_utc(value)).total_seconds() / 3600.0)


@dataclass
class Article:
    """One normalized article. `id` is derived, so sources never invent one."""

    title: str
    source: str
    url: str
    published_at: datetime | None = None
    author: str | None = None
    # Excerpt only -- feed description or the opening of the page. Never a full
    # article body; see README "Attribution and copyright".
    description: str = ""
    source_topics: list[str] = field(default_factory=list)
    source_weight: float = 1.0
    fetched_at: datetime = field(default_factory=utcnow)

    # --- derived ---
    canonical: str = ""
    id: str = ""

    # --- LLM enrichment, filled in by the enrich stage ---
    summary: str | None = None
    why_it_matters: str | None = None
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    importance: float | None = None
    # The LLM's short label for the underlying event ("uk budget 2026"). Used as
    # a clustering signal, never shown to the user.
    event_label: str | None = None
    enriched_by: str | None = None
    enriched_at: datetime | None = None

    story_id: str | None = None

    def __post_init__(self) -> None:
        self.title = (self.title or "").strip()
        self.canonical = canonical_url(self.url)
        if not self.id:
            self.id = hashlib.sha1(self.canonical.encode("utf-8")).hexdigest()[:16]

    @property
    def enriched(self) -> bool:
        return self.summary is not None

    @property
    def site(self) -> str:
        return domain(self.url)

    def content_hash(self) -> str:
        """Keys the LLM cache: same text in, same enrichment out, no API call."""
        payload = f"{self.title}\x00{self.description}".encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:16]

    def best_summary(self) -> str:
        return self.summary or self.description

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "url": self.url,
            "published_at": iso(self.published_at),
            "author": self.author,
            "summary": self.best_summary(),
            "topics": self.topics or self.source_topics,
            "importance": self.importance,
        }


@dataclass
class Story:
    """A cluster of articles covering the same event."""

    id: str
    headline: str
    summary: str = ""
    why_it_matters: str = ""
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    importance: float = 0.0
    score: float = 0.0
    article_ids: list[str] = field(default_factory=list)
    # Token fingerprint used to match tomorrow's articles onto this story.
    keywords: list[str] = field(default_factory=list)
    first_seen: datetime = field(default_factory=utcnow)
    last_updated: datetime = field(default_factory=utcnow)
    # Provider that wrote the merged headline/summary, or None if a single
    # article's own enrichment was reused verbatim.
    written_by: str | None = None

    @staticmethod
    def make_id(seed: str) -> str:
        return "s" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:15]

    def to_json(self, articles: list[Article]) -> dict[str, Any]:
        ordered = sorted(
            articles, key=lambda a: (a.published_at or a.fetched_at), reverse=True
        )
        published = [a.published_at for a in ordered if a.published_at]
        return {
            "id": self.id,
            "headline": self.headline,
            "summary": self.summary,
            "why_it_matters": self.why_it_matters,
            "topics": self.topics,
            "entities": self.entities[:8],
            "importance": round(self.importance, 3),
            "score": round(self.score, 3),
            "source_count": len({a.source for a in ordered}),
            "first_published": iso(min(published)) if published else None,
            "last_published": iso(max(published)) if published else None,
            "first_seen": iso(self.first_seen),
            "articles": [a.to_json() for a in ordered],
        }


@dataclass
class SourceReport:
    """Per-source outcome for one run. One broken source must not stop a run."""

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
    enriched: int = 0
    enrich_cached: int = 0
    enrich_failed: int = 0
    llm_calls: int = 0
    stories_total: int = 0
    stories_new: int = 0
    stories_published: int = 0

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
            "enriched": self.enriched,
            "enrich_cached": self.enrich_cached,
            "enrich_failed": self.enrich_failed,
            "llm_calls": self.llm_calls,
            "stories_total": self.stories_total,
            "stories_new": self.stories_new,
            "stories_published": self.stories_published,
        }
