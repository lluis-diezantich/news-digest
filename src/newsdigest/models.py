"""The common schema: emails in, articles and stories out, one digest a week."""

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
    """Best-effort date parsing across the many shapes mail headers use."""
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
class Email:
    """One newsletter message, stored so it is never parsed or fetched twice.

    The body is kept locally (and inside the runner) but never published: the
    digest carries article URLs and summaries, not mailbox contents.
    """

    message_id: str
    source: str
    subject: str = ""
    sender: str = ""
    sender_name: str = ""
    newsletter: str = ""
    received_at: datetime = field(default_factory=utcnow)
    html_body: str = ""
    text_body: str = ""

    # --- derived ---
    id: str = ""
    #: Set once articles have been extracted, so a re-run skips the parse.
    parsed_at: datetime | None = None
    article_count: int = 0

    def __post_init__(self) -> None:
        self.subject = (self.subject or "").strip()
        if not self.id:
            # Keyed on Message-ID, which is globally unique by definition and
            # stable across mailbox moves and re-downloads.
            self.id = hashlib.sha1(self.message_id.encode("utf-8")).hexdigest()[:16]

    @property
    def body(self) -> str:
        """The richest body available. HTML wins: the plain-text alternative of a
        newsletter is usually a stripped courtesy copy with the links flattened
        out, and links are most of what we are here for."""
        return self.html_body or self.text_body

    @property
    def parsed(self) -> bool:
        return self.parsed_at is not None

    def content_hash(self) -> str:
        """Keys the parse cache: the same message is never re-extracted."""
        return hashlib.sha1(self.body.encode("utf-8")).hexdigest()[:16]


@dataclass
class Article:
    """One news item extracted from one newsletter. `id` is derived, so the
    extractor never invents one."""

    title: str
    source: str
    url: str
    published_at: datetime | None = None
    author: str | None = None
    #: The blurb the newsletter wrote about this item. Never a full article body.
    description: str = ""
    #: Longer excerpt, if one was ever resolved from the article page.
    content: str = ""
    #: Language code, filled in at parse time. None means undetermined.
    language: str | None = None
    #: Outlet this newsletter belongs to; several newsletters may share one.
    publisher: str = ""
    #: Newsletter this item came from, e.g. "Saturday Edition".
    newsletter: str = ""
    #: The email this was extracted from.
    email_id: str = ""
    received_at: datetime | None = None
    #: Where this item sat in the newsletter, and how many items it held.
    #: A newsletter is hand-ordered by an editor who picked the items in the
    #: first place, so position 0 is a stronger claim than a feed's position 0:
    #: nothing arrives here by publication time alone. Stored raw so the
    #: normalisation can change without re-parsing.
    #: -1 means unknown.
    item_position: int = -1
    item_count: int = 0
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
    #: Filled by the cheap classification pass (section 8), not by enrichment.
    #: `region` feeds the diversity pass; `newsworthy` is False for the things a
    #: newsletter carries that are not news. None means "not yet classified",
    #: which is deliberately different from False -- an unclassified item is kept.
    region: str = ""
    newsworthy: bool | None = None
    classified_by: str | None = None
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

    def editorial_rank(self) -> float:
        """Prominence in [0, 1] from position in the newsletter: 1.0 leads, 0.0 trails.

        A curated newsletter is a stronger version of the signal feed order gives
        an aggregator. An editor chose these ten items out of the day's hundreds
        AND chose which one opens, so position 0 is two judgements rather than
        one, and it is available in every language for free.

        Newsletters also fail differently from feeds. Feed junk is structural and
        sits at predictable paths, which is what `exclude_url_patterns` catches.
        A newsletter's junk is the sponsor slot and the housekeeping block, which
        can sit anywhere -- including first. Two controls keep those out of reach
        of this method rather than any cleverness here: the extractor drops
        sponsored and housekeeping blocks before they become articles, and topic
        classification drops what survives. LOOSENING EITHER PUTS A SPONSOR SLOT
        AT THE TOP OF THE DIGEST.

        Normalised within the newsletter, because position 3 of 8 items and
        position 3 of 40 are not the same claim. Unknown position returns the
        neutral 0.5 -- a newsletter we could not order should not be penalised as
        though every item trailed.
        """
        if self.item_position < 0 or self.item_count <= 1:
            return 0.5
        return 1.0 - (min(self.item_position, self.item_count - 1) / (self.item_count - 1))

    def content_hash(self) -> str:
        """Keys the LLM and embedding caches: same text in, no new API call."""
        payload = f"{self.title}\x00{self.excerpt()}".encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:16]

    def best_summary(self) -> str:
        return self.summary or self.excerpt()

    def embedding_text(self) -> str:
        """What gets embedded. Title carries most of the signal; the excerpt
        disambiguates. Deliberately excludes source and date so the same event
        from two newsletters lands in the same place."""
        return f"{self.title}\n\n{self.excerpt()}".strip()

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "publisher": self.publisher,
            "newsletter": self.newsletter,
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
    #: Regions this story is about, for the diversity pass. See `regions.py`.
    regions: list[str] = field(default_factory=list)
    #: Where the sources disagree, in their own words. Empty when they do not.
    #: Never merged into `summary`: silently resolving a disagreement is the one
    #: thing a digest of several outlets must not do.
    disagreements: list[str] = field(default_factory=list)
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
            "regions": self.regions,
            "disagreements": self.disagreements,
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
    #: Stories below the main cut, published as a short list. See section 14 of
    #: the specification: "Also worth knowing".
    minor_story_ids: list[str] = field(default_factory=list)
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
        """The human date range, e.g. "14-20 September 2026".

        `period_end` is EXCLUSIVE -- a Monday-to-Sunday week ends at midnight on
        the following Monday -- so the label shows the day before it. Printing
        `period_end` directly claimed a week of eight days, and always named a day
        the digest did not cover.
        """
        start = to_utc(self.period_start)
        last = to_utc(self.period_end) - timedelta(days=1)
        if start.year != last.year:
            return f"{start:%-d %b %Y} – {last:%-d %b %Y}"
        if start.month == last.month:
            return f"{start.day}–{last.day} {last:%B %Y}"
        return f"{start:%-d %b} – {last:%-d %b %Y}"

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "period_start": iso(self.period_start),
            "period_end": iso(self.period_end),
            "generated_at": iso(self.generated_at),
            "story_count": len(self.story_ids),
            "minor_story_count": len(self.minor_story_ids),
            "article_count": self.article_count,
        }


@dataclass
class SourceReport:
    """Per-source outcome for one ingestion run."""

    name: str
    ok: bool
    #: Emails matched to this source in the window.
    emails: int = 0
    #: Emails not seen before.
    new_emails: int = 0
    #: Articles extracted from them.
    extracted: int = 0
    new: int = 0
    duplicates: int = 0
    #: Dropped by exclude_url_patterns / exclude_title_patterns. Counted apart
    #: from duplicates so a blocklist that quietly eats a source is visible.
    excluded: int = 0
    error: str | None = None
    elapsed_ms: int = 0


@dataclass
class RunStats:
    """Counters for one run. Printed by `--dry-run`, stored with every run."""

    started_at: datetime = field(default_factory=utcnow)
    sources: list[SourceReport] = field(default_factory=list)
    emails_fetched: int = 0
    emails_new: int = 0
    emails_unmatched: int = 0
    emails_parsed: int = 0
    #: Items the extractor found in the newsletters, before exclusion.
    articles_extracted: int = 0
    #: Items stored, after exclusion and deduplication.
    articles_new: int = 0
    #: Articles the weekly stage READ back for its window, which includes
    #: everything stored on earlier runs. Deliberately distinct from
    #: `articles_extracted`: reporting both under one name made a re-run look
    #: like it had extracted articles it had only re-read.
    articles_in_window: int = 0
    duplicates: int = 0
    excluded: int = 0
    #: Dropped by topic classification (section 8), as against by regex.
    filtered: int = 0
    languages: dict[str, int] = field(default_factory=dict)
    embedded: int = 0
    embed_cached: int = 0
    embed_calls: int = 0
    clusters: int = 0
    cross_language_clusters: int = 0
    llm_checks: int = 0
    classified: int = 0
    classify_cached: int = 0
    enriched: int = 0
    enrich_cached: int = 0
    enrich_failed: int = 0
    llm_calls: int = 0
    stories_total: int = 0
    stories_new: int = 0
    stories_published: int = 0
    minor_stories: int = 0
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
            "emails_fetched": self.emails_fetched,
            "emails_new": self.emails_new,
            "emails_unmatched": self.emails_unmatched,
            "emails_parsed": self.emails_parsed,
            "articles_extracted": self.articles_extracted,
            "articles_in_window": self.articles_in_window,
            "articles_new": self.articles_new,
            "duplicates": self.duplicates,
            "excluded": self.excluded,
            "filtered": self.filtered,
            "languages": self.languages,
            "embedded": self.embedded,
            "embed_cached": self.embed_cached,
            "embed_calls": self.embed_calls,
            "clusters": self.clusters,
            "cross_language_clusters": self.cross_language_clusters,
            "llm_checks": self.llm_checks,
            "classified": self.classified,
            "classify_cached": self.classify_cached,
            "enriched": self.enriched,
            "enrich_cached": self.enrich_cached,
            "enrich_failed": self.enrich_failed,
            "llm_calls": self.llm_calls,
            "stories_total": self.stories_total,
            "stories_new": self.stories_new,
            "stories_published": self.stories_published,
            "minor_stories": self.minor_stories,
            "digest_id": self.digest_id,
        }
