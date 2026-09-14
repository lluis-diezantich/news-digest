"""RSS/Atom adapter -- the preferred path for every source that offers a feed."""

from __future__ import annotations

import logging

import feedparser

from ..config import Source
from ..models import Article, parse_date
from ..text import strip_boilerplate, strip_html, truncate
from .base import SourceAdapter

log = logging.getLogger(__name__)


def _first_text(entry: dict, *keys: str) -> str:
    """First non-empty value among `keys`, handling feedparser's shapes."""
    for key in keys:
        value = entry.get(key)
        if isinstance(value, list) and value:
            value = value[0]
        if isinstance(value, dict):
            value = value.get("value") or value.get("href")
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _author(entry: dict) -> str | None:
    name = _first_text(entry, "author")
    if not name:
        detail = entry.get("author_detail") or {}
        name = detail.get("name", "") if isinstance(detail, dict) else ""
    if not name:
        authors = entry.get("authors") or []
        if authors and isinstance(authors[0], dict):
            name = authors[0].get("name", "")
    name = strip_html(name)
    return name or None


def _published(entry: dict):
    for key in ("published", "updated", "created", "pubDate", "dc:date"):
        parsed = parse_date(entry.get(key))
        if parsed:
            return parsed
    for key in ("published_parsed", "updated_parsed"):
        struct = entry.get(key)
        if struct:
            import calendar

            return parse_date(calendar.timegm(struct))
    return None


class RSSAdapter(SourceAdapter):
    method = "rss"

    def fetch(self, source: Source) -> list[Article]:
        url = source.rss or ""
        # feedparser can fetch on its own, but going through our Fetcher keeps
        # rate limiting, the User-Agent and robots handling in one place.
        response = self.fetcher.get(url)
        parsed = feedparser.parse(response.content)

        if parsed.bozo and not parsed.entries:
            reason = getattr(parsed, "bozo_exception", "unparseable feed")
            raise ValueError(f"could not parse feed: {reason}")

        feed_title = strip_html((parsed.feed or {}).get("title", "")) or source.name
        log.debug("%s: %d entries from %s", source.name, len(parsed.entries), feed_title)

        articles: list[Article] = []
        # Feed order is the newsroom's ranking, not just recency: measured
        # 80-100% concordant with the section page on eight of these sources.
        # `feed_size` is what we actually took, so the rank is relative to the
        # window we saw rather than to a number that changes between runs.
        taken = parsed.entries[: source.max_items]
        for index, entry in enumerate(taken):
            link = _first_text(entry, "link", "id")
            title = strip_html(_first_text(entry, "title"))
            if not link or not title:
                continue

            description = strip_html(
                _first_text(entry, "summary", "description", "subtitle")
            )
            if not description:
                content = entry.get("content") or []
                if content and isinstance(content[0], dict):
                    description = strip_html(content[0].get("value", ""))

            tags = [
                strip_html(t.get("term", "")).lower()
                for t in (entry.get("tags") or [])
                if isinstance(t, dict) and t.get("term")
            ]

            articles.append(
                Article(
                    title=title,
                    source=source.name,
                    url=link.strip(),
                    published_at=_published(entry),
                    author=_author(entry),
                    description=truncate(
                        strip_boilerplate(description), source.excerpt_chars
                    ),
                    source_topics=sorted(set(source.topics + tags[:4])),
                    source_weight=source.weight,
                    feed_position=index,
                    feed_size=len(taken),
                )
            )
        return articles
