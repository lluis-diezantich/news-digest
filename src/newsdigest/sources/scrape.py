"""HTML scrape adapter -- the fallback for sources with no feed or API.

Two passes: read the configured index page for article links, then fetch each
linked page and read its metadata. Only metadata and a short excerpt are kept;
we never store the article body.

robots.txt is honoured for both passes (a disallowed link is skipped, not
fetched), and requests are rate limited per host by `Fetcher`.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from ..config import Source
from ..models import Article, parse_date
from ..text import strip_boilerplate, strip_html, truncate
from .base import SourceAdapter
from .http import RobotsDisallowed

log = logging.getLogger(__name__)


def _meta(soup: BeautifulSoup, *candidates: tuple[str, str]) -> str:
    """First matching <meta> content, given (attribute, value) pairs."""
    for attr, value in candidates:
        tag = soup.find("meta", attrs={attr: value})
        if tag and tag.get("content"):
            return strip_html(tag["content"])
    return ""


def _lead_text(soup: BeautifulSoup, limit: int) -> str:
    """Opening paragraphs, capped -- enough to summarize from, no more."""
    root = soup.find("article") or soup.find("main") or soup
    collected: list[str] = []
    total = 0
    for para in root.find_all("p", limit=12):
        text = strip_html(para.get_text(" "))
        if len(text) < 40:
            continue
        collected.append(text)
        total += len(text)
        if total >= limit:
            break
    return truncate(" ".join(collected), limit)


class ScrapeAdapter(SourceAdapter):
    method = "scrape"

    def fetch(self, source: Source) -> list[Article]:
        index_url = source.url or ""
        index = self.fetcher.get(index_url)
        soup = BeautifulSoup(index.text, "html.parser")

        pattern = re.compile(source.link_pattern) if source.link_pattern else None
        links: list[str] = []
        seen: set[str] = set()
        for anchor in soup.select(source.link_selector):
            href = anchor.get("href")
            if not href or href.startswith(("#", "mailto:", "javascript:")):
                continue
            absolute = urljoin(index.url, href.strip())
            if pattern and not pattern.search(absolute):
                continue
            if absolute in seen:
                continue
            seen.add(absolute)
            links.append(absolute)
            if len(links) >= source.max_links:
                break

        if not links:
            raise ValueError(
                f"selector {source.link_selector!r} matched no usable links on {index_url} "
                "(the site's markup probably changed)"
            )

        articles: list[Article] = []
        for link in links:
            try:
                article = self._fetch_article(source, link)
            except RobotsDisallowed:
                log.info("%s: robots.txt disallows %s, skipping", source.name, link)
                continue
            except Exception as exc:  # one bad page must not lose the others
                log.warning("%s: %s failed (%s)", source.name, link, exc)
                continue
            if article:
                articles.append(article)
        return articles

    def _fetch_article(self, source: Source, url: str) -> Article | None:
        page = self.fetcher.get(url)
        soup = BeautifulSoup(page.text, "html.parser")

        title = (
            _meta(soup, ("property", "og:title"), ("name", "twitter:title"))
            or strip_html(soup.title.get_text() if soup.title else "")
        )
        if not title:
            return None

        description = _meta(
            soup, ("property", "og:description"), ("name", "description")
        )
        lead = _lead_text(soup, source.excerpt_chars)
        if len(lead) > len(description):
            description = lead

        published = parse_date(
            _meta(
                soup,
                ("property", "article:published_time"),
                ("name", "pubdate"),
                ("itemprop", "datePublished"),
            )
        )
        if published is None:
            time_tag = soup.find("time")
            if time_tag:
                published = parse_date(time_tag.get("datetime") or time_tag.get_text())

        author = _meta(soup, ("name", "author"), ("property", "article:author")) or None

        return Article(
            title=title,
            source=source.name,
            url=page.url,
            published_at=published,
            author=author,
            description=truncate(
                strip_boilerplate(description), source.excerpt_chars
            ),
            source_topics=list(source.topics),
            source_weight=source.weight,
        )
