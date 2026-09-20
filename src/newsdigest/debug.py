"""Intermediate data on disk (section 22).

    debug/emails.json     what arrived, and which source it matched
    debug/articles.json   what came out of the newsletters
    debug/filtered.json   what topic filtering dropped, and why
    debug/kept.json       what survived it
    debug/clusters.json   how the survivors grouped
    debug/stories.json    what was published, with the ranking breakdown

Written only with `--debug`, so a scheduled run leaves nothing behind.

Everything here is SANITIZED (section 28). Email bodies, sender addresses and
mailbox credentials never reach these files: the debug export is the artefact
most likely to be pasted into an issue or a chat window, so it must be safe to
paste. What it keeps is what makes a decision reviewable -- subjects, headlines,
URLs, positions, scores, and the reason each item was dropped.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .models import Article, Email, Story, iso

log = logging.getLogger(__name__)


def _article(article: Article) -> dict[str, Any]:
    return {
        "id": article.id,
        "title": article.title,
        "url": article.url,
        "source": article.source,
        "publisher": article.publisher,
        "newsletter": article.newsletter,
        "language": article.language,
        "position": article.item_position,
        "of": article.item_count,
        "editorial_rank": round(article.editorial_rank(), 3),
        "topics": article.topics,
        "region": article.region,
        "newsworthy": article.newsworthy,
        "content_type": article.content_type,
        "importance": article.importance,
        "excerpt": article.excerpt()[:300],
        "received_at": iso(article.received_at),
    }


def _email(message: Email, *, matched: str | None = None) -> dict[str, Any]:
    """One message, WITHOUT its body or its sender address.

    The sender is reduced to its domain: that is the part that explains a match
    or a miss, and the local part is a mailbox address that does not belong in a
    file meant to be shared.
    """
    return {
        "id": message.id,
        "subject": message.subject,
        "sender_domain": message.sender.rpartition("@")[2],
        "sender_name": message.sender_name,
        "matched_source": matched if matched is not None else (message.source or None),
        "newsletter": message.newsletter,
        "received_at": iso(message.received_at),
        "has_html": bool(message.html_body),
        "has_text": bool(message.text_body),
        "body_chars": len(message.body),
        "articles": message.article_count,
        "parsed_at": iso(message.parsed_at),
    }


class DebugWriter:
    """Writes the intermediate files. A no-op when disabled, so callers need no
    conditional at each stage."""

    def __init__(self, directory: Path | str | None):
        self.directory = Path(directory) if directory else None
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
            log.info("debug output in %s", self.directory)

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    def _write(self, name: str, payload: Any) -> None:
        if not self.directory:
            return
        path = self.directory / f"{name}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        log.debug("wrote %s", path)

    def dump(self, name: str, articles: list[Article]) -> None:
        self._write(name, [_article(a) for a in articles])

    def dump_emails(
        self, name: str, emails: list[Email], unmatched: list[Email] | None = None
    ) -> None:
        payload = [_email(m) for m in emails]
        payload += [_email(m, matched=None) for m in (unmatched or [])]
        self._write(name, payload)

    def dump_clusters(self, name: str, groups: list[list[Article]]) -> None:
        self._write(
            name,
            [
                {
                    "size": len(group),
                    "publishers": sorted({a.publisher for a in group}),
                    "languages": sorted({a.language for a in group if a.language}),
                    "titles": [a.title for a in group],
                }
                for group in groups
            ],
        )

    def dump_stories(
        self,
        name: str,
        main: list[tuple[Story, list[Article]]],
        minor: list[tuple[Story, list[Article]]] | None = None,
    ) -> None:
        def entry(pair, tier):
            story, articles = pair
            payload = story.to_json(articles)
            payload["tier"] = tier
            return payload

        self._write(
            name,
            [entry(pair, "main") for pair in main]
            + [entry(pair, "minor") for pair in (minor or [])],
        )
