"""Static output for GitHub Pages.

    docs/index.html          the page shell (copied from web/)
    docs/index.json          archive index: every digest, newest first
    docs/digests/<id>.json   one file per week, e.g. 2026-W37.json
    docs/feed.xml            RSS of the latest digest

The page is static and loads JSON at runtime, so there is no build step and the
frontend can be edited without touching Python. Every digest in the database is
rewritten each run, which makes the output directory reproducible from the
database alone -- delete docs/ and one `news-digest build` restores it.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections import Counter
from datetime import datetime
from email.utils import format_datetime
from pathlib import Path
from xml.sax.saxutils import escape

from .config import WEB_DIR, Config
from .digest import DigestResult
from .models import Article, Digest, Story, iso, to_utc, utcnow
from .store import Store

log = logging.getLogger(__name__)


def digest_payload(
    config: Config,
    digest: Digest,
    stories: list[tuple[Story, list[Article]]],
) -> dict:
    publishers: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    for story, articles in stories:
        for article in articles:
            publishers[article.publisher] += 1
            if article.language:
                languages[article.language] += 1

    return {
        **digest.to_json(),
        "title": config.digest.title,
        "subtitle": config.digest.subtitle,
        "output_language": config.settings.output_language,
        "publishers": [
            {"name": name, "articles": count} for name, count in publishers.most_common()
        ],
        "languages": [
            {"code": code, "articles": count} for code, count in languages.most_common()
        ],
        "stories": [story.to_json(articles) for story, articles in stories],
    }


def build_rss(config: Config, payload: dict, *, link: str = "") -> str:
    """The digest as RSS, so it is readable in any reader too."""
    items: list[str] = []
    for story in payload["stories"]:
        articles = story.get("articles") or []
        target = articles[0]["url"] if articles else link
        body = story["summary"]
        if story.get("why_it_matters"):
            body += f" Why it matters: {story['why_it_matters']}"
        outlets = ", ".join(story.get("publishers") or [])
        if outlets:
            body += f" (Sources: {outlets})"

        pub_date = ""
        published = story.get("last_published")
        if published:
            try:
                pub_date = (
                    f"<pubDate>{format_datetime(to_utc(datetime.fromisoformat(published)))}"
                    "</pubDate>"
                )
            except ValueError:
                pub_date = ""
        items.append(
            "<item>"
            f"<title>{escape(story['headline'])}</title>"
            f"<link>{escape(target)}</link>"
            f'<guid isPermaLink="false">{escape(story["id"])}</guid>'
            f"<description>{escape(body)}</description>"
            f"{pub_date}"
            "</item>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0"><channel>'
        f"<title>{escape(config.digest.title)}</title>"
        f"<link>{escape(link or 'https://example.invalid/')}</link>"
        f"<description>{escape(config.digest.subtitle or 'Personal weekly news digest')}"
        "</description>"
        f"<lastBuildDate>{format_datetime(utcnow())}</lastBuildDate>"
        f"{''.join(items)}"
        "</channel></rss>\n"
    )


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


def write_site(
    out_dir: Path,
    config: Config,
    store: Store,
    result: DigestResult | None = None,
    *,
    site_url: str = "",
) -> dict:
    """Write every digest the database holds, plus the index and page shell."""
    out_dir = Path(out_dir)
    digests_dir = out_dir / "digests"
    out_dir.mkdir(parents=True, exist_ok=True)

    limit = config.digest.archive_limit if config.digest.archive else 1
    all_digests = store.list_digests(limit=limit)

    # The just-built digest may not be re-readable identically (stories were
    # ranked in memory), so prefer the in-memory result for that one week.
    fresh_id = result.digest.id if result else None
    index: list[dict] = []
    latest_payload: dict | None = None

    for digest in all_digests:
        if fresh_id and digest.id == fresh_id and result is not None:
            payload = digest_payload(config, result.digest, result.stories)
        else:
            stories = store.digest_stories(digest.id)
            grouped = store.articles_by_story([s.id for s in stories])
            payload = digest_payload(
                config, digest, [(s, grouped.get(s.id, [])) for s in stories]
            )
        _write_json(digests_dir / f"{digest.id}.json", payload)
        index.append(
            {
                "id": digest.id,
                "label": digest.label,
                "period_start": iso(digest.period_start),
                "period_end": iso(digest.period_end),
                "generated_at": iso(digest.generated_at),
                "story_count": payload["story_count"],
                "article_count": payload["article_count"],
            }
        )
        if latest_payload is None:
            latest_payload = payload

    if latest_payload is None:
        # No digest yet: still write a valid index so the page explains itself.
        latest_payload = {"stories": [], "story_count": 0, "article_count": 0}

    _write_json(
        out_dir / "index.json",
        {
            "title": config.digest.title,
            "subtitle": config.digest.subtitle,
            "generated_at": iso(utcnow()),
            "archive": config.digest.archive,
            "digests": index,
        },
    )
    (out_dir / "feed.xml").write_text(
        build_rss(config, latest_payload, link=site_url), encoding="utf-8"
    )
    # Tells Pages to serve the directory as-is instead of running Jekyll.
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")

    shell = WEB_DIR / "index.html"
    if shell.exists():
        shutil.copyfile(shell, out_dir / "index.html")
    else:  # pragma: no cover - only if the repo is incomplete
        log.warning("no page shell at %s; wrote data files only", shell)

    log.info(
        "wrote %d digest(s) to %s (latest %s, %d stories)",
        len(index), out_dir, index[0]["id"] if index else "none",
        latest_payload.get("story_count", 0),
    )
    return latest_payload
