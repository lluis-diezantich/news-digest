"""Static output: docs/feed.json, docs/feed.xml and the page shell.

The frontend is a single static page that fetches feed.json, so the site needs
no build step and the HTML can be edited without touching Python. GitHub Pages
serves docs/ straight from the default branch.
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
from .models import Article, RunStats, Story, iso, to_utc, utcnow

log = logging.getLogger(__name__)


def build_feed(
    config: Config,
    stories: list[tuple[Story, list[Article]]],
    stats: RunStats | None = None,
) -> dict:
    interests = config.interests
    topic_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for story, articles in stories:
        topic_counts.update(story.topics)
        source_counts.update(a.source for a in articles)

    return {
        "generated_at": iso(utcnow()),
        "title": interests.title,
        "subtitle": interests.subtitle,
        "story_count": len(stories),
        "article_count": sum(len(articles) for _, articles in stories),
        "topics": [t for t, _ in topic_counts.most_common(24)],
        "sources": [
            {"name": name, "articles": count}
            for name, count in source_counts.most_common()
        ],
        "run": stats.to_json() if stats else None,
        "stories": [story.to_json(articles) for story, articles in stories],
    }


def build_rss(config: Config, feed: dict, *, link: str = "") -> str:
    """A digest RSS feed, so the output is readable in any reader too."""
    now = format_datetime(utcnow())
    items: list[str] = []
    for story in feed["stories"]:
        articles = story.get("articles") or []
        target = articles[0]["url"] if articles else link
        published = story.get("last_published") or story.get("first_seen")
        pub_date = ""
        if published:
            try:
                pub_date = f"<pubDate>{format_datetime(to_utc(datetime.fromisoformat(published)))}</pubDate>"
            except ValueError:
                pub_date = ""
        body = story["summary"]
        if story.get("why_it_matters"):
            body += f" Why it matters: {story['why_it_matters']}"
        sources = ", ".join(sorted({a["source"] for a in articles}))
        if sources:
            body += f" (Sources: {sources})"
        items.append(
            "<item>"
            f"<title>{escape(story['headline'])}</title>"
            f"<link>{escape(target)}</link>"
            f"<guid isPermaLink=\"false\">{escape(story['id'])}</guid>"
            f"<description>{escape(body)}</description>"
            f"{pub_date}"
            "</item>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0"><channel>'
        f"<title>{escape(feed['title'])}</title>"
        f"<link>{escape(link or 'https://example.invalid/')}</link>"
        f"<description>{escape(feed['subtitle'] or 'Personal news digest')}</description>"
        f"<lastBuildDate>{now}</lastBuildDate>"
        f"{''.join(items)}"
        "</channel></rss>\n"
    )


def write_site(
    out_dir: Path,
    config: Config,
    stories: list[tuple[Story, list[Article]]],
    stats: RunStats | None = None,
    *,
    site_url: str = "",
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feed = build_feed(config, stories, stats)
    (out_dir / "feed.json").write_text(
        json.dumps(feed, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    (out_dir / "feed.xml").write_text(
        build_rss(config, feed, link=site_url), encoding="utf-8"
    )
    # Tells Pages to serve the directory as-is instead of running Jekyll.
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")

    shell = WEB_DIR / "index.html"
    if shell.exists():
        shutil.copyfile(shell, out_dir / "index.html")
    else:  # pragma: no cover - only if the repo is incomplete
        log.warning("no page shell at %s; wrote data files only", shell)

    log.info("wrote %d stories to %s", len(stories), out_dir)
    return feed
