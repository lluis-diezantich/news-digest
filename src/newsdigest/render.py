"""Markdown output (sections 14 and 24).

    digests/2026/2026-W38.md   one file per week
    README.md                  the newest digest, inlined

Markdown, and only Markdown. The RSS version of this project published a static
site, and section 29 puts any interface in phase 8 -- "only after the core
pipeline works". Everything needed to build a site later is in the database, and
`news-digest build` regenerates every file from it, so nothing here is a
one-way decision.

The target is five to ten minutes of reading. That is what `max_stories` and the
brief prompt's word limits are for; this module's job is only to not add to it.
Nothing is padded, no story gets a section it has nothing to put in, and the
per-story metadata is one line.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .config import Config
from .digest import DigestResult
from .models import Article, Digest, Story
from .urls import display_url
from .store import Store

log = logging.getLogger(__name__)

#: Article links listed under one story. A digest entry needs enough links to
#: show the coverage was real, not every copy of it.
MAX_LINKS = 5
#: Marker pair delimiting the digest inside README.md, so the prose around it
#: survives a rewrite.
README_START = "<!-- digest:start -->"
README_END = "<!-- digest:end -->"


def _escape(text: str) -> str:
    """Neutralise the few characters that would break a link or a heading.

    Deliberately minimal. Headlines legitimately contain brackets, quotes and
    asterisks, and escaping all of Markdown's punctuation makes them unreadable
    in the source; what matters is that a `]` in a headline cannot end a link and
    a leading `#` cannot become a heading.
    """
    return (text or "").replace("[", "\\[").replace("]", "\\]").strip().lstrip("#").strip()


def _story_sources(articles: list[Article]) -> list[str]:
    """One line per publisher, linking its coverage.

    Grouped by publisher rather than listed per article, because several
    newsletters from one outlet are one outlet's coverage -- the same reason
    corroboration counts publishers.
    """
    by_publisher: dict[str, Article] = {}
    for article in sorted(
        articles, key=lambda a: (a.published_at or a.collected_at), reverse=True
    ):
        by_publisher.setdefault(article.publisher, article)

    lines = []
    for publisher, article in list(by_publisher.items())[:MAX_LINKS]:
        lines.append(f"- [{_escape(publisher)}]({display_url(article.url)})")
    return lines


def story_markdown(story: Story, articles: list[Article], index: int) -> str:
    """One numbered story."""
    parts = [f"## {index}. {_escape(story.headline)}", ""]

    if story.summary:
        parts += [story.summary.strip(), ""]
    if story.why_it_matters:
        parts += [f"**Why it matters:** {story.why_it_matters.strip()}", ""]

    # Section 15: never folded into the summary, and never omitted when present.
    if story.disagreements:
        parts.append("**Where sources differ:**")
        parts += [f"- {line.strip()}" for line in story.disagreements]
        parts.append("")

    sources = _story_sources(articles)
    if sources:
        parts.append("Sources:")
        parts += sources
        parts.append("")

    return "\n".join(parts)


def minor_markdown(minor: list[tuple[Story, list[Article]]]) -> str:
    """Section 14's "Also worth knowing": one line each, no write-up."""
    if not minor:
        return ""
    lines = ["## Also worth knowing", ""]
    for story, articles in minor:
        first = next(iter(_publishers(articles)), "")
        link = display_url(articles[0].url) if articles else ""
        headline = _escape(story.headline)
        entry = f"- [{headline}]({link})" if link else f"- {headline}"
        lines.append(f"{entry} — {first}" if first else entry)
    lines.append("")
    return "\n".join(lines)


def _publishers(articles: list[Article]) -> list[str]:
    seen: list[str] = []
    for article in articles:
        if article.publisher not in seen:
            seen.append(article.publisher)
    return seen


def digest_markdown(config: Config, result: DigestResult) -> str:
    """The whole digest, as one Markdown document."""
    digest = result.digest
    parts = [f"# {config.digest.title}", f"{digest.label}", ""]
    if config.digest.subtitle:
        parts += [f"*{config.digest.subtitle}*", ""]

    if not result.stories:
        parts += [
            "No stories were published for this week. The newsletters that "
            "arrived are stored; `news-digest inspect` says what happened to them.",
            "",
        ]
        return "\n".join(parts)

    for index, (story, articles) in enumerate(result.stories, start=1):
        parts.append(story_markdown(story, articles, index))

    minor = minor_markdown(result.minor)
    if minor:
        parts += ["---", "", minor]

    parts += ["---", "", _footer(result), ""]
    return "\n".join(parts)


def _footer(result: DigestResult) -> str:
    """Provenance, in one line.

    Which outlets and languages this was built from, because a digest that merges
    ten newsletters should say so -- and because a week where one publisher
    supplied everything is worth noticing from the output alone.
    """
    publishers = sorted(
        {a.publisher for _, articles in result.stories for a in articles}
    )
    languages = sorted(
        {a.language for _, articles in result.stories for a in articles if a.language}
    )
    stats = result.digest.stats or {}
    bits = [
        f"{len(result.stories)} stories",
        f"{result.digest.article_count} items",
        f"{len(publishers)} publishers",
    ]
    if languages:
        bits.append("/".join(languages))
    line = f"*Built from {', '.join(bits)}.*"
    if publishers:
        line += f"  \n*Sources: {', '.join(publishers)}.*"
    if stats.get("articles_filtered"):
        line += f"  \n*{stats['articles_filtered']} items filtered as not news.*"
    return line


def digest_path(out: Path, digest: Digest) -> Path:
    """`digests/2026/2026-W38.md`, per section 24."""
    year = digest.id.split("-")[0]
    return Path(out) / year / f"{digest.id}.md"


def write_digest(out: Path, config: Config, result: DigestResult) -> Path:
    path = digest_path(out, result.digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(digest_markdown(config, result), encoding="utf-8")
    log.info("wrote %s", path)
    return path


def update_readme(readme: Path, config: Config, result: DigestResult) -> None:
    """Inline the newest digest between the markers in README.md.

    If the markers are absent the README is left completely alone rather than
    appended to or overwritten -- this runs unattended every Monday, and a
    workflow that rewrites hand-written documentation is worse than one that
    silently does nothing.
    """
    if not readme.exists():
        log.info("no README at %s; not creating one", readme)
        return
    text = readme.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        log.info(
            "README has no %s / %s markers; leaving it untouched",
            README_START, README_END,
        )
        return
    before = text.split(README_START)[0]
    after = text.split(README_END, 1)[1]
    body = digest_markdown(config, result).strip()
    readme.write_text(
        f"{before}{README_START}\n\n{body}\n\n{README_END}{after}", encoding="utf-8"
    )
    log.info("updated %s", readme)


def index_markdown(digests: list[Digest]) -> str:
    """`digests/README.md`: every published week, newest first."""
    lines = ["# Digest archive", ""]
    for digest in digests:
        year = digest.id.split("-")[0]
        lines.append(
            f"- [{digest.id}]({year}/{digest.id}.md) — {digest.label} "
            f"({len(digest.story_ids)} stories)"
        )
    lines.append("")
    return "\n".join(lines)


def write_all(
    out: Path,
    config: Config,
    store: Store,
    result: DigestResult,
    *,
    readme: Path | None = None,
) -> Path:
    """Write this week's digest, refresh the archive index, update the README."""
    out = Path(out)
    path = write_digest(out, config, result)

    index = out / "README.md"
    index.write_text(index_markdown(store.list_digests(limit=520)), encoding="utf-8")

    if config.digest.update_readme and readme is not None:
        update_readme(readme, config, result)
    return path


def rebuild(out: Path, config: Config, store: Store, *, readme: Path | None = None) -> int:
    """Regenerate every digest file from the database.

    This is what makes `digests/` disposable: delete the directory and one
    `news-digest build` restores it, so the Markdown is an artefact of the
    database rather than a second copy of the truth.
    """
    written = 0
    latest: DigestResult | None = None
    for digest in store.list_digests(limit=520):
        main = store.digest_stories(digest.id, tier="main")
        minor = store.digest_stories(digest.id, tier="minor")
        by_story = store.articles_by_story(
            [s.id for s in main] + [s.id for s in minor]
        )
        result = DigestResult(
            digest=digest,
            stories=[(s, by_story.get(s.id, [])) for s in main],
            minor=[(s, by_story.get(s.id, [])) for s in minor],
            output_language=(digest.stats or {}).get("output_language", "en"),
        )
        write_digest(Path(out), config, result)
        written += 1
        latest = latest or result

    index = Path(out) / "README.md"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(index_markdown(store.list_digests(limit=520)), encoding="utf-8")

    if latest is not None and config.digest.update_readme and readme is not None:
        update_readme(readme, config, latest)
    return written
