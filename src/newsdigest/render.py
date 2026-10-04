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
import re
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .digest import DigestResult
from .models import Article, Digest, Story, Theme
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
    return [
        f"- [{_escape(publisher)}]({display_url(article.url)})"
        for publisher, article in _by_publisher(articles).items()
    ]


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


def theme_markdown(
    theme: Theme,
    members: list[tuple[Story, list[Article]]],
    index: int,
    *,
    lead: bool = False,
) -> str:
    """One theme: the narrative, then the stories under it.

    The theme's own write-up carries the entry. Its member stories follow as
    headlines with their sources rather than full write-ups, because repeating
    each story's summary underneath a narrative that already states it is how a
    themed digest turns back into the list it was meant to replace.

    A theme whose write-up failed falls back to its label and its members, which
    is no worse than the flat output it replaced.
    """
    heading = theme.headline or theme.label
    parts = [f"## {index}. {_escape(heading)}", ""]
    if lead and theme.label and theme.label != heading:
        # The grouping pass names the connection; the writer names the story.
        # Both are useful on the lead and only there, where a reader is deciding
        # whether to keep going.
        parts += [f"*{_escape(theme.label)}*", ""]

    if theme.narrative:
        parts += [theme.narrative.strip(), ""]
    if theme.why_it_matters:
        parts += [f"**Why it matters:** {theme.why_it_matters.strip()}", ""]
    if theme.open_questions:
        parts.append("**Not answered by this week's coverage:**")
        parts += [f"- {q.strip()}" for q in theme.open_questions]
        parts.append("")

    # Disagreements are per-story and must not be swallowed by the narrative:
    # Section 15 applies whether a story is published alone or inside a theme.
    for story, _ in members:
        for line in story.disagreements:
            parts.append(f"**Where sources differ:** {line.strip()}")
    if any(story.disagreements for story, _ in members):
        parts.append("")

    if members:
        parts.append("In this story:")
        for story, articles in members:
            links = ", ".join(
                f"[{_escape(publisher)}]({display_url(article.url)})"
                for publisher, article in _by_publisher(articles).items()
            )
            parts.append(f"- {_escape(story.headline)} — {links}")
        parts.append("")

    return "\n".join(parts)


def _by_publisher(articles: list[Article]) -> dict[str, Article]:
    """One representative article per publisher, newest first."""
    out: dict[str, Article] = {}
    for article in sorted(
        articles, key=lambda a: (a.published_at or a.collected_at), reverse=True
    ):
        out.setdefault(article.publisher, article)
    return dict(list(out.items())[:MAX_LINKS])


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


#: Characters GitHub drops when it builds a heading anchor. Everything that is
#: not a word character, whitespace or a hyphen -- so accents survive and
#: punctuation does not, which is why a Spanish headline anchors on its accents.
_SLUG_DROP = re.compile(r"[^\w\s-]", re.UNICODE)


def heading_slug(heading: str) -> str:
    """The anchor GitHub will generate for this heading.

    Each whitespace character becomes one hyphen, NOT each run of them:
    github-slugger substitutes per character, so "city - and" (where the dash is
    dropped, leaving two spaces) anchors as `city--and`. Collapsing runs here
    would produce `city-and` and a link that silently goes nowhere, which is the
    whole risk of a contents list -- a dead anchor looks exactly like a live one
    until someone clicks it.
    """
    return re.sub(r"\s", "-", _SLUG_DROP.sub("", heading.lower()).strip())


@dataclass
class Entry:
    """One line of the digest, and the section it expands to when asked."""

    title: str
    #: publisher -> the article to link for it, newest first and capped.
    sources: dict
    #: Every article in the entry, so each outlet's own headline can be shown.
    raw: list
    #: Numbered, for the anchor and the expanded heading.
    heading: str
    block: str


def _entries(result: DigestResult) -> list[Entry]:
    """The week in published order.

    The ONE place the running order is decided, so the list and the expanded
    sections cannot drift apart: both are built from this, and an anchor is
    always the slug of a heading that exists.
    """
    by_id = {story.id: (story, articles) for story, articles in result.stories}
    themed = {sid for theme in result.themes for sid in theme.story_ids}
    entries: list[Entry] = []

    for position, theme in enumerate(result.themes):
        members = [by_id[sid] for sid in theme.story_ids if sid in by_id]
        if not members:
            continue
        index = len(entries) + 1
        title = theme.headline or theme.label
        member_articles = [a for _, articles in members for a in articles]
        entries.append(Entry(
            title=title,
            sources=_by_publisher(member_articles),
            raw=member_articles,
            heading=f"{index}. {title}",
            block=theme_markdown(theme, members, index, lead=position == 0),
        ))

    # Stories in no narrative, in ranked order. Most weeks there are several,
    # and forcing them into a theme would be the failure the grouping prompt is
    # written to avoid.
    for story, articles in result.stories:
        if story.id in themed:
            continue
        index = len(entries) + 1
        entries.append(Entry(
            title=story.headline,
            sources=_by_publisher(articles),
            raw=list(articles),
            heading=f"{index}. {story.headline}",
            block=story_markdown(story, articles, index),
        ))
    return entries

#: Months, for the short date beside each headline. Rendered rather than
#: localized: three letters reads the same to a Spanish or English reader and
#: needs no locale on the runner.
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _short_date(article: Article) -> str:
    """The day this item is from, as "3 Oct".

    ABSOLUTE, not relative. A digest is a file that is committed and read for
    months, so "15 hours ago" is wrong the morning after it is written -- it
    describes when the file was generated rather than when the story ran.
    """
    when = article.published_at or article.received_at or article.collected_at
    if not when:
        return ""
    return f"{when.day} {_MONTHS[when.month - 1]}"


def _coverage_lines(articles: list[Article]) -> list[str]:
    """One line per outlet, carrying that outlet's OWN headline.

    This is the point of the format: several newspapers covered one thing and
    each wrote its own headline, so showing all of them is showing the coverage.
    Collapsing them into one synthesized headline with a row of source links --
    which this did until 2026-10-04 -- throws away the thing a reader is
    scanning for.

    One line per ARTICLE, newest first -- deliberately not one per publisher.
    Collapsing by publisher was the first attempt and it defeats the format: a
    topic's three headlines frequently come from one outlet across its several
    newsletters, and keeping one of them leaves a "full coverage" list with a
    single line in it.
    """
    ordered = sorted(
        articles,
        key=lambda a: (a.published_at or a.received_at or a.collected_at),
        reverse=True,
    )
    lines = []
    for article in ordered[:MAX_LINKS]:
        when = _short_date(article)
        stamp = f" · {when}" if when else ""
        lines.append(
            f"- **{_escape(article.publisher)}**{stamp} — "
            f"[{_escape(article.title)}]({display_url(article.url)})"
        )
    return lines


def contents_markdown(entries: list["Entry"], *, detail: bool) -> str:
    """The digest body, in whichever shape the two modes need.

    They are genuinely different documents rather than one with a section
    hidden, which is why this branches instead of sharing a renderer:

      * Default: a topic heading per entry, then every outlet's OWN headline
        beneath it. The body IS this -- nothing expands.
      * `detail`: a bullet contents list anchored to the expanded sections
        below. Headings here would collide with those sections' headings and
        GitHub would silently suffix the duplicate slugs, so the index is a
        list and the headings belong to the sections.
    """
    if not entries:
        return ""
    if detail:
        lines = ["## This week", ""]
        for entry in entries:
            who = ", ".join(
                f"[{_escape(publisher)}]({display_url(article.url)})"
                for publisher, article in entry.sources.items()
            )
            lines.append(
                f"- [{_escape(entry.title)}](#{heading_slug(entry.heading)})"
                + (f" — {who}" if who else "")
            )
        return "\n".join(lines + [""])

    blocks: list[str] = []
    for entry in entries:
        if len(entry.raw) == 1:
            # Its headline IS the topic, so a heading plus a line repeating it
            # word for word would be the same text twice.
            article = entry.raw[0]
            when = _short_date(article)
            blocks.append("\n".join([
                f"## [{_escape(article.title)}]({display_url(article.url)})",
                "",
                f"**{_escape(article.publisher)}**" + (f" · {when}" if when else ""),
                "",
            ]))
            continue
        blocks.append("\n".join(
            [f"## {_escape(entry.title)}", ""] + _coverage_lines(entry.raw) + [""]
        ))
    return "\n".join(blocks)

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

    entries = _entries(result)
    detail = config.digest.detail
    contents = contents_markdown(entries, detail=detail)
    if contents:
        parts.append(contents)
    if detail:
        parts += ["---", ""] + [entry.block for entry in entries]
        # Section 14's "Also worth knowing" belongs to the expanded form. The
        # list is a scan of what mattered, and a second ranked list below it
        # reads as more of the same rather than less.
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

    # What the digest could not see. A reader comparing two weeks cannot
    # otherwise tell a quiet week from one where half the sources went silent,
    # and a story resting on a single outlet is not the same claim as one three
    # outlets agree on -- corroboration is the whole basis of the ranking.
    single = stats.get("single_publisher_stories") or 0
    if single:
        line += (
            f"  \n*{single} of {len(result.stories)} stories rest on a single "
            f"publisher.*"
        )
    silent = stats.get("silent_sources") or []
    if silent:
        line += (
            f"  \n*Silent this week: {', '.join(silent)} "
            f"({len(silent)} configured source{'s' if len(silent) != 1 else ''} "
            f"contributed nothing).*"
        )
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
            # Restored from the free-form stats blob, so a rebuilt digest is the
            # same document rather than a flat version of it.
            themes=[
                Theme.from_dict(t)
                for t in ((digest.stats or {}).get("themes") or [])
                if isinstance(t, dict)
            ],
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
