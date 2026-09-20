"""Command line interface.

    news-digest run                     # fetch, parse, and publish the week
    news-digest fetch --from 2026-09-14 # just read the mailbox
    news-digest parse --force           # re-extract, after changing the parser
    news-digest digest --days 7         # rebuild from what is already stored
    news-digest inspect --week 2026-W38 # what happened, changing nothing
    news-digest sources --check         # do the match rules actually match?
    news-digest explain <story-id>      # why did this rank where it did?

Section 26 of the specification lists `process` and `generate` as separate
commands. They are one command here, `digest`, because clustering, summarizing,
ranking and writing share a transaction: stopping between them would mean
persisting a digest with no summaries in it, which is not a state worth being
able to reach. `process` and `generate` are accepted as aliases so the names in
the specification still work.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

from . import pipeline, render, scoring
from .config import (
    DEFAULT_DB,
    DEFAULT_DEBUG,
    DEFAULT_FILTERS,
    DEFAULT_OUT,
    DEFAULT_PREFERENCES,
    DEFAULT_SOURCES,
    REPO_ROOT,
    ConfigError,
    load_config,
)
from .digest import select_candidates, weekly_window
from .inbox import MailboxError, Matcher, get_mailbox, parse_message
from .models import utcnow
from .store import Store

log = logging.getLogger(__name__)


def _add_global_args(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Options accepted either side of the subcommand.

    Added to the top-level parser with real defaults and to every subparser with
    SUPPRESS defaults, so `--db X digest` and `digest --db X` both work and the
    subparser only overrides what was actually typed. Without SUPPRESS the
    subparser's default silently overwrites a value given before the subcommand,
    which is the shape every recipe in the docs uses.
    """

    def default(value):
        return argparse.SUPPRESS if suppress else value

    parser.add_argument("--sources", type=Path, default=default(DEFAULT_SOURCES),
                        help=f"sources file (default: {DEFAULT_SOURCES})")
    parser.add_argument("--preferences", type=Path, default=default(DEFAULT_PREFERENCES),
                        help=f"preferences file (default: {DEFAULT_PREFERENCES})")
    parser.add_argument("--filters", type=Path, default=default(DEFAULT_FILTERS),
                        help=f"filters file (default: {DEFAULT_FILTERS})")
    parser.add_argument("--db", type=Path, default=default(DEFAULT_DB),
                        help=f"SQLite database (default: {DEFAULT_DB})")
    parser.add_argument("--out", type=Path, default=default(DEFAULT_OUT),
                        help=f"digest output directory (default: {DEFAULT_OUT})")
    parser.add_argument("--debug", nargs="?", const=str(DEFAULT_DEBUG),
                        default=default(None), metavar="DIR",
                        help="export intermediate JSON (default: debug/)")
    parser.add_argument("-v", "--verbose", action="store_true", default=default(False))
    parser.add_argument("-q", "--quiet", action="store_true", default=default(False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="news-digest",
        description="Turn a week of newsletters into one digest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_global_args(parser, suppress=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    _add_global_args(common, suppress=True)

    # Window options, shared by every command that has a window.
    window = argparse.ArgumentParser(add_help=False)
    group = window.add_mutually_exclusive_group()
    group.add_argument("--week", help="an ISO week like 2026-W38")
    group.add_argument("--days", type=int, metavar="N",
                       help="the last N days, ending now")
    window.add_argument("--from", dest="from_date", metavar="DATE",
                        help="window start, YYYY-MM-DD")
    window.add_argument("--to", dest="to_date", metavar="DATE",
                        help="window end, YYYY-MM-DD (exclusive)")
    window.add_argument("--source", action="append", default=[], dest="only",
                        metavar="NAME", help="restrict to one source; repeatable")

    fetch = subparsers.add_parser("fetch", parents=[common, window],
                                  help="read the mailbox into the database")
    fetch.add_argument("--dry-run", action="store_true",
                       help="report what would be stored, store nothing")

    parse_cmd = subparsers.add_parser("parse", parents=[common, window],
                                      help="extract articles from stored messages")
    parse_cmd.add_argument("--force", action="store_true",
                           help="re-parse messages already parsed, for a changed parser")
    parse_cmd.add_argument("--no-links", action="store_true",
                           help="do not resolve tracking links over the network")
    parse_cmd.add_argument("--dry-run", action="store_true")

    digest = subparsers.add_parser(
        "digest", parents=[common, window], aliases=["process", "generate"],
        help="cluster, summarize, rank and write the digest",
    )
    digest.add_argument("--no-llm", action="store_true",
                        help="offline heuristics instead of a model")
    digest.add_argument("--no-embeddings", action="store_true",
                        help="skip embeddings; clustering becomes within-language")
    digest.add_argument("--dry-run", action="store_true",
                        help="build and report, write no digest and no stories")

    run = subparsers.add_parser("run", parents=[common, window],
                                help="fetch, parse and publish in one go")
    run.add_argument("--no-llm", action="store_true")
    run.add_argument("--no-embeddings", action="store_true")
    run.add_argument("--no-links", action="store_true")
    run.add_argument("--no-prune", action="store_true",
                     help="keep data past retention")
    run.add_argument("--force", action="store_true")
    run.add_argument("--dry-run", action="store_true")

    inspect = subparsers.add_parser("inspect", parents=[common, window],
                                    help="what is stored for a window; writes nothing")
    inspect.add_argument("--json", action="store_true")

    subparsers.add_parser("build", parents=[common],
                          help="regenerate digests/ from the database")

    sources_cmd = subparsers.add_parser(
        "sources", parents=[common], help="list sources, or test them against the mailbox"
    )
    sources_cmd.add_argument("--check", action="store_true",
                             help="open the mailbox and report what each rule matches")
    sources_cmd.add_argument("--days", type=int, default=30,
                             help="days of mail to check against (default: 30)")

    stats_cmd = subparsers.add_parser("stats", parents=[common],
                                      help="summarize the database")
    stats_cmd.add_argument("--json", action="store_true")

    explain = subparsers.add_parser("explain", parents=[common],
                                    help="per-term ranking breakdown for one story")
    explain.add_argument("story_id")

    prune = subparsers.add_parser("prune", parents=[common],
                                  help="apply retention now")
    prune.add_argument("--days", type=int, default=None,
                       help="override storage.retention_days")

    return parser


def setup_logging(verbose: bool, quiet: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# --------------------------------------------------------------------------- #
# windows
# --------------------------------------------------------------------------- #

def parse_week(value: str) -> tuple[datetime, datetime]:
    """Turn '2026-W38' into that ISO week's [Monday, next Monday) window."""
    try:
        year_part, week_part = value.upper().split("-W")
        year, week = int(year_part), int(week_part)
        monday = datetime.combine(
            datetime.fromisocalendar(year, week, 1).date(), time.min, tzinfo=timezone.utc
        )
    except (ValueError, TypeError) as exc:
        raise ConfigError(
            f"--week {value!r} is not an ISO week like 2026-W38 ({exc})"
        ) from None
    return monday, monday + timedelta(days=7)


def _date(value: str, flag: str) -> datetime:
    try:
        return datetime.combine(
            datetime.strptime(value, "%Y-%m-%d").date(), time.min, tzinfo=timezone.utc
        )
    except ValueError:
        raise ConfigError(f"{flag} {value!r} is not a date like 2026-09-14") from None


def resolve_window(args: argparse.Namespace, config) -> tuple[datetime, datetime]:
    """The window a command should operate on.

    Four ways to say it, in precedence order: `--from/--to`, `--week`, `--days`,
    and nothing at all.

    Nothing at all means the last FINISHED Monday-to-Sunday week, which is what a
    scheduled run needs and almost never what you want while editing the
    pipeline: on a Friday it rebuilds a week that ended five days ago. `--days`
    is the rolling alternative -- and a rolling window's digest id comes from the
    day it ends on, so a rolling run persisted into the real database claims the
    id of the calendar week it lands in. Pair it with `--db`/`--out` pointing
    somewhere scratch, or with `--dry-run`.
    """
    from_date = getattr(args, "from_date", None)
    to_date = getattr(args, "to_date", None)
    if from_date or to_date:
        start = _date(from_date, "--from") if from_date else utcnow() - timedelta(days=7)
        end = _date(to_date, "--to") if to_date else utcnow()
        if end <= start:
            raise ConfigError(f"--to {to_date} is not after --from {from_date}")
        return start, end

    if getattr(args, "week", None):
        return parse_week(args.week)

    days = getattr(args, "days", None)
    if days:
        if days < 1:
            raise ConfigError(f"--days {days} must be at least 1")
        end = utcnow()
        return end - timedelta(days=days), end

    return weekly_window(week_ends_on=config.digest.week_ends_on)


def _options(args: argparse.Namespace, config) -> pipeline.Options:
    return pipeline.Options(
        db=args.db,
        out=args.out,
        readme=REPO_ROOT / "README.md",
        debug_dir=Path(args.debug) if getattr(args, "debug", None) else None,
        dry_run=getattr(args, "dry_run", False),
        prune=not getattr(args, "no_prune", False),
        force=getattr(args, "force", False),
        only=list(getattr(args, "only", []) or []),
        resolve_links=not getattr(args, "no_links", False),
    )


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def print_summary(stats, *, dry_run: bool = False) -> None:
    """The counters section 22 asks for, in the order the pipeline produces them."""
    rows = [
        ("Fetched", f"{stats.emails_fetched} emails"
                    + (f", {stats.emails_new} new" if stats.emails_new else "")
                    + (f" ({stats.emails_unmatched} unmatched)"
                       if stats.emails_unmatched else "")),
        ("Parsed", f"{stats.emails_parsed} emails"),
        ("Extracted", f"{stats.articles_extracted} items"
                      + (f" ({stats.excluded} excluded, {stats.duplicates} dupes)"
                         if stats.excluded or stats.duplicates else "")),
        ("Stored", f"{stats.articles_new} articles"),
        ("In window", f"{stats.articles_in_window} articles"),
        ("Filtered", f"{stats.filtered} not news"),
        ("Clusters", f"{stats.clusters}"
                     + (f" ({stats.cross_language_clusters} cross-language)"
                        if stats.cross_language_clusters else "")),
        ("Selected", f"{stats.stories_published} stories"
                     + (f" (+{stats.minor_stories} minor)"
                        if stats.minor_stories else "")),
    ]
    # Rows that belong to a stage this command did not run are dropped, so
    # `parse` does not print a confident "Clusters: 0".
    empty_is_noise = {
        "Fetched", "Parsed", "Extracted", "Stored", "In window", "Clusters",
        "Selected",
    }
    rows = [
        (label, value) for label, value in rows
        if not (label in empty_is_noise and value.split()[0] == "0")
    ] or rows
    width = max(len(label) for label, _ in rows) + 2
    print()
    if dry_run:
        print("DRY RUN — nothing was written\n")
    for label, value in rows:
        print(f"{label + ':':<{width}}{value}")
    if stats.languages:
        print(f"{'Languages:':<{width}}"
              + ", ".join(f"{k} {v}" for k, v in stats.languages.items()))
    if stats.llm_calls or stats.embed_calls:
        print(f"{'API calls:':<{width}}{stats.llm_calls} llm, "
              f"{stats.embed_calls} embedding")
    failed = stats.sources_failed
    if failed:
        print("\nSources with errors:")
        for report in failed:
            print(f"  {report.name}: {report.error}")
    print()


def _dispatch(args: argparse.Namespace, config, log: logging.Logger) -> int:
    command = args.command
    if command in ("process", "generate"):
        command = "digest"
    options = _options(args, config)

    if command == "fetch":
        start, end = resolve_window(args, config)
        stats = pipeline.run_fetch(config, options, since=start, until=end)
        print_summary(stats, dry_run=options.dry_run)
        return 0

    if command == "parse":
        stats = pipeline.run_parse(
            config, options, window=resolve_window(args, config)
        )
        print_summary(stats, dry_run=options.dry_run)
        return 0

    if command == "digest":
        stats = pipeline.run_weekly(
            config, options, window=resolve_window(args, config)
        )
        print_summary(stats, dry_run=options.dry_run)
        return 0 if stats.stories_published or options.dry_run else 1

    if command == "run":
        start, end = resolve_window(args, config)
        stats = pipeline.run_all(
            config, options, since=start, until=end, window=(start, end)
        )
        print_summary(stats, dry_run=options.dry_run)
        return 0 if stats.stories_published or options.dry_run else 1

    if command == "inspect":
        return _inspect(args, config, log)

    if command == "build":
        with Store(args.db) as store:
            written = render.rebuild(
                args.out, config, store, readme=options.readme
            )
        print(f"rebuilt {written} digest file(s) in {args.out}")
        return 0

    if command == "sources":
        return _sources(args, config, log)

    if command == "stats":
        with Store(args.db) as store:
            summary = store.summary()
        if args.json:
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        else:
            for key, value in summary.items():
                if key == "last_run":
                    continue
                print(f"{key + ':':<22}{value}")
        return 0

    if command == "explain":
        return _explain(args, config, log)

    if command == "prune":
        days = args.days if args.days is not None else config.storage.retention_days
        with Store(args.db) as store:
            removed = store.prune(
                days,
                config.storage.embedding_retention_days,
                config.storage.email_body_retention_days,
            )
        print(f"pruned {removed} rows (retention {days}d)")
        return 0

    log.error("unknown command %r", args.command)
    return 2


def _inspect(args: argparse.Namespace, config, log: logging.Logger) -> int:
    """Read-only: what is stored for a window, and how it would cluster.

    Writes nothing -- no story, no digest, no cache row -- and builds no provider,
    so it works with no API key, no mailbox and an exhausted quota. This is the
    command for answering "why is that story not in the digest" without paying
    for a run.
    """
    start, end = resolve_window(args, config)
    with Store(args.db) as store:
        emails = store.emails_in_window(start, end)
        articles = select_candidates(
            store.articles_in_window(
                start, end, languages=config.settings.supported_languages
            )
        )
        digest = store.get_digest(store.latest_digest().id) if store.latest_digest() else None

    publishers = sorted({a.publisher for a in articles})
    by_source: dict[str, int] = {}
    for article in articles:
        by_source[article.source] = by_source.get(article.source, 0) + 1

    payload = {
        "window": [start.date().isoformat(), end.date().isoformat()],
        "emails": len(emails),
        "emails_unparsed": sum(1 for m in emails if not m.parsed),
        "articles": len(articles),
        "publishers": len(publishers),
        "by_source": dict(sorted(by_source.items(), key=lambda kv: -kv[1])),
        "languages": {
            lang: sum(1 for a in articles if a.language == lang)
            for lang in sorted({a.language for a in articles if a.language})
        },
        "unclassified": sum(1 for a in articles if a.newsworthy is None),
        "not_news": sum(1 for a in articles if a.newsworthy is False),
        "regions": {
            region: sum(1 for a in articles if a.region == region)
            for region in sorted({a.region for a in articles if a.region})
        },
        "latest_digest": digest.id if digest else None,
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0 if articles else 1

    print(f"\n{start.date()} .. {end.date()}\n")
    for key in ("emails", "emails_unparsed", "articles", "publishers",
                "unclassified", "not_news"):
        print(f"{key + ':':<18}{payload[key]}")
    if payload["languages"]:
        print(f"{'languages:':<18}"
              + ", ".join(f"{k} {v}" for k, v in payload["languages"].items()))
    if payload["regions"]:
        print(f"{'regions:':<18}"
              + ", ".join(f"{k} {v}" for k, v in payload["regions"].items()))
    if by_source:
        print("\nby source:")
        for name, count in payload["by_source"].items():
            print(f"  {name:<24}{count}")
    if not articles:
        print("\nNothing stored for this window.")
        return 1

    # Publisher spread is the number to look at before touching
    # `corroboration_saturation`, which is uncalibrated for this source list: the
    # term saturates at that value, so it has to sit at the top of the range the
    # weeks actually produce rather than above or below it.
    print(f"\n{'publishers:':<18}{len(publishers)} -- {', '.join(publishers)}")
    print(
        f"{'saturation:':<18}{config.preferences.corroboration_saturation:g} "
        f"(corroboration reaches 1.0 here)"
    )
    print()
    return 0


def _sources(args: argparse.Namespace, config, log: logging.Logger) -> int:
    """List the configured sources, or test the match rules against real mail."""
    sources = config.sources
    if not args.check:
        print()
        for source in sources:
            flag = " " if source.enabled else "-"
            print(f"{flag} {source.name:<24}{source.publisher:<18}"
                  f"{','.join(source.languages) or '?':<6}"
                  f"{', '.join(source.senders)}")
            if source.subject_patterns:
                print(f"    subject: {', '.join(source.subject_patterns)}")
            if source.sender_name_patterns:
                print(f"    name   : {', '.join(source.sender_name_patterns)}")
        print(f"\n{len(config.enabled_sources)} of {len(sources)} enabled\n")
        return 0

    # --check is the command for the question that actually bites: the rules look
    # right, so why did nothing match? It reports per source AND lists the senders
    # that matched nothing, which is where a wrong address shows up.
    matcher = Matcher(config.enabled_sources)
    since = utcnow() - timedelta(days=args.days)
    counts: dict[str, int] = {s.name: 0 for s in config.enabled_sources}
    unmatched: dict[str, int] = {}

    try:
        with get_mailbox(config.mailbox) as mailbox:
            raw = mailbox.fetch(since, None)
    except MailboxError as exc:
        log.error("%s", exc)
        return 2

    for message in raw:
        try:
            parsed = parse_message(message.raw)
        except ValueError:
            continue
        source = matcher.match(parsed)
        if source is None:
            key = parsed.sender or "(no sender)"
            unmatched[key] = unmatched.get(key, 0) + 1
        else:
            counts[source.name] += 1

    print(f"\n{len(raw)} messages in the last {args.days} days\n")
    for name, count in counts.items():
        mark = "ok  " if count else "NONE"
        print(f"  {mark} {name:<24}{count}")
    if unmatched:
        print("\nsenders matching no source:")
        for sender, count in sorted(unmatched.items(), key=lambda kv: -kv[1]):
            print(f"  {count:>4}  {sender}")
    silent = [name for name, count in counts.items() if not count]
    print()
    return 1 if silent else 0


def _explain(args: argparse.Namespace, config, log: logging.Logger) -> int:
    with Store(args.db) as store:
        story = store.get_story(args.story_id)
        if story is None:
            log.error("no story %r; ids appear in the digest JSON and debug output",
                      args.story_id)
            return 1
        articles = store.articles_by_story([story.id]).get(story.id, [])

    breakdown = scoring.explain(story, articles, config.preferences)
    print(f"\n{story.headline}\n")
    print(f"{len(articles)} articles from {len({a.publisher for a in articles})} "
          f"publishers, {'/'.join(story.languages) or '?'}")
    if story.regions:
        print(f"region: {story.regions[0]}")
    print()
    total = breakdown.pop("total")
    for term, value in sorted(breakdown.items(), key=lambda kv: -abs(kv[1])):
        weight = config.preferences.ranking.get(term)
        suffix = f"  (weight {weight})" if weight else ""
        print(f"  {term:<22}{value:>9.4f}{suffix}")
    print(f"  {'TOTAL':<22}{total:>9.4f}\n")
    if story.disagreements:
        print("sources differ:")
        for line in story.disagreements:
            print(f"  - {line}")
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose, args.quiet)
    log = logging.getLogger("newsdigest")

    try:
        config = load_config(args.sources, args.preferences, args.filters)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2

    if getattr(args, "no_llm", False):
        config.llm.provider = "none"
    if getattr(args, "no_embeddings", False):
        config.embeddings.provider = "none"

    try:
        return _dispatch(args, config, log)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2
    except MailboxError as exc:
        log.error("mailbox: %s", exc)
        return 3
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
