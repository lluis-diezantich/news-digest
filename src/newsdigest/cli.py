"""Command line entry point.

    news-digest run                 # the whole pipeline (what Actions runs)
    news-digest run --no-llm        # same, with offline heuristic enrichment
    news-digest fetch               # just collect new articles
    news-digest enrich --limit 20   # just enrich, with a tighter budget
    news-digest cluster             # just re-cluster and re-score
    news-digest build               # just regenerate docs/ from the database
    news-digest sources --check     # verify every configured source responds
    news-digest stats               # what is in the database
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from . import pipeline
from .config import (
    DEFAULT_DB,
    DEFAULT_INTERESTS,
    DEFAULT_OUT,
    DEFAULT_SOURCES,
    ConfigError,
    load_config,
)
from .llm import get_provider
from .models import RunStats, utcnow
from .sources import Fetcher, adapter_for
from .store import Store


def _add_global_args(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Shared options.

    Added to the top-level parser with real defaults and to every subparser with
    SUPPRESS defaults, so `--db X run` and `run --db X` both work and the
    subparser only overrides what was actually typed.
    """

    def default(value):
        return argparse.SUPPRESS if suppress else value

    parser.add_argument("--sources", type=Path, default=default(DEFAULT_SOURCES),
                        help=f"sources YAML (default: {DEFAULT_SOURCES})")
    parser.add_argument("--interests", type=Path, default=default(DEFAULT_INTERESTS),
                        help=f"interests YAML (default: {DEFAULT_INTERESTS})")
    parser.add_argument("--db", type=Path, default=default(DEFAULT_DB),
                        help=f"SQLite database (default: {DEFAULT_DB})")
    parser.add_argument("--out", type=Path, default=default(DEFAULT_OUT),
                        help=f"static site output directory (default: {DEFAULT_OUT})")
    parser.add_argument("-v", "--verbose", action="store_true",
                        default=default(False), help="debug logging")
    parser.add_argument("-q", "--quiet", action="store_true",
                        default=default(False), help="warnings and errors only")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="news-digest",
        description="Personal news aggregator: fetch, enrich, cluster, publish.",
    )
    _add_global_args(parser, suppress=False)

    common = argparse.ArgumentParser(add_help=False)
    _add_global_args(common, suppress=True)

    subparsers = parser.add_subparsers(dest="command", required=True, parser_class=argparse.ArgumentParser)

    run_cmd = subparsers.add_parser("run", parents=[common], help="run the full pipeline")
    run_cmd.add_argument("--no-fetch", action="store_true", help="skip fetching")
    run_cmd.add_argument("--no-llm", action="store_true",
                         help="use offline heuristic enrichment instead of the LLM")
    run_cmd.add_argument("--no-enrich", action="store_true", help="skip enrichment entirely")
    run_cmd.add_argument("--no-build", action="store_true", help="skip writing the site")
    run_cmd.add_argument("--no-prune", action="store_true", help="keep articles past retention")
    run_cmd.add_argument("--limit", type=int, default=None,
                         help="max articles to send to the LLM this run")
    run_cmd.add_argument("--site-url", default=os.environ.get("SITE_URL", ""),
                         help="public URL of the site, used in the RSS output")
    run_cmd.add_argument("--dry-run", action="store_true",
                         help="fetch and enrich, but write no site files and record no run")

    subparsers.add_parser("fetch", parents=[common], help="fetch and store new articles only")

    enrich_cmd = subparsers.add_parser("enrich", parents=[common], help="enrich stored articles only")
    enrich_cmd.add_argument("--limit", type=int, default=None, help="article budget")
    enrich_cmd.add_argument("--no-llm", action="store_true", help="offline heuristics")

    subparsers.add_parser("cluster", parents=[common], help="re-cluster and re-score stored articles")

    build_cmd = subparsers.add_parser("build", parents=[common], help="regenerate docs/ from the database")
    build_cmd.add_argument("--site-url", default=os.environ.get("SITE_URL", ""))

    sources_cmd = subparsers.add_parser("sources", parents=[common], help="list configured sources")
    sources_cmd.add_argument("--check", action="store_true",
                             help="fetch each enabled source once and report the result")

    stats_cmd = subparsers.add_parser("stats", parents=[common], help="summarize the database")
    stats_cmd.add_argument("--json", action="store_true", help="machine-readable output")

    prune_cmd = subparsers.add_parser("prune", parents=[common], help="drop articles past the retention window")
    prune_cmd.add_argument("--days", type=int, default=None, help="override retention_days")

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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose, args.quiet)
    log = logging.getLogger("newsdigest")

    try:
        config = load_config(args.sources, args.interests)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2

    if getattr(args, "no_llm", False):
        config.llm.provider = "none"

    try:
        return _dispatch(args, config, log)
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130


def _dispatch(args: argparse.Namespace, config, log: logging.Logger) -> int:
    if args.command == "run":
        options = pipeline.Options(
            db=args.db,
            out=args.out,
            fetch=not args.no_fetch,
            do_enrich=not args.no_enrich,
            build=not args.no_build,
            prune=not args.no_prune,
            enrich_limit=args.limit,
            site_url=args.site_url,
            dry_run=args.dry_run,
        )
        stats = pipeline.run(config, options)
        _print_summary(stats)
        # A run where every source failed is a real failure worth a red build.
        if stats.sources and stats.sources_ok == 0:
            log.error("every source failed")
            return 1
        return 0

    if args.command == "fetch":
        stats = RunStats(started_at=utcnow())
        with Store(args.db) as store:
            pipeline.fetch_stage(config, store, stats)
            store.record_run(stats)
        _print_summary(stats)
        return 0 if stats.sources_ok else 1

    if args.command == "enrich":
        stats = RunStats(started_at=utcnow())
        provider = get_provider(config.llm)
        with Store(args.db) as store:
            pipeline.enrich_stage(config, store, provider, stats, args.limit)
        print(f"enriched {stats.enriched}, cached {stats.enrich_cached}, "
              f"failed {stats.enrich_failed}, llm calls {stats.llm_calls}")
        return 0

    if args.command == "cluster":
        stats = RunStats(started_at=utcnow())
        provider = get_provider(config.llm)
        with Store(args.db) as store:
            pipeline.cluster_stage(config, store, provider, stats)
        print(f"{stats.stories_total} stories ({stats.stories_new} new)")
        return 0

    if args.command == "build":
        stats = RunStats(started_at=utcnow())
        with Store(args.db) as store:
            feed = pipeline.build_stage(config, store, args.out, stats, args.site_url)
        print(f"wrote {feed['story_count']} stories "
              f"({feed['article_count']} articles) to {args.out}")
        return 0

    if args.command == "sources":
        return _sources_command(args, config)

    if args.command == "stats":
        with Store(args.db) as store:
            summary = store.summary()
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print(f"database        {summary['db']} ({summary['size_kb']} KB)")
            print(f"articles        {summary['articles']} "
                  f"({summary['unenriched']} awaiting enrichment)")
            print(f"stories         {summary['stories']}")
            print(f"cached llm      {summary['cached_enrichments']}")
            print(f"runs recorded   {summary['runs']}")
            last = summary["last_run"]
            if last:
                print(f"last run        {last['finished_at']}  "
                      f"{last['articles_new']} new, "
                      f"{last['stories_published']} published, "
                      f"{last['llm_calls']} llm calls")
                for failure in last["sources_failed"]:
                    print(f"  ! {failure['name']}: {failure['error']}")
        return 0

    if args.command == "prune":
        days = args.days if args.days is not None else config.interests.retention_days
        with Store(args.db) as store:
            removed = store.prune(days)
        print(f"pruned {removed} rows older than {days} days")
        return 0

    return 2


def _sources_command(args: argparse.Namespace, config) -> int:
    health = {}
    if args.db.exists():
        with Store(args.db) as store:
            health = {row["name"]: row for row in store.source_health()}

    print(f"{'source':24} {'method':7} {'w':>4}  status")
    print("-" * 78)
    failures = 0
    for source in config.sources:
        state = "disabled" if not source.enabled else "enabled"
        row = health.get(source.name)
        if row and row["consecutive_failures"]:
            state = f"failing x{row['consecutive_failures']}: {(row['last_error'] or '')[:34]}"
        elif row and row["last_ok"]:
            state = f"ok, {row['total_articles']} articles, last {row['last_ok'][:16]}"
        print(f"{source.name[:23]:24} {source.method:7} {source.weight:4.1f}  {state}")

    if not args.check:
        return 0

    print("\nchecking enabled sources...")
    with Fetcher(config.user_agent) as fetcher:
        for source in config.enabled_sources:
            try:
                articles = adapter_for(source, fetcher).fetch(source)
                newest = max(
                    (a.published_at for a in articles if a.published_at), default=None
                )
                print(f"  ok    {source.name[:23]:24} {len(articles):3d} articles"
                      f"{f', newest {newest:%Y-%m-%d %H:%M}' if newest else ''}")
            except Exception as exc:
                failures += 1
                print(f"  FAIL  {source.name[:23]:24} {type(exc).__name__}: {exc}")
    return 1 if failures else 0


def _print_summary(stats: RunStats) -> None:
    print(
        f"sources {stats.sources_ok}/{len(stats.sources)} ok | "
        f"{stats.articles_new} new articles ({stats.duplicates} dupes) | "
        f"enriched {stats.enriched} (+{stats.enrich_cached} cached) | "
        f"{stats.stories_published} stories published | "
        f"{stats.llm_calls} llm calls"
    )
    for failure in stats.sources_failed:
        print(f"  ! {failure.name}: {failure.error}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
