"""Command line entry point.

The two pipelines are separate commands because they run on different schedules:

    news-digest collect                 # daily: fetch, detect language, store
    news-digest digest                  # weekly: embed, cluster, LLM, publish
    news-digest digest --no-llm         # same, offline heuristics, no API calls
    news-digest digest --week 2026-W36  # rebuild a specific past week
    news-digest digest --days 7         # rolling window ending now, for local runs
    news-digest build                   # regenerate docs/ from the database
    news-digest sources --check         # verify every configured source responds
    news-digest stats                   # what is in the database
    news-digest explain <story-id>      # why a story ranked where it did
    news-digest themes                  # what the week was ABOUT, no model needed
    news-digest themes iran             # every article on one theme
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

from . import pipeline, render, scoring, themes as themes_module
from .config import (
    DEFAULT_DB,
    DEFAULT_OUT,
    DEFAULT_PREFERENCES,
    DEFAULT_SOURCES,
    ConfigError,
    load_config,
)
from .models import Digest, utcnow
from .sources import Fetcher, adapter_for
from .store import Store


def _add_global_args(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Shared options.

    Added to the top-level parser with real defaults and to every subparser with
    SUPPRESS defaults, so `--db X digest` and `digest --db X` both work and the
    subparser only overrides what was actually typed.
    """

    def default(value):
        return argparse.SUPPRESS if suppress else value

    parser.add_argument("--sources", type=Path, default=default(DEFAULT_SOURCES),
                        help=f"sources YAML (default: {DEFAULT_SOURCES})")
    parser.add_argument("--preferences", type=Path, default=default(DEFAULT_PREFERENCES),
                        help=f"preferences YAML (default: {DEFAULT_PREFERENCES})")
    parser.add_argument("--db", type=Path, default=default(DEFAULT_DB),
                        help=f"SQLite database (default: {DEFAULT_DB})")
    parser.add_argument("--out", type=Path, default=default(DEFAULT_OUT),
                        help=f"static site output directory (default: {DEFAULT_OUT})")
    parser.add_argument("-v", "--verbose", action="store_true", default=default(False),
                        help="debug logging")
    parser.add_argument("-q", "--quiet", action="store_true", default=default(False),
                        help="warnings and errors only")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="news-digest",
        description="Personal multilingual news aggregator: collect daily, digest weekly.",
    )
    _add_global_args(parser, suppress=False)
    common = argparse.ArgumentParser(add_help=False)
    _add_global_args(common, suppress=True)

    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", parents=[common],
                                   help="daily collection: fetch, normalize, store")
    collect.add_argument("--no-prune", action="store_true", help="keep articles past retention")
    collect.add_argument("--dry-run", action="store_true",
                         help="fetch and store, but record no run and prune nothing")

    digest = subparsers.add_parser("digest", parents=[common],
                                   help="weekly processing: embed, cluster, LLM, publish")
    digest.add_argument("--no-llm", action="store_true",
                        help="offline heuristic enrichment instead of the LLM")
    digest.add_argument("--no-embeddings", action="store_true",
                        help="skip embeddings; clustering becomes within-language only")
    window_group = digest.add_mutually_exclusive_group()
    window_group.add_argument("--week",
                              help="ISO week to build, e.g. 2026-W36 (default: last week)")
    window_group.add_argument("--days", type=int, metavar="N",
                              help="rolling window of the last N days, ending now, instead "
                                   "of a calendar week -- for local experiments, not the "
                                   "scheduled run (see rolling_window)")
    digest.add_argument("--site-url", default=os.environ.get("SITE_URL", ""),
                        help="public URL of the site, used in the RSS output")
    digest.add_argument("--dry-run", action="store_true",
                        help="process but persist nothing: no stories, digest, site "
                             "or run record (caches are still filled)")

    build = subparsers.add_parser("build", parents=[common],
                                  help="regenerate docs/ from the database")
    build.add_argument("--site-url", default=os.environ.get("SITE_URL", ""))

    sources_cmd = subparsers.add_parser("sources", parents=[common],
                                        help="list configured sources")
    sources_cmd.add_argument("--check", action="store_true",
                             help="fetch each enabled source once and report the result")

    stats_cmd = subparsers.add_parser("stats", parents=[common], help="summarize the database")
    stats_cmd.add_argument("--json", action="store_true", help="machine-readable output")

    explain = subparsers.add_parser("explain", parents=[common],
                                    help="show a story's ranking breakdown")
    explain.add_argument("story_id", help="story id, as shown in the digest JSON")

    themes_cmd = subparsers.add_parser(
        "themes", parents=[common],
        help="what the week was about, keyed on the names in the headlines",
    )
    themes_cmd.add_argument(
        "theme", nargs="?",
        help="show every article on one theme instead of the list (any of its keys)",
    )
    themes_window = themes_cmd.add_mutually_exclusive_group()
    themes_window.add_argument("--week", help="an ISO week like 2026-W36")
    themes_window.add_argument("--days", type=int, metavar="N",
                               help="rolling window of N days ending now")
    themes_cmd.add_argument("--top", type=int, default=None,
                            help="themes to list (default: themes.top)")
    themes_cmd.add_argument("--min-outlets", type=int, default=None, dest="min_outlets",
                            help="outlets a theme needs (default: themes.min_publishers)")
    themes_cmd.add_argument("--json", action="store_true", help="machine-readable output")

    prune = subparsers.add_parser("prune", parents=[common],
                                  help="drop articles past the retention window")
    prune.add_argument("--days", type=int, default=None, help="override retention_days")

    return parser


def setup_logging(verbose: bool, quiet: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)-24s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def parse_week(value: str) -> tuple[datetime, datetime]:
    """Turn '2026-W36' into that ISO week's [Monday, next Monday) window."""
    try:
        year_part, week_part = value.upper().split("-W")
        year, week = int(year_part), int(week_part)
        monday = datetime.combine(
            datetime.fromisocalendar(year, week, 1).date(), time.min, tzinfo=timezone.utc
        )
    except (ValueError, TypeError) as exc:
        raise ConfigError(
            f"--week {value!r} is not an ISO week like 2026-W36 ({exc})"
        ) from None
    return monday, monday + timedelta(days=7)


def rolling_window(days: int) -> tuple[datetime, datetime]:
    """The last `days` days ending right now: "what happened lately".

    Deliberately not the default, and deliberately not aligned to midnight.

    Not the default because the scheduled run needs a window that is the same
    whenever it fires; `digest.weekly_window` gives it a finished calendar week
    for that reason. This is the opposite trade -- a window that moves with the
    clock, which is what you want when you are changing the pipeline and re-running
    it against whatever has been collected so far.

    Not aligned to midnight because two runs on the same day would then cover the
    same window, and the digest id is derived from the window's last day
    (`Digest.week_id`), so a rolling run persisted into the real database would
    overwrite the calendar week that shares that id. Ending at `now` keeps every
    run distinct, but the id collision is still there: pair this with `--db` and
    `--out` pointing somewhere scratch, or `--dry-run`. cli.md has the recipe.
    """
    if days < 1:
        raise ConfigError(f"--days {days} must be at least 1")
    end = utcnow()
    return end - timedelta(days=days), end


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose, args.quiet)
    log = logging.getLogger("newsdigest")

    try:
        config = load_config(args.sources, args.preferences)
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
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130


def _cached_vectors(store, config, articles) -> dict:
    """Vectors already in the cache, and never a model call.

    `themes` is a reading command, so it reads the embedding_cache table the way it
    reads the articles table. Building the provider is free -- the local model
    loads lazily on its first `embed`, which never happens here -- and only its
    `cache_key` is wanted, since vectors from different models are not comparable.
    """
    from .embeddings import get_provider

    try:
        key = get_provider(config.embeddings).cache_key()
    except Exception:  # a provider we cannot even name has no cache to read
        return {}
    hashes = {a.id: a.content_hash() for a in articles}
    cached = store.cached_vectors(list(set(hashes.values())), key)
    return {aid: cached[h] for aid, h in hashes.items() if h in cached}


#: Vector coverage below which dispersion is not attempted. Under this, clustering
#: would fall back to comparing 1000+ articles by text similarity -- slow, and a
#: reading command should not do that for a column.
_MIN_VECTOR_COVERAGE = 0.5


def _theme_spread(found, articles, vectors, config) -> dict:
    """{theme key: (sub_clusters, mean similarity)} for every theme."""
    if not articles or len(vectors) / len(articles) < _MIN_VECTOR_COVERAGE:
        return {}
    from . import clustering

    events = clustering.cluster(
        articles, vectors,
        similarity_threshold=config.embeddings.similarity_threshold,
        ambiguous_threshold=config.embeddings.ambiguous_threshold,
        provider=None,
    )
    return {t.key: themes_module.dispersion(t, events, vectors) for t in found}


def _spread_json(value, vectors) -> dict:
    subs, mean = value
    if not vectors:
        return {"sub_clusters": None, "spread": None, "dispersed": None}
    return {
        "sub_clusters": subs,
        "spread": round(mean, 4) if mean is not None else None,
        "dispersed": themes_module.is_dispersed(subs, mean),
    }


def _themes_command(args: argparse.Namespace, config, log: logging.Logger) -> int:
    """List what the window was about, or expand one theme.

    Read-only by construction: it opens the store, reads articles, and prints. No
    story, digest or cache row is written, and no provider is built -- which is why
    it still works with no API key and an exhausted quota.
    """
    from .digest import select_candidates, weekly_window

    if args.days:
        window = rolling_window(args.days)
    elif args.week:
        window = parse_week(args.week)
    else:
        window = weekly_window(week_ends_on=config.digest.week_ends_on)

    with Store(args.db) as store:
        articles = select_candidates(
            store.articles_in_window(
                *window, languages=config.settings.supported_languages
            )
        )
        vectors = _cached_vectors(store, config, articles)
    if not articles:
        log.warning("no articles between %s and %s", window[0].date(), window[1].date())
        return 1

    found = themes_module.group(
        articles,
        containers=config.themes.containers,
        min_publishers=args.min_outlets or config.themes.min_publishers,
    )
    spread = _theme_spread(found, articles, vectors, config)

    if args.theme:
        theme = themes_module.find(found, args.theme)
        if theme is None:
            log.error("no theme matching %r; run without an argument to list them",
                      args.theme)
            return 1
        if args.json:
            payload = theme.to_json()
            payload.update(_spread_json(spread.get(theme.key, (0, None)), vectors))
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0
        subs, mean = spread.get(theme.key, (0, None))
        print(f"{theme.key} · {len(theme.publishers)} outlets · "
              f"{len(theme.articles)} articles · {theme.days}d · "
              f"{'+'.join(theme.languages)}")
        if vectors and subs:
            note = " -- no single summary represents it" if themes_module.is_dispersed(
                subs, mean) else ""
            story = "story" if subs == 1 else "stories"
            print(f"{subs} separate multi-outlet {story} inside it"
                  + (f", spread {mean:.2f}{note}" if mean is not None else note))
        if len(theme.keys) > 1:
            shown = ", ".join(theme.keys[:8])
            extra = len(theme.keys) - 8
            print(f"merged keys: {shown}" + (f", and {extra} more" if extra > 0 else ""))
        print()
        for article in theme.newest_first():
            stamp = (article.published_at or article.collected_at).strftime("%d %b %H:%M")
            print(f"{stamp}  {article.language or '??'}  {article.publisher:<14} "
                  f"{article.title}")
            print(f"{'':<32}{article.url}")
        print()
        print("outlets: " + ", ".join(f"{p} ({n})" for p, n in theme.by_publisher()))
        return 0

    top = args.top or config.themes.top
    if args.json:
        print(json.dumps(
            {
                "window": [window[0].isoformat(), window[1].isoformat()],
                "articles": len(articles),
                "themes": [
                    {**t.to_json(),
                     **_spread_json(spread.get(t.key, (0, None)), vectors)}
                    for t in found[:top]
                ],
            },
            indent=2, ensure_ascii=False,
        ))
        return 0

    print(f"{window[0].date()} .. {window[1].date()} · {len(articles)} articles · "
          f"{len({a.publisher for a in articles})} outlets · {len(found)} themes")
    print()
    print(f"{'#':>3}  {'outlets':>7}  {'arts':>4}  {'days':>4}  {'sub':>4}  "
          f"{'spread':>7}  theme")
    dispersed_seen = False
    for index, theme in enumerate(found[:top], 1):
        label = theme.key
        if len(theme.keys) > 1:
            label += f"  ({', '.join(k for k in theme.keys[1:4] if k != theme.key)})"
        subs, mean = spread.get(theme.key, (0, None))
        flag = themes_module.is_dispersed(subs, mean)
        dispersed_seen = dispersed_seen or flag
        subs_col = "-" if not vectors else str(subs)
        mean_col = "-" if mean is None else f"{mean:.2f}"
        print(f"{index:>3}  {len(theme.publishers):>7}  {len(theme.articles):>4}  "
              f"{theme.days:>4}  {subs_col:>4}  {mean_col:>7}  "
              f"{'! ' if flag else '  '}{label}")
    print()
    if not vectors:
        print("sub/spread need the embedding cache, which holds nothing for this "
              "window; run `digest` for it, or ignore the columns")
    else:
        print("sub = separate multi-outlet stories inside the theme; spread = how "
              "alike they are")
        if dispersed_seen:
            print("!   = holds several unlike stories, so no single summary "
                  "represents it")
    print(f"`{args.db.name if hasattr(args.db, 'name') else 'news.db'}` unchanged — "
          f"run `news-digest themes <key>` for one theme's articles")
    return 0


def _dispatch(args: argparse.Namespace, config, log: logging.Logger) -> int:
    options = pipeline.Options(
        db=args.db,
        out=args.out,
        site_url=getattr(args, "site_url", ""),
        dry_run=getattr(args, "dry_run", False),
        prune=not getattr(args, "no_prune", False),
    )

    if args.command == "collect":
        stats = pipeline.run_collect(config, options)
        print(
            f"sources {stats.sources_ok}/{len(stats.sources)} ok | "
            f"{stats.articles_new} new articles ({stats.duplicates} dupes) | "
            f"languages {stats.languages or '{}'}"
        )
        for failure in stats.sources_failed:
            print(f"  ! {failure.name}: {failure.error}")
        if stats.sources and stats.sources_ok == 0:
            log.error("every source failed")
            return 1
        return 0

    if args.command == "digest":
        window = None
        if args.days:
            window = rolling_window(args.days)
        elif args.week:
            window = parse_week(args.week)
        stats = pipeline.run_weekly(config, options, window=window)
        print(
            f"digest {stats.digest_id} | {stats.stories_published} stories from "
            f"{stats.clusters} clusters ({stats.cross_language_clusters} cross-language) | "
            f"embeddings {stats.embedded} new/{stats.embed_cached} cached | "
            f"enriched {stats.enriched} (+{stats.enrich_cached} cached) | "
            f"{stats.llm_calls} llm + {stats.embed_calls} embed calls"
        )
        return 0

    if args.command == "build":
        with Store(args.db) as store:
            payload = render.write_site(
                args.out, config, store, None, site_url=args.site_url
            )
        print(f"wrote {payload.get('story_count', 0)} stories to {args.out}")
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
                  f"({summary['unenriched']} never enriched)")
            print(f"languages       {summary['languages']}")
            print(f"stories         {summary['stories']}")
            print(f"digests         {summary['digests']} "
                  f"(latest {summary['latest_digest'] or 'none'})")
            print(f"cached llm      {summary['cached_enrichments']}")
            print(f"cached vectors  {summary['cached_vectors']}")
            last = summary["last_run"]
            if last:
                print(f"last run        {summary['last_run_kind']} at {last['finished_at']}")
                for failure in last["sources_failed"]:
                    print(f"  ! {failure['name']}: {failure['error']}")
        return 0

    if args.command == "themes":
        return _themes_command(args, config, log)

    if args.command == "explain":
        with Store(args.db) as store:
            story = store.get_story(args.story_id)
            if story is None:
                log.error("no story %r in %s", args.story_id, args.db)
                return 1
            articles = store.articles_by_story([story.id]).get(story.id, [])
        print(f"{story.headline}\n")
        print(f"  {len(articles)} articles from {len({a.publisher for a in articles})} "
              f"outlets in {sorted({a.language for a in articles if a.language})}\n")
        breakdown = scoring.explain(story, articles, config.preferences)
        width = max(len(k) for k in breakdown)
        for name, value in breakdown.items():
            if name == "total":
                print(f"  {'-' * (width + 8)}")
            weight = config.preferences.ranking.get(name)
            suffix = f"   (weight {weight})" if weight is not None else ""
            print(f"  {name:<{width}}  {value:+.4f}{suffix}")
        return 0

    if args.command == "prune":
        days = args.days if args.days is not None else config.storage.retention_days
        with Store(args.db) as store:
            removed = store.prune(days, config.storage.embedding_retention_days)
        print(f"pruned {removed} rows older than {days} days")
        return 0

    return 2


def _sources_command(args: argparse.Namespace, config) -> int:
    health = {}
    if args.db.exists():
        with Store(args.db) as store:
            health = {row["name"]: row for row in store.source_health()}

    print(f"{'source':26} {'publisher':16} {'lang':6} {'w':>4}  status")
    print("-" * 92)
    for source in config.sources:
        state = "enabled" if source.enabled else "disabled"
        row = health.get(source.name)
        if row and row["consecutive_failures"]:
            state = f"failing x{row['consecutive_failures']}: {(row['last_error'] or '')[:28]}"
        elif row and row["last_ok"]:
            state = f"ok, {row['total_articles']} articles, last {row['last_ok'][:16]}"
        print(f"{source.name[:25]:26} {source.publisher[:15]:16} "
              f"{','.join(source.languages)[:5]:6} {source.weight:4.1f}  {state}")

    if not args.check:
        return 0

    print("\nchecking enabled sources...")
    failures = 0
    with Fetcher(config.user_agent) as fetcher:
        for source in config.enabled_sources:
            try:
                articles = adapter_for(source, fetcher).fetch(source)
                newest = max((a.published_at for a in articles if a.published_at), default=None)
                print(f"  ok    {source.name[:25]:26} {len(articles):3d} articles"
                      f"{f', newest {newest:%Y-%m-%d %H:%M}' if newest else ''}")
            except Exception as exc:
                failures += 1
                print(f"  FAIL  {source.name[:25]:26} {type(exc).__name__}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
