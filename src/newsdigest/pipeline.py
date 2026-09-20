"""The pipeline, in the order section 5 lays it out.

    EMAIL -> INGEST -> PARSE -> NORMALIZE -> FILTER -> DEDUPLICATE
          -> CLUSTER -> RANK -> SUMMARIZE -> DIGEST -> STORE

One cadence, not two. The RSS version of this project split collection from
processing because feeds churn hourly and a scheduled job that called a model
every hour would be unaffordable. Newsletters arrive weekly, so the split buys
nothing here and the whole thing is one `run` -- which is also what sections 4
and 25 ask for: the same command locally and in Actions, differing only in where
the secrets come from.

The stages remain separately invocable (`fetch`, `parse`, `digest`) because that
is what makes iterating on any one of them cheap. Each is idempotent: a message is
fetched once, parsed once, and its articles classified and enriched once, however
many times you re-run.

Resilient by construction: one unreadable message, one newsletter that yields
nothing, one failed batch or one exhausted quota degrades the digest rather than
aborting the run. The mailbox itself is the single exception -- with no mail there
is no digest, so a mailbox that will not open is fatal and says why.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import digest as digest_module
from . import lang, render
from .config import Config
from .debug import DebugWriter
from .dedupe import LOOKBACK_HOURS, dedupe
from .embeddings import get_provider as get_embedder
from .extract import extract, to_articles
from .inbox import Matcher, assign, get_mailbox, parse_message
from .llm import build_context, get_provider as get_llm
from .models import Email, RunStats, SourceReport, to_utc, utcnow
from .net import Resolver
from .store import Store
from .urls import exclude_by_title, exclude_by_url

log = logging.getLogger(__name__)


@dataclass
class Options:
    db: Path
    out: Path
    readme: Path | None = None
    debug_dir: Path | None = None
    dry_run: bool = False
    prune: bool = True
    #: Re-parse messages that have already been parsed. For when the extractor
    #: has changed, which is the only reason to want it.
    force: bool = False
    #: Restrict to these source names. For working on one newsletter at a time.
    only: list[str] = field(default_factory=list)
    #: Resolve tracking links over the network. Off makes the whole parse stage
    #: offline and deterministic, at the cost of unresolved attribution.
    resolve_links: bool = True


def _selected(config: Config, options: Options) -> list:
    sources = config.enabled_sources
    if not options.only:
        return sources
    wanted = {name.lower() for name in options.only}
    chosen = [s for s in sources if s.name.lower() in wanted]
    missing = wanted - {s.name.lower() for s in chosen}
    if missing:
        log.warning("--source named %s, which is not an enabled source", sorted(missing))
    return chosen


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #

def fetch(
    config: Config,
    store: Store,
    stats: RunStats,
    *,
    since: datetime | None,
    until: datetime | None,
    options: Options,
    debug: DebugWriter | None = None,
) -> list[Email]:
    """Read the mailbox and store any message we have not seen.

    Returns the messages that were new, which is what `parse` would work on next
    -- though `parse` reads them back from the database rather than taking them
    from here, so the two commands behave identically whether they run together
    or days apart.
    """
    sources = _selected(config, options)
    if not sources:
        log.warning("no enabled sources in config")
        return []

    matcher = Matcher(sources)
    store.register_sources(sources)

    with get_mailbox(config.mailbox) as mailbox:
        raw = mailbox.fetch(since, until)

    matched: list[Email] = []
    unmatched: list[Email] = []
    per_source: dict[str, SourceReport] = {
        s.name: SourceReport(name=s.name, ok=True) for s in sources
    }

    for message in raw:
        try:
            parsed = parse_message(message.raw)
        except ValueError as exc:
            # One unreadable message must not cost the week.
            log.warning("message %s: %s", message.uid, exc)
            continue

        # IMAP's date search is day-granular in a timezone we do not know, so the
        # window is enforced here, on the parsed header, where it is exact.
        received = to_utc(parsed.received_at)
        if since and received < to_utc(since):
            continue
        if until and received >= to_utc(until):
            continue

        source = matcher.match(parsed)
        if source is None:
            unmatched.append(parsed)
            continue
        assign(parsed, source)
        matched.append(parsed)
        per_source[source.name].emails += 1

    stats.emails_fetched = len(matched)
    stats.emails_unmatched = len(unmatched)

    known = store.known_email_ids([m.id for m in matched])
    fresh = [m for m in matched if m.id not in known]
    if not options.dry_run:
        stats.emails_new = store.insert_emails(fresh)
    else:
        stats.emails_new = len(fresh)

    for message in fresh:
        per_source[message.source].new_emails += 1

    for report in per_source.values():
        stats.sources.append(report)

    if debug is not None and debug.enabled:
        debug.dump_emails("emails", matched, unmatched)

    log.info(
        "%d messages matched %d sources (%d new, %d unmatched)",
        len(matched), len(sources), stats.emails_new, len(unmatched),
    )
    if unmatched:
        log.info(
            "unmatched senders: %s",
            ", ".join(sorted({m.sender.rpartition("@")[2] for m in unmatched})),
        )
    return fresh


# --------------------------------------------------------------------------- #
# parse
# --------------------------------------------------------------------------- #

def parse(
    config: Config,
    store: Store,
    stats: RunStats,
    *,
    window: tuple[datetime, datetime],
    options: Options,
    debug: DebugWriter | None = None,
) -> int:
    """Extract articles from stored messages. Returns how many were new."""
    sources = _selected(config, options)
    by_name = {s.name: s for s in sources}
    if not sources:
        return 0

    lang.configure(config.settings.supported_languages)
    if options.force:
        # Scoped to the window and the selected sources, never the whole
        # database: --force is for re-running one week against a changed
        # extractor, and re-parsing three years of mail is not that.
        reset = store.reset_parsed(
            [
                m.id
                for m in store.emails_in_window(
                    *window, sources=list(by_name)
                )
            ]
        )
        log.info("--force: %d messages marked unparsed", reset)

    messages = store.emails_in_window(
        *window, sources=list(by_name), unparsed_only=not options.force
    )
    if not messages:
        log.info("no unparsed messages in %s .. %s", window[0].date(), window[1].date())
        return 0

    reports = {report.name: report for report in stats.sources}
    for source in sources:
        reports.setdefault(source.name, SourceReport(name=source.name, ok=True))
        if reports[source.name] not in stats.sources:
            stats.sources.append(reports[source.name])

    fingerprints = store.recent_fingerprints(LOOKBACK_HOURS)
    all_articles = []
    unresolved_total = 0

    resolver = Resolver(config.user_agent) if options.resolve_links else None
    try:
        for message in messages:
            source = by_name.get(message.source)
            if source is None:
                continue
            report = reports[source.name]
            started = time.monotonic()
            try:
                items = extract(message, excerpt_chars=source.excerpt_chars)
                articles, unresolved = to_articles(
                    message,
                    items,
                    publisher=source.publisher,
                    weight=source.weight,
                    topics=source.topics,
                    resolver=resolver,
                )
                unresolved_total += unresolved
                report.extracted += len(articles)

                kept = exclude_by_url(articles, source.exclude_url_patterns)
                by_url = len(articles) - len(kept)
                kept = exclude_by_title(kept, source.exclude_title_patterns)
                report.excluded += len(articles) - len(kept)
                articles = kept

                for article in articles:
                    article.language = lang.detect_article(
                        article.title,
                        article.excerpt(),
                        allowed=source.languages,
                        fallback=source.languages[0] if source.languages else None,
                    )

                known = store.known_ids([a.id for a in articles])
                keep, dropped = dedupe(
                    articles, known_ids=known, known_fingerprints=fingerprints
                )
                report.duplicates += len(dropped)
                report.new += (
                    store.insert_articles(keep) if not options.dry_run else len(keep)
                )
                if not options.dry_run:
                    store.mark_parsed(message.id, len(keep))
                message.article_count = len(keep)

                for article in keep:
                    fingerprints.append((article.id, "", article.source))
                    key = article.language or "unknown"
                    stats.languages[key] = stats.languages.get(key, 0) + 1
                all_articles.extend(keep)

                log.info(
                    "%-22s %-38s %3d items  %3d new  %2d dupes  %2d excluded",
                    source.name, message.subject[:38], len(items),
                    len(keep), len(dropped), by_url,
                )
            except Exception as exc:  # deliberately broad: isolate the message
                report.ok = False
                report.error = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "%-22s FAILED on %r: %s",
                    source.name, message.subject[:40], report.error,
                )
            report.elapsed_ms += int((time.monotonic() - started) * 1000)
    finally:
        if resolver is not None:
            resolver.close()

    stats.emails_parsed = len(messages)
    stats.articles_extracted = sum(r.extracted for r in reports.values())
    stats.articles_new = sum(r.new for r in reports.values())
    stats.duplicates = sum(r.duplicates for r in reports.values())
    stats.excluded = sum(r.excluded for r in reports.values())

    if unresolved_total:
        log.info(
            "%d links could not be resolved past their tracker; they are kept, "
            "but section blocklists cannot see them",
            unresolved_total,
        )
    if stats.sources_failed:
        log.warning(
            "%d of %d sources hit an error: %s",
            len(stats.sources_failed), len(sources),
            ", ".join(s.name for s in stats.sources_failed),
        )
    if debug is not None and debug.enabled:
        debug.dump("extracted", all_articles)
    return stats.articles_new


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #

def run_fetch(
    config: Config,
    options: Options,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
) -> RunStats:
    stats = RunStats(started_at=utcnow())
    debug = DebugWriter(options.debug_dir)
    log.info("mailbox: %s", config.mailbox.describe())
    with Store(options.db) as store:
        fetch(config, store, stats, since=since, until=until,
              options=options, debug=debug)
        if not options.dry_run:
            store.record_run(stats, kind="fetch")
    return stats


def run_parse(
    config: Config,
    options: Options,
    *,
    window: tuple[datetime, datetime],
) -> RunStats:
    stats = RunStats(started_at=utcnow())
    debug = DebugWriter(options.debug_dir)
    with Store(options.db) as store:
        parse(config, store, stats, window=window, options=options, debug=debug)
        if not options.dry_run:
            store.record_run(stats, kind="parse")
    return stats


def run_weekly(
    config: Config,
    options: Options,
    *,
    now: datetime | None = None,
    window: tuple[datetime, datetime] | None = None,
) -> RunStats:
    """Cluster, summarize, rank and publish, from what is already stored."""
    stats = RunStats(started_at=utcnow())
    debug = DebugWriter(options.debug_dir)

    with Store(options.db) as store:
        start, end = window or digest_module.weekly_window(
            now, week_ends_on=config.digest.week_ends_on
        )
        # `output_language: auto` is resolved here, before the context is built:
        # a Context carrying the literal "auto" would reach the prompts.
        counts = store.language_counts_in_window(start, end)
        output_language = config.settings.resolve_output_language(counts)
        config.llm.output_language = output_language

        llm = get_llm(config.llm)
        embedder = get_embedder(config.embeddings)
        context = build_context(
            config.settings, config.preferences, output_language=output_language
        )

        log.info(
            "llm: %s (%s) | embeddings: %s (%s, %dd) | writing in: %s%s",
            llm.name, config.llm.model if llm.name != "none" else "offline heuristics",
            embedder.name, embedder.model, embedder.dimensions, output_language,
            " (auto)" if config.settings.output_language == "auto" else "",
        )

        result = digest_module.build_digest(
            config, store, llm, embedder, context, stats,
            window=(start, end), persist=not options.dry_run,
            debug=debug if debug.enabled else None,
        )
        if not options.dry_run:
            render.write_all(
                options.out, config, store, result, readme=options.readme
            )
            store.record_run(stats, kind="digest")

    log.info(
        "digest %s: %d stories, %d llm calls, %d embed calls",
        stats.digest_id, stats.stories_published, stats.llm_calls, stats.embed_calls,
    )
    return stats


def run_all(
    config: Config,
    options: Options,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    window: tuple[datetime, datetime] | None = None,
    now: datetime | None = None,
) -> RunStats:
    """fetch -> parse -> digest, sharing one set of counters.

    This is what the scheduled workflow runs and what section 25 means by no
    meaningful difference between local and production.

    A mailbox failure stops the run here rather than falling through to a digest
    of last week's leftovers: publishing a stale digest as though it were this
    week's is the one silent failure worth being loud about.
    """
    stats = RunStats(started_at=utcnow())
    debug = DebugWriter(options.debug_dir)
    period = window or digest_module.weekly_window(
        now, week_ends_on=config.digest.week_ends_on
    )

    log.info("mailbox: %s", config.mailbox.describe())
    with Store(options.db) as store:
        # Deliberately not caught: a mailbox failure stops the run here rather
        # than falling through to a digest of last week's leftovers. Publishing a
        # stale digest as though it were this week's is the one silent failure
        # worth being loud about. The CLI reports it and exits 3.
        fetch(
            config, store, stats,
            since=since or period[0], until=until or period[1],
            options=options, debug=debug,
        )
        parse(config, store, stats, window=period, options=options, debug=debug)

        if options.prune and not options.dry_run:
            removed = store.prune(
                config.storage.retention_days,
                config.storage.embedding_retention_days,
                config.storage.email_body_retention_days,
            )
            if removed:
                log.info("pruned %d rows past retention", removed)

    weekly = run_weekly(config, options, window=period, now=now)

    # One report for the whole run: the digest half carries the counters the
    # ingestion half does not, and the reverse.
    for name in (
        "clusters", "cross_language_clusters", "embedded", "embed_cached",
        "embed_calls", "classified", "classify_cached", "filtered", "enriched",
        "enrich_cached", "enrich_failed", "llm_calls", "llm_checks",
        "stories_total", "stories_new", "stories_published", "minor_stories",
        "digest_id",
    ):
        setattr(stats, name, getattr(weekly, name))
    stats.articles_in_window = weekly.articles_in_window
    return stats
