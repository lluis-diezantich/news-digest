# CLI

```
news-digest run                     # fetch, parse, and publish the week
news-digest fetch                   # just read the mailbox
news-digest parse                   # just extract articles from stored messages
news-digest digest                  # just cluster, summarize, rank, write
news-digest inspect                 # what is stored for a window; writes nothing
news-digest sources --check         # do the match rules actually match?
news-digest explain <story-id>      # why did this rank where it did?
news-digest build                   # regenerate digests/ from the database
news-digest stats                   # summarize the database
news-digest prune                   # apply retention now
```

Section 26 of the specification lists `process` and `generate` as separate
commands. They are one command here, `digest`, because clustering, summarizing,
ranking and writing share a transaction — stopping between them would mean
persisting a digest with no summaries in it, which is not a state worth being able
to reach. `process` and `generate` are accepted as aliases.

## Windows

Four ways to say which week, in precedence order:

| Flag | Window |
|---|---|
| `--from DATE --to DATE` | exactly that, `--to` exclusive |
| `--week 2026-W38` | that ISO week, Monday to Monday |
| `--days 7` | the last 7 days, ending **now** |
| *(nothing)* | the last **finished** Monday-to-Sunday week |

The default is what a scheduled run needs and almost never what you want while
editing: on a Friday it rebuilds a week that ended five days ago, and anything
collected since is invisible.

`--days` is the rolling alternative, and it has one trap. A digest's id comes
from the last day of its window, so a rolling run *persisted* into the real
database claims the id of the calendar week it happens to end in and overwrites
it. Pair it with `--db` and `--out` pointing somewhere scratch, or `--dry-run`.

## Recipes

**Try a pipeline change without touching your digest.**

```bash
cp data/news.db /tmp/try.db
news-digest --db /tmp/try.db --out /tmp/out digest --days 7 --debug
```

**Work on one newsletter's extraction.** `--source` is repeatable, and `--force`
re-parses messages already parsed — which is the only reason to want it, and is
scoped to the window and the named sources rather than the whole database.

```bash
news-digest parse --source guardian-saturday --week 2026-W38 --force --debug
```

**Find out why a story is missing.** `inspect` is read-only by construction: no
story, no digest, no cache row, and no provider is built, so it works with no API
key, no mailbox and an exhausted quota.

```bash
news-digest inspect --week 2026-W38
```

```
2026-09-14 .. 2026-09-21

emails:           7
emails_unparsed:  0
articles:        58
publishers:       6
unclassified:     0
not_news:        14
languages:        en 31, es 27
regions:          europe 19, middle east 11, north america 9, ...
```

`not_news` is what topic filtering dropped and `unclassified` is what it never
saw. If a story is in `articles` but not in the digest, `news-digest explain` on
its id gives the per-term ranking breakdown.

**See everything the run did.** `--debug` writes sanitized JSON per stage:

```
debug/emails.json      what arrived, and which source it matched
debug/extracted.json   what came out of the newsletters
debug/filtered.json    what topic filtering dropped
debug/kept.json        what survived it
debug/clusters.json    how the survivors grouped
debug/stories.json     what was published
```

Sender addresses, email bodies and credentials are never in these files — they
are the artefact most likely to be pasted into an issue, so they have to be safe
to paste.

**Run with no API calls at all.**

```bash
news-digest digest --no-llm --no-embeddings
```

**Report without writing.** `--dry-run` fills the caches — derived data that makes
the next real run cheaper — but writes no story, digest or run.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | fine |
| 1 | nothing to publish, or `sources --check` found a silent source |
| 2 | bad configuration, named in the error |
| 3 | the mailbox could not be opened |

A mailbox failure during `run` stops it rather than falling through to a digest of
last week's leftovers. Publishing a stale digest as though it were this week's is
the one silent failure worth being loud about.
