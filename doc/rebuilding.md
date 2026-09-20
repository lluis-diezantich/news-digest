# Rebuilding

## Regenerate the Markdown

`digests/` is an artefact of the database, not a second copy of the truth. Delete
it and restore it:

```bash
rm -rf digests
news-digest build
```

This rewrites every digest file and the archive index from the stored stories. It
writes no new stories and makes no API calls. Use it after changing `render.py`.

## Re-parse a week after changing the extractor

Messages are parsed once. To make a changed extractor see them again:

```bash
news-digest parse --week 2026-W38 --force
```

`--force` is scoped to the window and any `--source` you name, never the whole
database — re-parsing three years of mail is not what you meant.

Their existing articles are left in place. The extractor is deterministic, so a
re-parse produces the same ids and the insert is a no-op; what changes is what a
*new* extractor finds.

## Re-cluster and re-rank a week

```bash
news-digest digest --week 2026-W38
```

Story ids derive from the cluster's member URLs, so re-running a week refreshes
stories in place rather than duplicating them. Cached classifications, enrichments
and embeddings are reused, so this is close to free.

## Start over completely

```bash
rm -rf digests debug data/news.db
news-digest run --from 2026-01-01 --to 2026-09-21
```

Everything is re-fetched from the mailbox, which is why the mailbox is opened
read-only and nothing is ever marked seen: the inbox is the source of truth and
this project is a cache over it.

Bear in mind your provider may not hold a year of mail, and `--from`/`--to` in one
run means every stage processes the whole range at once — one very large
classification and enrichment bill. Better to walk it a week at a time.

## Shorten the retention window

Changing `email_body_retention_days` affects the next `prune`, not the history.
Apply it now:

```bash
news-digest prune
```

Bodies older than the window are emptied and the rows kept. If the database has
been committed, **the old bodies are still in your git history** — shortening the
window does not clean it. Rewriting history is the only way, and for a personal
repository it is usually not worth it; keeping the window short from the start is.

## What must survive

| Do not delete | Why |
|---|---|
| the `emails` rows | they are what stops every newsletter being re-ingested |
| `parsed_at` | the "already parsed" flag; `--force` is the way to clear it deliberately |
| the `llm_cache` rows | the only reason re-running a week is affordable |

## Two failure modes that look like bugs

**A rolling window overwrote a calendar week.** A digest's id comes from the last
day of its window, so `--days 7` run on a Friday produces an id for the calendar
week it happens to end in and replaces that week's digest. Point scratch runs at
`--db` and `--out` somewhere temporary, or use `--dry-run`.

**Nothing is published and nothing is obviously wrong.** Almost always one of:

- The source rules do not match. `news-digest sources --check`.
- The messages are already parsed, so `parse` has nothing to do. That is correct;
  `digest` is the command you want, or `--force` if the extractor changed.
- `min_articles` is above what the week corroborates. It falls back to the ranked
  list rather than yielding nothing, so this shows up as fewer stories than
  expected rather than zero.
- Everything was filtered. If filtering would remove every article it is ignored
  for that run and says so in the log — but a partial over-filter is quieter. Check
  `not_news` in `news-digest inspect` and `debug/filtered.json`.
