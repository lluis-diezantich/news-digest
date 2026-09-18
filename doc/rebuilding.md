# Rebuilding digests from scratch

Sometimes a digest is wrong in a way that re-running does not fix. A run made
with `--no-embeddings` publishes almost-unclustered output; a bad threshold welds
unrelated events together. Because stories are keyed by their cluster's URLs,
re-running a week *replaces* that week — but it leaves every other week alone,
and that is usually not what you want when you are starting over.

This is the reset. It removes every digest and story while keeping the two things
that are expensive or impossible to get back.

## What survives, and why

| Kept | Why |
|---|---|
| `articles` | Collected coverage. **Unrecoverable** — once an item scrolls out of a feed it is gone, and no re-run brings it back. |
| `embedding_cache` | ~1 KB per article and slow to rebuild. Keeping it makes the next clustering pass instant. |
| `llm_cache` | Enrichments keyed by content hash. This is what makes the optional step below free. |
| `sources` | Your configuration and each source's health. |

Everything else is derived and safe to drop: stories, the digests that point at
them, the story↔article join, and the story-side topic and entity vocabularies.
Article topics and entities live in JSON columns on `articles`, not in those
tables, so clearing them does not touch collection.

## Back up first

```bash
cp data/news.db data/news.db.bak
```

Not optional. The delete below is one typo away from the articles table.

## Clear the digests and stories

```bash
sqlite3 data/news.db "
DELETE FROM digest_stories;
DELETE FROM weekly_digests;
DELETE FROM story_entities;
DELETE FROM story_topics;
DELETE FROM article_story;
DELETE FROM stories;
DELETE FROM entities;
DELETE FROM topics;
DELETE FROM runs WHERE kind != 'collect';
"
```

Children before parents, so it works whether or not `PRAGMA foreign_keys` is on.
The `runs` filter keeps collection history, which is how you tell when the feeds
last worked.

## Clear the published files

Deleting rows does not delete what was already written to `docs/`. Stale files
keep appearing in the archive:

```bash
rm docs/digests/*.json
```

`index.html`, `index.json` and `feed.xml` are rewritten by the next build, so
leave them.

## Optional: re-enrich the articles too

Enrichment skips anything already enriched (`pending = [a for a in articles if
not a.enriched]`), so a fresh digest reuses whatever the articles already carry.
To make it genuinely from scratch:

```bash
sqlite3 data/news.db "
UPDATE articles SET summary=NULL, why_it_matters=NULL, key_facts='[]',
  topics='[]', entities='[]', importance=NULL, relevance=NULL,
  content_type=NULL, enriched_by=NULL, enriched_at=NULL;
"
```

This is cheaper than it looks. `llm_cache` still holds those enrichments keyed by
content hash, so the next run's cache pass refills them with **zero LLM calls**.
It only costs real requests for articles that were never enriched successfully.

## Verify, then rebuild

```bash
sqlite3 data/news.db "SELECT
  (SELECT count(*) FROM articles) articles,
  (SELECT count(*) FROM stories) stories,
  (SELECT count(*) FROM weekly_digests) digests;"

.venv/bin/news-digest digest --week 2026-W37 2>&1 | tee /tmp/digest.log
```

Article count unchanged, the other two zero. Piping to a log matters: the
warnings are where failures show up, and they scroll away otherwise.

## Two things that will bite

**The newest week wins, however bad it is.** The site's latest digest is the one
with the highest week id, not the best one. On 2026-09-17 three good rebuilds of
`2026-W37` published 8 stories each while the site kept showing `2026-W38` — two
stories left over from a `--no-embeddings` run — because W38 is a later week and
was still in the database. If the site looks wrong after a rebuild, check which
week is latest before re-running anything:

```bash
sqlite3 data/news.db "SELECT d.digest_id, count(*) FROM digest_stories d
  GROUP BY d.digest_id ORDER BY d.digest_id DESC;"
```

**Enrichment can time out silently.** `LLMSettings.timeout` is 90 seconds and has
no environment variable, while a batch of 16 enrichments on a local 8B model
needs longer. It fails twice, trips `MAX_CONSECUTIVE_FAILURES = 2`, and abandons
the phase — the digest still publishes, so the only sign is `enriched: 0` in the
run stats and a `batch failed` warning in the log. Losing enrichment costs you
`topics`, which is the half of `excluded_topics` that catches what headline text
cannot: `sports` appears nowhere in a Catalan football headline.

## The database is committed

`data/news.db` is in git on purpose ([pipeline.md](pipeline.md) explains why), so
a reset shows up as a large binary diff along with the deleted `docs/digests/`
files. Collection also writes that file every six hours from a GitHub Action, so
commit before the next run rather than after, or you will be resolving the
conflict described in the `.gitignore` note — and only ever in the direction that
keeps the bot's articles.
