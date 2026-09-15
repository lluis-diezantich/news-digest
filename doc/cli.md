# CLI

| Command | What it does |
|---|---|
| `news-digest collect` | Daily: fetch, detect language, dedupe, store. No model calls. |
| `news-digest digest` | Weekly: embed, cluster, LLM, rank, publish. |
| `news-digest digest --week 2026-W36` | Build a specific week. Needed for the current, partial week. |
| `news-digest digest --no-llm --no-embeddings` | Fully offline. |
| `news-digest digest --dry-run` | Process, persist nothing (caches still fill). |
| `news-digest build` | Regenerate `docs/` from the database. |
| `news-digest sources --check` | Fetch every enabled source once and report. |
| `news-digest stats` | Articles, languages, digests, cache sizes, last run. |
| `news-digest explain <story-id>` | Per-term ranking breakdown. |
| `news-digest prune --days 30` | Drop articles past a retention window. |

`--db`, `--out`, `--sources`, `--preferences`, `-v`, `-q` work before or after the
subcommand. `make help` lists the equivalent shortcuts.

## Recipes

**Look at a week that is not the last completed one.** Plain `digest` builds the
previous Monday–Sunday week; the current partial week needs naming:

```bash
news-digest digest --week $(date -u +%G-W%V)
```

**Try a config change without touching your real data or site:**

```bash
cp data/news.db /tmp/t.db
news-digest --db /tmp/t.db --out /tmp/out --preferences /tmp/alt.yaml digest --week 2026-W37
```

Both `--db` and `--out` matter. A bare `digest` writes into `docs/`, overwriting
the published site.

**See why a story ranked where it did:**

```bash
news-digest explain s60bcd7af55eab81
```

**Reset the derived data but keep the articles** — re-collecting only recovers
what is currently in the feeds, so this is usually what "start over" should mean:

```bash
sqlite3 data/news.db "DELETE FROM embedding_cache; DELETE FROM llm_cache;
                      DELETE FROM digest_stories;  DELETE FROM weekly_digests;
                      DELETE FROM article_story;   DELETE FROM stories; VACUUM;"
```

**Browse the database** — `datasette data/news.db --open` is the quickest way to
see it; `sqlite3 -header -column data/news.db` if you would rather query.
