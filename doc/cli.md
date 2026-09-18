# CLI

| Command | What it does |
|---|---|
| `news-digest collect` | Daily: fetch, detect language, dedupe, store. No model calls. |
| `news-digest digest` | Weekly: embed, cluster, LLM, rank, publish. |
| `news-digest digest --week 2026-W36` | Build a specific week. Needed for the current, partial week. |
| `news-digest digest --days 7` | Build a rolling window ending now, not a calendar week. Local runs only -- see the recipe below. |
| `news-digest digest --no-llm --no-embeddings` | No model at all, built-in heuristics instead. |
| `news-digest digest --dry-run` | Process, persist nothing (caches still fill). |
| `news-digest build` | Regenerate `docs/` from the database. |
| `news-digest sources --check` | Fetch every enabled source once and report. |
| `news-digest stats` | Articles, languages, digests, cache sizes, last run. |
| `news-digest explain <story-id>` | Per-term ranking breakdown. |
| `news-digest themes` | What the week was ABOUT, by subject. No model, read-only. |
| `news-digest themes iran` | Every article on one theme. |
| `news-digest prune --days 30` | Drop articles past a retention window. |

`--db`, `--out`, `--sources`, `--preferences`, `-v`, `-q` work before or after the
subcommand. `make help` lists the equivalent shortcuts.

## Themes — what the week was about

`digest` ranks **events**. `themes` groups by **subject**, keyed on the proper
nouns in the headlines, over the whole window.

The two disagree, and the disagreement is the point. Measured on 2026-W38, the
Iran war's 39 headlines sat in 31 separate event clusters, 27 of them a single
outlet, the largest reaching two publishers — so nothing about it could clear the
digest's ranking and the week's biggest international story was invisible. Keyed
on `iran` it is one subject with 7 outlets across three languages. Ceuta fails the
other way: 161 articles in 67 clusters, four of which are big enough to take
separate digest slots for one subject.

```bash
news-digest themes                     # the week that just finished
news-digest themes --days 3            # rolling window ending now
news-digest themes --week 2026-W37     # a specific finished week
news-digest themes iran                # every article on one theme, newest first
news-digest themes puig                # a prefix or any merged key works
news-digest themes --top 20 --min-outlets 2
news-digest themes --json
```

```
2026-09-14 .. 2026-09-21 · 1365 articles · 16 outlets · 96 themes

  #  outlets  arts  days   sub   spread  theme
  1       13   194     5    17     0.45  ! ceuta  (nacional, audiencia, ...)
  2       13    82     5     9     0.31  ! trump  (canada, donald, donald trump)
  3       10    17     4     1        -    puigdemont  (llarena, tjue)
  4        9    34     4     7     0.41  ! sanchez  (david, david sanchez)
  7        7    26     5     1        -    iran
```

`sub` is how many separate multi-outlet stories the theme holds; `spread` is how
alike they are. A `!` means the theme holds several unlike stories, so **no single
summary represents it** — `trump` above is nine unrelated stories averaging 0.31
similarity. `iran` and `puigdemont` hold one each and would summarize cleanly.

That column is why themes are not fed to the LLM. Measured across six windows,
18 of 48 top-eight theme slots were dispersed — 38% — so briefing "the theme"
would have produced one arbitrary sub-story more than a third of the time. It gets
worse as a window fills: the quiet week had none, the busy week four.

`sub` and `spread` are the only part that needs the embedding cache. They read it,
never write to it and never call a model; with nothing cached for the window they
show `-` and the rest of the output is unaffected.

Watch the merged keys on a dominant theme. `ceuta` at 194 articles absorbed 34 of
them, including `psoe`, `feijoo`, `cis` and `melilla`, because almost any Spanish
political key clears the 60% overlap rule against something that large. The
biggest theme is therefore also reliably the most dispersed one.

**No model, no embeddings.** It is a proper-noun regex over stored headlines plus
set arithmetic, so it runs in about a second, works with no API key and an
exhausted quota, and reads the same whether or not the week was enriched.

**Read-only.** No story, digest or cache row is written and no provider is built,
so it is safe against `data/news.db` directly rather than a scratch copy.

**Separate from the digest.** Nothing in the weekly pipeline calls it. Feeding it
back as a one-story-per-theme diversity cap was measured and rejected: it removed
the Trump tariff threat, the Fed's rate rise and Carney's EU visit from one top
eight — three unrelated stories that entity-keying files under `trump` — and
replaced them with a schools announcement and an opinion piece. `trump` is a
container in the way `catalunya` is, and unlike a place you cannot enumerate those
in advance.

Tune it in `config/preferences.yaml` under `themes:` — `containers` is the list of
names that locate a story rather than being one. It is hand-maintained because no
metric replaces it: article coherence scores `catalunya` 0.588 against `trump`
0.566, so any automatic floor that drops the places drops the subjects too.

## Recipes

**Look at a week that is not the last completed one.** Plain `digest` builds the
previous Monday–Sunday week; the current partial week needs naming:

```bash
news-digest digest --week $(date -u +%G-W%V)
```

**See what happened lately, not what happened last week.** `--days N` ends the
window at *now* rather than on a week boundary, so it picks up everything collected
since — the window you want while you are changing the pipeline:

```bash
cp data/news.db /tmp/try.db
news-digest --db /tmp/try.db --out /tmp/out digest --days 7
```

The scratch `--db` and `--out` are not optional here. The digest id is derived from
the last day of the window, so a rolling run that ends inside 2026-W38 is *stored
as* `2026-W38` and replaces that week's stories. Keep it out of `data/news.db`.

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
