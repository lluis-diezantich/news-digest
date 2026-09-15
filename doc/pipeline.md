# How it works

```
DAILY    sources → fetch → normalize → detect language → deduplicate → SQLite
WEEKLY   SQLite  → embed → cluster across languages → LLM → rank → digest → site
```

Keeping those apart is the whole architecture. Collection must be cheap and
reliable enough to run constantly, so it calls **no model at all**. The expensive
semantic work happens once, over a week that has already finished.

## Daily

**Fetch.** One adapter per `method` in `config/sources.yaml`. `rss` is preferred;
`scrape` is the fallback for sources with no usable feed. All HTTP goes through
one client that sends an identifying User-Agent, rate limits per host, and honours
`robots.txt` including `Crawl-delay`. Each source runs in its own error boundary —
a broken feed is recorded in the `sources` table and the run continues.

Both adapters record `feed_position` and `feed_size`: where the article sat in its
source's listing, and how long that listing was. For RSS that is feed order, for
scrape it is DOM order on the section page — and both are the newsroom's own
ranking. See [configuration.md](configuration.md) for what that is used for, and
what it must not be used for.

**Normalize + detect.** Everything becomes an `Article` with title, publisher,
URL, publication date, author, language and a **short excerpt**. The id is a hash
of the canonicalized URL, so tracking parameters cannot create phantom articles.

**Filter.** `exclude_url_patterns` drops structural junk before it reaches the
database — see [configuration.md](configuration.md).

**Deduplicate.** Narrow on purpose: only the *same article* is removed — a
canonical URL already stored, or one outlet re-running a near-identical headline.
Two outlets covering one event are **not** duplicates, and neither are a
publisher's Spanish and English editions of one story. That is clustering's job.

## Weekly

**Embed.** One vector per article, cached by content hash, so re-running a week is
free. This is the only thing that can match a Catalan headline to an English one.

**Cluster.** Three tiers, cheapest first: cosine ≥ `similarity_threshold` is a
merge; the band down to `ambiguous_threshold` is referred to the LLM, capped at
`max_cluster_checks` pairs per run; with no embeddings at all, within-language
text similarity only — which barely works, and
[providers.md](providers.md) has the numbers.

**Pre-rank.** Clusters are ordered using only signals collection already provided
— publisher count, coverage volume, source weights, recency — and only the top
`2 × max_stories` are enriched. That is the difference between summarizing six
stories and summarizing five hundred articles.

**LLM.** Briefs first, then per-article enrichment with whatever budget remains.
The brief is the only LLM output a reader sees on the page; enrichment is
scaffolding that feeds ranking and gets cached. The order used to be reversed,
and on a free tier that spent the whole daily allowance on scaffolding.

**Rank + publish.** `scoring.py` applies the formula from
`config/preferences.yaml`, then writes the digest, the archive and RSS.

## Multilingual

Articles keep their original language, title and URL — nothing is translated on
collection. The **output** language is separate and configurable.

Detection runs during collection, restricted to the languages you configured.
Asking "en, es or ca?" is a far easier question than picking from 97, and that
restriction is what makes the es/ca pair reliable. A source's declared `languages`
both constrains the answer and supplies the fallback when a headline is too short
to judge.

Measured on 440 collected articles from single-language sources, detecting
*without* the source constraint and comparing against each source's own
declaration: **440/440**. Benchmarking the underlying detector alone on 195
headlines gave 99.5%; the wrapper's length and confidence guards account for the
rest. `lingua` was tried and rejected — 99.0% at 307 MB against py3langid's
4.6 MB.

One story gathers every outlet covering the event, each link marked with the
language it opens in:

```
EU announces new sanctions against Russia                          3 outlets

  EN  BBC World      EU announces new sanctions against Russia
  ES  El País        La UE anuncia nuevas sanciones contra Rusia
  CA  Ara            La UE anuncia noves sancions contra Rússia
```

The per-link marker is the only language the page shows. A story-level
`[EN] [ES] [CA]` badge row and a per-digest `EN 12 · ES 30 · CA 8` tally were both
removed: which languages a story happened to be covered in is an artefact of the
source list, not something the reader is choosing between. On a link it is
different — it says what you get if you click.

## Tags

Tags come from four places, in this order: a feed's own `<category>` terms plus
the `topics:` you set on that source in `config/sources.yaml`; the LLM, per
article; the offline provider's keyword matcher when no LLM is configured; and
`clustering.py`, which counts what a cluster's articles were tagged with and
keeps the top four. A multi-source brief then overwrites the story's tags with
its own.

The vocabulary is **closed** — the ten topics in `llm/base.TOPICS`. Both prompts
list them and forbid anything else, `normalize_topics` folds known synonyms and
drops whatever is left, and `Enrichment.clamp` / `Brief.clamp` put every path
through it. The offline matcher keys on exactly the same ten, asserted at import
so the two cannot drift.

It is closed because it was once open — "2-4 broad lowercase topics" with no
list. Over one archive that produced 48 distinct tags for 95 stories: `ai`
beside `artificial intelligence`, `politics` beside `spanish politics` and
`us politics`, `world` beside `geopolitics`, `diplomacy` and `international
relations`, and 27 tags used exactly once, most of them place names that were
already in `entities`. Folded onto the vocabulary the same archive is 10 tags.
The chips are a filter, and a filter whose values are mostly unique is not one.

Dropping an unrecognized tag is deliberate: it is either a synonym of a tag that
already exists, splitting one filter in two, or an entity wearing a topic's
clothes. A story left with nothing falls back to its feed-declared topics. There
is no `general` — it was the second most common tag in that archive and told the
reader nothing.

## Known limits

- Without embeddings, cross-language coverage stays split. Measured, documented,
  and asserted in tests rather than hidden.
- Clustering is O(n²) inside the weekly window. At ~2000 articles that is 2M
  cosine comparisons — one numpy matmul, milliseconds. It needs an index long
  before it needs a rewrite.
- `data/news.db` is committed as a binary blob, and vectors dominate its size.
  `embedding_retention_days` is the knob. Switching embedding provider leaves both
  sets cached, since `cache_key` includes the provider — deliberate, so nothing
  silently mixes, but it doubles that storage until retention prunes it.
- The heuristic provider cannot translate, so `output_language` is only honoured
  with a real LLM.
- `min_articles` counts articles rather than publishers, so a multi-feed publisher
  can satisfy it alone.
- The scrape adapter depends on per-site CSS selectors and will break when a site
  redesigns. `news-digest sources --check` tells you which.
- Tags already written to `data/news.db` keep whatever vocabulary was in force
  when they were written; the closed vocabulary applies to new runs. A story
  whose only tag was `general` shows none until it is re-enriched.
