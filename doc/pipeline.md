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

The story-level output names every language and outlet covering an event:

```
EU announces new sanctions against Russia          [EN] [ES] [CA]  3 outlets

  EN  BBC World      EU announces new sanctions against Russia
  ES  El País        La UE anuncia nuevas sanciones contra Rusia
  CA  Ara            La UE anuncia noves sancions contra Rússia
```

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
- The offline provider cannot translate, so `output_language` is only honoured
  with a real LLM.
- `min_articles` counts articles rather than publishers, so a multi-feed publisher
  can satisfy it alone.
- The scrape adapter depends on per-site CSS selectors and will break when a site
  redesigns. `news-digest sources --check` tells you which.
