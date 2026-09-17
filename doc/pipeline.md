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
merge; the band down to `ambiguous_threshold` is referred to the LLM in batches of
`batch_size`, capped at `max_cluster_checks` pairs per run and asked in descending
similarity order; with no embeddings at all, within-language text similarity only —
which barely works, and [providers.md](providers.md) has the numbers.

Merging is transitive (union-find), which is what makes a single false-positive
edge expensive: one wrong link welds two otherwise-clean clusters together. That
is the argument for a *narrow* auto-merge tier and a *wide* adjudicated band
rather than one finely-tuned threshold — see providers.md.

Centroid linkage — requiring two clusters' *average* vectors to match before they
merge, so an outlier has to resemble the whole group rather than its nearest
member — was implemented, measured and removed. On the 2026-W38 window at 0.80 it
rejected 4 of 175 merges, and one was plainly wrong: a 14th article about the
Morelos murder kept out of the 13-article cluster covering that murder, on a
centroid score of 0.755 — the same score as a rejection that was correct, so no
threshold below it separated the two. It rejected 17 of 509 at 0.70 and 31 of 770
at 0.65, so it would only earn its place if the auto-merge tier were loosened.
Don't re-add it as a global gate without that change.

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

## Evaluation

`tests/fixtures/cluster_eval.json` is the ground truth: 27 articles from two
stories dissected by hand, grouped into the 13 real events they actually cover.
Both stories were welded out of unrelated events by the pre-fix clustering —
`s9b661259ac146ae` mixed Ceuta/Morocco with the Podemos primaries and a Junts
piece, `scf09754259aef38` mixed four separate court matters.

Labels are stored as **events, not pairs**: two articles in one event are a
positive, two in different events a negative. Twelve event pairs are listed as
`unsure` and excluded from scoring rather than guessed at — they are all "same
broader affair, but is it one event?" calls. That yields **307 labelled pairs:
43 same, 264 different, 9 of the positives cross-language.**

The fixture carries its own cached vectors, so `python -m newsdigest.eval` needs
no embedder, no ollama and no digest run, and finishes in about a second.
`tests/test_cluster_eval.py` turns the same numbers into regression guards.
Both cover the **embedding tiers only** — with no provider the adjudicated band
is left alone, so recall is a floor.

Measured 2026-09-17:

```
label      languages          n     min     p50     max
different  cross-language    87   0.259   0.491   0.736
different  same-language    177   0.130   0.433   0.705
same       cross-language     9   0.556   0.694   0.782
same       same-language     34   0.536   0.722   0.859

similarity   precision  recall     f1   clusters
      0.80       1.000   0.163  0.280         22
      0.78       1.000   0.372  0.542         20   <- current
      0.76       0.660   0.721  0.689         14
      0.75       0.684   0.907  0.780         12
      0.72       0.457   0.977  0.622          6
      0.70       0.297   1.000  0.457          2
```

Three things that follow, and they are the reason the tiers are shaped as they
are:

- **The bands overlap by 0.20.** Genuine pairs run down to 0.536, unrelated ones
  up to 0.736. No threshold separates them, so the adjudicated band is
  structural, not a stopgap — any threshold low enough to catch the positives
  welds unrelated events. Both fixture stories are that failure.
- **Cross-language coverage is effectively unreachable by the auto-merge tier.**
  Eight of the nine genuine cross-language pairs score below 0.78; only the best
  reaches it, at 0.782. So almost every merge that spans a language comes from
  adjudication or from chaining, not from the embedding — and a threshold anywhere
  near the same-language one cannot change that, because cross-language negatives
  run up to 0.736 while its positives start at 0.556. Bridging languages needs
  adjudication, not a number.
- **0.80 → 0.78 was free, and was adopted on 2026-09-17.** It more than doubles
  recall (0.163 → 0.372) with precision still 1.000 and no documented weld
  returning. The recorded welds come back at 0.72, not before, so there is margin
  left — but below 0.78 precision falls off a cliff (1.000 at 0.78, 0.660 at
  0.76), which is where the adjudicated band has to take over.

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

## Topics

**Topics are not shown to the reader.** There are no tag chips on a story and no
topic filter on the page; both were removed on 2026-09-15. Topics survive as an
internal signal with exactly one job: `excluded_topics`.

They still get computed, in this order: a feed's own `<category>` terms plus the
`topics:` you set on that source in `config/sources.yaml`; the LLM, per article;
the offline provider's keyword matcher when no LLM is configured; and
`clustering.py`, which counts what a cluster's articles were tagged with and keeps
the top four. A multi-source brief then overwrites the story's topics with its own.

The vocabulary is **closed** — the ten topics in `llm/base.TOPICS`. Both prompts
list them and forbid anything else, `normalize_topics` folds known synonyms and
drops whatever is left, and `Enrichment.clamp` / `Brief.clamp` put every path
through it. The offline matcher keys on exactly the same ten, asserted at import
so the two cannot drift.

The reason to keep any of this once nothing is displayed is
`scoring.is_excluded`, which matches `excluded_topics` against `story.topics`
*and* the headline and summary. The topic half is what catches the case the text
half cannot: `sports` appears nowhere in a Catalan football headline, so without a
`sports` topic a Barça match is not excluded at all. A closed vocabulary matters
for the same reason — `excluded_topics: [sports]` has to match the tag the model
actually emitted, and it cannot match `football`.

Ranking does not read topics. `interest` and `relevance` both left the formula on
2026-09-14 and `preferences.topics` is empty, so a topic changes a story's score
only by triggering the `excluded_penalty`.

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
- Topics already written to `data/news.db` keep whatever vocabulary was in force
  when they were written; the closed vocabulary applies to new runs. Until a story
  is re-enriched, `excluded_topics` matches against its old topics.
