# How it works

```
EMAIL → INGEST → PARSE → NORMALIZE → FILTER → DEDUPLICATE
      → CLUSTER → RANK → SUMMARIZE → DIGEST → MARKDOWN
```

One cadence. Newsletters arrive weekly, so the whole thing runs once a week. The
stages remain separately invocable because that is what makes iterating on any one
of them cheap, and each is idempotent: a message is fetched once, parsed once, and
its articles classified and enriched once, however many times you re-run.

## Ingest — `inbox/`

The mailbox is opened read-only over IMAP4 with TLS. Messages are searched by
date, parsed, matched to a configured source, and stored.

Two details carry weight:

**The date search is widened by a day at each end, and the window enforced in
Python.** IMAP's `SINCE`/`BEFORE` compare against the server's own `INTERNALDATE`
at day granularity in a timezone we do not know, so a newsletter that arrived at
23:40 UTC can sit on the wrong side of the server's midnight. Over-fetching costs
a few messages; under-fetching loses a newsletter for good.

**A message is keyed on its `Message-ID`**, which is globally unique by definition
and stable across mailbox moves and re-downloads. When a newsletter has none, an
id is synthesised from a hash of the bytes — not from the clock and not from the
server's UID, either of which would re-ingest the same newsletter every week.

Unmatched senders are counted and logged, never guessed at. A digest built from
"probably The Economist" is worse than one built from nine sources.

## Parse — `extract/`

The stage with no schema on its input, and the only one with no equivalent in a
feed reader. A feed hands over titled entries; a newsletter hands over a table
layout from 2006 and expects a human to read it.

The approach is structural rather than per-publisher. Every newsletter item is the
same shape — a link to an article, a headline, and usually a sentence or two — so
the extractor finds the links that could be articles and takes, for each, the
**largest enclosing block that still holds only that one article**.

Largest rather than smallest, because the blurb is a sibling of the headline
rather than a child of it. And *one article* rather than *one link*: a newsletter
links its lead story three times from one block — the image, the headline, and a
"read more" — and counting anchors made that block look like three items, so the
block never grew and the lead story was published with no summary. It is the item
most worth getting right, and the failure was invisible in the counts.

Nothing depends on a class name or a table depth, both of which change without
notice. There are no per-source strategies yet; `tests/fixtures/emails/` is where
to add one when a real newsletter defeats the structural rule.

### What is not news

A newsletter's furniture has exactly the shape the extractor is looking for. A
sponsor slot in particular *is* a headline, a blurb and a link. Three defences:

1. **Structural**, in `boilerplate.strip_chrome` — the tags that cannot hold a
   story, and the block around an unsubscribe or copyright line.
2. **Per link** — social hosts, unsubscribe and preference links, "read more",
   and the publisher's own masthead, which is a link with no path.
3. **Per block** — anything announcing itself as paid placement. Publishers are
   required to print that label, and it is the only reliable signal, because
   advertorial copy is written to look like editorial.

### Tracking links — `extract/links.py`

Newsletter links go through a click tracker:

```
https://link.mail.elpais.com/c/eJx1kM...
```

Three things stay broken while it does: attribution points at a URL that expires;
the same article carries a different opaque token in every newsletter, so
`canonical_url` cannot tell they are one article; and `exclude_url_patterns`
matches section paths — `/deportes/`, `/horoscopo/` — which a tracker does not
have, so every blocklist entry silently matches nothing.

Two mechanisms, cheapest first. Most trackers put the destination in a query
parameter, which costs nothing to unwrap and is applied recursively, because a
link is sometimes wrapped twice — a publisher's tracker inside a mail vendor's.
Only an opaque token needs the network.

`resolve()` returns the URL **and whether it is real**, and the caller acts on the
flag. An unresolved tracker is still stored and still published — a story we can
only link through a tracker is a story, and dropping it would silently shrink the
digest whenever a publisher changed mailers — but the run says how many there
were.

## Normalize

Language is detected per article, with the candidate set restricted to the
languages you configured. Asking "is this English or Spanish?" is a far easier
question than asking which of 97 languages it is, and a source's declared
`languages` acts as both filter and fallback: publishers know what language they
publish in, while the detector has a headline and a sentence.

`collected_at` is the **email's** timestamp, not the clock. It is the honest answer
— the item reached us when the newsletter did — and it makes every window and
recency calculation work without pretending to know a publication time nobody
sent.

## Filter — `classify.py`

One cheap batched call per `batch_size` articles, before clustering, answering
three things: topics, region, and is-this-news-at-all. Cached on the article's
content hash, so iterating on the prompt is free after the first run.

It runs at the **article** level. A source that runs one sports item has not
disqualified its front page.

Every rule needs a positive signal to drop. An unclassified article is kept, a
failed batch is kept, an exhausted quota keeps everything left, and an article is
dropped on its topics only when *every* topic it carries is excluded — a story
tagged `politics, sports` is a head of state at a stadium opening, and dropping it
on the sports tag alone loses the political story. If filtering would remove
everything, it is ignored for that run: that is far more likely to be a broken
classifier than a week with no news in it.

## Deduplicate — `dedupe.py`

Deliberately narrow: this removes articles that are **the same article**, and
nothing else. Two outlets covering one event are not duplicates — they are the
corroboration the digest is built on, and clustering handles them. Collapsing them
here would throw away the "sources covering this story" list.

## Cluster — `clustering.py`

Three tiers, cheapest first:

| | |
|---|---|
| cosine ≥ `similarity_threshold` | same event, free |
| between the two thresholds | ask the LLM, capped at `max_cluster_checks` pairs, highest similarity first |
| no embeddings available | within-language text similarity, which leaves cross-language coverage split — and says so |

Embeddings are not optional for the thing this project exists to do. Measured on
one headline in three languages, token overlap scores 0.00 (en/es) and 0.06
(en/ca) against a 0.60 merge threshold. No threshold rescues that.

The text fallback threshold of 0.45 was calibrated on 37,776 real within-language
pairs **from RSS feeds**: genuine same-event pairs that text can detect at all
scored 0.49–0.78, the highest unrelated pair reached 0.34. Reworded coverage of one
event lands at 0.30–0.33, inside the noise, and is not recoverable by text at any
threshold. That calibration has not been redone on newsletter text.

## Rank — `scoring.py`

```
final_score = sum(weight × signal for each term in ranking.terms)
              − excluded_penalty (if the story hits an excluded topic)
```

Deterministic, and driven entirely by `config/preferences.yaml`. Dropping a term
removes that signal from the maths entirely; a misspelled one is a startup error.
`news-digest explain` prints the breakdown, and it provably sums to the score used
for ranking — computed from the same rounded terms it displays, because computing
it twice let the two disagree.

The shipped formula is four terms:

- **`editorial_position` (2.0)** — where the editor put it. A curated newsletter is
  two judgements rather than one: they chose these items out of the day's hundreds
  *and* chose which one opens. Primary by intent; the digest should read like the
  newsletters it is built from.
- **`corroboration` (1.5)** — how many independent publishers carried it. The honest
  measure of newsworthiness, and 0.00 on every story until embeddings exist.
- **`recency` (0.3)** and **`story_size` (0.2)** — tiebreaks.

`importance` and `relevance` are computed, shown, and deliberately **not** in the
formula. On a measured week of the RSS project, qwen3:8b returned 0.80–0.85 for
every briefed story — a spread of 0.05, so at any weight it was a constant in
signal's clothing. Worse, it inverted: stories that were never briefed kept a
higher derived value, so being briefed cost a story rank.

Geographic spread is applied at **selection**, never to the score
(`regions.diversify`). The order stays by score — a story does not become the
week's lead because of where it happened — and the cap only decides which stories
make the cut. It is not a quota: when a region genuinely holds most of the week's
major news, the cap runs out of alternatives and those stories publish anyway.

## Summarize — `enrich.py`, `prompts/`

Briefs are written **before** per-article enrichment. The order used to be the
other way round, and on a free tier that spent the whole daily allowance analysing
articles and then had nothing left to write the digest with. Enrichment is
scaffolding — it feeds ranking and is cached for next time — while the brief is the
only LLM output a reader sees.

Only the articles in candidate clusters are ever enriched, and they are
interleaved round-robin so the per-run cap is shared. Flat concatenation gave the
whole budget to the first cluster: on one measured run all 40 enriched articles
landed in one 52-article cluster and seven of the eight published stories had none
at all.

Disagreements are a separate field, and a separate section in the output. Never
folded into the summary, and never carried over from a previous run — an empty list
is the model's answer that the coverage does not conflict, and preserving an old
disagreement over that would assert one it just denied.

## Known limits

- **The ranking constants are uncalibrated.** Carried over from a source list that
  no longer exists. `corroboration_saturation` in particular: ten newsletters
  cannot exceed ten publishers, and the default of 5 is a guess.
- **Publication times are unknown.** Newsletters rarely date their items, so
  `recency` mostly measures when the newsletter arrived. Over a weekly window with
  a 72h half-life that matters less than it sounds, but it is not what the term
  claims to measure.
- **Per-source extraction is not implemented.** The structural rule handles the
  fixtures; a newsletter built differently enough will need its own strategy.
- **The text-similarity fallback threshold is inherited, not measured.**
- **Offline filtering is weak.** With no model, the keyword classifier answers
  topics only — it never claims something is not news, because "final" and "corona"
  would drop a court ruling and a public-health story.
