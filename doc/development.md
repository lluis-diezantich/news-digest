# Development

## Tests

```bash
pytest -q
```

353 tests, no network and no API key required. Providers are exercised against
stubbed transports, so nothing downloads a model or calls an API.

They cover URL canonicalization and the similarity metric, language detection
including es/ca, dedupe boundaries, cross-language clustering with stubbed
vectors, the ambiguous-band LLM adjudication and its budget, all four providers
(Gemini and Ollama for the LLM, Gemini and local for embeddings) including the
aggregated-embedding trap, the per-minute/per-day 429 split and its wait budgets,
the thinking-level nesting and its fallback, URL exclusion, feed-position
recording, brief-before-enrichment ordering, the configurable ranking formula, the
store with v1→v2→v3 migrations, the archive renderer, and both pipelines end to
end — with a broken source, an exhausted quota, and cache reuse.

Some tests exist to guard a finding rather than a behaviour. `max_items <= 10` and
the `/especials/` blocklist are asserted because `editorial_position` promotes
advertorial without them; `editorial_position` itself is asserted absent from
nowhere and present in the formula. Read the docstring before "fixing" one.

## Layout

```
config/sources.yaml       what to read, and what to drop on sight
config/preferences.yaml   languages, the ranking formula, digest size, retention
doc/                      this documentation
src/newsdigest/
  cli.py                  argparse entry point
  pipeline.py             the two pipelines and their error boundaries
  digest.py               weekly window, pre-ranking, digest assembly
  config.py               YAML + env loading, validation
  models.py               Article / Story / Digest / RunStats
  lang.py                 language detection
  sources/                rss.py, scrape.py, http.py (robots, rate limiting)
  embeddings/             base.py contract, local.py, gemini.py, none.py
  llm/                    base.py contract, ollama.py, gemini.py, heuristic.py
  dedupe.py               same-article removal
  clustering.py           cross-language grouping
  ratelimit.py            telling a per-minute 429 from a per-day one
  scoring.py              the configurable ranking formula
  store.py                SQLite schema, migrations, queries
  render.py               index.json, digests/*.json, feed.xml
  text.py, urls.py        similarity, normalization, canonicalization
web/index.html            page shell, copied into docs/ on build
docs/                     generated — what GitHub Pages serves
data/news.db              generated — pipeline state, committed by Actions
```

## Adding a provider

Implement the contract, add a registry entry, done — nothing else changes.

- **LLM** (`llm/base.py`): `enrich`, `write_brief`, `same_event`. Register in
  `llm/__init__.py`. `cache_key` includes the provider name, so switching does
  not reuse another provider's enrichments.
- **Embeddings** (`embeddings/base.py`): `embed(texts) -> vectors`, plus a
  `dimensions` attribute. Register in `embeddings/__init__.py`.

Two things to get right, both learned the hard way:

**Similarity thresholds are model-specific.** Cosine values do not transfer
between embedding models, and a stale threshold is a silent quality regression
rather than an error. `embeddings/local.py` warns when it sees one; do the same.

**Batch APIs need checking.** Gemini's `embedContent` returns *one aggregated
vector* for several inputs rather than an error, which would produce
plausible-looking nonsense clusters. `batchEmbedContents` is required, and a test
asserts the count matches.

## Schema changes

Bump `SCHEMA_VERSION` in `store.py` and add a migration step. Currently at 3;
v3 added `feed_position` and `feed_size`. Existing rows keep `-1`, which reads as
"unknown" and scores neutral — backfilling was impossible, and guessing from
`published_at` would have invented a signal.
