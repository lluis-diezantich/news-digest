# news-digest

A personal news aggregator that runs entirely on GitHub. It watches the RSS
feeds you list, normalizes what it finds, removes duplicates, uses an LLM to
summarize and classify each article, groups articles covering the same event into
one story, ranks stories against your stated interests, and publishes a static
page on GitHub Pages.

```
fetch → normalize → deduplicate → store → LLM enrich → cluster → score → publish
```

Deterministic work happens in Python. The LLM is used only for language
understanding — summarizing, classifying topics, extracting entities, estimating
importance, and naming the underlying event so coverage can be clustered. Ranking
is plain arithmetic over weights you control, so the feed's ordering is
inspectable and reproducible.

---

## Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

# No API key needed: --no-llm uses offline heuristic enrichment.
news-digest run --no-llm
open docs/index.html
```

With an LLM (Google Gemini's free tier by default):

```bash
cp .env.example .env
# put your key from https://aistudio.google.com/apikey into GEMINI_API_KEY
news-digest run
```

Run `news-digest sources --check` to confirm every configured feed responds
before you trust a schedule with it.

---

## Setting it up on GitHub

1. Push this directory to a new repository.
2. **Secret** — Settings → Secrets and variables → Actions → *New repository
   secret*: `GEMINI_API_KEY`. Without it the workflow still succeeds, falling
   back to offline enrichment.
3. **Pages** — Settings → Pages → Source: *Deploy from a branch*, branch `main`,
   folder `/docs`.
4. **Variables** (optional, same screen as secrets): `SITE_URL` so the generated
   RSS carries absolute links; `LLM_MODEL`, `LLM_ARTICLES_PER_RUN` to override
   defaults without editing the workflow.
5. Actions → *Update digest* → **Run workflow** for the first run. After that it
   runs every three hours.

The workflow commits `docs/` (what Pages serves) and `data/news.db` (the state
that makes deduplication and story continuity work) back to the branch.

---

## How each stage works

**Fetch.** One adapter per `method` in `config/sources.yaml`. `rss` is preferred
and is the only method enabled out of the box; `scrape` is the fallback for
sources with no feed. All HTTP goes through one client that sends an identifying
User-Agent, rate limits per host, and reads `robots.txt` (including
`Crawl-delay`). Each source is fetched inside its own error boundary — a broken
feed is recorded in `source_health` and the run continues.

**Normalize.** Everything becomes an `Article` with title, source, URL,
publication date, author, and a **short excerpt** (see *Attribution* below).
The article id is a hash of the canonicalized URL, so tracking parameters and
`www.` cannot create phantom new articles.

**Deduplicate.** Narrow on purpose: only the *same article* is removed — a
canonical URL already stored, or the same outlet re-running a near-identical
headline (measured threshold, see `dedupe.py`). Two outlets covering one event
are deliberately **not** duplicates; that is the corroboration the digest is
built on, and clustering handles it.

**Enrich.** Articles are batched (default 8 per request) and the LLM returns a
summary, why-it-matters clause, topics, entities, an importance score, and an
`event_label`. Results are cached by content hash, so re-running costs nothing.

**Cluster.** Matching `event_label`s merge articles into one story; text
similarity catches near-verbatim wire copy; shared named entities rescue
middling text matches. New clusters are matched against recent stories by
keyword overlap, so a story that gains coverage tomorrow keeps its identity
instead of appearing twice.

**Score and publish.** `scoring.py` combines importance, interest match,
recency decay and corroboration using the weights in `config/interests.yaml`,
then writes `docs/feed.json`, `docs/feed.xml` and the page shell.

---

## Configuration

Adding or removing a source never requires touching code.

```yaml
# config/sources.yaml
sources:
  - name: BBC World
    rss: https://feeds.bbci.co.uk/news/world/rss.xml
    weight: 1.2          # multiplies this source's contribution to a score
    topics: [world]      # hints merged into every article from this source

  - name: Example Site
    url: https://example.com/news/
    method: scrape
    enabled: false
    link_selector: "a.headline"   # CSS selector for article links
    link_pattern: "/2026/"        # optional regex an href must match
    max_links: 8
```

`config/interests.yaml` holds topic weights, keyword boosts, muted terms, the
scoring weights, recency half-life, and how many stories the page shows. All of
it is optional — an empty file gives you importance plus recency.

Secrets and tuning that shouldn't live in git go in the environment (`.env`
locally, Actions secrets/variables in CI): see `.env.example`.

---

## LLM providers

Everything the pipeline needs is behind `LLMProvider` in `src/newsdigest/llm/base.py`:

| Provider | `LLM_PROVIDER` | Notes |
|---|---|---|
| Google Gemini | `gemini` | Default. Free tier; `gemini-2.5-flash` by default. Uses `responseSchema` so the JSON contract is enforced server-side. |
| Offline heuristic | `none` | No key, no network, no cost. Extractive summaries and keyword topics. Used automatically when no key is configured, and by `--no-llm`. |

To add one: implement `enrich()` and `write_brief()` in a new module under
`llm/`, register it in `PROVIDERS` in `llm/__init__.py`. Nothing else changes.

**Keeping inside a free tier.** Three guards, all configurable:

- **Cache** — enrichment is keyed by article content hash, so nothing is ever
  summarized twice (`llm_cache` table).
- **Batching** — `LLM_BATCH_SIZE` articles per request, not one each.
- **Per-run cap** — `LLM_ARTICLES_PER_RUN` (default 60) bounds a busy news day.
  Anything left over is picked up next run.

Merged headlines for multi-source stories cost one extra call each, and only
when a story is new or has gained coverage.

A quota error (HTTP 429) stops LLM work for that run without failing it: the
feed still builds, unenriched articles appear with their source's own
description, and enrichment resumes next run.

---

## CLI

| Command | What it does |
|---|---|
| `news-digest run` | The whole pipeline. What Actions runs. |
| `news-digest run --no-llm` | Same, with offline enrichment. |
| `news-digest run --dry-run` | Fetch and enrich, write no files, record no run. |
| `news-digest fetch` | Collect and store new articles only. |
| `news-digest enrich --limit 20` | Enrich stored articles with a tighter budget. |
| `news-digest cluster` | Re-cluster and re-score what is stored. |
| `news-digest build` | Regenerate `docs/` from the database. |
| `news-digest sources --check` | Fetch every enabled source once and report. |
| `news-digest stats` | What is in the database, and the last run's numbers. |
| `news-digest prune --days 14` | Drop articles past a retention window. |

`--db`, `--out`, `--sources`, `--interests`, `-v` and `-q` work before or after
the subcommand.

---

## Attribution and copyright

The point of the digest is to send you to the reporting, not to replace it.

- Only metadata and a **short excerpt** are ever stored — the feed description,
  or the opening paragraphs for scraped pages, capped by `excerpt_chars`
  (default 1200). Full article bodies are never stored or sent to the LLM.
- Every story links to the original articles and names every outlet covering it.
  The original URL is preserved verbatim; canonicalization is used only as an
  internal identity key.
- Summaries are LLM-written from those excerpts, not copied, and the page says
  so.
- Before enabling a `scrape` source, read that site's terms of service. The
  fetcher honours `robots.txt`, `Crawl-delay` and a per-host rate limit, but
  that is a technical control, not permission. Prefer an official feed or API
  whenever one exists.

---

## Layout

```
config/sources.yaml     what to read
config/interests.yaml   what to prioritize, and the scoring weights
src/newsdigest/
  cli.py                argparse entry point
  pipeline.py           stage orchestration and error boundaries
  config.py             YAML + env loading, validation
  models.py             Article / Story / RunStats — the common schema
  sources/              rss.py, scrape.py, http.py (robots, rate limiting)
  llm/                  base.py contract, gemini.py, heuristic.py, registry
  dedupe.py             same-article removal
  clustering.py         same-event grouping and story continuity
  scoring.py            deterministic ranking
  store.py              SQLite schema and queries
  render.py             feed.json, feed.xml, page shell
  text.py, urls.py      normalization, similarity, canonicalization
web/index.html          the page shell, copied into docs/ on build
docs/                   generated — what GitHub Pages serves
data/news.db            generated — pipeline state, committed by the workflow
```

## Tests

```bash
pytest -q
```

82 tests, no network and no API key required. Covers URL canonicalization,
similarity and stemming, dedupe boundaries, clustering (including a test
documenting what text similarity *cannot* do), scoring, the store, the RSS
adapter against a fixture feed, the Gemini provider against a stubbed
transport, config validation, and the pipeline end to end — including a broken
source, an exhausted quota, and cache reuse.

## Known limits

- Without a working LLM, reworded coverage of one event stays split into
  separate stories. Text similarity cannot separate that case from unrelated
  stories about the same organisation; `clustering.py` documents the measured
  numbers. This is degradation, not breakage.
- Clustering is O(n²) inside a 48-hour window — a few thousand comparisons at
  this scale. It will need an index long before it needs a rewrite.
- `data/news.db` is committed as a binary blob. Retention pruning keeps it
  small; if the history ever gets uncomfortable, squash it.
- The scrape adapter depends on per-site CSS selectors and will break when a
  site redesigns. `news-digest sources --check` tells you which.
