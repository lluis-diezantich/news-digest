# news-digest

A personal multilingual news aggregator that runs entirely on GitHub. It collects
articles from your sources **every day**, and once a **week** turns them into one
curated digest: coverage of the same event grouped across English, Spanish and
Catalan, summarized and ranked against your interests, published as a static page
on GitHub Pages.

```
DAILY    sources → fetch → normalize → detect language → deduplicate → SQLite
WEEKLY   SQLite  → embed → cluster across languages → LLM → rank → digest → site
```

Keeping those two apart is the whole architecture. Collection must be cheap and
reliable enough to run constantly, so it calls **no model at all**. The expensive
semantic work happens once, over a week that has already finished.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

news-digest collect                        # takes ~20s across 22 feeds
news-digest digest --no-llm --no-embeddings --week $(date -u +%G-W%V)
make serve                                 # then open http://localhost:8000
```

> The page must be served over **http**, not opened as a file. It fetches
> `index.json` at runtime, and browsers block `fetch()` from `file://` origins,
> so `open docs/index.html` shows an empty page. `make serve` is just
> `python3 -m http.server` inside `docs/`. GitHub Pages serves it over http, so
> this only affects local viewing.

That runs with no API key at all — degraded, but end to end. For the real thing:

```bash
cp .env.example .env    # put your key in GEMINI_API_KEY
news-digest collect
news-digest digest      # builds last completed Mon–Sun week
```

`news-digest sources --check` verifies every configured feed responds before you
trust a schedule with it.

> **A fresh install has nothing to digest.** `digest` covers the last *completed*
> week, which is earlier than anything you have collected. Pass
> `--week $(date -u +%G-W%V)` to build the current partial week while testing.

---

## Setting it up on GitHub

1. Push this directory to a new repository.
2. **Secret** — Settings → Secrets and variables → Actions → *New repository
   secret*: `GEMINI_API_KEY`. Without it both workflows still succeed; the digest
   is just built with offline heuristics and within-language clustering.
3. **Pages** — Settings → Pages → Source: *Deploy from a branch*, branch `main`,
   folder `/docs`.
4. **Variables** (optional): `SITE_URL` for absolute RSS links;
   `LLM_MODEL`, `EMBEDDING_DIMENSIONS`, `LLM_ARTICLES_PER_RUN`,
   `LLM_RATE_LIMIT_RETRIES`, `LLM_RATE_LIMIT_WAIT` to override defaults without
   editing workflows.
5. Actions → *Collect articles* → **Run workflow**. Then let it run for a week,
   or trigger *Build weekly digest* with a `week` input to see output immediately.

| Workflow | Schedule | Calls a model? | Commits |
|---|---|---|---|
| `collect.yml` | every 6 hours | no | `data/news.db` |
| `digest.yml` | Mondays 06:41 UTC | yes | `docs/` + `data/news.db` |
| `tests.yml` | push / PR | no | — |

Collection runs four times a day rather than once, deliberately: most feeds
expose only their ~25 most recent items, so a busy source can publish more than
one feed-window between two daily runs and those articles are gone for good.
Collection costs nothing but HTTP requests. Set `cron: "23 5 * * *"` for strictly
daily.

---

## Multilingual

Articles keep their original language, title and URL — nothing is translated on
collection. The **output** language is separate and configurable:

```yaml
settings:
  supported_languages: [en, es, ca]
  output_language: en          # the LLM reads Catalan, writes English
```

Detection runs during collection, restricted to the languages you configured —
asking "en, es or ca?" is a far easier question than picking from 97, and that
restriction is what makes the es/ca pair reliable. A source's declared
`languages` both constrains the answer and supplies the fallback when a headline
is too short to judge.

Measured on 440 collected articles from single-language sources, detecting
*without* the source constraint and comparing against each source's own
declaration: **440/440**. (Benchmarking the underlying detector alone on 195
headlines gave 99.5%; the wrapper's length and confidence guards account for the
rest. `lingua` was tried and rejected — it scored 99.0% at 307 MB against
py3langid's 4.6 MB.)

The story-level output names every language and outlet covering an event:

```
EU announces new sanctions against Russia          [EN] [ES] [CA]  3 outlets

The European Union adopted a further package…
Why it matters: …

  EN  BBC World      EU announces new sanctions against Russia
  ES  El País        La UE anuncia nuevas sanciones contra Rusia
  CA  Ara            La UE anuncia noves sancions contra Rússia
```

---

## How each stage works

**Fetch (daily).** One adapter per `method` in `config/sources.yaml`. `rss` is
preferred; `scrape` is the fallback for sources with no feed. All HTTP goes
through one client that sends an identifying User-Agent, rate limits per host,
and honours `robots.txt` including `Crawl-delay`. Each source runs in its own
error boundary — a broken feed is recorded in the `sources` table and the run
continues.

**Normalize + detect (daily).** Everything becomes an `Article` with title,
publisher, URL, publication date, author, language and a **short excerpt**. The
id is a hash of the canonicalized URL, so tracking parameters cannot create
phantom articles.

**Deduplicate (daily).** Narrow on purpose: only the *same article* is removed —
a canonical URL already stored, or one outlet re-running a near-identical
headline. Two outlets covering one event are **not** duplicates, and neither are
a publisher's Spanish and English editions of one story; that is what clustering
is for.

**Embed (weekly).** One vector per article, cached by content hash, so re-running
a week is free. This is the only thing that can match a Catalan headline to an
English one.

**Cluster (weekly).** Three tiers, cheapest first: cosine ≥ `similarity_threshold`
is a merge; the band down to `ambiguous_threshold` is referred to the LLM, capped
at `max_cluster_checks` pairs per run; with no embeddings at all, within-language
text similarity only.

**LLM (weekly).** Clusters are pre-ranked using only signals collection already
provided — publisher count, coverage volume, source weights, recency, feed topic
hints — and **only the top clusters are enriched**. That is the difference between
summarizing 30 stories and summarizing 3000 articles.

**Rank + publish (weekly).** `scoring.py` combines the LLM's `importance` and
`relevance` with deterministic signals using the formula in
`config/preferences.yaml`, then writes the digest, the archive and RSS.

---

## Configuration

Adding or removing a source never requires touching code.

```yaml
sources:
  - name: El País Economía
    publisher: El País        # several feeds, one outlet
    rss: https://feeds.elpais.com/…/economia/portada
    languages: [es]
    weight: 1.1
    topics: [economics]

  - name: Example Site
    url: https://example.com/news/
    method: scrape
    enabled: false
    link_selector: "a.headline"
    link_pattern: "/2026/"
```

`publisher` matters more than it looks. Corroboration — "how many independent
outlets covered this" — is a ranking signal, and it counts **publishers, not
feeds**. Without it, seven El País sections covering one story would read as
seven independent outlets and one publisher could dominate the digest.

### The ranking formula is configuration

```yaml
ranking:
  terms:
    importance: 1.0         # LLM: how consequential
    relevance: 1.0          # LLM: how relevant to your topics
    interest: 0.8           # deterministic topic/keyword match
    recency: 0.6            # exponential decay across the week
    corroboration: 0.4      # distinct publishers
    source_preference: 0.3  # source weights + preferred_sources
    story_size: 0.2         # how much coverage there is
  excluded_penalty: 1.5
  recency_half_life_hours: 72
```

`final_score = Σ weight × signal`. Delete a line to remove that signal from the
formula entirely; a misspelled term is a startup error rather than a silent
no-op. `news-digest explain <story-id>` prints the per-term breakdown, and the
numbers shown provably sum to the score used for ranking.

---

## Providers

Both the LLM and the embedder sit behind small interfaces
(`llm/base.py`, `embeddings/base.py`). Adding one means a new module and a
registry entry; nothing else changes.

| Role | Default | Offline fallback |
|---|---|---|
| LLM | Gemini `gemini-2.5-flash`, `responseSchema`-enforced JSON | extractive summaries, keyword topics, no translation |
| Embeddings | Gemini `gemini-embedding-2`, 256 dims, `batchEmbedContents` | none — clustering drops to within-language |

**Keeping inside a free tier.** Collection never calls a model. Weekly, four
guards apply: enrichment and embeddings are cached by content hash; requests are
batched; only pre-ranked candidate clusters are enriched; and
`LLM_ARTICLES_PER_RUN` bounds a busy week.

**Two kinds of 429.** Google answers a per-minute and a per-day limit with the
same status code, and the difference decides what to do about it. A weekly run
fires its requests in one burst, so the per-minute allowance is the one it
actually trips — and waiting a few seconds clears it, which on a batch job with
no reader waiting costs nothing. So `ratelimit.py` reads the body: a
`QuotaFailure` naming a `...PerMinute...` quota is waited out and retried,
honouring the server's own `RetryInfo` delay, while a `...PerDay...` quota is
terminal because no amount of waiting helps. Total waiting is bounded per run
(`max_rate_limit_wait`, 600s) so a low allowance cannot spend the workflow's
timeout asleep. Once a limit is genuinely terminal the run degrades exactly as
before — whatever succeeded still publishes.

> The pricing page lists Gemini Embedding 2 text input as free of charge on the
> free tier, though the rate-limit page still only tabulates it for paid tiers.
> Check your own quota in AI Studio before relying on it. If embeddings are
> unavailable the digest still builds — see the next section.

---

## What breaks without a key, precisely

Everything still runs; two things get worse, and both are visible.

1. **Summaries are extractive, not written**, and are not translated — a Spanish
   article keeps a Spanish summary under `output_language: en`.
2. **Cross-language coverage stays split.** This is not a tuning problem. Over
   37,776 real within-language pairs from these feeds, text similarity tops out
   at 0.28 for genuine same-event pairs it can see at all, and the same event in
   three languages scores **0.00 (en/es)** and **0.06 (en/ca)**. No threshold
   separates that from unrelated articles. `clustering.py` records the numbers,
   and a test asserts the limitation so nobody "fixes" it by lowering a
   threshold.

---

## CLI

| Command | What it does |
|---|---|
| `news-digest collect` | Daily: fetch, detect language, dedupe, store. No model calls. |
| `news-digest digest` | Weekly: embed, cluster, LLM, rank, publish. |
| `news-digest digest --week 2026-W36` | Rebuild a specific past week. |
| `news-digest digest --no-llm --no-embeddings` | Fully offline. |
| `news-digest digest --dry-run` | Process, persist nothing (caches still fill). |
| `news-digest build` | Regenerate `docs/` from the database. |
| `news-digest sources --check` | Fetch every enabled source once and report. |
| `news-digest stats` | Articles, languages, digests, cache sizes, last run. |
| `news-digest explain <story-id>` | Per-term ranking breakdown. |
| `news-digest prune --days 30` | Drop articles past a retention window. |

`--db`, `--out`, `--sources`, `--preferences`, `-v`, `-q` work before or after
the subcommand.

---

## Attribution and copyright

The point of the digest is to send you to the reporting, not to replace it.

- Only metadata and a **short excerpt** are stored — the feed description, or the
  opening paragraphs for scraped pages, capped by `excerpt_chars` (default 1200).
  Full article bodies are never stored, embedded or sent to the LLM.
- Every story links to the original articles and names every outlet and language.
  The original URL and headline are preserved verbatim; canonicalization is used
  only as an internal identity key.
- Summaries are LLM-written from those excerpts, and the page says so.
- `robots.txt` is honoured per the modern convention: 4xx other than 429 means
  "no robots.txt, no restrictions" (per Google's spec, explicitly including 401
  and 403), while 429 and 5xx mean back off. The strict old reading turns a CDN
  misconfiguration into a silent source blackout — `feeds.elpais.com` answers
  `robots.txt` with a Varnish 403 while publishing feeds for readers.
- Before enabling a `scrape` source, read that site's terms of service. The
  fetcher's robots handling is a technical control, not permission.

Two configured sources are disabled with the reason recorded in
`config/sources.yaml`: **RTVE** (its robots allowlist excludes the news feed;
only audio/video bulletin feeds are permitted) and **CTXT** (its feed host
returns 403 to every user agent). **Público** publishes no feed at all.

---

## Layout

```
config/sources.yaml       what to read
config/preferences.yaml   languages, interests, the ranking formula, retention
src/newsdigest/
  cli.py                  argparse entry point
  pipeline.py             the two pipelines and their error boundaries
  digest.py               weekly window, pre-ranking, digest assembly
  config.py               YAML + env loading, validation
  models.py               Article / Story / Digest / RunStats
  lang.py                 language detection
  sources/                rss.py, scrape.py, http.py (robots, rate limiting)
  embeddings/             base.py contract, gemini.py, none.py
  llm/                    base.py contract, gemini.py, heuristic.py
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

## Tests

```bash
pytest -q
```

259 tests, no network and no API key required. They cover URL canonicalization
and the similarity metric, language detection including es/ca, dedupe
boundaries, cross-language clustering with stubbed vectors, the ambiguous-band
LLM adjudication and its budget, both providers against stubbed transports
(including the aggregated-embedding trap), the per-minute/per-day 429 split and
its wait budgets, the configurable ranking formula, the store with a v1→v2
migration, the archive renderer, and both pipelines end
to end — with a broken source, an exhausted quota, and cache reuse.

## Known limits

- Without embeddings, cross-language coverage stays split. Measured, documented,
  and asserted in tests rather than hidden.
- Clustering is O(n²) inside the weekly window. At ~2000 articles that is 2M
  cosine comparisons — one numpy matmul, milliseconds. It needs an index long
  before it needs a rewrite.
- `data/news.db` is committed as a binary blob. Embeddings dominate its size
  (~1 KB per article at 256 dims); `embedding_retention_days` and
  `EMBEDDING_DIMENSIONS` are the knobs. Squash history if it gets uncomfortable.
- The offline provider cannot translate, so `output_language` is only honoured
  when a real LLM is configured.
- The scrape adapter depends on per-site CSS selectors and will break when a site
  redesigns. `news-digest sources --check` tells you which.
