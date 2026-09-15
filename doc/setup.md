# Setup

## Locally, with no API key

The recommended setup: real embeddings and a real model, nothing hosted.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,local]'

brew install ollama                    # or the installer from ollama.com
ollama serve &
ollama pull qwen3:8b
```

`.env`:

```bash
EMBEDDING_PROVIDER=local
EMBEDDING_SIMILARITY_THRESHOLD=0.70    # model-specific, see providers.md
EMBEDDING_AMBIGUOUS_THRESHOLD=0.60
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:8b
```

```bash
news-digest collect
news-digest digest --week $(date -u +%G-W%V)
make serve            # then open http://localhost:8000
```

Pin the embedding cache too, or it lands in `$TMPDIR` and macOS eventually clears
it — which quietly turns the next run into a 220 MB download:

```bash
FASTEMBED_CACHE_PATH=$HOME/.cache/fastembed
```

Collection needs a connection; nothing after it does. See
[providers.md](providers.md) for what each provider costs and what it is good at.

To see the pipeline work before installing anything, `pip install -e '.[dev]'` and
`news-digest digest --no-llm --no-embeddings --week $(date -u +%G-W%V)` runs end to
end on built-in heuristics — meaningfully worse, and measured as such.

> **Serve it over http, not `file://`.** The page fetches `index.json` at
> runtime and browsers block `fetch()` from file origins, so `open
> docs/index.html` shows an empty page. `make serve` is `python3 -m http.server`
> inside `docs/`. GitHub Pages serves http, so this only affects local viewing.

> **A fresh install has nothing to digest.** Plain `digest` covers the last
> *completed* Monday–Sunday week, which is earlier than anything you have just
> collected. Pass `--week $(date -u +%G-W%V)` for the current partial week.

`news-digest sources --check` fetches every enabled source once and reports, which
is worth doing before trusting a schedule with them.

## On GitHub

1. Push to a new repository.
2. **Pages** — Settings → Pages → Source: *Deploy from a branch*, branch `main`,
   folder `/docs`.
3. **Secret**, only if using a hosted model — Settings → Secrets and variables →
   Actions: `GEMINI_API_KEY`. Both workflows succeed without it; the digest is
   just built with the built-in heuristics.
4. **Variables** (optional) — anything in [providers.md](providers.md) or
   [configuration.md](configuration.md) can be set here instead of editing
   workflows. `SITE_URL` gives the RSS feed absolute links.
5. Actions → *Collect articles* → **Run workflow**, then let it accumulate.

| Workflow | Schedule | Calls a model? | Commits |
|---|---|---|---|
| `collect.yml` | every 6 hours | no | `data/news.db` |
| `digest.yml` | Mondays 06:41 UTC | yes | `docs/` + `data/news.db` |
| `tests.yml` | push / PR | no | — |

### Why collection runs four times a day

Not for freshness — for coverage. A source only exposes its most recent items,
and `max_items_per_source` (default 10) takes the top of that list, so anything a
busy outlet publishes beyond ten between two runs is gone for good.

Measured across one real week, articles lost to the cap:

| Cadence | Lost |
|---|---|
| every 6h | 19% |
| every 12h | 32% |
| daily | 45% |

There is a second reason, less obvious. `feed_position` is recorded as
**first seen**, and articles enter near the top of a section page and sink over
hours. Polling less often means first meeting an article after it has already
slipped down — which biases `editorial_position`, the digest's joint-primary
ranking signal, downward and adds noise nothing in the output would reveal.

Collection calls no model, so the cost is HTTP requests and a couple of minutes
of Actions time. Set `cron: "23 5 * * *"` if you want strictly daily anyway.
