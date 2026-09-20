# Providers

Two pluggable things, both optional, each degrading in a specific and documented
way.

## Embeddings

These do the cross-language clustering, and that is the thing this project exists
to do. Measured on one headline in three languages, token overlap scores **0.00**
(en/es) and **0.06** (en/ca) against a 0.60 merge threshold. No threshold rescues
that.

| `EMBEDDING_PROVIDER` | |
|---|---|
| `gemini` | Google API. Free tier, but the per-minute limit is the binding one |
| `local` | fastembed/ONNX on your machine. No API, no quota. `pip install 'news-digest[local]'` |
| `none` | no embeddings; clustering falls back to within-language text similarity |

Measured on 551 real articles from the RSS version of this project: **local** gave
442 clusters and 20 cross-language groups in 8.5 seconds, against gemini's 445 / 10
in about five minutes — most of which was waiting out per-minute 429s. Local also
stops embeddings competing with the LLM for one free-tier budget.

### Thresholds are model-specific

Cosine values do not transfer between embedding models.

| Model | `SIMILARITY` | `AMBIGUOUS` |
|---|---|---|
| `gemini-embedding-2` | 0.82 | 0.72 |
| local MiniLM, adjudication **on** | 0.80 | 0.65 |

Do not narrow the band while `LLM_MAX_CLUSTER_CHECKS=0`: with checks off the
ambiguous band merges nothing, so the two knobs have to move together. Below 0.60
MiniLM collapses into one enormous blob. The local provider warns if it sees a
gemini-shaped threshold.

`EMBEDDING_DIMENSIONS=256` keeps the committed database small, about 1 KB per
article; 768 triples that. Gemini embeddings degrade gracefully when truncated.

### Without embeddings

Everything still runs. Clustering falls back to within-language text similarity at
a threshold of 0.45, calibrated on 37,776 real within-language pairs — from RSS
feeds, not newsletters. What you lose is the whole point: an EU sanctions story
covered by The Guardian and EL PAÍS stays two separate stories, `corroboration`
reads 0.00 for every story, and the digest is a list rather than a synthesis.

## LLM

| `LLM_PROVIDER` | |
|---|---|
| `gemini` | Google API. `gemini-3.6-flash` and `gemini-3.5-flash-lite` both work |
| `ollama` | a local model. No API and no quota, but it needs real memory |
| `none` | built-in heuristics. No model, no cost |

With no key, `gemini` falls back to `none` automatically rather than failing.

`gemini-2.5-flash` was retired for new API keys — 404, verified 2026-09-14.

### Ollama

A GitHub runner (2 vCPU, no GPU) cannot do this inside the 45-minute job. An
Apple-silicon laptop can.

```bash
brew install ollama
brew services start ollama       # background server on :11434, logs to a file
ollama pull qwen3:8b             # ~5 GB, once
```

Start it as a **service**, not `ollama serve &`. A backgrounded `serve` writes
llama-server's startup dump, per-token timings and one `[GIN]` line per request to
whatever terminal you launched it from, interleaved with the digest's own output.
Without brew, redirect it yourself:

```bash
ollama serve >/tmp/ollama.log 2>&1 &
```

Then in `.env`:

```bash
EMBEDDING_PROVIDER=local
EMBEDDING_SIMILARITY_THRESHOLD=0.80
EMBEDDING_AMBIGUOUS_THRESHOLD=0.65
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:8b
```

Do not lower `LLM_TIMEOUT` for a local run — it defaults to 300s for ollama and 90s
otherwise, because a local model generates every token on your machine. At 90s a
batch of 16 on qwen3:8b times out, and both enrichment and cluster adjudication
give up quietly while the digest still publishes. The only signs are `enriched: 0`
and a `batch failed` warning.

With 16 GB or more, a 12–14B model is a clear upgrade for the briefs. It is also
the way back to a usable `importance` signal, which qwen3:8b does not provide: it
returned 0.80–0.85 for every story, a spread of 0.05, which is why the term is out
of the ranking formula.

### Budgeting a free tier

The scarce resource is the **request count**, not the tokens. The four jobs, in the
order the pipeline spends on them:

| Job | Cost | Turn it off with |
|---|---|---|
| classify | one request per `LLM_BATCH_SIZE` items, ~600 items/week | `filters.classify: false` |
| same_event | one request per batch of ambiguous pairs | `LLM_MAX_CLUSTER_CHECKS=0` |
| write_brief | **one request per candidate cluster** | `write_story_briefs: false` |
| enrich | one request per `LLM_BATCH_SIZE` articles, capped by `LLM_ARTICLES_PER_RUN` | `LLM_ARTICLES_PER_RUN=0` |

Briefs are paid for **first**. On a measured run of the RSS project, 13 enrichment
requests plus retries exhausted a whole day's allowance and all six briefs fell
back to raw article text — enrichment is scaffolding that is cached for next time,
while the brief is the only LLM output a reader ever sees.

Everything is cached on content hash, so a second run over the same week costs
nothing. That is what makes iterating on a prompt affordable.

A per-minute 429 is waited out (`LLM_RATE_LIMIT_RETRIES`, `LLM_RATE_LIMIT_WAIT`);
a per-day one is terminal and stops LLM work for the run, degrading the digest
rather than aborting it.

`LLM_THINKING_LEVEL=low` is the cheapest level the 3.x Flash models accept. Thought
tokens are billed as output, count against the per-minute token allowance, and come
out of `maxOutputTokens` — so a thinking-heavy reply can exhaust the budget before
writing any JSON, and this is schema-enforced extraction.

### Without a model

`--no-llm`, or no key. What you lose:

- **Filtering.** The offline classifier answers topics only, by keyword. It never
  claims something is not news, because "final" and "corona" would drop a court
  ruling and a public-health story. So sport and horoscopes are caught only by the
  URL and title patterns in `config/sources.yaml`.
- **Summaries.** Extractive, and untranslated — with `output_language: en` and a
  Spanish source, the summary stays Spanish.
- **Merged headlines.** A multi-source story borrows the highest-importance
  article's own headline instead.
- **Disagreements.** Never detected.
- **Cluster adjudication.** The ambiguous band keeps the embedding's own verdict.

Ranking holds up, because it runs on what parsing already provides.
