# Providers

The LLM and the embedder sit behind small interfaces (`llm/base.py`,
`embeddings/base.py`). Adding one means a new module and a registry entry.

| Role | Options | Set with |
|---|---|---|
| Embeddings | `local` (fastembed/ONNX), `gemini`, `none` | `EMBEDDING_PROVIDER` |
| LLM | `gemini`, `ollama` (local), `none` (offline heuristics) | `LLM_PROVIDER` |

The two halves have genuinely different answers, and it is worth knowing why.

## Embeddings: local is the better option

Cross-lingual sentence embedding is a small-model problem, so a local model is
not a compromise here. Measured on 551 articles from one real week, same set:

| | Clusters | Cross-language | Time |
|---|---|---|---|
| `gemini-embedding-2`, 256d | 445 | 10 | ~300s, mostly waiting out 429s |
| **local MiniLM, 384d** | **442** | **20** | **8.5s** |

Same granularity, twice the cross-language grouping, no rate limit, no key. It
also stops embeddings competing with the LLM for one free-tier budget.

```bash
pip install 'news-digest[local]'
EMBEDDING_PROVIDER=local
EMBEDDING_SIMILARITY_THRESHOLD=0.70
EMBEDDING_AMBIGUOUS_THRESHOLD=0.60
```

**Thresholds do not transfer between embedding models.** Cosine values are not
comparable, and this is a silent failure — a stale threshold produces a worse
digest with no error. Measured for MiniLM:

| sim / amb | Clusters | Cross-language | Largest |
|---|---|---|---|
| 0.50 / 0.40 | 178 | 10 | **327** ← collapsed into one blob |
| **0.70 / 0.60** | **442** | **20** | 25 |
| 0.82 / 0.72 (Gemini's) | 522 | 9 | 8 ← under-merged |

The local provider warns if it sees a Gemini-shaped threshold. In CI, cache the
model or every run re-downloads ~220 MB; `digest.yml` sets
`FASTEMBED_CACHE_PATH` explicitly because fastembed otherwise caches under
`$TMPDIR`, which a runner wipes between jobs.

`fastembed` is pinned to `>=0.8,<0.9` deliberately: 0.8 changed MiniLM from CLS
to mean pooling, which changes every vector, and `cache_key` does not include the
library version.

## LLM: hosted or local, and it is a hardware question

### Gemini

```bash
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.6-flash
```

`gemini-2.5-flash` was retired for new API keys (404, verified 2026-09-14);
`gemini-3.6-flash` and `gemini-3.5-flash-lite` both work.

Thinking is sent as `generationConfig.thinkingConfig.thinkingLevel` — nested. The
flatter spellings 400 with "Unknown name", so an implementation that puts it one
level too high silently loses the setting to the retry-without-it fallback.
`LLM_THINKING_LEVEL` defaults to `low`: thought tokens are billed as output,
count against the per-minute token allowance, and come out of `maxOutputTokens`,
so on schema-enforced extraction they cost three ways and buy nothing.

### Ollama, locally

No API, no key, no quota — but it needs a machine with real memory. A GitHub
runner (2 vCPU, no GPU) cannot do this inside the 45-minute job; an
Apple-silicon laptop can.

```bash
brew install ollama && ollama serve
ollama pull qwen3:8b
LLM_PROVIDER=ollama
```

`LLM_MODEL` selects the model; a 12–14B model is a clear upgrade for the
`importance` judgements if there is memory for it. `LLM_BASE_URL` points
elsewhere, `LLM_NUM_CTX` sets the window.

Three details in `llm/ollama.py` are load-bearing. Ollama is used rather than
llama.cpp directly because it enforces a **JSON Schema** through `format`, and a
local model reliably wanders out of JSON on the array-of-objects enrich call
without it. `think: false` is always sent, because hybrid models otherwise emit
reasoning into the response and break parsing. And `num_ctx` defaults to 8192,
because a batch of 16 articles at 900-char excerpts overflows 4k and an
overflowed prompt is **truncated silently** — surfacing as missing ids, not an
error.

## Staying inside a free tier

The scarce resource is the **request count**, not tokens. One run on 2026-09-15
spent a whole day's allowance on 13 enrichment requests plus retries, and the six
briefs — the only LLM output a reader sees — got nothing.

Four structural guards: enrichment and embeddings are cached by content hash;
requests are batched; only pre-ranked candidate clusters are enriched; and
**briefs are requested before per-article enrichment**, so the part that reaches
the page is paid for first and enrichment takes the remainder.

Then the budget knobs:

```bash
LLM_BATCH_SIZE=16            # halves requests for the same articles
LLM_ARTICLES_PER_RUN=40      # six published stories do not need 100 analysed
LLM_MAX_CLUSTER_CHECKS=0     # adjudication costs a request for a nicety
```

Together, roughly 4 requests per digest instead of 17.

### Two kinds of 429

Google answers a per-minute and a per-day limit with the same status code, and
the difference decides what to do. A weekly run fires its requests in one burst,
so the per-minute allowance is the one it trips — and waiting clears it, which on
a batch job costs nothing. `ratelimit.py` reads the body: a `QuotaFailure` naming
a `...PerMinute...` quota is waited out, honouring the server's own `RetryInfo`
delay, while `...PerDay...` is terminal. Total waiting is bounded per run
(`LLM_RATE_LIMIT_WAIT`, 600s) so a low allowance cannot spend the workflow's
timeout asleep.

## What degrades without a model

Everything still runs. Two things get worse, both visible:

1. **Summaries are extractive and untranslated** — a Spanish article keeps a
   Spanish summary under `output_language: en`.
2. **`importance` becomes near-useless.** Offline it is `0.35 + 0.12` per hit on a
   33-word list; 91% of articles match nothing, so it is a constant wearing a
   signal's clothes. It sits in `ranking.terms` at weight 2.0 and only earns that
   with a real model.

Without **embeddings** specifically, cross-language coverage stays split, and
that is not a tuning problem: over 37,776 real within-language pairs, text
similarity tops out at 0.28 for genuine same-event pairs, while the same event in
three languages scores 0.00 (en/es) and 0.06 (en/ca). `clustering.py` records the
numbers and a test asserts the limitation so nobody "fixes" it by lowering a
threshold.
