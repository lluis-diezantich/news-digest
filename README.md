# news-digest

A personal multilingual news aggregator that runs on GitHub, or entirely on your
own machine. It collects articles from your sources every few hours, and once a
week turns them into one short digest: coverage of the same event grouped across
English, Spanish and Catalan, summarized, ranked, and published as a static page.

```
DAILY    sources → fetch → normalize → detect language → filter → dedupe → SQLite
WEEKLY   SQLite  → embed → cluster across languages → LLM → rank → digest → site
```

Keeping those two apart is the whole architecture. Collection must be cheap and
reliable enough to run constantly, so it calls **no model at all**. The expensive
semantic work happens once, over a week that has already finished.

It can run at zero cost, and with no third-party API at all if you want: local
embeddings via ONNX, a local LLM via Ollama. See [Providers](doc/providers.md).

---

## Try it in five minutes

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

news-digest collect
news-digest digest --no-llm --no-embeddings --week $(date -u +%G-W%V)
make serve            # then open http://localhost:8000
```

No API key, no model — degraded but end to end. [Setup](doc/setup.md) explains
what to do next, and why that command needs `--week`.

---

## Documentation

| | |
|---|---|
| **[Setup](doc/setup.md)** | Running it locally, deploying to GitHub, and why collection runs four times a day |
| **[Providers](doc/providers.md)** | Embeddings and LLM: local, Gemini, or Ollama. Free-tier budgeting, rate limits, and what degrades without a model |
| **[Configuration](doc/configuration.md)** | Sources, junk filters, the ranking formula, digest size, retention |
| **[How it works](doc/pipeline.md)** | Each stage in turn, the multilingual design, and the known limits |
| **[CLI](doc/cli.md)** | Every command, plus recipes for trying things without breaking your site |
| **[Attribution](doc/attribution.md)** | What is stored, robots.txt, terms of service, and per-source findings |
| **[Development](doc/development.md)** | Tests, layout, adding a provider, schema changes |

---

## The parts worth knowing up front

**Configuration is not code.** Sources, the ranking formula, junk filters and
digest size all live in two YAML files. Deleting a line from `ranking.terms`
removes that signal from the maths entirely, and a misspelled term is a startup
error rather than a silent no-op. `news-digest explain <story-id>` prints the
per-term breakdown, and the numbers provably sum to the score used for ranking.

**Nothing fails loudly that could fail quietly instead.** A broken feed is
recorded and the run continues. An exhausted quota degrades the digest rather than
aborting it. A per-minute rate limit is waited out; a per-day one is not.

**Findings are recorded where they were learned.** Several sources cannot be used,
several ideas were measured and rejected, and one ranking signal was built,
removed, and restored under different conditions. Those reasons sit in the config
and the docstrings, so the next reader does not rediscover them. If a comment
explains why something is *not* done, it is usually load-bearing.

**It counts publishers, not feeds.** "The main news of the week" is close to a
definition of corroboration, and in one measured week 411 of 440 story clusters
were a single outlet reporting alone. The digest is sized to what the week
actually corroborates rather than to a round number.
