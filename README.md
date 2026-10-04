# news-digest

It reads your newsletter inbox, pulls the articles out, throws away the junk,
groups what is about the same thing, and writes one ranked summary you can read
in ten minutes.

The goal is not to reproduce ten newsletters. It is to answer one question:
**what happened in the world this week?**

## What it does

1. Read the inbox. Read-only -- nothing is marked or deleted.
2. Work out which newsletter each email is.
3. Pull the individual articles out of each email.
4. Unwrap the tracking links to find the real article URLs.
5. Bin the junk: sentence fragments, boilerplate, mastheads, sport.
6. Drop articles already stored.
7. Ask the model what each article is about, and drop what is not news.
8. Turn each one into numbers, so they can be compared across languages.
9. Group articles covering the same event.
10. Score and rank the groups.
11. Write a short summary for each.
12. Group related stories into bigger narratives.
13. Pick the top 12 plus 5 extras, without letting one region or topic take over.
14. Write the Markdown file.

`news-digest stages` prints these fourteen steps for a given week with what went
into each and what came out, so a missing story can be traced to the step that
dropped it.

About 3-5 minutes on Gemini, about 13 locally on Ollama.

It runs at zero cost with no third-party API at all: local embeddings via ONNX,
a local model via Ollama. See [Providers](doc/providers.md).

## Quick start

You need an inbox that receives the newsletters. Subscribe from a dedicated
address -- everything in the box is read, so personal mail there is just noise.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env          # then fill in NEWS_EMAIL_*
```

Check that the source rules match your mail before anything else. This is the
step that bites: the rules look right, and nothing matches.

```bash
news-digest sources --check   # fix config/sources.yaml until no source says NONE
```

Then:

```bash
news-digest run --week $(date -u +%G-W%V)
```

With no model or API key at all, a couple of minutes end to end:

```bash
news-digest run --no-llm --no-embeddings
```

## Documentation

| | |
|---|---|
| **[Setup](doc/setup.md)** | The inbox, the credentials, running locally, deploying to GitHub |
| **[Providers](doc/providers.md)** | Embeddings and LLM: local, Gemini or Ollama, and what degrades without each |
| **[Configuration](doc/configuration.md)** | Sources and match rules, filters, the ranking formula, digest size, retention |
| **[How it works](doc/pipeline.md)** | Each stage in turn, the multilingual design, and the known limits |
| **[Design notes](doc/design-notes.md)** | Why the pipeline is shaped this way, and which numbers are still guesses |
| **[CLI](doc/cli.md)** | Every command, plus recipes for trying things without breaking your digest |
| **[Attribution](doc/attribution.md)** | What is stored, what is published, and what is deliberately not |
| **[Rebuilding](doc/rebuilding.md)** | Starting over, what must survive, and the failure modes that look like bugs |
| **[Development](doc/development.md)** | Tests, layout, adding a provider, schema changes |

The digests themselves live in [`digests/`](digests/), one Markdown file per
week, newest listed in [digests/README.md](digests/README.md).
