# Development

```bash
make install     # .venv with dev extras
make test        # pytest -q
```

588 tests, no network, no API keys, no mailbox. Nothing in the suite touches a
live service: the mailbox is a fake returning fixed RFC822 bytes, the LLM is a
stub or the offline heuristic provider, and embeddings come from a deterministic
stub that maps articles to events.

## Layout

```
src/newsdigest/
  inbox/         IMAP, message parsing, source matching
  extract/       newsletter HTML -> items; boilerplate; tracking links
  llm/           provider interface + gemini, ollama, heuristic
  embeddings/    provider interface + gemini, local (ONNX), none
  classify.py    topic filtering (section 8)
  clustering.py  same-event grouping, across languages
  regions.py     geographic spread (section 12)
  scoring.py     the ranking formula
  digest.py      the weekly build
  pipeline.py    fetch / parse / digest, and `run`
  render.py      Markdown output
  store.py       SQLite
  debug.py       the --debug export
  prompts.py     prompt loading
  eval.py        clustering evaluation harness
prompts/         the prompts themselves, versioned separately
config/          sources, filters, preferences
tests/fixtures/emails/   real newsletter structure, one file per source
```

## Adding a source

Config only — `config/sources.yaml`, then `news-digest sources --check`. No code.

If a newsletter defeats the structural extractor, add its `.eml` to
`tests/fixtures/emails/` and a test to `test_extract.py::TestRealFixtures` first,
so the failure is pinned before it is fixed. Per-source extraction strategies are
deliberately not implemented yet; the fixture is how you find out whether one is
actually needed.

## Adding an LLM provider

Implement `LLMProvider` (`llm/base.py`) and register it in `PROVIDERS`. Three
abstract methods: `enrich`, `write_brief`, `same_event`.

`classify` is deliberately **not** abstract. Its default offers no opinion, which
the pipeline reads as "keep everything", so a provider that cannot triage degrades
to no topic filtering rather than to an empty digest — and can be added without
implementing four methods at once.

## Adding a mailbox provider

Implement `Mailbox` (`inbox/base.py`) — one method, `fetch(since, until)`, returning
raw RFC822 bytes — and register it in `inbox/__init__.py` `PROVIDERS`. Matching and
message parsing are shared and need no changes.

The contract is deliberately one-sided: implementations may **over**-return, and
the caller filters precisely on the parsed timestamp. Under-returning would
silently lose a newsletter.

## Schema changes

Bump `SCHEMA_VERSION` in `store.py` and add a migration step.

There are currently no migrations, on purpose: this schema has never been
deployed, so there is nothing to migrate from, and carrying dead migration code
only invites someone to trust it. What *is* there is a guard — opening the RSS
project's database raises and says so, because nothing here would otherwise fail
on it. `CREATE TABLE IF NOT EXISTS` is silent about an existing table, the missing
columns read as empty, and the run would publish a digest built from articles whose
provenance it had invented.

## Evaluating clustering changes

```bash
python -m newsdigest.eval
```

Scores clustering against hand-labelled pairs in
`tests/fixtures/cluster_eval.json`, and prints the cosine distribution split by
same-language against cross-language — which is the only way to see that a
threshold tuned on Spanish-to-Spanish pairs is far too high for
Spanish-to-Catalan ones.

It runs offline in about a second; the fixture carries its own cached vectors. It
measures the **embedding tiers only**: with no provider the ambiguous band is left
to the embedding's own verdict, so recall is a floor.

The fixture is inherited from the RSS version of this project. Rebuilding it from
newsletter articles is the highest-value calibration work available, and it is what
`corroboration_saturation` and the text-fallback threshold are both waiting on.

## Conventions

Comments explain **why**, especially why something is *not* done — those are usually
load-bearing. Several numbers in the config are annotated with the measurement that
produced them, and several are annotated as uncalibrated guesses. Keep that
distinction: it is the difference between a value you may change freely and one
that cost a week of observation.
