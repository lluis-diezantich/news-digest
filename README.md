# news-digest

A personal weekly digest built from newsletters. It reads a dedicated inbox,
pulls the individual stories out of each newsletter, throws away what is not
news, groups coverage of the same event across English and Spanish, and writes one
Markdown file you can read in five to ten minutes.

```
EMAIL → ingest → parse → filter → dedupe → cluster → rank → summarize → digests/2026/2026-W38.md
```

The goal is not to reproduce ten newsletters. It is to answer one question:
**what happened in the world this week?**

It can run at zero cost and with no third-party API at all: local embeddings via
ONNX, a local model via Ollama. See [Providers](doc/providers.md).

---

## Try it locally

You need an inbox that receives the newsletters. Subscribe from a dedicated
address — everything in the box is read, and personal mail there is just noise.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env          # then fill in NEWS_EMAIL_*
```

First, check that the source rules actually match your mail. This is the step
that bites: the rules look right, and nothing matches.

```bash
news-digest sources --check
```

It reports what each rule matched over the last 30 days and lists every sender
that matched nothing — which is where a wrong address shows up. Fix
`config/sources.yaml` until no source says `NONE`.

Then run it:

```bash
news-digest run --week $(date -u +%G-W%V)
```

### Without a model

```bash
news-digest run --no-llm --no-embeddings
```

End to end in a couple of minutes, and meaningfully worse in two specific ways:
coverage of one event in English and Spanish stays split into two separate
stories, and nothing filters the sport and the horoscopes except the URL patterns
in `config/sources.yaml`. [Providers](doc/providers.md) has the numbers.

### While you are still changing it

`run` on its own builds the last *finished* Monday-to-Sunday week, which is what
a scheduled run needs and almost never what you want while editing: on a Friday
it rebuilds a week that ended five days ago. Point a scratch run at a scratch
database:

```bash
cp data/news.db /tmp/try.db
news-digest --db /tmp/try.db --out /tmp/out digest --days 7 --debug
```

`--debug` writes the intermediate data to `debug/` — what arrived, what was
extracted, what the filter dropped and why, how it clustered. That directory is
where you look when a story you expected is missing. [CLI](doc/cli.md) has the
rest of the recipes.

---

## Documentation

| | |
|---|---|
| **[Setup](doc/setup.md)** | The inbox, the credentials, running locally, deploying to GitHub |
| **[Providers](doc/providers.md)** | Embeddings and LLM: local, Gemini or Ollama, and what degrades without each |
| **[Configuration](doc/configuration.md)** | Sources and match rules, filters, the ranking formula, digest size, retention |
| **[How it works](doc/pipeline.md)** | Each stage in turn, the multilingual design, and the known limits |
| **[CLI](doc/cli.md)** | Every command, plus recipes for trying things without breaking your digest |
| **[Attribution](doc/attribution.md)** | What is stored, what is published, and what is deliberately not |
| **[Rebuilding](doc/rebuilding.md)** | Starting over, what must survive, and the failure modes that look like bugs |
| **[Development](doc/development.md)** | Tests, layout, adding a provider, schema changes |

---

## The parts worth knowing up front

**A newsletter is a curated list, and that is the best signal here.** An editor
chose ten items out of the day's hundreds *and* chose which one opens, so position
in the newsletter is two judgements rather than one — and it is free, and available
in every language. It is the highest-weighted term in the ranking formula.

**Tracking links are not a cosmetic problem.** Newsletter links go through a click
tracker, and until one is resolved back to `elpais.com/deportes/...` three things
are broken: attribution points at a URL that expires, the same article carries a
different opaque token in every newsletter so deduplication cannot see it is one
article, and every section blocklist silently matches nothing.

**Filtering has to ask to drop something.** An unclassified item is kept, a failed
batch is kept, an exhausted quota keeps everything left, and an item is only
dropped on its topics when *every* topic is excluded. The worst case is a noisy
digest, never an empty one — a wrongly dropped story is invisible in the output,
whereas a wrongly kept one merely ranks low.

**Configuration is not code.** Sources, filters, the ranking formula and digest
size live in three YAML files, and the prompts live in `prompts/`. Deleting a line
from `ranking.terms` removes that signal from the maths entirely, and a misspelled
term is a startup error rather than a silent no-op. `news-digest explain
<story-id>` prints the per-term breakdown, and the numbers provably sum to the
score used for ranking.

**Where sources disagree, the digest says so.** Disagreements are a separate field
on the story and a separate section in the output, never folded into the summary.
Silently resolving a factual conflict is the one thing a digest of ten outlets
must not do.

**The ranking constants are not yet calibrated.** They are carried over from the
RSS-based version of this project, where each was measured against a published top
ten — against a source list that no longer exists. Ten newsletters cannot exceed
ten publishers, so `corroboration_saturation` in particular is a guess until a real
week has been measured. Every such number says so where it is defined.

**Re-running is free and safe.** A message is fetched once, parsed once, and its
articles classified and enriched once, keyed on content hash. The mailbox is opened
read-only, so a run can neither delete a newsletter nor mark one read.

<!-- digest:start -->

# The Week in Global News
14–20 September 2026

*Synthesized from ten newsletters, in two languages*

## 1. Los líderes europeos acuerdan un nuevo paquete de sanciones

El acuerdo llega tras nueve horas de negociación en Bruselas y afecta principalmente a las exportaciones de energía.

Sources:
- [EL PAÍS](https://link.email.elpais.com/c/aGVsbG8x)

## 2. EU leaders agree new sanctions package after Brussels summit

The package targets energy exports and was agreed after nine hours of negotiation, with two member states abstaining.

Sources:
- [The Guardian](https://link.email.theguardian.com/c/eJxVkMtuAyEMRb-Gpa1)

## 3. Floods displace thousands in northern Nigeria

Aid agencies say at least 40,000 people have left their homes after the Niger river breached its banks.

Sources:
- [The Guardian](https://www.theguardian.com/world/2026/sep/18/floods-displace-thousands)

## 4. Las inundaciones desplazan a miles de personas en el norte de Nigeria

Las agencias humanitarias cifran en 40.000 los desplazados por la crecida del río Níger.

Sources:
- [EL PAÍS](https://elpais.com/internacional/2026-09-18/inundaciones-nigeria.html)

## 5. Antarctic ice loss faster than models predicted, study finds

Antarctic ice loss faster than models predicted, study finds

Sources:
- [The Guardian](https://www.theguardian.com/science/2026/sep/17/antarctic-ice-study)

## 6. La inflación de la eurozona baja al 1,9% en agosto

El dato refuerza las expectativas de un recorte de tipos en octubre.

Sources:
- [EL PAÍS](https://elpais.com/economia/2026-09-16/inflacion-eurozona.html)

## 7. Eurozone inflation falls to 1.9% in August

Eurozone inflation falls to 1.9% in August

Sources:
- [The Guardian](https://www.theguardian.com/business/2026/sep/16/inflation-eurozone)

---

*Built from 7 stories, 7 items, 2 publishers, en/es.*  
*Sources: EL PAÍS, The Guardian.*

<!-- digest:end -->
