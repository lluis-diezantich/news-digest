# Configuration

Two files, and adding or removing a source never requires touching code.

- `config/sources.yaml` — what to read, and what to drop on sight
- `config/preferences.yaml` — languages, the ranking formula, digest size, retention

## Sources

```yaml
defaults:
  max_items_per_source: 10
  excerpt_chars: 1200

sources:
  - name: El País Economía
    publisher: El País        # several feeds, one outlet
    rss: https://feeds.elpais.com/…/economia/portada
    languages: [es]
    weight: 1.1

  - name: RTVE Noticias
    publisher: RTVE
    url: https://www.rtve.es/noticias/
    method: scrape
    languages: [ca]
    link_selector: "article.cell a"
    link_pattern: "\\.shtml$"
    max_links: 12
```

`publisher` matters more than it looks. Corroboration counts **publishers, not
feeds**, so without it seven El País sections covering one story would read as
seven independent outlets and one publisher could dominate the digest.

`weight` no longer touches the final score — `source_preference` was removed from
the formula. It still feeds pre-ranking (which clusters get enriched) and breaks
ties over whose wording represents a story.

**`max_items_per_source: 10` is a top-ten filter, not a truncation.** Feed order
is the newsroom's own ranking: measured 80–100% concordant with each site's front
page across eight sources. So the cap takes each desk's top ten rather than ten
arbitrary items — and it is what keeps advertorial out, since Ara's sponsored
items sit at positions 14–24 of 131. Raising it puts advertising back in reach.

## Dropping junk at collection

```yaml
exclude_url_patterns:
  - /especials/          # Ara native advertising
  - /loterias?/
  - /horoscopo?/
  - /el-tiempo/
  - /deportes?/          # sports, es
  - /esports?/           # sports, ca
  - /sports?/            # sports, en
  - /football/
  - /futbol/
```

Regexes matched against the original URL, applied **before anything reaches the
database**. This is for structural junk, not taste: native advertising and
service content live at predictable paths, and no ranking signal catches them
reliably — advertorial is written to match whatever topics score well, so a topic
match actively promotes it. Broken patterns are a startup error.

A per-source `exclude_url_patterns:` is merged with the global list. The Guardian
entry used one for `/australia-news/`, which its world feed carries.

### By headline

Some junk has no section path to match. `exclude_title_patterns` catches it:

```yaml
exclude_title_patterns:
  - previsi[oó] del temps    # RAC1's daily forecast, filed under its news path
  - el tiempo hoy            # RTVE's equivalent
  - lamine yamal
  - \bbar[çc]a\b
  - \bfutbol\b
```

Two things needed it. RAC1 publishes a weather forecast every single day — six in
the 2026-W38 window, and one of them topped the `prominence` signal outright —
under its ordinary news path, where `/el-tiempo/` never sees it. And VilaWeb files
football under `/noticies/`, 3Cat under `/3catinfo/`, and 20minutos put a Lamine
Yamal family piece under `/gente/`, so `/esports?/` misses all three.

Matched against the **title only**, lowercased and accent-stripped on both sides —
write `previsió` or `prevision`, either matches. The excerpt is deliberately not
matched: an article whose background paragraph mentions football is not a football
article.

The weather patterns target **forecasts, not weather**. The same week's "Un muerto
en Barcelona y grave caos de transporte por las lluvias torrenciales" is a story
about a death and a shut-down metro; the floods, the ES-Alert and the school
closures are news. Matching `lluvia`/`pluja`/`tormenta` would take all of it. Note
the trap the tests pin: "Alerta per la previsió de pluges: el Govern demana
limitar els desplaçaments" contains *previsió* but is a government instruction.

Measured over the 1,938 stored articles, these 17 patterns drop 33. Three are
known false positives, accepted on the grounds that less football is the point:
an El Salto election analysis titled "entre el fútbol y la abstención", a VilaWeb
interview on Barça and the Catalan language, and — if these sources ever cover it
— Russia's Yamal gas field.

Not exhaustive, by design: La Vanguardia filed one cycling piece under `/clic/`,
so `excluded_topics` remains the backstop for strays.

## The ranking formula is configuration

```yaml
ranking:
  terms:
    editorial_position: 2.0  # where its source placed it
    corroboration: 1.5       # distinct publishers covering it
    recency: 0.3
    story_size: 0.2
  excluded_penalty: 1.5
  recency_half_life_hours: 72
```

`final_score = Σ weight × signal`. Delete a line to remove that signal entirely;
a misspelled term is a startup error, not a silent no-op.
`news-digest explain <story-id>` prints the per-term breakdown, and the numbers
shown provably sum to the score used for ranking.

`editorial_position` and `corroboration` carry the formula: what a desk led with,
and how many desks led with it. Both come from collection, so a degraded run with
no model available ranks exactly as well as a full one — which is the point.

`importance` used to sit alongside them at weight 2.0 and was removed on
2026-09-17. Measured over a stored week, qwen3:8b returned 0.80–0.85 for every
briefed story, so the term was a constant that could not reorder anything; worse,
stories that were never briefed kept a higher article-derived value and beat
briefed ones on it, which put a 3-publisher story above three rivals with 5 and 6.
It is still computed and still shown on the page, like `interest` and `relevance`.
Restore it when a model spreads the scale — `news-digest explain` makes that a
measurement rather than a hope.

Four terms were removed after measuring what each contributed — the first three
across a published top ten, `importance` across a stored week. The measurements
are recorded in the config file itself:

| Term | Spread | Why it went |
|---|---|---|
| `relevance` | 0.40 | "how well does this match THIS reader" — the personalization this digest is meant not to do, and the only term still moving the order |
| `interest` | 0.08 | topic matching over tags too broad to mean much |
| `importance` | 0.05 | removed 2026-09-17: qwen3:8b returns 0.80–0.85 for every briefed story, so it could not reorder anything — and it inverted, because unbriefed stories kept a higher article-derived value and beat briefed ones on it |
| `source_preference` | — | an assertion that four outlets are trustworthy, not evidence; its +0.5 favourites bonus dwarfed the 0.8–1.2 weight spread, and dropping it took one digest from 4 distinct publishers to 6 |

`editorial_position` is the counter-example worth knowing about: it was built,
measured, **removed** because Ara's advertorial ranked 0.82–0.89 on it, then
restored once `max_items: 10` and the URL blocklist put that advertorial out of
reach. Two tests fail if either control is weakened.

## Digest size

```yaml
digest:
  max_stories: 6
  min_articles: 3
```

"The main news of the week" is close to a definition of corroboration. Of 440
clusters in one week, **411 were a single outlet reporting alone**; 29 had two or
more outlets, 9 had three or more, 5 had six or more. Asking for ten stories
therefore reached into pairs and singletons.

`min_articles` counts **articles, not publishers**, so four El País sections on
one story would satisfy a floor of 3 while being one outlet. It has not bitten
yet; `min_publishers` would be the strict version.

## Languages and retention

```yaml
settings:
  supported_languages: [en, es, ca]
  output_language: en          # the LLM reads Catalan, writes English

preferences:
  topics: {}                   # emptied: every entry was too broad to mean much
  excluded_topics: [sports, celebrity, horoscope]

storage:
  retention_days: 45
  embedding_retention_days: 21   # vectors dominate the database's size
```

`excluded_topics` applies a penalty (`excluded_penalty`) rather than a hard
filter, so a genuinely enormous sports story can still surface. It is independent
of the ranking terms, so it works even with `topics` empty.
