# Configuration

Three files, none of which contain a credential.

```
config/sources.yaml      which newsletters, and how to recognise them
config/filters.yaml      what is not news
config/preferences.yaml  languages, the ranking formula, digest size, retention
prompts/*.txt            what the model is asked (versioned separately)
```

Malformed config is a **startup error that names the offending entry**, never a
silent default. That applies to unknown ranking terms, broken regexes, an
`output_language` outside the allowed set, a topic in both the include and exclude
lists, and an enabled source with no sender rule.

## sources.yaml

```yaml
sources:
  - name: guardian-saturday
    newsletter: Saturday Edition
    publisher: The Guardian
    languages: [en]
    senders: ["@theguardian.com", "@email.theguardian.com"]
    subject_patterns: ["saturday edition"]
```

| Key | |
|---|---|
| `name` | unique; identifies this newsletter |
| `senders` | addresses or domains it arrives from. **Required** if enabled |
| `subject_patterns` | regexes over the subject, accent-stripped and lowercased |
| `newsletter` | human name. Defaults to `name` |
| `publisher` | outlet this belongs to. Defaults to `name` |
| `languages` | what it publishes in; constrains detection and is the fallback |
| `enabled` | default true. A disabled source may omit `senders` |
| `weight` | 0.0–2.0, default 1.0. Feeds `source_preference` and pre-ranking |
| `topics` | hints merged into every item from this source |
| `excerpt_chars` | cap on the blurb stored per item |

**A bare domain matches subdomains.** `@theguardian.com` matches
`news.email.theguardian.com`, because bulk mailers move between hosts without
notice. A rule written with no `@` at all is treated the same way — that is the
obvious mistake to make, and a silent no-match is the worst possible response to
it.

**`publisher` is what corroboration counts.** Two EL PAÍS newsletters covering one
story are one outlet's view of it. Give them the same publisher or the digest will
read two as independent confirmation.

**Subject patterns are how you narrow one address.** Most publishers send every
newsletter they have from one address. Where several entries could match a message,
the one whose subject pattern matched wins over a catch-all, so config file order
never decides it.

### exclude_url_patterns / exclude_title_patterns

Regexes matched against an item's URL and headline, dropping it at parse time
before it reaches the database. Global lists apply to every source; per-source
lists are added to them. Duplicates collapse, and a broken regex is a startup
error.

These still work with newsletters — an extracted item links the publisher's own
article — **but only on a resolved URL.** A tracking link has no section path, so
every pattern here is inert until `extract/links.py` has unwrapped it. With
`--no-links` the whole list does nothing.

Title patterns are for junk with no section path: a weekly weather round-up, a
football result at an outlet that files sport under its main path. Title only,
deliberately — matching the excerpt would drop any article whose background
paragraph happens to mention football.

## filters.yaml

```yaml
filters:
  classify: true
  max_items: 600
  excluded_topics: [sports, celebrity]
  include_topics: [politics, world, economics, science, technology, climate, health]
  drop_content_types: []
```

`excluded_topics` must use words the classifier can actually emit — the controlled
vocabulary in `llm/base.py` `TOPICS`. A topic outside it can never be returned and
therefore can never be excluded; that silently disarmed this list once already.
There is a test asserting it.

An item is dropped only when **every** topic it carries is excluded.

`include_topics` is advisory, not a whitelist: an item matching none of them is
still kept, because the vocabulary will always lag the news.

`drop_content_types` is empty on purpose. Section 8 of the specification excludes
opinion that carries no news, which is a per-item judgement the classifier makes,
not a property of the type — and a blanket `opinion` drop would lose elDiario.es's
Boletín del director, which is a director's column and one of the ten sources.

With `classify: false` filtering is the regex lists alone, which newsletters defeat
more easily than feeds did.

## preferences.yaml

### settings

```yaml
settings:
  supported_languages: [en, es]
  output_language: en        # auto | en | es
```

`auto` is resolved per run, after the window's languages are counted: whichever
language most of the week's items were in. Ties and an empty week fall back to the
first supported language, so the result never depends on dict ordering. A Context
carrying the literal string "auto" would reach the prompts and ask the model to
write in a language called Auto, which is why it is resolved before the run starts.

### ranking

```yaml
ranking:
  terms:
    editorial_position: 2.0
    corroboration: 1.5
    recency: 0.3
    story_size: 0.2
  corroboration_saturation: 5
  recency_half_life_hours: 72
```

Eight terms are available: `editorial_position`, `corroboration`, `recency`,
`story_size`, `importance`, `relevance`, `interest`, `source_preference`. Only the
ones listed are in the formula; an unlisted one is still computed and still shown.
An unknown name is a startup error.

**`corroboration_saturation` is the number most worth measuring.** It is where the
curve reaches 1.0, and it has to sit at the top of the range your weeks actually
produce. Too high and the term flattens toward nothing; too low and every
multi-outlet story ties at exactly 1.000 — silently, since it still computes and
still appears in `explain`. Ten newsletters cannot exceed ten publishers and will
rarely pass five. `news-digest inspect` prints the publisher spread.

**All the weights are uncalibrated.** Each was measured against a published top ten
on the RSS version of this project, against a source list that no longer exists.
The shape should hold; the numbers are a starting point.

### digest

```yaml
digest:
  max_stories: 12       # section 11 asks for 10-15
  minor_stories: 5      # section 14's "Also worth knowing"
  min_articles: 1
  week_ends_on: 6       # 0 = Monday ... 6 = Sunday
  update_readme: true
```

`min_articles` is **1**, not the RSS project's 3. Ten hand-curated newsletters
rarely triple-cover anything, and at 3 most weeks would publish nothing. Raise it
only once a real week shows enough overlap. The digest falls back to the ranked
list if the filter would empty it, so it cannot yield nothing.

The pipeline enriches `2 × (max_stories + minor_stories)` candidate clusters, with
a floor of 10, and publishes from those.

### regions

```yaml
regions:
  enabled: true
  max_share: 0.5
```

Largest share of the main stories one region may hold before others are promoted.
At 0.5 with `max_stories: 12`, no region takes more than 6. Not a quota — see
[How it works](pipeline.md#rank--scoringpy).

A story whose region is unknown is never deferred: that would punish a
classification failure rather than a real imbalance.

### storage

```yaml
storage:
  retention_days: 120
  email_body_retention_days: 30
  embedding_retention_days: 60
```

Email **bodies** are emptied earliest and the rows kept. The row is what stops a
newsletter being ingested again on the next run; the body is the bulkiest and only
sensitive part, and nothing needs it once articles are extracted. See
[Attribution](attribution.md).

## Prompts

`prompts/*.txt`, one per job:

```
enrich_article.txt    per-article detail, for ranking and entity lists
classify_story.txt    the cheap triage pass
write_brief.txt       the entry a reader actually sees
cluster_stories.txt   same-event adjudication
```

Placeholders are `$name`, not `{name}` — prompts are full of prose that may contain
braces, and a stray brace in a `.format()` template raises mid-run, after the
mailbox has already been read. A missing *value* is an error rather than sending
the model the literal text `$output_language`.

A prompt file that is missing or empty is a loud failure, not a fallback to inline
text.
