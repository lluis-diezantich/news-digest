# Attribution, robots and terms

The point of the digest is to send you to the reporting, not to replace it.

- Only metadata and a **short excerpt** are stored — the feed description, or the
  opening paragraphs for scraped pages, capped by `excerpt_chars` (default 1200).
  Full article bodies are never stored, embedded or sent to a model.
- Every story links to the original articles and names every outlet and language.
  The original URL and headline are preserved verbatim; canonicalization is used
  only as an internal identity key.
- Summaries are model-written from those excerpts, and the page says so.
- The digest publishes to a **public** GitHub Pages site. That is republishing
  excerpts, not private reading, and it is the part that carries risk.

## robots.txt

Honoured per the modern convention: 4xx other than 429 means "no robots.txt, no
restrictions" (per Google's spec, explicitly including 401 and 403), while 429 and
5xx mean back off. The strict old reading turns a CDN misconfiguration into a
silent source blackout — `feeds.elpais.com` answers `robots.txt` with a Varnish
403 while publishing feeds for readers.

**A technical control is not permission.** Before enabling a `scrape` source, read
that site's terms of service. `robots.txt` saying yes is not the publisher saying
yes.

## Findings worth not rediscovering

Recorded in `config/sources.yaml` beside each entry, because each cost real time:

- **RTVE** — all 42 of its feeds are `.xml`, which both its robots files disallow:
  `api2.rtve.es` via an allowlist omitting the news feeds, `www.rtve.es` via
  `Disallow: /*.xml$` next to `Allow: /*.rss$`. A `.rss` news feed would be
  permitted; RTVE publishes none. Its news sitemap is allowed but useless — no
  titles, newest entry 2025-03. The HTML listing is permitted and carries good
  metadata, so it is scraped.
- **Reuters** — removed. Its `robots.txt` is `Allow: /plus/` then `Disallow: /`,
  with access granted only through a named allowlist. It retired its feeds *and*
  disallows scraping, so the entry could never run.
- **El Periódico (CAT)** — disallowed for everyone, probably by accident: line 50
  of its `robots.txt` is a malformed bare `User-agent:` followed by `Disallow: /`,
  and an empty token is a substring of every user agent. Not worked around.
- **Cadena SER** — no feed at any path, and bot protection returns 403
  intermittently. The scrape config is correct and the links are in the static
  HTML; the 403s make it unfit for a schedule.
- **CTXT** — its feed host and main site both return 403 to every user agent
  tried. Genuinely closed.
