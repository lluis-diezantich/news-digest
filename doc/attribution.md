# Attribution, privacy and copyright

## What is stored

| | |
|---|---|
| Email metadata | sender, subject, date, `Message-ID` — kept indefinitely |
| Email bodies | kept for `email_body_retention_days` (default 30), then **emptied** |
| Articles | headline, the newsletter's own blurb (capped at `excerpt_chars`), URL, position |
| Stories | the merged headline and summary a model wrote |

No article page is ever fetched. The only HTTP this project makes is following a
tracking link to find out which article it points at, which is a redirect, not a
crawl — so `robots.txt` and crawl-delay do not apply and the code that handled them
is gone.

## What is published

`digests/` and `README.md` carry headlines, model-written summaries, publisher
names and article URLs. They do not carry:

- raw newsletter content
- sender addresses or the mailbox address
- email subjects
- anything from `debug/`, which is gitignored

## Why bodies are emptied rather than rows deleted

The row is the record that we have seen this message, and it is what stops the same
newsletter being ingested again next month. Deleting it would make the mailbox look
new. The body is the privacy-relevant half, and once articles are extracted nothing
downstream reads it.

So `prune` empties `html_body` and `text_body` and keeps everything else. Set
`email_body_retention_days` to the shortest value that still lets you re-parse a
week after changing the extractor — 30 days is a month of slack.

## The committed database

`data/news.db` is committed, because it is how deduplication and the "already
parsed" flag survive between GitHub Actions runs. **That means email bodies newer
than the retention window are in your git history.** Two consequences:

1. Keep `email_body_retention_days` short. It is the only thing bounding how much
   newsletter content the history holds.
2. If the repository is public, treat it as public. Newsletters are not private
   correspondence, but the fact that *you* subscribe to them is inferable, and so
   is the mailbox address if a newsletter happens to print it.

Shortening the retention window does not retroactively clean the history. See
[Rebuilding](rebuilding.md).

## Credentials

Never in a config file, never in a log line, never in the debug export.

`MailboxSettings.describe()` is what appears in logs, and it prints the local part
of the username only. The IMAP login error is deliberately not chained and not
interpolated, because some servers echo the login line back — password included.
The debug export reduces a sender to its domain for the same reason: it is the part
that explains a match or a miss, and the local part is a mailbox address that has
no business in a file people paste into issues.

## Copyright

Only enough text to summarize from is stored — the newsletter's own blurb, capped.
No full article bodies, no reproduction of a newsletter's layout or editorial
selection as such. Summaries are written, not extracted, whenever a model is
available; with `--no-llm` they are extractive and that is a documented degradation.

Source attribution is preserved throughout: every story lists the publishers that
covered it and links each one's article. Where outlets disagree on a material fact,
the digest prints the disagreement rather than resolving it — the
`disagreements` field exists precisely so that merging cannot quietly pick a side.
