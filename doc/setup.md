# Setup

## 1. The inbox

One account, used for nothing else, subscribed to the newsletters.

Not for tidiness -- unmatched senders are counted and ignored, so a shared inbox
works fine. For blast radius. The app password ends up in GitHub Actions secrets,
and a Gmail app password reads the **entire account**, bypasses two-factor
authentication, and lasts until revoked. Pointed at a catch-all signup account
that receives password-reset mail, a leaked digest credential becomes an
account-takeover path into unrelated services. Pointed at a box containing
nothing but newsletters, it is worth almost nothing.

The connection is opened read-only (`EXAMINE`, not `SELECT`), so a run can neither
delete a newsletter nor mark one seen. Re-running a week is always safe.

---

## 2. Setting up a new Gmail account, start to finish

Work through it in order. Steps 1-4 are Google's interface, 5-7 are this
repository, 8 is deployment. Budget half an hour, most of it waiting for
newsletters to arrive.

### 1. Create the account

Nothing special -- any address will do, since matching is on the **From** header
and never on yours. Google will probably ask for a phone number; one number can
only verify a limited number of accounts, so use one you have.

Add nothing else to this account. No contacts, no other subscriptions, no
recovery mail you care about. Its emptiness is the point.

### 2. Turn on 2-Step Verification

[myaccount.google.com/security](https://myaccount.google.com/security)

**Do this first, on day one, before anything else in this guide.** On a new
account Google applies a security delay to a newly added second factor, and its
own documentation puts the ceiling at a week:

> "It may take up to 7 days for Google Authenticator to show up as an available
> option for sign in."

> "If you add a phone number for 2-Step Verification, it may take up to 7 days for
> Google to trust the phone number."

It is "up to", and often much quicker, but it is not a few minutes and it gates
everything downstream: no 2FA means no app password, which means no IMAP. Using
the phone number you already verified at signup is the most likely way to avoid
the wait, since it may be trusted already; a freshly added number is not.

Neither page documents when app passwords themselves become available.

Being blocked here costs less than it looks. Weekly newsletters take up to a week
to arrive anyway, so the delay and the first issues run down in parallel -- and the
sender addresses you need for `config/sources.yaml` can be read straight off each
message in Gmail's web interface while you wait. `sources --check` automates that
lookup; it does not have to be the way you do it.

Do this **before** looking for app passwords. Google does not offer them at all
until 2FA is on, and the app-passwords page says only:

> App passwords
> The setting that you are looking for is not available for your account.

Which reads like the feature is unavailable to you, rather than like a
prerequisite you have not met yet. On a personal Gmail account it almost always
means 2-Step Verification is off. (The other causes are a Workspace admin having
disabled app passwords, or enrolment in the Advanced Protection Program.)

Watch the account too. The app-passwords page is per-account and Google will
happily show you your *default* account's page after you signed in to a new one,
so an app password generated for the wrong mailbox looks completely valid and
fails at login with `Invalid credentials`. Check the avatar in the top right
before you read the list.

### 3. Create the app password

[myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)

Name it something you will recognise in six months -- `news-digest` -- and copy the
16 characters. **It is shown once.** Spaces in it are optional; both forms work.

If the page is missing while 2FA is on, the account is managed by a Workspace
admin who has disabled app passwords.

Two things worth knowing now rather than later:

* It can be revoked individually, at any time, without changing the account
  password. That is your kill switch.
* It cannot be scoped to one label. `NEWS_EMAIL_FOLDER` limits what this project
  reads, not what the credential can reach.

#### Where to keep it

Three places, for three purposes. Do the first one **before** pasting it anywhere,
because Google will not show it again and you need it twice.

| Where | Why |
|---|---|
| Your password manager | The durable record. Needed again for CI, and after any reinstall. |
| `.env` | Where local runs read it. Gitignored. |
| GitHub Actions secrets | Where the scheduled run reads it. Write-only once set. |

`.env` is plaintext on disk, which is the trade this project makes and worth
tightening rather than pretending otherwise:

```bash
chmod 600 .env          # it is created world-readable
```

Never put it in: a tracked file, a config YAML, a commit, or a shell command --
`export NEWS_EMAIL_PASSWORD=...` on one line writes it to your shell history. It
is already excluded from the `--debug` export by design, since that is the artefact
most likely to be pasted into an issue.

#### Keeping it out of plaintext entirely (macOS)

An exported environment variable beats `.env` -- `load_dotenv` only fills in what is
absent -- so the password can live in the Keychain instead, and `.env` need not
contain it at all.

```bash
# -w LAST, with no value, so it prompts and never enters your shell history
security add-generic-password -a "$USER" -s news-digest-imap -w

# then, in ~/.zshrc or a wrapper script
export NEWS_EMAIL_PASSWORD="$(security find-generic-password -s news-digest-imap -w)"
```

`find-generic-password -w` returns the value with no trailing newline, so it needs
no trimming. This changes nothing about CI, which reads the Actions secret.

### 4. Check IMAP is on

Gmail → Settings → See all settings → **Forwarding and POP/IMAP** → IMAP access.

Google has been making IMAP always-on, so there may be no toggle left. If the
setting exists and is off, nothing will connect.

### 5. Point the newsletters at it

Two routes, and mixing them is fine.

**New subscriptions** -- just subscribe with the new address. Nothing else to do.

**Subscriptions you already have elsewhere** -- forward them, rather than
re-subscribing eleven times:

1. In the OLD account: Settings → **Forwarding and POP/IMAP** → *Add a forwarding
   address* → the new address. Google mails a confirmation code to the new box;
   confirm it.
2. Leave the radio on **"Disable forwarding"**. You do not want the whole account
   forwarded, only the newsletters.
3. Settings → **Filters and Blocked Addresses** → *Create a new filter*. In
   **From**, list the senders separated by `OR`:

   ```
   from:(redes@elsaltodiario.com OR news@publisher-news.com OR newsletters@email.theguardian.com)
   ```

4. Action: **Forward it to** → the new address. Do **not** tick "Delete it" --
   keep the originals until the new box is proven.

Forwarding applies to **new mail only**; do not expect it to backfill. See
"Backfilling an existing archive" below if you want the weeks you already have.

While you are in here: unsubscribe from Público's *daily* ("Hoy en Público") and
subscribe to the weekly **"En pocas palabras"**. `config/sources.yaml` matches the
weekly on purpose -- this is a weekly digest, and the daily arrives from the same
sender and display name, so only the subject rule separates them.

### 6. Put the credentials in `.env`

`.env` is gitignored. Never commit a filled-in copy.

```bash
NEWS_EMAIL_HOST=imap.gmail.com
NEWS_EMAIL_PORT=993
NEWS_EMAIL_USERNAME=your-new-address@gmail.com
NEWS_EMAIL_PASSWORD=the16charapppassword
```

Leave `NEWS_EMAIL_FOLDER` unset. On a dedicated account `INBOX` is everything,
and a label only adds a thing that can be misspelled. (Gmail's Promotions tab is
still `INBOX`, so newsletters landing there are found normally. A filter that
*skips* the inbox is the one arrangement that hides them -- if you use one, name
the label in `NEWS_EMAIL_FOLDER`.)

### 7. Prove it works

```bash
news-digest sources --check
```

This is the step that saves the most time, so do it before anything else. It
opens the mailbox, runs every rule over the last 30 days, and prints both what
matched and every sender that matched nothing:

```
12 messages in the last 30 days

  ok   el-salto                   1
  NONE reuters-world              0
  ...
senders matching no source:
     1  newsletters@newsletters.reuters.com
```

Expect most sources to say `NONE` at first -- nine of the eleven ship with
placeholder addresses, because a newsletter's real sending address is rarely the
one on its website. The loop is: copy an address from the bottom list into that
source's `senders` in `config/sources.yaml`, run again, repeat. It exits non-zero
while any source is silent, so it works in a script.

Failures are distinguishable on purpose:

| What you see | What it means |
|---|---|
| `login rejected for …` (exit 3) | wrong password, or the account password instead of an app password |
| `cannot reach imap.gmail.com:993` (exit 3) | network, or a typo in the host |
| `0 messages match` | connected fine; the mail is not in this folder |
| every source `NONE` but senders listed | connected and reading; the rules are wrong, which is the normal starting state |

The password never appears in any of these. The IMAP error is deliberately not
chained, because some servers echo the login line back.

### 8. Deploy

Only once step 7 is clean. See "GitHub Actions" below.

---

### Other providers

Gmail is what this guide assumes. The rest, with the column that actually matters:

| Provider | Credential | Scoped? | Works in Actions? |
|---|---|---|---|
| Gmail | App Password (needs 2-Step Verification first) | No — full account access | Yes |
| Fastmail | Settings → Privacy & Security → App passwords | **Yes**, restrict it to IMAP | Yes |
| Outlook | App passwords; Microsoft has been withdrawing basic IMAP auth | No | Check first |
| Proton | Bridge, which listens on `127.0.0.1` | n/a | **No** |

"Scoped" is the difference between a credential that can read one protocol and one
that can read everything. Fastmail's can be limited to IMAP; Gmail's cannot, which
is the whole reason section 1 asks for an empty account. Fastmail is paid-only
(30-day trial, no free tier), so a second Gmail holding nothing is the free way to
get most of the same safety.

Proton is local-only: Bridge runs on your machine, so a Proton mailbox can be read
by a local run and never by a scheduled one. Do not plan a deployment around it.

Adding a provider that is not IMAP at all -- a Gmail or Graph API client -- means
implementing `Mailbox` in one module and registering it. See
[Development](development.md).

---

### Backfilling an existing archive

Forwarding only moves new mail, so the digest starts from the week you set it up.
If you want the weeks already sitting in the old account, read them **once,
locally**, before switching over:

```bash
# app password on the OLD account, in .env, temporarily
news-digest fetch --from 2026-08-01 --to 2026-09-21
news-digest parse --from 2026-08-01 --to 2026-09-21
```

Then put the new account's credentials back, and **revoke the old account's app
password immediately**. The messages are already in `data/news.db`, keyed by
`Message-ID`, so they are never fetched again and every past week can be built
with `news-digest digest --week 2026-W36`.

This is a deliberate, time-boxed exception to everything section 1 says. It is
acceptable only because the credential stays on your machine and never reaches
GitHub. If that trade does not appeal, skip it -- one week's wait costs nothing.

## 3. Sources

`config/sources.yaml` ships with the eleven configured newsletters and, for nine of
them, **placeholder sender addresses** -- a newsletter's real sending address is
rarely the one printed on its website. `news-digest sources --check` is how you
replace them; see step 7 above for the loop.

What the file expresses is worth understanding before you edit it.

**Recognition is two-part, and both parts matter.** `senders` is necessary but
rarely sufficient: most publishers send every newsletter they have from one
address, so `subject_patterns` is what separates the weekly digest you want from
the daily briefing you do not. Every rule a source names must pass.

**The sender is often not the publisher at all.** Público arrives from
`news@publisher-news.com`, a shared bulk-mail domain that identifies nothing and
could serve any number of outlets. Matching on it alone would let this project
claim someone else's newsletter as Público's, so that entry also carries
`sender_name_patterns: ["publico"]` -- the From display name is the only reliable
discriminator there. **Always pair a shared mailer domain with one.**

**A bare domain matches subdomains.** `@theguardian.com` matches
`news.email.theguardian.com`, because bulk mailers move between hosts without
notice. A rule written with no `@` at all is treated the same way, since that is
the obvious mistake to make and a silent no-match is the worst response to it.

**`publisher` is what corroboration counts.** Two newsletters from one outlet
covering one story are one outlet's view of it. Give them the same `publisher` or
the digest will read two as independent confirmation.

Where several entries could match one message, the one that actively matched a
subject or display-name pattern wins over a catch-all, so config file order never
decides it.

Full key reference: [Configuration](configuration.md).

## 4. Run it

```bash
news-digest run
```

That is fetch, parse, filter, cluster, summarize, rank and write, in one command.
With no `--week` or `--days` it covers the last **finished** Monday-to-Sunday
week -- which on a Wednesday means the week that ended on Sunday, not the days
since. Use `--days 7` while you are still changing things, pointed at a scratch
database; see [CLI](cli.md).

A first run that worked looks roughly like this:

```
Fetched:   7 emails, 7 new
Parsed:    7 emails
Extracted: 84 items (6 excluded, 3 dupes)
Stored:    75 articles
In window: 75 articles
Filtered:  19 not news
Clusters:  41 (6 cross-language)
Selected:  12 stories (+5 minor)
```

Things to look at rather than skip past:

* **`0 cross-language`** with embeddings enabled means coverage of one event in
  English and Spanish is staying split into separate stories, which is most of the
  point. Check `EMBEDDING_PROVIDER`.
* **`Filtered: 0`** means topic classification is not running -- no API key, or
  `classify: false` -- so the sport and the horoscopes are getting through on the
  URL patterns alone.
* **A line about unresolved links** means some newsletter's tracker could not be
  followed. Those stories are kept, but their URLs are trackers and section
  blocklists cannot see them.

Then read `digests/2026/2026-Wxx.md`. If it does not answer "what happened in the
world this week" in five to ten minutes, the thing to change is source selection,
filtering and the ranking weights -- not the infrastructure.

## 5. GitHub Actions

`.github/workflows/weekly.yml` runs the same command on Monday morning and commits
the result.

Add under **Settings → Secrets and variables → Actions**:

| Secret | Value |
|---|---|
| `NEWS_EMAIL_HOST` | `imap.gmail.com` |
| `NEWS_EMAIL_USERNAME` | the new address |
| `NEWS_EMAIL_PASSWORD` | the app password, same one as in `.env` |
| `GEMINI_API_KEY` | optional; without it the run degrades rather than failing |

Same values as `.env`, which is not read in CI. `NEWS_EMAIL_PORT` and
`NEWS_EMAIL_FOLDER` are repository *variables* with defaults of `993` and `INBOX`,
so leave them alone unless you need to change them. Variables (not secrets) also
override the model and embedding settings — see [Providers](providers.md).

Run it once by hand first: **Actions → Weekly digest → Run workflow**. A mailbox
failure exits 3 and publishes nothing, rather than falling through to a digest of
last week's leftovers, so a red run here means the secrets are wrong and not that
your data is damaged.

The workflow commits `digests/`, `data/news.db` and `README.md`. The database is
committed on purpose: it is how deduplication and the "already parsed" flag
survive between runs. Raw email bodies live in it until `prune` empties them, so
keep `storage.email_body_retention_days` short — see [Attribution](attribution.md).

### Why one workflow, not two

The RSS version of this project collected four times a day and processed weekly,
because feeds expose only their most recent items and a busy source could publish
more than one feed-window between two daily runs. Mail does not expire from a
mailbox, so there is nothing to poll for and the whole thing is one weekly job.
