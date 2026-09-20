"""SQLite persistence.

Stores four things, in the order the pipeline produces them: the emails we have
seen, the articles extracted from them, the stories those cluster into, and the
digests published from those. Every stage is keyed so that re-running a week is
cheap and idempotent -- an email is fetched once, parsed once, and its articles
enriched once, however many times the run is repeated.

It is committed to the repository by GitHub Actions, so size matters. `prune()`
drops old articles, drops embeddings sooner, and clears email BODIES soonest of
all: they are the bulkiest rows and the only sensitive ones.

Schema changes bump SCHEMA_VERSION and add a migration step.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .embeddings.base import from_blob, to_blob
from .llm.base import Classification, Enrichment
from .models import (
    Article, Digest, Email, RunStats, Story, iso, parse_date, utcnow,
)
from .text import title_key

log = logging.getLogger(__name__)

# v1 of the newsletter schema. The RSS project's v1-v3 migrations were dropped
# with the database they upgraded: no deployment of this schema has ever existed,
# so there is nothing to migrate FROM, and carrying dead migration code would
# only invite someone to trust it.
SCHEMA_VERSION = 1

SCHEMA = """
-- One row per newsletter message. Kept forever (minus the body) so a message is
-- never ingested or parsed twice, which is what makes re-running a week free.
CREATE TABLE IF NOT EXISTS emails (
    id            TEXT PRIMARY KEY,
    message_id    TEXT NOT NULL UNIQUE,
    source        TEXT NOT NULL DEFAULT '',
    newsletter    TEXT NOT NULL DEFAULT '',
    sender        TEXT NOT NULL DEFAULT '',
    sender_name   TEXT NOT NULL DEFAULT '',
    subject       TEXT NOT NULL DEFAULT '',
    received_at   TEXT NOT NULL,
    -- Bodies are the bulkiest rows here and the only ones carrying mailbox
    -- content. `prune` empties them early and keeps the row, so the message
    -- stays known without the newsletter staying stored.
    html_body     TEXT NOT NULL DEFAULT '',
    text_body     TEXT NOT NULL DEFAULT '',
    body_hash     TEXT NOT NULL DEFAULT '',
    -- Set once extraction has run. NULL means "fetched, not yet parsed", which
    -- is what `unparsed_emails` looks for and what --force resets.
    parsed_at     TEXT,
    article_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_emails_received ON emails(received_at DESC);
CREATE INDEX IF NOT EXISTS idx_emails_source   ON emails(source);
CREATE INDEX IF NOT EXISTS idx_emails_parsed   ON emails(parsed_at);

CREATE TABLE IF NOT EXISTS articles (
    id            TEXT PRIMARY KEY,
    canonical     TEXT NOT NULL UNIQUE,
    url           TEXT NOT NULL,
    title         TEXT NOT NULL,
    title_key     TEXT NOT NULL,
    source        TEXT NOT NULL,
    publisher     TEXT NOT NULL DEFAULT '',
    newsletter    TEXT NOT NULL DEFAULT '',
    -- The email this was extracted from. Deliberately NOT a foreign key: an
    -- email row can be pruned, and its articles must survive that.
    email_id      TEXT NOT NULL DEFAULT '',
    received_at   TEXT,
    author        TEXT,
    language      TEXT,
    published_at  TEXT,
    collected_at  TEXT NOT NULL,
    description   TEXT NOT NULL DEFAULT '',
    content       TEXT NOT NULL DEFAULT '',
    content_hash  TEXT NOT NULL DEFAULT '',
    source_topics TEXT NOT NULL DEFAULT '[]',
    source_weight REAL NOT NULL DEFAULT 1.0,
    -- Where the item sat in its newsletter, and how many items it held.
    -- -1 = unknown. An editor both picked and ordered these, so position 0 is
    -- the strongest free signal in the pipeline.
    item_position INTEGER NOT NULL DEFAULT -1,
    item_count    INTEGER NOT NULL DEFAULT 0,
    summary        TEXT,
    why_it_matters TEXT,
    key_facts      TEXT NOT NULL DEFAULT '[]',
    topics         TEXT NOT NULL DEFAULT '[]',
    entities       TEXT NOT NULL DEFAULT '[]',
    importance     REAL,
    relevance      REAL,
    content_type   TEXT,
    -- Classification (section 8), separate from enrichment because it runs over
    -- every article rather than only the candidates. NULL newsworthy means "not
    -- classified", which is not the same as 0 and must not be filtered on.
    region         TEXT NOT NULL DEFAULT '',
    newsworthy     INTEGER,
    classified_by  TEXT,
    enriched_by    TEXT,
    enriched_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_articles_title_key ON articles(title_key);
CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_collected ON articles(collected_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_language  ON articles(language);
CREATE INDEX IF NOT EXISTS idx_articles_hash      ON articles(content_hash);
CREATE INDEX IF NOT EXISTS idx_articles_email     ON articles(email_id);

CREATE TABLE IF NOT EXISTS stories (
    id             TEXT PRIMARY KEY,
    headline       TEXT NOT NULL,
    summary        TEXT NOT NULL DEFAULT '',
    why_it_matters TEXT NOT NULL DEFAULT '',
    key_facts      TEXT NOT NULL DEFAULT '[]',
    regions        TEXT NOT NULL DEFAULT '[]',
    -- Where the sources disagreed, in their own words. A separate column rather
    -- than prose folded into `summary`, because section 15 forbids resolving a
    -- disagreement silently and a column cannot be quietly paraphrased away.
    disagreements  TEXT NOT NULL DEFAULT '[]',
    importance     REAL NOT NULL DEFAULT 0,
    relevance      REAL NOT NULL DEFAULT 0,
    score          REAL NOT NULL DEFAULT 0,
    first_seen     TEXT NOT NULL,
    last_updated   TEXT NOT NULL,
    written_by     TEXT
);
CREATE INDEX IF NOT EXISTS idx_stories_score ON stories(score DESC);

-- Membership lives here rather than on articles, so there is one source of
-- truth. An article belongs to at most one story per the spec, which the
-- UNIQUE constraint on article_id enforces.
CREATE TABLE IF NOT EXISTS article_story (
    article_id TEXT NOT NULL UNIQUE REFERENCES articles(id) ON DELETE CASCADE,
    story_id   TEXT NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    PRIMARY KEY (article_id, story_id)
);
CREATE INDEX IF NOT EXISTS idx_article_story_story ON article_story(story_id);

CREATE TABLE IF NOT EXISTS topics (
    name TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS story_topics (
    story_id TEXT NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    topic    TEXT NOT NULL REFERENCES topics(name),
    PRIMARY KEY (story_id, topic)
);
CREATE TABLE IF NOT EXISTS entities (
    name TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS story_entities (
    story_id TEXT NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    entity   TEXT NOT NULL REFERENCES entities(name),
    rank     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (story_id, entity)
);

CREATE TABLE IF NOT EXISTS weekly_digests (
    id            TEXT PRIMARY KEY,
    period_start  TEXT NOT NULL,
    period_end    TEXT NOT NULL,
    generated_at  TEXT NOT NULL,
    article_count INTEGER NOT NULL DEFAULT 0,
    stats         TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS digest_stories (
    digest_id TEXT NOT NULL REFERENCES weekly_digests(id) ON DELETE CASCADE,
    story_id  TEXT NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    rank      INTEGER NOT NULL,
    -- 'main' for a full write-up, 'minor' for section 14's "Also worth knowing".
    tier      TEXT NOT NULL DEFAULT 'main',
    PRIMARY KEY (digest_id, story_id)
);

-- Both caches are keyed by content hash plus a provider/model key, because
-- results from different models are not interchangeable.
CREATE TABLE IF NOT EXISTS llm_cache (
    content_hash TEXT NOT NULL,
    cache_key    TEXT NOT NULL,
    payload      TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (content_hash, cache_key)
);
CREATE TABLE IF NOT EXISTS embedding_cache (
    content_hash TEXT NOT NULL,
    cache_key    TEXT NOT NULL,
    vector       BLOB NOT NULL,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (content_hash, cache_key)
);

CREATE TABLE IF NOT EXISTS sources (
    name                 TEXT PRIMARY KEY,
    publisher            TEXT NOT NULL DEFAULT '',
    languages            TEXT NOT NULL DEFAULT '[]',
    last_ok              TEXT,
    last_error_at        TEXT,
    last_error           TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    total_articles       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL DEFAULT 'collect',
    started_at  TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    stats       TEXT NOT NULL
);
"""


def _int_or(row, key: str, default: int) -> int:
    """Read an int column that may be absent on a database opened read-only."""
    try:
        value = row[key]
    except (IndexError, KeyError):
        return default
    return default if value is None else int(value)


def _dumps(value: Any) -> str:
    return json.dumps(value if value is not None else [], ensure_ascii=False)


def _loads(value: str | None) -> list:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


class Store:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    # -- schema -------------------------------------------------------------

    def _migrate(self) -> None:
        """Create the schema, and refuse to open a database from the RSS project.

        That database has an `articles.source_url` and no `emails` table, and
        nothing here would fail on it -- `CREATE TABLE IF NOT EXISTS` is silent,
        the missing columns read as empty, and the run would publish a digest
        built from articles whose provenance it had invented. So it is detected
        and named instead.
        """
        with closing(self.conn.cursor()) as cur:
            tables = {
                row[0]
                for row in cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "articles" in tables and "emails" not in tables:
                raise sqlite3.DatabaseError(
                    f"{self.path} holds the RSS project's schema, which this "
                    f"version cannot read. Move it aside and let a fresh "
                    f"database be created."
                )
            version = cur.execute("PRAGMA user_version").fetchone()[0]

        self.conn.executescript(SCHEMA)
        if version != SCHEMA_VERSION:
            self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.conn.commit()

    def _columns(self, table: str) -> set[str]:
        return {
            row["name"]
            for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        }

    def close(self) -> None:
        # Collapse the WAL so the committed .db file is self-contained.
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- emails -------------------------------------------------------------

    def known_email_ids(self, ids: Sequence[str]) -> set[str]:
        """Which of these messages we already have. The re-ingestion guard."""
        found: set[str] = set()
        for chunk in _chunks(list(ids), 500):
            marks = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT id FROM emails WHERE id IN ({marks})", chunk
            ).fetchall()
            found.update(r["id"] for r in rows)
        return found

    def insert_emails(self, emails: Iterable[Email]) -> int:
        """Store messages we have not seen. Returns how many were new.

        Existing rows are left alone rather than updated. A message is immutable
        -- its Message-ID keys it -- so a second sighting carries no new
        information, and overwriting would clobber `parsed_at` and re-parse the
        whole newsletter.
        """
        rows = [
            (
                e.id, e.message_id, e.source, e.newsletter, e.sender, e.sender_name,
                e.subject, iso(e.received_at), e.html_body, e.text_body,
                e.content_hash(),
            )
            for e in emails
        ]
        if not rows:
            return 0
        with self.conn:
            before = self.conn.total_changes
            self.conn.executemany(
                """
                INSERT INTO emails (
                    id, message_id, source, newsletter, sender, sender_name,
                    subject, received_at, html_body, text_body, body_hash
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO NOTHING
                """,
                rows,
            )
            return self.conn.total_changes - before

    def emails_in_window(
        self,
        start: datetime,
        end: datetime,
        *,
        sources: Sequence[str] | None = None,
        unparsed_only: bool = False,
    ) -> list[Email]:
        """Stored messages received in [start, end)."""
        params: list[Any] = [iso(start), iso(end)]
        clause = ""
        if sources:
            marks = ",".join("?" * len(sources))
            clause += f" AND source IN ({marks})"
            params.extend(sources)
        if unparsed_only:
            clause += " AND parsed_at IS NULL"
        rows = self.conn.execute(
            f"""
            SELECT * FROM emails
            WHERE received_at >= ? AND received_at < ?{clause}
            ORDER BY received_at
            """,
            params,
        ).fetchall()
        return [_row_to_email(r) for r in rows]

    def mark_parsed(self, email_id: str, article_count: int) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE emails SET parsed_at = ?, article_count = ? WHERE id = ?",
                (iso(utcnow()), article_count, email_id),
            )

    def reset_parsed(self, email_ids: Sequence[str] | None = None) -> int:
        """Mark messages unparsed again, for `--force`.

        Their articles are left in place: the extractor is deterministic, so a
        re-parse produces the same ids and the insert is a no-op. What changes is
        that a NEW extractor sees the newsletter again, which is the only reason
        to ask for this.
        """
        with self.conn:
            before = self.conn.total_changes
            if email_ids is None:
                self.conn.execute("UPDATE emails SET parsed_at = NULL")
            else:
                for chunk in _chunks(list(email_ids), 500):
                    marks = ",".join("?" * len(chunk))
                    self.conn.execute(
                        f"UPDATE emails SET parsed_at = NULL WHERE id IN ({marks})",
                        chunk,
                    )
            return self.conn.total_changes - before

    def email_counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT source, COUNT(*) AS n FROM emails GROUP BY source ORDER BY n DESC"
        ).fetchall()
        return {r["source"] or "unmatched": r["n"] for r in rows}

    # -- collection ---------------------------------------------------------

    def known_ids(self, ids: Sequence[str]) -> set[str]:
        found: set[str] = set()
        for chunk in _chunks(list(ids), 500):
            marks = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT id FROM articles WHERE id IN ({marks})", chunk
            ).fetchall()
            found.update(r["id"] for r in rows)
        return found

    def recent_fingerprints(self, hours: float) -> list[tuple[str, str, str]]:
        """(id, title_key, source) for the same-source re-run check."""
        cutoff = iso(utcnow() - timedelta(hours=hours))
        rows = self.conn.execute(
            "SELECT id, title_key, source FROM articles "
            "WHERE COALESCE(published_at, collected_at) >= ?",
            (cutoff,),
        ).fetchall()
        return [(r["id"], r["title_key"], r["source"]) for r in rows]

    def insert_articles(self, articles: Iterable[Article]) -> int:
        rows = [
            (
                a.id, a.canonical, a.url, a.title, title_key(a.title), a.source,
                a.publisher, a.newsletter, a.email_id, iso(a.received_at),
                a.author, a.language,
                iso(a.published_at), iso(a.collected_at),
                a.description, a.content, a.content_hash(),
                _dumps(a.source_topics), a.source_weight,
                a.item_position, a.item_count,
            )
            for a in articles
        ]
        if not rows:
            return 0
        with self.conn:
            before = self.conn.total_changes
            self.conn.executemany(
                """
                INSERT INTO articles (
                    id, canonical, url, title, title_key, source, publisher,
                    newsletter, email_id, received_at, author, language,
                    published_at, collected_at,
                    description, content, content_hash, source_topics, source_weight,
                    item_position, item_count
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO NOTHING
                """,
                rows,
            )
            return self.conn.total_changes - before

    def language_counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT COALESCE(language, 'unknown') AS lang, COUNT(*) AS n "
            "FROM articles GROUP BY lang ORDER BY n DESC"
        ).fetchall()
        return {r["lang"]: r["n"] for r in rows}

    # -- weekly window ------------------------------------------------------

    def articles_in_window(
        self, start: datetime, end: datetime, *, languages: Sequence[str] | None = None
    ) -> list[Article]:
        """Articles published (or, lacking a date, collected) in the window."""
        params: list[Any] = [iso(start), iso(end)]
        clause = ""
        if languages:
            marks = ",".join("?" * len(languages))
            clause = f" AND (language IN ({marks}) OR language IS NULL)"
            params.extend(languages)
        rows = self.conn.execute(
            f"""
            SELECT * FROM articles
            WHERE COALESCE(published_at, collected_at) >= ?
              AND COALESCE(published_at, collected_at) < ?{clause}
            ORDER BY COALESCE(published_at, collected_at) DESC
            """,
            params,
        ).fetchall()
        return [_row_to_article(r) for r in rows]

    def language_counts_in_window(
        self, start: datetime, end: datetime
    ) -> dict[str, int]:
        """language -> article count for a window, without hydrating the rows.

        Used to resolve `output_language: auto` before the run builds its prompt
        context, which is why it is a count rather than a pass over the articles.
        """
        rows = self.conn.execute(
            """
            SELECT COALESCE(language, 'unknown') AS lang, COUNT(*) AS n
            FROM articles
            WHERE COALESCE(published_at, collected_at) >= ?
              AND COALESCE(published_at, collected_at) < ?
            GROUP BY lang ORDER BY n DESC
            """,
            (iso(start), iso(end)),
        ).fetchall()
        return {r["lang"]: r["n"] for r in rows}

    def get_articles(self, ids: Sequence[str]) -> list[Article]:
        out: list[Article] = []
        for chunk in _chunks(list(ids), 400):
            marks = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT * FROM articles WHERE id IN ({marks})", chunk
            ).fetchall()
            out.extend(_row_to_article(r) for r in rows)
        return out

    def save_enrichment(self, article_id: str, enrichment: Enrichment, provider: str) -> None:
        with self.conn:
            self.conn.execute(
                """
                UPDATE articles SET
                    summary = ?, why_it_matters = ?, key_facts = ?, topics = ?,
                    entities = ?, importance = ?, relevance = ?, content_type = ?,
                    enriched_by = ?, enriched_at = ?
                WHERE id = ?
                """,
                (
                    enrichment.summary, enrichment.why_it_matters,
                    _dumps(enrichment.key_facts), _dumps(enrichment.topics),
                    _dumps(enrichment.entities), enrichment.importance,
                    enrichment.relevance, enrichment.content_type,
                    provider, iso(utcnow()), article_id,
                ),
            )

    def save_classification(
        self, article_id: str, classification: Classification, provider: str
    ) -> None:
        """Persist a triage verdict.

        `topics` is written only when the article has none yet. Enrichment writes
        a better-informed set from the full excerpt, and it may run either side of
        classification depending on the cache -- so the cheap pass must not
        overwrite the expensive one's answer.
        """
        with self.conn:
            self.conn.execute(
                """
                UPDATE articles SET
                    region = ?, newsworthy = ?, classified_by = ?,
                    content_type = COALESCE(content_type, ?),
                    topics = CASE WHEN topics IN ('[]', '') THEN ? ELSE topics END
                WHERE id = ?
                """,
                (
                    classification.region,
                    int(classification.newsworthy),
                    provider,
                    classification.content_type,
                    _dumps(classification.topics),
                    article_id,
                ),
            )

    # -- caches -------------------------------------------------------------

    def cached_classifications(
        self, hashes: Sequence[str], cache_key: str
    ) -> dict[str, Classification]:
        out: dict[str, Classification] = {}
        for chunk in _chunks(list(hashes), 400):
            marks = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT content_hash, payload FROM llm_cache "
                f"WHERE cache_key = ? AND content_hash IN ({marks})",
                [cache_key, *chunk],
            ).fetchall()
            for row in rows:
                try:
                    data = json.loads(row["payload"])
                except json.JSONDecodeError:
                    continue
                out[row["content_hash"]] = Classification(
                    id="",
                    topics=data.get("topics") or [],
                    region=data.get("region", ""),
                    newsworthy=bool(data.get("newsworthy", True)),
                    content_type=data.get("content_type"),
                )
        return out

    def cache_classification(
        self, content_hash: str, cache_key: str, classification: Classification
    ) -> None:
        payload = json.dumps(
            {
                "topics": classification.topics,
                "region": classification.region,
                "newsworthy": classification.newsworthy,
                "content_type": classification.content_type,
            },
            ensure_ascii=False,
        )
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO llm_cache (content_hash, cache_key, payload, created_at)
                VALUES (?,?,?,?)
                ON CONFLICT(content_hash, cache_key) DO UPDATE SET
                    payload = excluded.payload, created_at = excluded.created_at
                """,
                (content_hash, cache_key, payload, iso(utcnow())),
            )


    def cached_enrichments(
        self, hashes: Sequence[str], cache_key: str
    ) -> dict[str, Enrichment]:
        out: dict[str, Enrichment] = {}
        for chunk in _chunks(list(hashes), 400):
            marks = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT content_hash, payload FROM llm_cache "
                f"WHERE cache_key = ? AND content_hash IN ({marks})",
                [cache_key, *chunk],
            ).fetchall()
            for row in rows:
                try:
                    data = json.loads(row["payload"])
                except json.JSONDecodeError:
                    continue
                out[row["content_hash"]] = Enrichment(
                    id="",
                    summary=data.get("summary", ""),
                    why_it_matters=data.get("why_it_matters", ""),
                    key_facts=data.get("key_facts") or [],
                    topics=data.get("topics") or [],
                    entities=data.get("entities") or [],
                    importance=data.get("importance", 0.5),
                    relevance=data.get("relevance", 0.5),
                    content_type=data.get("content_type"),
                )
        return out

    def cache_enrichment(
        self, content_hash: str, cache_key: str, enrichment: Enrichment
    ) -> None:
        payload = json.dumps(
            {
                "summary": enrichment.summary,
                "why_it_matters": enrichment.why_it_matters,
                "key_facts": enrichment.key_facts,
                "topics": enrichment.topics,
                "entities": enrichment.entities,
                "importance": enrichment.importance,
                "relevance": enrichment.relevance,
                "content_type": enrichment.content_type,
            },
            ensure_ascii=False,
        )
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO llm_cache (content_hash, cache_key, payload, created_at)
                VALUES (?,?,?,?)
                ON CONFLICT(content_hash, cache_key) DO UPDATE SET
                    payload = excluded.payload, created_at = excluded.created_at
                """,
                (content_hash, cache_key, payload, iso(utcnow())),
            )

    def cached_vectors(
        self, hashes: Sequence[str], cache_key: str
    ) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for chunk in _chunks(list(hashes), 400):
            marks = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT content_hash, vector FROM embedding_cache "
                f"WHERE cache_key = ? AND content_hash IN ({marks})",
                [cache_key, *chunk],
            ).fetchall()
            for row in rows:
                out[row["content_hash"]] = from_blob(row["vector"])
        return out

    def cache_vectors(
        self, vectors: dict[str, np.ndarray], cache_key: str
    ) -> None:
        if not vectors:
            return
        now = iso(utcnow())
        with self.conn:
            self.conn.executemany(
                """
                INSERT INTO embedding_cache (content_hash, cache_key, vector, created_at)
                VALUES (?,?,?,?)
                ON CONFLICT(content_hash, cache_key) DO UPDATE SET
                    vector = excluded.vector, created_at = excluded.created_at
                """,
                [(h, cache_key, to_blob(v), now) for h, v in vectors.items()],
            )

    # -- stories ------------------------------------------------------------

    def replace_story(self, story: Story, article_ids: Sequence[str]) -> bool:
        """Write a story and its membership. Returns True if the story is new."""
        existing = self.conn.execute(
            "SELECT first_seen FROM stories WHERE id = ?", (story.id,)
        ).fetchone()
        first_seen = parse_date(existing["first_seen"]) if existing else story.first_seen

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO stories (
                    id, headline, summary, why_it_matters, key_facts, regions,
                    disagreements, importance, relevance, score,
                    first_seen, last_updated, written_by
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    headline = excluded.headline, summary = excluded.summary,
                    why_it_matters = excluded.why_it_matters,
                    key_facts = excluded.key_facts, regions = excluded.regions,
                    disagreements = excluded.disagreements,
                    importance = excluded.importance, relevance = excluded.relevance,
                    score = excluded.score, last_updated = excluded.last_updated,
                    written_by = excluded.written_by
                """,
                (
                    story.id, story.headline, story.summary, story.why_it_matters,
                    _dumps(story.key_facts), _dumps(story.regions),
                    _dumps(story.disagreements),
                    story.importance, story.relevance,
                    story.score, iso(first_seen or story.first_seen),
                    iso(story.last_updated), story.written_by,
                ),
            )
            self.conn.execute("DELETE FROM article_story WHERE story_id = ?", (story.id,))
            self.conn.executemany(
                "INSERT OR REPLACE INTO article_story (article_id, story_id) VALUES (?,?)",
                [(aid, story.id) for aid in article_ids],
            )

            self.conn.execute("DELETE FROM story_topics WHERE story_id = ?", (story.id,))
            for topic in story.topics:
                self.conn.execute("INSERT OR IGNORE INTO topics (name) VALUES (?)", (topic,))
                self.conn.execute(
                    "INSERT OR IGNORE INTO story_topics (story_id, topic) VALUES (?,?)",
                    (story.id, topic),
                )
            self.conn.execute("DELETE FROM story_entities WHERE story_id = ?", (story.id,))
            for rank, entity in enumerate(story.entities):
                self.conn.execute("INSERT OR IGNORE INTO entities (name) VALUES (?)", (entity,))
                self.conn.execute(
                    "INSERT OR IGNORE INTO story_entities (story_id, entity, rank) VALUES (?,?,?)",
                    (story.id, entity, rank),
                )
        return existing is None

    def get_story(self, story_id: str) -> Story | None:
        row = self.conn.execute("SELECT * FROM stories WHERE id = ?", (story_id,)).fetchone()
        return self._hydrate_story(row) if row else None

    def _hydrate_story(self, row: sqlite3.Row) -> Story:
        story = _row_to_story(row)
        story.topics = [
            r["topic"]
            for r in self.conn.execute(
                "SELECT topic FROM story_topics WHERE story_id = ? ORDER BY topic",
                (story.id,),
            )
        ]
        story.entities = [
            r["entity"]
            for r in self.conn.execute(
                "SELECT entity FROM story_entities WHERE story_id = ? ORDER BY rank",
                (story.id,),
            )
        ]
        story.article_ids = [
            r["article_id"]
            for r in self.conn.execute(
                "SELECT article_id FROM article_story WHERE story_id = ?", (story.id,)
            )
        ]
        return story

    def articles_by_story(self, story_ids: Sequence[str]) -> dict[str, list[Article]]:
        grouped: dict[str, list[Article]] = {sid: [] for sid in story_ids}
        for chunk in _chunks(list(story_ids), 300):
            marks = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"""
                SELECT a.*, m.story_id AS _story FROM articles a
                JOIN article_story m ON m.article_id = a.id
                WHERE m.story_id IN ({marks})
                ORDER BY COALESCE(a.published_at, a.collected_at) DESC
                """,
                chunk,
            ).fetchall()
            for row in rows:
                grouped[row["_story"]].append(_row_to_article(row))
        return grouped

    # -- digests ------------------------------------------------------------

    def save_digest(self, digest: Digest) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO weekly_digests (
                    id, period_start, period_end, generated_at, article_count, stats
                ) VALUES (?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    period_start = excluded.period_start,
                    period_end = excluded.period_end,
                    generated_at = excluded.generated_at,
                    article_count = excluded.article_count,
                    stats = excluded.stats
                """,
                (
                    digest.id, iso(digest.period_start), iso(digest.period_end),
                    iso(digest.generated_at), digest.article_count,
                    json.dumps(digest.stats, ensure_ascii=False),
                ),
            )
            self.conn.execute("DELETE FROM digest_stories WHERE digest_id = ?", (digest.id,))
            rows = [
                (digest.id, sid, rank, "main")
                for rank, sid in enumerate(digest.story_ids)
            ]
            # Minor ranks continue the main sequence so `ORDER BY rank` alone
            # reproduces the published order across both tiers.
            rows += [
                (digest.id, sid, len(digest.story_ids) + rank, "minor")
                for rank, sid in enumerate(digest.minor_story_ids)
                if sid not in digest.story_ids
            ]
            self.conn.executemany(
                "INSERT INTO digest_stories (digest_id, story_id, rank, tier) "
                "VALUES (?,?,?,?)",
                rows,
            )

    def get_digest(self, digest_id: str) -> Digest | None:
        row = self.conn.execute(
            "SELECT * FROM weekly_digests WHERE id = ?", (digest_id,)
        ).fetchone()
        return self._hydrate_digest(row) if row else None

    def latest_digest(self) -> Digest | None:
        row = self.conn.execute(
            "SELECT * FROM weekly_digests ORDER BY period_end DESC LIMIT 1"
        ).fetchone()
        return self._hydrate_digest(row) if row else None

    def list_digests(self, limit: int = 52) -> list[Digest]:
        rows = self.conn.execute(
            "SELECT * FROM weekly_digests ORDER BY period_end DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._hydrate_digest(r) for r in rows]

    def _hydrate_digest(self, row: sqlite3.Row) -> Digest:
        members = self.conn.execute(
            "SELECT story_id, tier FROM digest_stories WHERE digest_id = ? "
            "ORDER BY rank",
            (row["id"],),
        ).fetchall()
        story_ids = [r["story_id"] for r in members if r["tier"] != "minor"]
        minor_ids = [r["story_id"] for r in members if r["tier"] == "minor"]
        try:
            stats = json.loads(row["stats"])
        except json.JSONDecodeError:
            stats = {}
        return Digest(
            id=row["id"],
            period_start=parse_date(row["period_start"]) or utcnow(),
            period_end=parse_date(row["period_end"]) or utcnow(),
            story_ids=story_ids,
            minor_story_ids=minor_ids,
            generated_at=parse_date(row["generated_at"]) or utcnow(),
            article_count=row["article_count"],
            stats=stats,
        )

    def digest_stories(self, digest_id: str, tier: str | None = None) -> list[Story]:
        clause = " AND d.tier = ?" if tier else ""
        params: list[Any] = [digest_id] + ([tier] if tier else [])
        rows = self.conn.execute(
            f"""
            SELECT s.* FROM stories s
            JOIN digest_stories d ON d.story_id = s.id
            WHERE d.digest_id = ?{clause} ORDER BY d.rank
            """,
            params,
        ).fetchall()
        return [self._hydrate_story(r) for r in rows]

    # -- bookkeeping --------------------------------------------------------

    def register_sources(self, sources: Iterable[Any]) -> None:
        """Keep the sources table in step with the config file."""
        with self.conn:
            for source in sources:
                self.conn.execute(
                    """
                    INSERT INTO sources (name, publisher, languages)
                    VALUES (?,?,?)
                    ON CONFLICT(name) DO UPDATE SET
                        publisher = excluded.publisher, languages = excluded.languages
                    """,
                    (source.name, source.publisher, _dumps(source.languages)),
                )

    def record_run(self, stats: RunStats, kind: str = "collect") -> None:
        payload = stats.to_json()
        with self.conn:
            self.conn.execute(
                "INSERT INTO runs (kind, started_at, finished_at, stats) VALUES (?,?,?,?)",
                (kind, payload["started_at"], payload["finished_at"], json.dumps(payload)),
            )
            for report in stats.sources:
                if report.ok:
                    self.conn.execute(
                        """
                        INSERT INTO sources (name, last_ok, consecutive_failures, total_articles)
                        VALUES (?,?,0,?)
                        ON CONFLICT(name) DO UPDATE SET
                            last_ok = excluded.last_ok, consecutive_failures = 0,
                            total_articles = sources.total_articles + excluded.total_articles
                        """,
                        (report.name, iso(utcnow()), report.new),
                    )
                else:
                    self.conn.execute(
                        """
                        INSERT INTO sources (name, last_error_at, last_error, consecutive_failures)
                        VALUES (?,?,?,1)
                        ON CONFLICT(name) DO UPDATE SET
                            last_error_at = excluded.last_error_at,
                            last_error = excluded.last_error,
                            consecutive_failures = sources.consecutive_failures + 1
                        """,
                        (report.name, iso(utcnow()), report.error),
                    )

    def source_health(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM sources ORDER BY consecutive_failures DESC, name"
        ).fetchall()
        return [dict(r) for r in rows]

    def prune(
        self,
        retention_days: int,
        embedding_retention_days: int = 60,
        email_body_retention_days: int = 30,
    ) -> int:
        """Drop old articles and stale cache rows; empty old email bodies.

        Bodies are EMPTIED rather than their rows deleted. The row is the record
        that we have seen this message, and it is what stops the same newsletter
        being ingested again on the next run -- deleting it would make the
        mailbox look new every month. It is also the privacy-relevant half
        (section 28): once articles are extracted, keeping the newsletter itself
        buys nothing.
        """
        articles_cutoff = iso(utcnow() - timedelta(days=retention_days))
        vectors_cutoff = iso(utcnow() - timedelta(days=embedding_retention_days))
        bodies_cutoff = iso(utcnow() - timedelta(days=email_body_retention_days))
        with self.conn:
            before = self.conn.total_changes
            self.conn.execute(
                """
                UPDATE emails SET html_body = '', text_body = ''
                WHERE received_at < ? AND (html_body != '' OR text_body != '')
                """,
                (bodies_cutoff,),
            )
            self.conn.execute(
                "DELETE FROM articles WHERE COALESCE(published_at, collected_at) < ?",
                (articles_cutoff,),
            )
            # Keep any story a published digest still points at, so the archive
            # does not develop holes.
            self.conn.execute(
                """
                DELETE FROM stories WHERE id NOT IN (SELECT story_id FROM article_story)
                  AND id NOT IN (SELECT story_id FROM digest_stories)
                """
            )
            self.conn.execute(
                "DELETE FROM embedding_cache WHERE created_at < ?", (vectors_cutoff,)
            )
            self.conn.execute("DELETE FROM llm_cache WHERE created_at < ?", (articles_cutoff,))
            self.conn.execute("DELETE FROM runs WHERE started_at < ?", (articles_cutoff,))
            removed = self.conn.total_changes - before
        self.conn.execute("VACUUM")
        return removed

    def summary(self) -> dict[str, Any]:
        def scalar(sql: str) -> int:
            return int(self.conn.execute(sql).fetchone()[0] or 0)

        last_run = self.conn.execute(
            "SELECT kind, stats FROM runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        latest = self.latest_digest()
        return {
            "db": str(self.path),
            "size_kb": round(self.path.stat().st_size / 1024, 1) if self.path.exists() else 0,
            "emails": scalar("SELECT COUNT(*) FROM emails"),
            "emails_unparsed": scalar(
                "SELECT COUNT(*) FROM emails WHERE parsed_at IS NULL"
            ),
            "emails_by_source": self.email_counts(),
            "articles": scalar("SELECT COUNT(*) FROM articles"),
            "unenriched": scalar("SELECT COUNT(*) FROM articles WHERE enriched_at IS NULL"),
            "languages": self.language_counts(),
            "stories": scalar("SELECT COUNT(*) FROM stories"),
            "digests": scalar("SELECT COUNT(*) FROM weekly_digests"),
            "latest_digest": latest.id if latest else None,
            "cached_enrichments": scalar("SELECT COUNT(*) FROM llm_cache"),
            "cached_vectors": scalar("SELECT COUNT(*) FROM embedding_cache"),
            "runs": scalar("SELECT COUNT(*) FROM runs"),
            "last_run_kind": last_run["kind"] if last_run else None,
            "last_run": json.loads(last_run["stats"]) if last_run else None,
        }


def _row_to_email(row: sqlite3.Row) -> Email:
    message = Email(
        message_id=row["message_id"],
        source=row["source"] or "",
        subject=row["subject"] or "",
        sender=row["sender"] or "",
        sender_name=row["sender_name"] or "",
        newsletter=row["newsletter"] or "",
        received_at=parse_date(row["received_at"]) or utcnow(),
        html_body=row["html_body"] or "",
        text_body=row["text_body"] or "",
    )
    message.id = row["id"]
    message.parsed_at = parse_date(row["parsed_at"])
    message.article_count = _int_or(row, "article_count", 0)
    return message


def _row_to_article(row: sqlite3.Row) -> Article:
    article = Article(
        title=row["title"],
        source=row["source"],
        url=row["url"],
        published_at=parse_date(row["published_at"]),
        author=row["author"],
        description=row["description"] or "",
        content=row["content"] or "",
        language=row["language"],
        publisher=row["publisher"] or row["source"],
        newsletter=row["newsletter"] or "",
        email_id=row["email_id"] or "",
        received_at=parse_date(row["received_at"]),
        source_topics=_loads(row["source_topics"]),
        source_weight=row["source_weight"],
        item_position=_int_or(row, "item_position", -1),
        item_count=_int_or(row, "item_count", 0),
        collected_at=parse_date(row["collected_at"]) or utcnow(),
    )
    article.id = row["id"]
    article.canonical = row["canonical"]
    article.summary = row["summary"]
    article.why_it_matters = row["why_it_matters"]
    article.key_facts = _loads(row["key_facts"])
    article.topics = _loads(row["topics"])
    article.entities = _loads(row["entities"])
    article.importance = row["importance"]
    article.relevance = row["relevance"]
    article.content_type = row["content_type"]
    article.region = row["region"] or ""
    article.newsworthy = (
        None if row["newsworthy"] is None else bool(row["newsworthy"])
    )
    article.classified_by = row["classified_by"]
    article.enriched_by = row["enriched_by"]
    article.enriched_at = parse_date(row["enriched_at"])
    return article


def _row_to_story(row: sqlite3.Row) -> Story:
    return Story(
        id=row["id"],
        headline=row["headline"],
        summary=row["summary"] or "",
        why_it_matters=row["why_it_matters"] or "",
        key_facts=_loads(row["key_facts"]),
        regions=_loads(row["regions"]),
        disagreements=_loads(row["disagreements"]),
        importance=row["importance"] or 0.0,
        relevance=row["relevance"] or 0.0,
        score=row["score"] or 0.0,
        first_seen=parse_date(row["first_seen"]) or utcnow(),
        last_updated=parse_date(row["last_updated"]) or utcnow(),
        written_by=row["written_by"],
    )


def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]
