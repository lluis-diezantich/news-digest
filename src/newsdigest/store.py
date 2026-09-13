"""SQLite persistence.

The database is what separates the two pipelines: daily collection writes
articles into it, and the weekly run reads a window back out. It is committed to
the repository by GitHub Actions, so size matters -- `prune()` drops old articles
and drops embeddings sooner still, since those are only working data for one
weekly run and are the bulkiest rows here.

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
from .llm.base import Enrichment
from .models import Article, Digest, RunStats, Story, iso, parse_date, utcnow
from .text import title_key

log = logging.getLogger(__name__)

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id            TEXT PRIMARY KEY,
    canonical     TEXT NOT NULL UNIQUE,
    url           TEXT NOT NULL,
    title         TEXT NOT NULL,
    title_key     TEXT NOT NULL,
    source        TEXT NOT NULL,
    publisher     TEXT NOT NULL DEFAULT '',
    source_url    TEXT NOT NULL DEFAULT '',
    author        TEXT,
    language      TEXT,
    published_at  TEXT,
    collected_at  TEXT NOT NULL,
    description   TEXT NOT NULL DEFAULT '',
    content       TEXT NOT NULL DEFAULT '',
    content_hash  TEXT NOT NULL DEFAULT '',
    source_topics TEXT NOT NULL DEFAULT '[]',
    source_weight REAL NOT NULL DEFAULT 1.0,
    summary        TEXT,
    why_it_matters TEXT,
    key_facts      TEXT NOT NULL DEFAULT '[]',
    topics         TEXT NOT NULL DEFAULT '[]',
    entities       TEXT NOT NULL DEFAULT '[]',
    importance     REAL,
    relevance      REAL,
    content_type   TEXT,
    enriched_by    TEXT,
    enriched_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_articles_title_key ON articles(title_key);
CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_collected ON articles(collected_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_language  ON articles(language);
CREATE INDEX IF NOT EXISTS idx_articles_hash      ON articles(content_hash);

CREATE TABLE IF NOT EXISTS stories (
    id             TEXT PRIMARY KEY,
    headline       TEXT NOT NULL,
    summary        TEXT NOT NULL DEFAULT '',
    why_it_matters TEXT NOT NULL DEFAULT '',
    key_facts      TEXT NOT NULL DEFAULT '[]',
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
        with closing(self.conn.cursor()) as cur:
            version = cur.execute("PRAGMA user_version").fetchone()[0]
            had_tables = bool(
                cur.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='articles'"
                ).fetchone()
            )
        if version == 0 and had_tables:
            version = 1  # v1 predates user_version being set on an existing db

        if version and version < 2:
            self._migrate_v1_to_v2()

        self.conn.executescript(SCHEMA)
        if version < SCHEMA_VERSION:
            self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.conn.commit()

    def _columns(self, table: str) -> set[str]:
        return {
            row["name"]
            for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        }

    def _migrate_v1_to_v2(self) -> None:
        """v1 stored one denormalized articles table and no digests.

        `CREATE TABLE IF NOT EXISTS` silently does nothing for a table that
        already exists, so every column v2 added to a v1 table has to be added
        explicitly here -- otherwise the schema looks current while a query for
        one of the new columns fails at runtime.
        """
        log.info("migrating database schema v1 -> v2")
        columns = self._columns("articles")
        with self.conn:
            for table, column, decl in [
                ("runs", "kind", "TEXT NOT NULL DEFAULT 'collect'"),
                ("stories", "relevance", "REAL NOT NULL DEFAULT 0"),
                ("stories", "key_facts", "TEXT NOT NULL DEFAULT '[]'"),
            ]:
                if self._columns(table) and column not in self._columns(table):
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

            if "fetched_at" in columns and "collected_at" not in columns:
                self.conn.execute(
                    "ALTER TABLE articles RENAME COLUMN fetched_at TO collected_at"
                )
            for column, decl in [
                ("publisher", "TEXT NOT NULL DEFAULT ''"),
                ("source_url", "TEXT NOT NULL DEFAULT ''"),
                ("language", "TEXT"),
                ("content", "TEXT NOT NULL DEFAULT ''"),
                ("content_hash", "TEXT NOT NULL DEFAULT ''"),
                ("key_facts", "TEXT NOT NULL DEFAULT '[]'"),
                ("relevance", "REAL"),
                ("content_type", "TEXT"),
            ]:
                if column not in columns:
                    self.conn.execute(f"ALTER TABLE articles ADD COLUMN {column} {decl}")
            self.conn.execute(
                "UPDATE articles SET publisher = source WHERE publisher = ''"
            )

            # v1's llm_cache had a `provider` column instead of `cache_key`.
            cache_columns = {
                row["name"]
                for row in self.conn.execute("PRAGMA table_info(llm_cache)").fetchall()
            }
            if cache_columns and "cache_key" not in cache_columns:
                self.conn.execute("ALTER TABLE llm_cache RENAME TO llm_cache_v1")

            # v1 kept story membership on articles.story_id.
            if "story_id" in columns:
                self.conn.executescript(SCHEMA)
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO article_story (article_id, story_id)
                    SELECT id, story_id FROM articles WHERE story_id IS NOT NULL
                      AND story_id IN (SELECT id FROM stories)
                    """
                )
                try:
                    self.conn.execute("ALTER TABLE articles DROP COLUMN story_id")
                except sqlite3.OperationalError:
                    # SQLite < 3.35: harmless to leave the column behind.
                    log.debug("could not drop legacy articles.story_id")

            # v1 called it source_health; v2 folds it into `sources`.
            if self._columns("source_health"):
                self.conn.executescript(SCHEMA)
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO sources (
                        name, last_ok, last_error_at, last_error,
                        consecutive_failures, total_articles
                    )
                    SELECT name, last_ok, last_error_at, last_error,
                           consecutive_failures, total_articles
                    FROM source_health
                    """
                )
                self.conn.execute("DROP TABLE source_health")

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
                a.publisher, a.source_url, a.author, a.language,
                iso(a.published_at), iso(a.collected_at),
                a.description, a.content, a.content_hash(),
                _dumps(a.source_topics), a.source_weight,
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
                    source_url, author, language, published_at, collected_at,
                    description, content, content_hash, source_topics, source_weight
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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

    # -- caches -------------------------------------------------------------

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
                    id, headline, summary, why_it_matters, key_facts,
                    importance, relevance, score, first_seen, last_updated, written_by
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    headline = excluded.headline, summary = excluded.summary,
                    why_it_matters = excluded.why_it_matters,
                    key_facts = excluded.key_facts,
                    importance = excluded.importance, relevance = excluded.relevance,
                    score = excluded.score, last_updated = excluded.last_updated,
                    written_by = excluded.written_by
                """,
                (
                    story.id, story.headline, story.summary, story.why_it_matters,
                    _dumps(story.key_facts), story.importance, story.relevance,
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
            self.conn.executemany(
                "INSERT INTO digest_stories (digest_id, story_id, rank) VALUES (?,?,?)",
                [(digest.id, sid, rank) for rank, sid in enumerate(digest.story_ids)],
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
        story_ids = [
            r["story_id"]
            for r in self.conn.execute(
                "SELECT story_id FROM digest_stories WHERE digest_id = ? ORDER BY rank",
                (row["id"],),
            )
        ]
        try:
            stats = json.loads(row["stats"])
        except json.JSONDecodeError:
            stats = {}
        return Digest(
            id=row["id"],
            period_start=parse_date(row["period_start"]) or utcnow(),
            period_end=parse_date(row["period_end"]) or utcnow(),
            story_ids=story_ids,
            generated_at=parse_date(row["generated_at"]) or utcnow(),
            article_count=row["article_count"],
            stats=stats,
        )

    def digest_stories(self, digest_id: str) -> list[Story]:
        rows = self.conn.execute(
            """
            SELECT s.* FROM stories s
            JOIN digest_stories d ON d.story_id = s.id
            WHERE d.digest_id = ? ORDER BY d.rank
            """,
            (digest_id,),
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

    def prune(self, retention_days: int, embedding_retention_days: int = 21) -> int:
        """Drop old articles, orphaned stories, and stale cache rows."""
        articles_cutoff = iso(utcnow() - timedelta(days=retention_days))
        vectors_cutoff = iso(utcnow() - timedelta(days=embedding_retention_days))
        with self.conn:
            before = self.conn.total_changes
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
        source_url=row["source_url"] or "",
        source_topics=_loads(row["source_topics"]),
        source_weight=row["source_weight"],
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
