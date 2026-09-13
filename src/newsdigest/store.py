"""SQLite persistence.

The database is the pipeline's memory: it is what makes deduplication work
across runs and lets a story accumulate coverage over days. In GitHub Actions
the file is committed back to the repo after each run, so keep it small --
`prune()` drops articles past the retention window.

Schema changes bump SCHEMA_VERSION and add a migration step.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

from .llm.base import Enrichment
from .models import Article, RunStats, Story, iso, parse_date, utcnow
from .text import title_key

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id            TEXT PRIMARY KEY,
    canonical     TEXT NOT NULL UNIQUE,
    url           TEXT NOT NULL,
    title         TEXT NOT NULL,
    title_key     TEXT NOT NULL,
    source        TEXT NOT NULL,
    author        TEXT,
    published_at  TEXT,
    fetched_at    TEXT NOT NULL,
    description   TEXT NOT NULL DEFAULT '',
    source_topics TEXT NOT NULL DEFAULT '[]',
    source_weight REAL NOT NULL DEFAULT 1.0,
    summary        TEXT,
    why_it_matters TEXT,
    topics         TEXT NOT NULL DEFAULT '[]',
    entities       TEXT NOT NULL DEFAULT '[]',
    importance     REAL,
    event_label    TEXT,
    enriched_by    TEXT,
    enriched_at    TEXT,
    story_id       TEXT
);
CREATE INDEX IF NOT EXISTS idx_articles_title_key ON articles(title_key);
CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_story ON articles(story_id);
CREATE INDEX IF NOT EXISTS idx_articles_unenriched ON articles(enriched_at)
    WHERE enriched_at IS NULL;

CREATE TABLE IF NOT EXISTS stories (
    id             TEXT PRIMARY KEY,
    headline       TEXT NOT NULL,
    summary        TEXT NOT NULL DEFAULT '',
    why_it_matters TEXT NOT NULL DEFAULT '',
    topics         TEXT NOT NULL DEFAULT '[]',
    entities       TEXT NOT NULL DEFAULT '[]',
    keywords       TEXT NOT NULL DEFAULT '[]',
    importance     REAL NOT NULL DEFAULT 0,
    score          REAL NOT NULL DEFAULT 0,
    first_seen     TEXT NOT NULL,
    last_updated   TEXT NOT NULL,
    written_by     TEXT
);
CREATE INDEX IF NOT EXISTS idx_stories_updated ON stories(last_updated DESC);

-- Enrichment results keyed by article content hash. A re-run over unchanged
-- text costs nothing; this is the main defence of the free-tier quota.
CREATE TABLE IF NOT EXISTS llm_cache (
    content_hash TEXT PRIMARY KEY,
    provider     TEXT NOT NULL,
    payload      TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    stats       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_health (
    name                 TEXT PRIMARY KEY,
    last_ok              TEXT,
    last_error_at        TEXT,
    last_error           TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    total_articles       INTEGER NOT NULL DEFAULT 0
);
"""


def _dumps(value: Any) -> str:
    return json.dumps(value or [], ensure_ascii=False)


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

    def _migrate(self) -> None:
        with closing(self.conn.cursor()) as cur:
            version = cur.execute("PRAGMA user_version").fetchone()[0]
        self.conn.executescript(SCHEMA)
        if version < SCHEMA_VERSION:
            # Future migrations: `if version < 2: self.conn.execute("ALTER ...")`
            self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.conn.commit()

    def close(self) -> None:
        # Collapse the WAL so the committed .db file is a single self-contained
        # artifact -- otherwise Actions would need to commit -wal/-shm too.
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- articles -----------------------------------------------------------

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
        """(id, title_key, source) for the near-duplicate check."""
        cutoff = iso(utcnow() - timedelta(hours=hours))
        rows = self.conn.execute(
            "SELECT id, title_key, source FROM articles "
            "WHERE COALESCE(published_at, fetched_at) >= ?",
            (cutoff,),
        ).fetchall()
        return [(r["id"], r["title_key"], r["source"]) for r in rows]

    def insert_articles(self, articles: Iterable[Article]) -> int:
        """Insert new articles. Existing canonical URLs are left untouched."""
        rows = [
            (
                a.id,
                a.canonical,
                a.url,
                a.title,
                title_key(a.title),
                a.source,
                a.author,
                iso(a.published_at),
                iso(a.fetched_at),
                a.description,
                _dumps(a.source_topics),
                a.source_weight,
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
                    id, canonical, url, title, title_key, source, author,
                    published_at, fetched_at, description, source_topics, source_weight
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO NOTHING
                """,
                rows,
            )
            return self.conn.total_changes - before

    def articles_needing_enrichment(self, limit: int) -> list[Article]:
        rows = self.conn.execute(
            """
            SELECT * FROM articles
            WHERE enriched_at IS NULL
            ORDER BY source_weight DESC, COALESCE(published_at, fetched_at) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [_row_to_article(r) for r in rows]

    def save_enrichment(self, article_id: str, enrichment: Enrichment, provider: str) -> None:
        with self.conn:
            self.conn.execute(
                """
                UPDATE articles SET
                    summary = ?, why_it_matters = ?, topics = ?, entities = ?,
                    importance = ?, event_label = ?, enriched_by = ?, enriched_at = ?
                WHERE id = ?
                """,
                (
                    enrichment.summary,
                    enrichment.why_it_matters,
                    _dumps(enrichment.topics),
                    _dumps(enrichment.entities),
                    enrichment.importance,
                    enrichment.event_label,
                    provider,
                    iso(utcnow()),
                    article_id,
                ),
            )

    def recent_articles(self, hours: float, *, enriched_only: bool = True) -> list[Article]:
        cutoff = iso(utcnow() - timedelta(hours=hours))
        clause = "AND enriched_at IS NOT NULL" if enriched_only else ""
        rows = self.conn.execute(
            f"""
            SELECT * FROM articles
            WHERE COALESCE(published_at, fetched_at) >= ? {clause}
            ORDER BY COALESCE(published_at, fetched_at) DESC
            """,
            (cutoff,),
        ).fetchall()
        return [_row_to_article(r) for r in rows]

    def assign_story(self, article_ids: Sequence[str], story_id: str) -> None:
        with self.conn:
            self.conn.executemany(
                "UPDATE articles SET story_id = ? WHERE id = ?",
                [(story_id, aid) for aid in article_ids],
            )

    # -- stories ------------------------------------------------------------

    def upsert_story(self, story: Story) -> bool:
        """Returns True if this story is new."""
        existing = self.conn.execute(
            "SELECT first_seen FROM stories WHERE id = ?", (story.id,)
        ).fetchone()
        first_seen = parse_date(existing["first_seen"]) if existing else story.first_seen
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO stories (
                    id, headline, summary, why_it_matters, topics, entities,
                    keywords, importance, score, first_seen, last_updated, written_by
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    headline = excluded.headline,
                    summary = excluded.summary,
                    why_it_matters = excluded.why_it_matters,
                    topics = excluded.topics,
                    entities = excluded.entities,
                    keywords = excluded.keywords,
                    importance = excluded.importance,
                    score = excluded.score,
                    last_updated = excluded.last_updated,
                    written_by = excluded.written_by
                """,
                (
                    story.id,
                    story.headline,
                    story.summary,
                    story.why_it_matters,
                    _dumps(story.topics),
                    _dumps(story.entities),
                    _dumps(story.keywords),
                    story.importance,
                    story.score,
                    iso(first_seen or story.first_seen),
                    iso(story.last_updated),
                    story.written_by,
                ),
            )
        return existing is None

    def get_story(self, story_id: str) -> Story | None:
        row = self.conn.execute(
            "SELECT * FROM stories WHERE id = ?", (story_id,)
        ).fetchone()
        return _row_to_story(row) if row else None

    def recent_stories(self, hours: float) -> list[Story]:
        cutoff = iso(utcnow() - timedelta(hours=hours))
        rows = self.conn.execute(
            "SELECT * FROM stories WHERE last_updated >= ? ORDER BY score DESC",
            (cutoff,),
        ).fetchall()
        return [_row_to_story(r) for r in rows]

    def story_article_count(self, story_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM articles WHERE story_id = ?", (story_id,)
        ).fetchone()
        return int(row[0] or 0)

    def articles_by_story(self, story_ids: Sequence[str]) -> dict[str, list[Article]]:
        """Every article belonging to the given stories, keyed by story id."""
        grouped: dict[str, list[Article]] = {sid: [] for sid in story_ids}
        for chunk in _chunks(list(story_ids), 400):
            marks = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT * FROM articles WHERE story_id IN ({marks}) "
                "ORDER BY COALESCE(published_at, fetched_at) DESC",
                chunk,
            ).fetchall()
            for row in rows:
                grouped[row["story_id"]].append(_row_to_article(row))
        return grouped

    # -- LLM cache ----------------------------------------------------------

    def cached_enrichments(self, hashes: Sequence[str]) -> dict[str, Enrichment]:
        out: dict[str, Enrichment] = {}
        for chunk in _chunks(list(hashes), 500):
            marks = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT content_hash, payload FROM llm_cache WHERE content_hash IN ({marks})",
                chunk,
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
                    topics=data.get("topics") or [],
                    entities=data.get("entities") or [],
                    importance=data.get("importance", 0.5),
                    event_label=data.get("event_label", ""),
                )
        return out

    def cache_enrichment(self, content_hash: str, provider: str, enrichment: Enrichment) -> None:
        payload = json.dumps(
            {
                "summary": enrichment.summary,
                "why_it_matters": enrichment.why_it_matters,
                "topics": enrichment.topics,
                "entities": enrichment.entities,
                "importance": enrichment.importance,
                "event_label": enrichment.event_label,
            },
            ensure_ascii=False,
        )
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO llm_cache (content_hash, provider, payload, created_at)
                VALUES (?,?,?,?)
                ON CONFLICT(content_hash) DO UPDATE SET
                    provider = excluded.provider,
                    payload = excluded.payload,
                    created_at = excluded.created_at
                """,
                (content_hash, provider, payload, iso(utcnow())),
            )

    # -- bookkeeping --------------------------------------------------------

    def record_run(self, stats: RunStats) -> None:
        payload = stats.to_json()
        with self.conn:
            self.conn.execute(
                "INSERT INTO runs (started_at, finished_at, stats) VALUES (?,?,?)",
                (payload["started_at"], payload["finished_at"], json.dumps(payload)),
            )
            for report in stats.sources:
                if report.ok:
                    self.conn.execute(
                        """
                        INSERT INTO source_health (name, last_ok, consecutive_failures, total_articles)
                        VALUES (?,?,0,?)
                        ON CONFLICT(name) DO UPDATE SET
                            last_ok = excluded.last_ok,
                            consecutive_failures = 0,
                            total_articles = source_health.total_articles + excluded.total_articles
                        """,
                        (report.name, iso(utcnow()), report.new),
                    )
                else:
                    self.conn.execute(
                        """
                        INSERT INTO source_health (name, last_error_at, last_error, consecutive_failures)
                        VALUES (?,?,?,1)
                        ON CONFLICT(name) DO UPDATE SET
                            last_error_at = excluded.last_error_at,
                            last_error = excluded.last_error,
                            consecutive_failures = source_health.consecutive_failures + 1
                        """,
                        (report.name, iso(utcnow()), report.error),
                    )

    def source_health(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM source_health ORDER BY consecutive_failures DESC, name"
        ).fetchall()
        return [dict(r) for r in rows]

    def prune(self, retention_days: int) -> int:
        """Drop old articles, then stories and cache rows left with nothing."""
        cutoff = iso(utcnow() - timedelta(days=retention_days))
        with self.conn:
            before = self.conn.total_changes
            self.conn.execute(
                "DELETE FROM articles WHERE COALESCE(published_at, fetched_at) < ?",
                (cutoff,),
            )
            self.conn.execute(
                "DELETE FROM stories WHERE id NOT IN "
                "(SELECT DISTINCT story_id FROM articles WHERE story_id IS NOT NULL)"
            )
            self.conn.execute("DELETE FROM llm_cache WHERE created_at < ?", (cutoff,))
            self.conn.execute("DELETE FROM runs WHERE started_at < ?", (cutoff,))
            removed = self.conn.total_changes - before
        self.conn.execute("VACUUM")
        return removed

    def summary(self) -> dict[str, Any]:
        def scalar(sql: str) -> int:
            return int(self.conn.execute(sql).fetchone()[0] or 0)

        last_run = self.conn.execute(
            "SELECT stats FROM runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return {
            "db": str(self.path),
            "size_kb": round(self.path.stat().st_size / 1024, 1) if self.path.exists() else 0,
            "articles": scalar("SELECT COUNT(*) FROM articles"),
            "unenriched": scalar("SELECT COUNT(*) FROM articles WHERE enriched_at IS NULL"),
            "stories": scalar("SELECT COUNT(*) FROM stories"),
            "cached_enrichments": scalar("SELECT COUNT(*) FROM llm_cache"),
            "runs": scalar("SELECT COUNT(*) FROM runs"),
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
        source_topics=_loads(row["source_topics"]),
        source_weight=row["source_weight"],
        fetched_at=parse_date(row["fetched_at"]) or utcnow(),
    )
    article.id = row["id"]
    article.canonical = row["canonical"]
    article.summary = row["summary"]
    article.why_it_matters = row["why_it_matters"]
    article.topics = _loads(row["topics"])
    article.entities = _loads(row["entities"])
    article.importance = row["importance"]
    article.event_label = row["event_label"]
    article.enriched_by = row["enriched_by"]
    article.enriched_at = parse_date(row["enriched_at"])
    article.story_id = row["story_id"]
    return article


def _row_to_story(row: sqlite3.Row) -> Story:
    return Story(
        id=row["id"],
        headline=row["headline"],
        summary=row["summary"] or "",
        why_it_matters=row["why_it_matters"] or "",
        topics=_loads(row["topics"]),
        entities=_loads(row["entities"]),
        keywords=_loads(row["keywords"]),
        importance=row["importance"] or 0.0,
        score=row["score"] or 0.0,
        first_seen=parse_date(row["first_seen"]) or utcnow(),
        last_updated=parse_date(row["last_updated"]) or utcnow(),
        written_by=row["written_by"],
    )


def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]
