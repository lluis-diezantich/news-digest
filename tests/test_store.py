"""Store: schema v2 round-trips, the v1 migration, digests and pruning."""

import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np

from newsdigest import clustering
from newsdigest.embeddings.base import normalize
from newsdigest.llm.base import Enrichment
from newsdigest.models import Digest, RunStats, SourceReport, utcnow
from newsdigest.store import SCHEMA_VERSION, Store

from conftest import make_article

CACHE_KEY = "gemini:model:en|ai"


class TestArticles:
    def test_insert_is_idempotent(self, store):
        articles = [make_article("One"), make_article("Two")]
        assert store.insert_articles(articles) == 2
        assert store.insert_articles(articles) == 0
        assert store.summary()["articles"] == 2

    def test_language_and_publisher_survive_a_roundtrip(self, store):
        store.insert_articles([
            make_article("Notícies en català", source="Ara", publisher="Ara", language="ca"),
            make_article("Noticias", source="El País Economía",
                         publisher="El País", language="es"),
        ])
        loaded = {a.source: a for a in store.articles_in_window(
            utcnow() - timedelta(days=1), utcnow() + timedelta(days=1))}
        assert loaded["Ara"].language == "ca"
        assert loaded["El País Economía"].publisher == "El País"
        assert store.language_counts() == {"ca": 1, "es": 1}

    def test_window_is_half_open(self, store):
        start = datetime(2026, 9, 7, tzinfo=timezone.utc)
        end = datetime(2026, 9, 14, tzinfo=timezone.utc)
        store.insert_articles([
            make_article("Before", published=start - timedelta(seconds=1)),
            make_article("First", published=start),
            make_article("Last", published=end - timedelta(seconds=1)),
            make_article("After", published=end),
        ])
        titles = {a.title for a in store.articles_in_window(start, end)}
        assert titles == {"First", "Last"}

    def test_window_filters_by_language(self, store):
        store.insert_articles([
            make_article("English", language="en"),
            make_article("Catalan", language="ca"),
            make_article("French", language="fr"),
        ])
        found = store.articles_in_window(
            utcnow() - timedelta(days=1), utcnow() + timedelta(days=1),
            languages=["en", "ca"],
        )
        assert {a.language for a in found} == {"en", "ca"}

    def test_enrichment_roundtrip(self, store):
        article = make_article("Rates rise")
        store.insert_articles([article])
        store.save_enrichment(article.id, Enrichment(
            id=article.id, summary="Rates went up.", why_it_matters="Loans cost more.",
            key_facts=["Half a point"], topics=["economics"], entities=["Central Bank"],
            importance=0.7, relevance=0.6, content_type="reporting"), "test")
        loaded = store.get_articles([article.id])[0]
        assert loaded.summary == "Rates went up."
        assert loaded.key_facts == ["Half a point"]
        assert loaded.relevance == 0.6
        assert loaded.content_type == "reporting"
        assert loaded.enriched_by == "test"


class TestCaches:
    def test_llm_cache_is_keyed_by_provider_and_language(self, store):
        enrichment = Enrichment(id="x", summary="cached", topics=["ai"], importance=0.4)
        store.cache_enrichment("h1", CACHE_KEY, enrichment)
        assert set(store.cached_enrichments(["h1"], CACHE_KEY)) == {"h1"}
        # A different output language is a different cache namespace.
        assert store.cached_enrichments(["h1"], "gemini:model:ca|ai") == {}

    def test_vector_cache_roundtrip(self, store):
        vector = normalize(np.array([1.0, 2.0, 3.0]))
        store.cache_vectors({"h1": vector}, "gemini:emb:256")
        got = store.cached_vectors(["h1", "h2"], "gemini:emb:256")
        assert set(got) == {"h1"}
        assert np.allclose(got["h1"], vector)

    def test_vector_cache_is_keyed_by_model(self, store):
        store.cache_vectors({"h1": normalize(np.array([1.0, 0.0]))}, "gemini:emb:256")
        assert store.cached_vectors(["h1"], "gemini:emb:768") == {}


class TestStories:
    def test_replace_story_writes_membership_topics_and_entities(self, store):
        articles = [
            make_article("EU sanctions", source="BBC", publisher="BBC", language="en",
                         importance=0.7, relevance=0.6, topics=["world"],
                         entities=["European Union"]),
            make_article("UE sanciones", source="El País", publisher="El País",
                         language="es", importance=0.6, relevance=0.7, topics=["world"],
                         entities=["European Union"]),
        ]
        store.insert_articles(articles)
        story = clustering.build_story(articles)
        assert store.replace_story(story, [a.id for a in articles]) is True

        loaded = store.get_story(story.id)
        assert loaded.topics == ["world"]
        assert loaded.entities == ["European Union"]
        assert len(loaded.article_ids) == 2
        grouped = store.articles_by_story([story.id])
        assert {a.language for a in grouped[story.id]} == {"en", "es"}

    def test_replace_story_is_idempotent_and_keeps_first_seen(self, store):
        articles = [make_article("Budget passes", importance=0.6)]
        store.insert_articles(articles)
        story = clustering.build_story(articles)
        store.replace_story(story, [a.id for a in articles])
        first_seen = store.get_story(story.id).first_seen

        story.headline = "Budget passes, revised"
        assert store.replace_story(story, [a.id for a in articles]) is False
        reloaded = store.get_story(story.id)
        assert reloaded.headline == "Budget passes, revised"
        assert reloaded.first_seen == first_seen

    def test_membership_is_replaced_not_appended(self, store):
        articles = [make_article("A"), make_article("B")]
        store.insert_articles(articles)
        story = clustering.build_story(articles)
        store.replace_story(story, [a.id for a in articles])
        store.replace_story(story, [articles[0].id])
        assert len(store.articles_by_story([story.id])[story.id]) == 1

    def test_an_article_belongs_to_one_story_only(self, store):
        article = make_article("Contested")
        store.insert_articles([article])
        first = clustering.build_story([article])
        second = clustering.build_story([article, make_article("Other")])
        store.replace_story(first, [article.id])
        store.replace_story(second, [article.id])
        rows = store.conn.execute(
            "SELECT COUNT(*) FROM article_story WHERE article_id = ?", (article.id,)
        ).fetchone()[0]
        assert rows == 1


class TestDigests:
    def _digest(self, store, week="2026-W37", stories=2):
        articles = [make_article(f"Story {i} of {week}", importance=0.5) for i in range(stories)]
        store.insert_articles(articles)
        ids = []
        for article in articles:
            story = clustering.build_story([article])
            store.replace_story(story, [article.id])
            ids.append(story.id)
        digest = Digest(
            id=week,
            period_start=datetime(2026, 9, 7, tzinfo=timezone.utc),
            period_end=datetime(2026, 9, 14, tzinfo=timezone.utc),
            story_ids=ids, article_count=len(articles), stats={"clusters": 5},
        )
        store.save_digest(digest)
        return digest

    def test_save_and_read_back(self, store):
        saved = self._digest(store)
        loaded = store.get_digest("2026-W37")
        assert loaded.story_ids == saved.story_ids
        assert loaded.stats == {"clusters": 5}
        assert loaded.label == "7–14 September 2026"

    def test_ranking_order_is_preserved(self, store):
        saved = self._digest(store, stories=4)
        assert store.get_digest("2026-W37").story_ids == saved.story_ids
        assert [s.id for s in store.digest_stories("2026-W37")] == saved.story_ids

    def test_latest_and_list(self, store):
        self._digest(store, week="2026-W36")
        store.save_digest(Digest(
            id="2026-W37",
            period_start=datetime(2026, 9, 14, tzinfo=timezone.utc),
            period_end=datetime(2026, 9, 21, tzinfo=timezone.utc)))
        assert store.latest_digest().id == "2026-W37"
        assert [d.id for d in store.list_digests()] == ["2026-W37", "2026-W36"]
        assert [d.id for d in store.list_digests(limit=1)] == ["2026-W37"]

    def test_resaving_a_week_replaces_its_stories(self, store):
        self._digest(store, stories=3)
        store.save_digest(Digest(id="2026-W37",
                                 period_start=datetime(2026, 9, 7, tzinfo=timezone.utc),
                                 period_end=datetime(2026, 9, 14, tzinfo=timezone.utc),
                                 story_ids=[]))
        assert store.get_digest("2026-W37").story_ids == []


class TestBookkeeping:
    def test_record_run_tracks_source_health(self, store):
        stats = RunStats()
        stats.sources = [SourceReport(name="Good", ok=True, new=3),
                         SourceReport(name="Bad", ok=False, error="HTTP 500")]
        store.record_run(stats)
        store.record_run(stats)
        health = {row["name"]: row for row in store.source_health()}
        assert health["Good"]["consecutive_failures"] == 0
        assert health["Good"]["total_articles"] == 6
        assert health["Bad"]["consecutive_failures"] == 2

    def test_register_sources_records_language_and_publisher(self, store, config):
        store.register_sources(config.sources)
        row = {r["name"]: r for r in store.source_health()}["Example"]
        assert row["languages"] == '["en"]'

    def test_run_kind_is_recorded(self, store):
        store.record_run(RunStats(), kind="weekly")
        assert store.summary()["last_run_kind"] == "weekly"


class TestPrune:
    def test_old_articles_go(self, store):
        store.insert_articles([make_article("Old", hours_ago=24 * 90),
                               make_article("New")])
        assert store.prune(45) >= 1
        assert store.summary()["articles"] == 1

    def test_embeddings_go_sooner_than_articles(self, store):
        store.insert_articles([make_article("Recent")])
        store.cache_vectors({"h1": normalize(np.array([1.0, 0.0]))}, "k")
        store.conn.execute(
            "UPDATE embedding_cache SET created_at = ?",
            ((utcnow() - timedelta(days=30)).isoformat(),),
        )
        store.conn.commit()
        store.prune(45, embedding_retention_days=21)
        assert store.summary()["cached_vectors"] == 0
        assert store.summary()["articles"] == 1

    def test_stories_in_a_published_digest_survive(self, store):
        """Otherwise the archive would develop holes as articles age out."""
        article = make_article("Archived story", hours_ago=24 * 90, importance=0.5)
        store.insert_articles([article])
        story = clustering.build_story([article])
        store.replace_story(story, [article.id])
        store.save_digest(Digest(id="2026-W20",
                                 period_start=utcnow() - timedelta(days=95),
                                 period_end=utcnow() - timedelta(days=88),
                                 story_ids=[story.id]))
        store.prune(45)
        assert store.summary()["articles"] == 0
        assert store.get_story(story.id) is not None


class TestMigration:
    def _make_v1(self, path):
        """Build a database in the v1 shape the first release shipped."""
        conn = sqlite3.connect(path)
        conn.executescript("""
            CREATE TABLE articles (
                id TEXT PRIMARY KEY, canonical TEXT NOT NULL UNIQUE, url TEXT NOT NULL,
                title TEXT NOT NULL, title_key TEXT NOT NULL, source TEXT NOT NULL,
                author TEXT, published_at TEXT, fetched_at TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                source_topics TEXT NOT NULL DEFAULT '[]',
                source_weight REAL NOT NULL DEFAULT 1.0,
                summary TEXT, why_it_matters TEXT, topics TEXT NOT NULL DEFAULT '[]',
                entities TEXT NOT NULL DEFAULT '[]', importance REAL,
                event_label TEXT, enriched_by TEXT, enriched_at TEXT, story_id TEXT);
            CREATE TABLE stories (
                id TEXT PRIMARY KEY, headline TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '', why_it_matters TEXT NOT NULL DEFAULT '',
                topics TEXT NOT NULL DEFAULT '[]', entities TEXT NOT NULL DEFAULT '[]',
                keywords TEXT NOT NULL DEFAULT '[]', importance REAL NOT NULL DEFAULT 0,
                score REAL NOT NULL DEFAULT 0, first_seen TEXT NOT NULL,
                last_updated TEXT NOT NULL, written_by TEXT);
            CREATE TABLE llm_cache (
                content_hash TEXT PRIMARY KEY, provider TEXT NOT NULL,
                payload TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL, stats TEXT NOT NULL);
            CREATE TABLE source_health (
                name TEXT PRIMARY KEY, last_ok TEXT, last_error_at TEXT,
                last_error TEXT, consecutive_failures INTEGER NOT NULL DEFAULT 0,
                total_articles INTEGER NOT NULL DEFAULT 0);
            PRAGMA user_version = 1;
        """)
        now = utcnow().isoformat()
        conn.execute(
            "INSERT INTO articles (id, canonical, url, title, title_key, source, "
            "fetched_at, description, summary, importance, story_id) "
            "VALUES ('a1','https://x.test/1','https://x.test/1','Old article','old article',"
            f"'Wire','{now}','An excerpt.','A summary.',0.6,'s1')")
        conn.execute(
            "INSERT INTO stories (id, headline, first_seen, last_updated) "
            f"VALUES ('s1','Old story','{now}','{now}')")
        conn.execute(
            "INSERT INTO llm_cache (content_hash, provider, payload, created_at) "
            f"VALUES ('h1','gemini','{{}}','{now}')")
        conn.commit()
        conn.close()

    def test_v1_database_migrates_without_losing_data(self, tmp_path):
        path = tmp_path / "v1.db"
        self._make_v1(path)

        with Store(path) as store:
            assert store.conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
            # The renamed column carried its data across.
            article = store.get_articles(["a1"])[0]
            assert article.title == "Old article"
            assert article.summary == "A summary."
            assert article.collected_at is not None
            assert article.publisher == "Wire"      # backfilled from source
            assert article.language is None         # v1 had no language

            # Story membership moved from articles.story_id to the join table.
            assert [a.id for a in store.articles_by_story(["s1"])["s1"]] == ["a1"]
            # The new tables exist and are usable.
            assert store.list_digests() == []
            store.cache_vectors({"h": normalize(np.array([1.0, 0.0]))}, "k")
            assert store.summary()["cached_vectors"] == 1

    def test_v1_tables_gain_the_columns_v2_added(self, tmp_path):
        """IF NOT EXISTS does nothing for an existing table, so ALTER must."""
        path = tmp_path / "v1cols.db"
        self._make_v1(path)
        with Store(path) as store:
            assert "kind" in store._columns("runs")
            assert {"relevance", "key_facts"} <= store._columns("stories")
            # These queries touch the new columns and must not raise.
            store.record_run(RunStats(), kind="weekly")
            assert store.summary()["last_run_kind"] == "weekly"

    def test_v1_source_health_is_folded_into_sources(self, tmp_path):
        path = tmp_path / "v1health.db"
        self._make_v1(path)
        conn = sqlite3.connect(path)
        conn.execute("INSERT INTO source_health (name, total_articles, "
                     "consecutive_failures) VALUES ('Wire', 42, 3)")
        conn.commit()
        conn.close()
        with Store(path) as store:
            health = {r["name"]: r for r in store.source_health()}
            assert health["Wire"]["total_articles"] == 42
            assert health["Wire"]["consecutive_failures"] == 3

    def test_migration_is_idempotent(self, tmp_path):
        path = tmp_path / "v1.db"
        self._make_v1(path)
        with Store(path):
            pass
        with Store(path) as store:
            assert store.get_articles(["a1"])[0].title == "Old article"

    def test_fresh_database_is_at_the_current_version(self, tmp_path):
        with Store(tmp_path / "new.db") as store:
            assert store.conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
