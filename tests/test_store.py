"""Store: emails, articles, stories, digests, pruning, and the schema guard."""

import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from newsdigest import clustering
from newsdigest.embeddings.base import normalize
from newsdigest.llm.base import Enrichment
from newsdigest.models import Digest, RunStats, SourceReport, utcnow
from newsdigest.store import SCHEMA_VERSION, Store

from conftest import make_article, make_email


def _wide_window():
    """A window that certainly contains anything just inserted."""
    return datetime(2000, 1, 1, tzinfo=timezone.utc), utcnow() + timedelta(days=1)

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
        # period_end is exclusive, so the label names the last day covered.
        assert loaded.label == "7–13 September 2026"

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


class TestSchemaGuard:
    """Opening the RSS project's database must fail loudly, not quietly work.

    Nothing in this code would raise on it: `CREATE TABLE IF NOT EXISTS` is
    silent about a table that already exists, the missing columns read as empty,
    and the run would publish a digest built from articles whose newsletter and
    position it had invented. There are no migrations because there is nothing to
    migrate from -- this schema has never been deployed -- so the only correct
    behaviour is to name the problem.
    """

    def _make_rss_database(self, path):
        """The shape the RSS project left behind: articles, no emails."""
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            CREATE TABLE articles (
                id TEXT PRIMARY KEY, canonical TEXT, url TEXT, title TEXT,
                source TEXT, source_url TEXT, feed_position INTEGER,
                collected_at TEXT
            );
            CREATE TABLE stories (id TEXT PRIMARY KEY, headline TEXT);
            PRAGMA user_version = 3;
            """
        )
        conn.execute(
            "INSERT INTO articles (id, canonical, url, title, source, collected_at) "
            "VALUES ('a1', 'https://x/1', 'https://x/1', 'Old', 'BBC', '2026-01-01')"
        )
        conn.commit()
        conn.close()

    def test_an_rss_database_is_refused_by_name(self, tmp_path):
        path = tmp_path / "old.db"
        self._make_rss_database(path)
        with pytest.raises(sqlite3.DatabaseError, match="RSS project"):
            Store(path)

    def test_the_error_says_what_to_do_about_it(self, tmp_path):
        path = tmp_path / "old.db"
        self._make_rss_database(path)
        with pytest.raises(sqlite3.DatabaseError, match="fresh database"):
            Store(path)

    def test_a_fresh_database_is_at_the_current_version(self, tmp_path):
        with Store(tmp_path / "new.db") as store:
            version = store.conn.execute("PRAGMA user_version").fetchone()[0]
            assert version == SCHEMA_VERSION
            assert "emails" in {
                row[0] for row in store.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }

    def test_reopening_a_current_database_is_a_no_op(self, tmp_path):
        path = tmp_path / "new.db"
        with Store(path) as store:
            store.insert_emails([make_email("A", message_id="<1@x>")])
        with Store(path) as store:
            assert len(store.emails_in_window(*_wide_window())) == 1
