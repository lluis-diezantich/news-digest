from newsdigest.llm.base import Enrichment
from newsdigest.models import RunStats, SourceReport
from newsdigest.store import Store

from conftest import make_article


def test_insert_is_idempotent(store):
    articles = [make_article("One"), make_article("Two")]
    assert store.insert_articles(articles) == 2
    assert store.insert_articles(articles) == 0
    assert store.summary()["articles"] == 2


def test_known_ids_and_fingerprints(store):
    article = make_article("Something notable")
    store.insert_articles([article])
    assert store.known_ids([article.id, "missing"]) == {article.id}
    fingerprints = store.recent_fingerprints(72)
    assert (article.id, *fingerprints[0][1:]) == fingerprints[0]
    assert fingerprints[0][2] == "Example"


def test_enrichment_roundtrip(store):
    article = make_article("Rates rise")
    store.insert_articles([article])
    assert len(store.articles_needing_enrichment(10)) == 1

    store.save_enrichment(
        article.id,
        Enrichment(id=article.id, summary="Rates went up.", topics=["economics"],
                   entities=["Central Bank"], importance=0.7, event_label="rate rise"),
        "test",
    )
    assert store.articles_needing_enrichment(10) == []
    loaded = store.recent_articles(48)[0]
    assert loaded.summary == "Rates went up."
    assert loaded.topics == ["economics"]
    assert loaded.importance == 0.7
    assert loaded.enriched_by == "test"


def test_llm_cache(store):
    enrichment = Enrichment(id="x", summary="cached text", topics=["ai"], importance=0.4)
    store.cache_enrichment("hash1", "test", enrichment)
    hit = store.cached_enrichments(["hash1", "hash2"])
    assert set(hit) == {"hash1"}
    assert hit["hash1"].summary == "cached text"
    assert hit["hash1"].topics == ["ai"]


def test_story_upsert_preserves_first_seen(store):
    from newsdigest import clustering

    articles = [make_article("Budget passes", importance=0.6)]
    store.insert_articles(articles)
    story = clustering.build_story(articles)

    assert store.upsert_story(story) is True
    original_first_seen = store.get_story(story.id).first_seen

    story.headline = "Budget passes, second look"
    assert store.upsert_story(story) is False
    reloaded = store.get_story(story.id)
    assert reloaded.headline == "Budget passes, second look"
    assert reloaded.first_seen == original_first_seen


def test_assign_story_and_group_lookup(store):
    articles = [make_article("A"), make_article("B")]
    store.insert_articles(articles)
    store.assign_story([a.id for a in articles], "story1")
    grouped = store.articles_by_story(["story1"])
    assert len(grouped["story1"]) == 2
    assert store.story_article_count("story1") == 2


def test_prune_removes_old_articles(store):
    store.insert_articles([make_article("Old", hours_ago=24 * 60), make_article("New")])
    removed = store.prune(retention_days=30)
    assert removed >= 1
    assert store.summary()["articles"] == 1


def test_record_run_tracks_source_health(store):
    stats = RunStats()
    stats.sources = [
        SourceReport(name="Good", ok=True, new=3),
        SourceReport(name="Bad", ok=False, error="HTTP 500"),
    ]
    store.record_run(stats)
    store.record_run(stats)
    health = {row["name"]: row for row in store.source_health()}
    assert health["Good"]["consecutive_failures"] == 0
    assert health["Good"]["total_articles"] == 6
    assert health["Bad"]["consecutive_failures"] == 2
    assert health["Bad"]["last_error"] == "HTTP 500"


def test_reopening_migrates_cleanly(tmp_path):
    path = tmp_path / "reopen.db"
    with Store(path) as first:
        first.insert_articles([make_article("Persisted")])
    with Store(path) as second:
        assert second.summary()["articles"] == 1
