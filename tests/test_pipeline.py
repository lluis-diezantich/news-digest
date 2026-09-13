"""End-to-end coverage of the stages after fetch, plus fetch resilience."""

import json

from newsdigest import pipeline
from newsdigest.config import Source
from newsdigest.llm import get_provider
from newsdigest.llm.base import Brief, LLMQuotaError
from newsdigest.llm.heuristic import HeuristicProvider
from newsdigest.models import RunStats

from conftest import LabelingProvider, make_article


def test_enrich_cluster_build_produces_a_feed(tmp_path, config, store):
    articles = [
        make_article("Parliament approves the budget", source="BBC"),
        make_article("Budget clears its final vote", source="Guardian"),
        make_article("Chip maker unveils new processor", source="Ars Technica",
                     description="A new processor for AI workloads was announced."),
    ]
    store.insert_articles(articles)
    provider = LabelingProvider()
    stats = RunStats()

    pipeline.enrich_stage(config, store, provider, stats, None)
    assert stats.enriched == 3
    assert store.articles_needing_enrichment(10) == []

    pipeline.cluster_stage(config, store, provider, stats)
    assert stats.stories_total == 2
    assert stats.stories_new == 2

    out = tmp_path / "site"
    feed = pipeline.build_stage(config, store, out, stats)

    assert (out / "feed.json").exists()
    assert (out / "feed.xml").exists()
    assert (out / "index.html").exists()
    assert (out / ".nojekyll").exists()

    written = json.loads((out / "feed.json").read_text())
    assert written["story_count"] == 2
    assert written["article_count"] == 3

    budget = next(s for s in written["stories"] if s["source_count"] == 2)
    assert {a["source"] for a in budget["articles"]} == {"BBC", "Guardian"}
    assert budget["summary"]
    # Interests weight technology above the default, so the chip story leads.
    assert written["stories"][0]["topics"]
    # Scores are sorted descending.
    scores = [s["score"] for s in written["stories"]]
    assert scores == sorted(scores, reverse=True)


def test_enrichment_is_cached_across_runs(tmp_path, config, store):
    class CountingProvider(HeuristicProvider):
        def __init__(self):
            super().__init__()
            self.enriched_count = 0

        def enrich(self, items):
            self.enriched_count += len(items)
            return super().enrich(items)

    store.insert_articles([make_article("A story worth summarizing")])
    provider = CountingProvider()
    pipeline.enrich_stage(config, store, provider, RunStats(), None)
    assert provider.enriched_count == 1

    # Same text, fresh row: the content-hash cache must answer instead.
    duplicate = make_article("A story worth summarizing", source="Other")
    store.insert_articles([duplicate])
    stats = RunStats()
    pipeline.enrich_stage(config, store, provider, stats, None)
    assert provider.enriched_count == 1
    assert stats.enrich_cached == 1


def test_offline_enrichment_ignores_the_quota_cap(config, store):
    """The per-run cap guards an API quota; offline runs have none to guard."""
    config.llm.articles_per_run = 2
    store.insert_articles([make_article(f"Distinct story number {i}") for i in range(6)])
    stats = RunStats()
    pipeline.enrich_stage(config, store, HeuristicProvider(), stats, None)
    assert stats.enriched == 6
    assert store.articles_needing_enrichment(10) == []


def test_api_provider_respects_the_quota_cap(config, store):
    class ApiProvider(HeuristicProvider):
        name = "gemini"

    config.llm.articles_per_run = 2
    store.insert_articles([make_article(f"Distinct story number {i}") for i in range(6)])
    stats = RunStats()
    pipeline.enrich_stage(config, store, ApiProvider(), stats, None)
    assert stats.enriched == 2
    assert len(store.articles_needing_enrichment(10)) == 4


def test_quota_error_leaves_the_rest_for_next_run(config, store):
    class QuotaProvider(HeuristicProvider):
        name = "quota"

        def enrich(self, items):
            raise LLMQuotaError("daily limit")

    store.insert_articles([make_article(f"Story {i}") for i in range(4)])
    stats = RunStats()
    pipeline.enrich_stage(config, store, QuotaProvider(), stats, None)
    assert stats.enriched == 0
    assert len(store.articles_needing_enrichment(10)) == 4


def test_unenriched_articles_still_reach_the_feed(tmp_path, config, store):
    """A quota problem must degrade the feed, not empty it."""
    store.insert_articles([make_article("Something happened today")])
    stats = RunStats()
    pipeline.cluster_stage(config, store, HeuristicProvider(), stats)
    feed = pipeline.build_stage(config, store, tmp_path / "site", stats)
    assert feed["story_count"] == 1
    assert feed["stories"][0]["summary"]


def test_brief_is_written_once_for_a_multi_source_story(config, store):
    class BriefProvider(LabelingProvider):
        name = "brief"

        def __init__(self):
            super().__init__()
            self.briefs = 0

        def write_brief(self, item):
            self.briefs += 1
            return Brief(headline="Merged headline", summary="Merged summary.",
                         why_it_matters="It matters.", topics=["politics"])

    store.insert_articles([
        make_article("Parliament approves the budget", source="BBC"),
        make_article("Budget clears its final vote", source="Guardian"),
    ])
    provider = BriefProvider()
    pipeline.enrich_stage(config, store, provider, RunStats(), None)

    pipeline.cluster_stage(config, store, provider, RunStats())
    assert provider.briefs == 1
    story = store.recent_stories(48)[0]
    assert story.headline == "Merged headline"
    assert story.written_by == "brief"

    # Nothing changed, so the second run must not pay for another brief.
    pipeline.cluster_stage(config, store, provider, RunStats())
    assert provider.briefs == 1
    assert store.recent_stories(48)[0].headline == "Merged headline"


def test_story_keeps_its_identity_when_coverage_grows(config, store):
    store.insert_articles([make_article("Parliament approves the budget", source="BBC")])
    provider = LabelingProvider()
    pipeline.enrich_stage(config, store, provider, RunStats(), None)
    pipeline.cluster_stage(config, store, provider, RunStats())
    first_id = store.recent_stories(48)[0].id

    store.insert_articles([make_article("Budget clears its final vote", source="Guardian")])
    pipeline.enrich_stage(config, store, provider, RunStats(), None)
    stats = RunStats()
    pipeline.cluster_stage(config, store, provider, stats)

    stories = store.recent_stories(48)
    assert len(stories) == 1
    assert stories[0].id == first_id
    assert stats.stories_new == 0
    assert store.story_article_count(first_id) == 2


def test_one_broken_source_does_not_stop_the_run(config, store, monkeypatch):
    good = Source(name="Good", rss="https://good.invalid/rss")
    bad = Source(name="Bad", rss="https://bad.invalid/rss")
    config.sources = [good, bad]

    class FakeAdapter:
        def __init__(self, source):
            self.source = source

        def fetch(self, source):
            if source.name == "Bad":
                raise ConnectionError("host unreachable")
            return [make_article("Working source story", source="Good")]

    monkeypatch.setattr(pipeline, "adapter_for", lambda source, fetcher: FakeAdapter(source))
    monkeypatch.setattr(pipeline, "Fetcher", lambda *a, **k: _NullFetcher())

    stats = RunStats()
    pipeline.fetch_stage(config, store, stats)

    assert stats.articles_new == 1
    assert stats.sources_ok == 1
    assert [s.name for s in stats.sources_failed] == ["Bad"]
    assert "ConnectionError" in stats.sources_failed[0].error


class _NullFetcher:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


def test_full_run_offline(tmp_path, config, monkeypatch):
    """`news-digest run` with a fake source and no network."""
    config.sources = [Source(name="Fake", rss="https://fake.invalid/rss")]

    class FakeAdapter:
        def fetch(self, source):
            return [
                make_article("Storm warning issued for the coast", source="Fake"),
                make_article("Tech firm reports record quarter", source="Fake"),
            ]

    monkeypatch.setattr(pipeline, "adapter_for", lambda source, fetcher: FakeAdapter())
    monkeypatch.setattr(pipeline, "Fetcher", lambda *a, **k: _NullFetcher())

    options = pipeline.Options(db=tmp_path / "run.db", out=tmp_path / "site")
    stats = pipeline.run(config, options)

    assert stats.articles_new == 2
    assert stats.stories_published == 2
    assert (tmp_path / "site" / "feed.json").exists()

    # A second run is a no-op on content but must still succeed.
    again = pipeline.run(config, options)
    assert again.articles_new == 0
    assert again.stories_published == 2
