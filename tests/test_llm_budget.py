"""Brief-first ordering and the request-count budget.

On a free tier the scarce resource is the REQUEST COUNT, not tokens. Observed
2026-09-15: 13 enrichment requests plus retries exhausted a whole day, and all
six briefs -- the only LLM output a reader sees on the page -- fell back to raw
article text. So the brief is paid for first and enrichment takes the remainder.
"""

import pytest

from conftest import make_article

from newsdigest import digest as digest_module
from newsdigest.config import (
    Config, DigestSettings, EmbeddingSettings, LLMSettings, Preferences,
    Settings, Source, StorageSettings, load_llm_settings,
)
from newsdigest.llm.base import Brief, Context, LLMProvider, LLMQuotaError
from newsdigest.models import RunStats


class Recorder(LLMProvider):
    """Counts requests and can run out of quota after a set number."""

    name = "recorder"
    model = "recorder-1"

    def __init__(self, budget=99):
        super().__init__()
        self.budget = budget
        self.order: list[str] = []

    def _spend(self, kind):
        self.order.append(kind)
        self.calls += 1
        if self.calls > self.budget:
            raise LLMQuotaError("daily quota exhausted")

    def enrich(self, items, context):
        self._spend("enrich")
        return []

    def write_brief(self, item, context):
        self._spend("brief")
        return Brief(headline="A written headline", summary="A written summary.",
                     why_it_matters="It matters.", key_facts=[], topics=["world"],
                     importance=0.8, relevance=0.5)

    def same_event(self, pairs, context):
        self._spend("adjudicate")
        return {}


def config_for(**llm):
    settings = dict(provider="recorder", write_story_briefs=True,
                    resolve_clusters=False, articles_per_run=100, batch_size=8)
    settings.update(llm)
    return Config(
        sources=[Source(name="S", rss="https://x.invalid/rss", languages=["en"])],
        settings=Settings(supported_languages=["en"], output_language="en"),
        preferences=Preferences(),
        digest=DigestSettings(max_stories=2, min_articles=1),
        storage=StorageSettings(),
        llm=LLMSettings(**settings),
        embeddings=EmbeddingSettings(provider="none"),
    )


def articles(n):
    return [make_article(f"Story number {i} about the world", source=f"Outlet {i}",
                         publisher=f"Outlet {i}") for i in range(n)]


class TestBriefFirst:
    def _run(self, provider, store, arts):
        cfg = config_for()
        ctx = Context(output_language="en", interests=[], excluded_topics=[])
        store.insert_articles(arts)
        stats = RunStats()
        from newsdigest.embeddings.none import NullEmbeddingProvider
        return digest_module.build_digest(
            cfg, store, provider, NullEmbeddingProvider(), ctx, stats,
            window=_wide(), persist=False,
        ), stats

    def test_briefs_are_requested_before_enrichment(self, store):
        p = Recorder()
        self._run(p, store, articles(4))
        assert "brief" in p.order and "enrich" in p.order
        assert p.order.index("brief") < p.order.index("enrich"), p.order

    def test_a_brief_survives_a_quota_that_dies_during_enrichment(self, store):
        """The regression this reordering exists for."""
        p = Recorder(budget=2)          # enough for briefs only
        result, _ = self._run(p, store, articles(4))
        written = [s for s, _ in result.stories if s.written_by]
        assert written, "no story got a written brief"
        assert all(s.headline == "A written headline" for s in written)

    def test_quota_during_briefs_still_publishes(self, store):
        p = Recorder(budget=0)
        result, _ = self._run(p, store, articles(3))
        assert result.stories, "nothing published"
        assert all(s.written_by is None for s, _ in result.stories)


def _wide():
    from datetime import datetime, timedelta, timezone
    from newsdigest.models import utcnow
    return datetime(2000, 1, 1, tzinfo=timezone.utc), utcnow() + timedelta(days=1)


class TestBudgetKnobs:
    """Both had no env wiring, so they were unreachable from CI."""

    def test_cluster_checks_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("LLM_MAX_CLUSTER_CHECKS", "0")
        assert load_llm_settings().max_cluster_checks == 0

    def test_cluster_checks_default_is_unchanged(self):
        assert load_llm_settings().max_cluster_checks == 40

    def test_resolve_clusters_can_be_turned_off(self, monkeypatch):
        monkeypatch.setenv("LLM_RESOLVE_CLUSTERS", "false")
        assert load_llm_settings().resolve_clusters is False

    @pytest.mark.parametrize("raw,expected",
                             [("1", True), ("yes", True), ("ON", True),
                              ("0", False), ("no", False), ("Off", False)])
    def test_boolean_spellings(self, monkeypatch, raw, expected):
        monkeypatch.setenv("LLM_RESOLVE_CLUSTERS", raw)
        assert load_llm_settings().resolve_clusters is expected

    def test_a_typo_keeps_the_default_and_warns(self, monkeypatch, caplog):
        """Silently reading as False would disable adjudication invisibly."""
        import logging
        monkeypatch.setenv("LLM_RESOLVE_CLUSTERS", "ture")
        with caplog.at_level(logging.WARNING):
            assert load_llm_settings().resolve_clusters is True
        assert "not a boolean" in caplog.text

    def test_batch_size_is_configurable(self, monkeypatch):
        monkeypatch.setenv("LLM_BATCH_SIZE", "16")
        assert load_llm_settings().batch_size == 16
