"""Weekly pipeline: windowing, pre-ranking, and the end-to-end offline path."""

from datetime import datetime, timedelta, timezone

import pytest

from newsdigest import digest as digest_module
from newsdigest.digest import (
    _interleave,
    build_digest,
    prerank_score,
    select_candidates,
    weekly_window,
)
from newsdigest.embeddings.none import NullEmbeddingProvider
from newsdigest.llm.heuristic import HeuristicProvider
from newsdigest.models import RunStats

from conftest import StubEmbedder, make_article

MONDAY = datetime(2026, 9, 7, tzinfo=timezone.utc)


class TestEnrichmentBudgetSharing:
    """`_interleave`: the per-run enrichment cap belongs to every candidate.

    Added 2026-09-18. Flat concatenation gave it all to the first cluster -- on
    the 2026-W38 run all 40 enriched articles landed in one 52-article cluster
    and seven of the eight published stories had none, which is also how
    `importance` came to mean "ten outlets" rather than "important".
    """

    def test_every_cluster_is_reached_before_any_is_finished(self):
        big = [make_article(f"Big {n}", source="A") for n in range(50)]
        small = [make_article("Small one", source="B")]
        other = [make_article("Other one", source="C")]
        order = _interleave([big, small, other])
        assert [a.title for a in order[:3]] == ["Big 0", "Small one", "Other one"]
        assert len(order) == 52

    def test_a_budget_smaller_than_the_biggest_cluster_still_spans_them(self):
        clusters = [[make_article(f"C{c} #{n}", source=f"S{c}") for n in range(20)]
                    for c in range(4)]
        first_ten = _interleave(clusters)[:10]
        assert len({a.source for a in first_ten}) == 4

    def test_no_article_is_lost_or_duplicated(self):
        clusters = [[make_article(f"C{c} #{n}", source=f"S{c}") for n in range(c + 1)]
                    for c in range(5)]
        flat = _interleave(clusters)
        assert len(flat) == 15
        assert len({a.id for a in flat}) == 15

    def test_no_clusters_is_not_an_error(self):
        assert _interleave([]) == []


class TestWeeklyWindow:
    @pytest.mark.parametrize("run_day", [0, 1, 3, 6])
    def test_always_covers_a_whole_finished_week(self, run_day):
        now = datetime(2026, 9, 14, 6, tzinfo=timezone.utc) + timedelta(days=run_day)
        start, end = weekly_window(now)
        assert (end - start).days == 7
        assert start.weekday() == 0 and end.weekday() == 0   # Monday to Monday
        assert end <= now

    def test_monday_run_covers_the_previous_week(self):
        start, end = weekly_window(datetime(2026, 9, 14, 6, tzinfo=timezone.utc))
        assert start == MONDAY
        assert end == datetime(2026, 9, 14, tzinfo=timezone.utc)

    def test_week_ends_on_shifts_the_boundary(self):
        now = datetime(2026, 9, 14, 6, tzinfo=timezone.utc)      # a Monday
        _, saturday_end = weekly_window(now, week_ends_on=5)
        assert saturday_end.weekday() == 6                       # ends Sunday 00:00


class TestSelection:
    def test_unusable_articles_are_dropped(self):
        good = make_article("A perfectly usable headline")
        short = make_article("Tiny")
        blank = make_article("Blank one")
        blank.title = ""
        assert select_candidates([good, short, blank]) == [good]


class TestPrerank:
    def test_more_publishers_outranks_more_articles(self, config):
        many_outlets = [
            make_article(f"Story {i}", source=f"Outlet {i}", publisher=f"Outlet {i}")
            for i in range(3)
        ]
        one_outlet = [
            make_article(f"Story {i}", source="Solo", publisher="Solo") for i in range(5)
        ]
        assert prerank_score(many_outlets, config) > prerank_score(one_outlet, config)

    def test_recency_matters(self, config):
        fresh = [make_article("Fresh", hours_ago=2)]
        stale = [make_article("Stale", hours_ago=150)]
        assert prerank_score(fresh, config) > prerank_score(stale, config)

    def test_feed_topic_hints_are_used_before_any_llm_output(self, config):
        tech = [make_article("Something happened", source="A")]
        tech[0].source_topics = ["technology"]
        plain = [make_article("Something happened", source="B")]
        assert prerank_score(tech, config) > prerank_score(plain, config)

    def test_excluded_topic_hint_is_penalised(self, config):
        sport = [make_article("The cup final ended in a draw", source="A")]
        sport[0].source_topics = ["sports"]
        other = [make_article("The council approved the bridge", source="B")]
        assert prerank_score(sport, config) < prerank_score(other, config)

    def test_preferred_publisher_lifts_the_cluster(self, config):
        preferred = [make_article("Story", source="BBC World", publisher="BBC World")]
        plain = [make_article("Story", source="Other", publisher="Other")]
        assert prerank_score(preferred, config) > prerank_score(plain, config)


class TestBuildDigest:
    #: Genuinely unrelated headlines with disjoint vocabulary. Templated titles
    #: like "Story number 3" share almost every token and correctly cluster
    #: together, which is not what these tests are measuring.
    HEADLINES = [
        ("Parliament approves the annual budget bill", "Members voted late on Tuesday after a long committee stage."),
        ("Wildfires force evacuations along the northern coast", "Emergency services moved residents from three villages overnight."),
        ("Central bank holds interest rates steady", "Policymakers cited easing inflation and weak wage growth."),
        ("Archaeologists date the shipwreck to the fourth century", "Timber samples were analysed at a laboratory in Marseille."),
        ("Regulator fines airline over baggage refunds", "The penalty follows two years of passenger complaints."),
        ("New telescope images reveal a distant nebula", "Astronomers described unexpected filaments of cold dust."),
        ("Cycling infrastructure plan doubles protected lanes", "Construction begins next spring across eleven districts."),
        ("Measles cases prompt a vaccination campaign", "Clinics will open on weekends through the end of October."),
        ("Court overturns a disputed planning permission", "Judges found the environmental assessment inadequate."),
        ("Semiconductor plant delays production targets", "Equipment installation slipped by roughly four months."),
        ("Ferry service resumes after engine repairs", "Two vessels returned to the route on Thursday morning."),
        ("Drought empties reservoirs in the south", "Water restrictions now cover more than forty municipalities."),
        ("Museum acquires a collection of textile fragments", "The donation includes pieces from six weaving traditions."),
        ("Housing starts fall for a third quarter", "Developers blamed borrowing costs and permit backlogs."),
        ("Rail strike disrupts commuter timetables", "Unions rejected the latest pay offer on Friday."),
    ]

    def _seed(self, store, count=6, publisher_each=True):
        assert count <= len(self.HEADLINES) * 3
        articles = []
        for i in range(count):
            title, description = self.HEADLINES[i % len(self.HEADLINES)]
            if i >= len(self.HEADLINES):
                title = f"{title} in the eastern region" if i < 2 * len(self.HEADLINES) \
                        else f"{title}, officials confirm"
            articles.append(make_article(
                title,
                source=f"Outlet {i}" if publisher_each else "Solo",
                publisher=f"Outlet {i}" if publisher_each else "Solo",
                description=description,
                published=MONDAY + timedelta(days=1, hours=i),
            ))
        store.insert_articles(articles)
        return articles

    def test_end_to_end_offline(self, store, config, context):
        self._seed(store)
        stats = RunStats()
        result = build_digest(
            config, store, HeuristicProvider(), NullEmbeddingProvider(), context, stats,
            window=(MONDAY, MONDAY + timedelta(days=7)),
        )
        assert result.digest.id == "2026-W37"
        assert len(result.stories) == 6
        assert stats.enriched == 6
        assert stats.stories_published == 6
        # Persisted and re-readable.
        assert store.get_digest("2026-W37").story_ids == [s.id for s, _ in result.stories]

    def test_empty_window_produces_an_empty_digest_not_a_crash(self, store, config, context):
        stats = RunStats()
        result = build_digest(
            config, store, HeuristicProvider(), NullEmbeddingProvider(), context, stats,
            window=(MONDAY, MONDAY + timedelta(days=7)),
        )
        assert result.stories == []
        assert store.get_digest(result.digest.id) is not None

    def test_cross_language_stories_merge_with_embeddings(self, store, config, context):
        # Three languages, though the shipped config collects two: the mechanism
        # is language-agnostic and this is the test that proves it, so it names
        # the languages it needs rather than inheriting them.
        config.settings.supported_languages = ["en", "es", "ca"]
        articles = [
            make_article("EU announces new sanctions against Russia", source="BBC",
                         publisher="BBC", language="en",
                         published=MONDAY + timedelta(days=1)),
            make_article("La UE anuncia nuevas sanciones contra Rusia",
                         source="El País", publisher="El País", language="es",
                         published=MONDAY + timedelta(days=1, hours=2)),
            make_article("La UE anuncia noves sancions contra Rússia", source="Ara",
                         publisher="Ara", language="ca",
                         published=MONDAY + timedelta(days=1, hours=3)),
            make_article("Barcelona metro strike enters its third day", source="Verge",
                         publisher="Verge", language="en",
                         published=MONDAY + timedelta(days=2)),
        ]
        store.insert_articles(articles)
        embedder = StubEmbedder({"sanctions": "eu-sanctions", "sanciones": "eu-sanctions",
                                 "sancions": "eu-sanctions"})
        stats = RunStats()
        result = build_digest(
            config, store, HeuristicProvider(), embedder, context, stats,
            window=(MONDAY, MONDAY + timedelta(days=7)),
        )
        assert stats.cross_language_clusters == 1
        assert len(result.stories) == 2          # the metro strike stays separate
        merged = next(s for s, arts in result.stories if len(arts) == 3)
        assert merged.languages == ["ca", "en", "es"]
        assert merged.publishers == ["Ara", "BBC", "El País"]
        solo = next(s for s, arts in result.stories if len(arts) == 1)
        assert solo.languages == ["en"]

    def test_max_stories_caps_the_digest(self, store, config, context):
        self._seed(store, count=15)
        config.digest.max_stories = 4
        stats = RunStats()
        result = build_digest(
            config, store, HeuristicProvider(), NullEmbeddingProvider(), context, stats,
            window=(MONDAY, MONDAY + timedelta(days=7)),
        )
        assert len(result.stories) == 4

    def test_only_candidate_clusters_are_enriched(self, store, config, context):
        """The cost control: 15 clusters must not mean 15 enrichments."""
        self._seed(store, count=15)
        config.digest.max_stories = 3

        class Counting(HeuristicProvider):
            def __init__(self):
                super().__init__()
                self.seen = 0

            def enrich(self, items, ctx):
                self.seen += len(items)
                return super().enrich(items, ctx)

        provider = Counting()
        build_digest(
            config, store, provider, NullEmbeddingProvider(), context, RunStats(),
            window=(MONDAY, MONDAY + timedelta(days=7)),
        )
        # Each seeded headline is distinct, so one cluster holds one article.
        # The candidate pool covers the minor-story list too, which is drawn from
        # the same enriched set.
        wanted = config.digest.max_stories + config.digest.minor_stories
        expected = max(digest_module.MIN_CANDIDATE_CLUSTERS,
                       int(wanted * digest_module.CANDIDATE_MULTIPLE))
        assert provider.seen == expected
        assert provider.seen < 15

    def test_stories_are_ordered_by_score(self, store, config, context):
        self._seed(store, count=8)
        result = build_digest(
            config, store, HeuristicProvider(), NullEmbeddingProvider(), context,
            RunStats(), window=(MONDAY, MONDAY + timedelta(days=7)),
        )
        scores = [story.score for story, _ in result.stories]
        assert scores == sorted(scores, reverse=True)

    def test_rerunning_a_week_is_idempotent(self, store, config, context):
        self._seed(store)
        window = (MONDAY, MONDAY + timedelta(days=7))
        first = build_digest(config, store, HeuristicProvider(), NullEmbeddingProvider(),
                            context, RunStats(), window=window)
        second = build_digest(config, store, HeuristicProvider(), NullEmbeddingProvider(),
                              context, RunStats(), window=window)
        assert [s.id for s, _ in first.stories] == [s.id for s, _ in second.stories]
        assert len(store.list_digests()) == 1

    def test_second_run_reuses_the_enrichment_cache(self, store, config, context):
        self._seed(store)
        window = (MONDAY, MONDAY + timedelta(days=7))
        build_digest(config, store, HeuristicProvider(), NullEmbeddingProvider(),
                     context, RunStats(), window=window)
        stats = RunStats()
        build_digest(config, store, HeuristicProvider(), NullEmbeddingProvider(),
                     context, stats, window=window)
        # Already enriched in the database, so nothing is re-sent.
        assert stats.enriched == 0

    def test_embeddings_are_cached_between_runs(self, store, config, context):
        self._seed(store)
        window = (MONDAY, MONDAY + timedelta(days=7))
        embedder = StubEmbedder()
        build_digest(config, store, HeuristicProvider(), embedder, context,
                     RunStats(), window=window)
        first_calls = embedder.calls
        stats = RunStats()
        build_digest(config, store, HeuristicProvider(), embedder, context,
                     stats, window=window)
        assert embedder.calls == first_calls      # no new API calls
        assert stats.embed_cached == 6 and stats.embedded == 0

    def test_multi_publisher_story_gets_a_merged_brief(self, store, config, context):
        articles = [
            make_article("EU announces new sanctions", source="BBC", publisher="BBC",
                         published=MONDAY + timedelta(days=1)),
            make_article("EU announces new sanctions", source="Guardian",
                         publisher="Guardian", published=MONDAY + timedelta(days=1)),
        ]
        store.insert_articles(articles)

        from newsdigest.llm.base import Brief

        class Briefing(HeuristicProvider):
            def __init__(self):
                super().__init__()
                self.briefs = 0

            def write_brief(self, item, ctx):
                self.briefs += 1
                return Brief(headline="Merged headline", summary="Merged summary.",
                             why_it_matters="It matters.", key_facts=["A fact"],
                             topics=["world"], importance=0.9, relevance=0.8)

        provider = Briefing()
        result = build_digest(config, store, provider, NullEmbeddingProvider(), context,
                             RunStats(), window=(MONDAY, MONDAY + timedelta(days=7)))
        assert provider.briefs == 1
        story = result.stories[0][0]
        assert story.headline == "Merged headline"
        assert story.key_facts == ["A fact"]
        # relevance and, since 2026-09-18, importance both come from the brief.
        assert story.relevance == 0.8
        # Copying importance over was removed on 2026-09-17 and restored the next
        # day. What this test used to assert -- build_story's article-derived value,
        # 0.0 plus the 0.03 second-publisher bonus -- is exactly the failure that
        # brought it back: with only 40 articles enriched per run, most published
        # stories have no enriched article, so that number IS the publisher bonus
        # wearing importance's name. Note what this line would have been:
        #     assert story.importance == pytest.approx(0.03)
        # A story-level number from a model that spreads is the point. If a model
        # returns a constant instead (qwen3:8b gave 0.80-0.85 for everything), the
        # page shows a worse number but the order does not move, because
        # `importance` is not in ranking.terms. Check the spread before it is.
        assert story.importance == 0.9
        assert story.written_by == provider.name

    def test_single_publisher_story_costs_no_brief(self, store, config, context):
        self._seed(store, count=3, publisher_each=False)

        class Briefing(HeuristicProvider):
            def __init__(self):
                super().__init__()
                self.briefs = 0

            def write_brief(self, item, ctx):
                self.briefs += 1
                return None

        provider = Briefing()
        build_digest(config, store, provider, NullEmbeddingProvider(), context,
                     RunStats(), window=(MONDAY, MONDAY + timedelta(days=7)))
        assert provider.briefs == 0

    def test_quota_exhaustion_still_produces_a_digest(self, store, config, context):
        from newsdigest.llm.base import LLMQuotaError

        self._seed(store)

        class Exhausted(HeuristicProvider):
            def enrich(self, items, ctx):
                raise LLMQuotaError("daily limit")

        stats = RunStats()
        result = build_digest(config, store, Exhausted(), NullEmbeddingProvider(),
                             context, stats, window=(MONDAY, MONDAY + timedelta(days=7)))
        assert stats.enriched == 0
        assert len(result.stories) == 6            # degraded, not empty
        assert all(story.summary for story, _ in result.stories)
