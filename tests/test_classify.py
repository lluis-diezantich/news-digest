"""Topic filtering (section 8).

Every assertion here is about the same property: a drop needs a POSITIVE signal.
Failure keeps the article, so the worst case is a noisy digest rather than an
empty one -- and a wrongly dropped story is invisible in the output, whereas a
wrongly kept one merely ranks low.
"""

import pytest

from newsdigest import classify as classify_module
from newsdigest.classify import FilterReport, apply_filters, classify_articles
from newsdigest.config import FilterSettings, LLMSettings
from newsdigest.llm.base import (
    Classification,
    ClassifyInput,
    Context,
    LLMError,
    LLMProvider,
    LLMQuotaError,
)
from newsdigest.llm.heuristic import HeuristicProvider

from conftest import make_article


class Stub(LLMProvider):
    """Returns whatever verdicts it is told to, and can fail on demand."""

    name = "stub"
    model = "stub-1"

    def __init__(self, verdicts=None, error=None):
        super().__init__()
        self.verdicts = verdicts or {}
        self.error = error
        self.batches = 0
        self.seen = 0

    def classify(self, items, context):
        self.batches += 1
        self.seen += len(items)
        if self.error:
            raise self.error
        self.calls += 1
        return [
            Classification(id=i.id, **self.verdicts[i.title]).clamp()
            for i in items if i.title in self.verdicts
        ]

    def enrich(self, items, context):
        return []

    def write_brief(self, item, context):
        return None

    def same_event(self, pairs, context):
        return {}


@pytest.fixture
def filters():
    return FilterSettings(
        classify=True, excluded_topics=["sports", "celebrity"], max_items=100
    )


@pytest.fixture
def llm_settings():
    return LLMSettings(provider="stub", batch_size=8)


class TestApplyFilters:
    def test_not_news_is_dropped(self, filters):
        keep = make_article("A real thing happened", newsworthy=True)
        junk = make_article("Your horoscope for the week", newsworthy=False)
        got = apply_filters([keep, junk], filters)
        assert [a.title for a in got] == ["A real thing happened"]

    def test_an_unclassified_article_is_kept(self, filters):
        """None is deliberately different from False."""
        article = make_article("Something not yet classified")
        assert article.newsworthy is None
        assert apply_filters([article], filters) == [article]

    def test_an_article_whose_every_topic_is_excluded_is_dropped(self, filters):
        article = make_article("A match ended in a draw", topics=["sports"],
                               newsworthy=True)
        assert apply_filters([article], filters) == []

    def test_a_mixed_topic_article_is_kept(self, filters):
        """EVERY topic must be excluded, not any. An item tagged
        `politics, sports` is a head of state at a stadium opening, and dropping
        it on the sports tag alone loses the political story."""
        article = make_article("The president opens the new stadium",
                               topics=["politics", "sports"], newsworthy=True)
        assert apply_filters([article], filters) == [article]

    def test_no_topics_at_all_is_kept(self, filters):
        article = make_article("An untagged thing happened", topics=[], newsworthy=True)
        assert apply_filters([article], filters) == [article]

    def test_content_types_are_only_dropped_when_configured(self, filters):
        opinion = make_article("A column about the budget", content_type="opinion",
                               newsworthy=True)
        assert apply_filters([opinion], filters) == [opinion]
        filters.drop_content_types = ["opinion"]
        assert apply_filters([opinion], filters) == []

    def test_the_report_says_which_rule_dropped_what(self, filters):
        report = FilterReport()
        apply_filters(
            [
                make_article("Your horoscope for the week", newsworthy=False),
                make_article("A match ended in a draw", topics=["sports"],
                             newsworthy=True),
            ],
            filters, report,
        )
        assert len(report.dropped_not_news) == 1
        assert len(report.dropped_topic) == 1
        assert report.dropped == 2


class TestClassifyArticles:
    def test_verdicts_are_applied_to_the_articles(self, store, filters, llm_settings):
        article = make_article("A real thing happened")
        llm = Stub({"A real thing happened": {
            "topics": ["politics"], "region": "europe", "newsworthy": True,
        }})
        report = classify_articles(
            store, llm, llm_settings, filters, Context(), [article]
        )
        assert report.classified == 1
        assert article.newsworthy is True
        assert article.region == "europe"
        assert article.topics == ["politics"]

    def test_a_second_run_is_served_from_the_cache(self, store, filters, llm_settings):
        """Iterating on the prompt must be free after the first run."""
        llm = Stub({"A real thing happened": {"newsworthy": True, "region": "europe"}})
        first = make_article("A real thing happened")
        classify_articles(store, llm, llm_settings, filters, Context(), [first])
        assert llm.calls == 1

        again = make_article("A real thing happened")
        report = classify_articles(store, llm, llm_settings, filters, Context(), [again])
        assert llm.calls == 1, "the same text must not be sent twice"
        assert report.cached == 1
        assert again.region == "europe"

    def test_an_article_missing_from_the_answer_is_kept(
        self, store, filters, llm_settings
    ):
        """Missing ids are retried next run, not dropped."""
        answered = make_article("A real thing happened")
        ignored = make_article("Something the model skipped")
        llm = Stub({"A real thing happened": {"newsworthy": True}})
        classify_articles(
            store, llm, llm_settings, filters, Context(), [answered, ignored]
        )
        assert ignored.newsworthy is None
        assert apply_filters([ignored], filters) == [ignored]

    def test_a_failed_batch_keeps_its_articles(self, store, filters, llm_settings):
        articles = [make_article(f"Thing {n}") for n in range(4)]
        llm = Stub(error=LLMError("transport blew up"))
        report = classify_articles(
            store, llm, llm_settings, filters, Context(), articles
        )
        assert report.failed == 4
        assert all(a.newsworthy is None for a in articles)
        assert apply_filters(articles, filters) == articles

    def test_it_gives_up_after_consecutive_failures(self, store, filters, llm_settings):
        """One slow provider must not mean one request per batch for the rest of
        the run."""
        llm_settings.batch_size = 1
        articles = [make_article(f"Thing {n}") for n in range(10)]
        llm = Stub(error=LLMError("timeout"))
        classify_articles(store, llm, llm_settings, filters, Context(), articles)
        assert llm.batches == classify_module.MAX_CONSECUTIVE_FAILURES

    def test_a_quota_error_stops_immediately(self, store, filters, llm_settings):
        llm_settings.batch_size = 1
        articles = [make_article(f"Thing {n}") for n in range(10)]
        llm = Stub(error=LLMQuotaError("out of quota"))
        report = classify_articles(
            store, llm, llm_settings, filters, Context(), articles
        )
        assert report.quota_exhausted
        assert llm.batches == 1

    def test_classify_off_makes_no_calls(self, store, llm_settings):
        llm = Stub({"A real thing happened": {"newsworthy": False}})
        article = make_article("A real thing happened")
        report = classify_articles(
            store, llm, llm_settings, FilterSettings(classify=False), Context(), [article]
        )
        assert (llm.batches, report.classified) == (0, 0)
        assert article.newsworthy is None

    def test_max_items_bounds_the_pass(self, store, filters, llm_settings):
        filters.max_items = 3
        articles = [make_article(f"Thing number {n}") for n in range(10)]
        llm = Stub({a.title: {"newsworthy": True} for a in articles})
        classify_articles(store, llm, llm_settings, filters, Context(), articles)
        assert llm.seen == 3

    def test_topics_do_not_overwrite_an_enriched_answer(
        self, store, filters, llm_settings
    ):
        """Enrichment sees the full excerpt and runs either side of this
        depending on the cache, so the cheap pass must not clobber it."""
        article = make_article("A real thing happened", topics=["economics"])
        llm = Stub({"A real thing happened": {"topics": ["politics"], "newsworthy": True}})
        classify_articles(store, llm, llm_settings, filters, Context(), [article])
        assert article.topics == ["economics"]

    def test_the_verdict_is_persisted(self, store, filters, llm_settings):
        article = make_article("A real thing happened")
        store.insert_articles([article])
        llm = Stub({"A real thing happened": {"newsworthy": False, "region": "asia"}})
        classify_articles(store, llm, llm_settings, filters, Context(), [article])

        from datetime import datetime, timedelta, timezone
        stored = store.articles_in_window(
            datetime(2000, 1, 1, tzinfo=timezone.utc),
            article.collected_at + timedelta(days=1),
        )
        assert stored[0].newsworthy is False
        assert stored[0].region == "asia"


class TestOfflineClassification:
    """With no model, topic filtering still works -- weakly, and in the safe
    direction."""

    def test_topics_come_from_keywords(self):
        provider = HeuristicProvider()
        got = provider.classify(
            [ClassifyInput(id="a", title="Central bank raises interest rates",
                           language="en", excerpt="Inflation data published")],
            Context(),
        )
        assert got[0].topics == ["economics"]

    def test_it_never_claims_something_is_not_news(self):
        """"final" and "corona" would drop a court ruling and a public-health
        story. A keyword matcher cannot make this call."""
        provider = HeuristicProvider()
        got = provider.classify(
            [ClassifyInput(id="a", title="The cup final ends in a draw",
                           language="en", excerpt="Extra time was needed")],
            Context(),
        )
        assert got[0].newsworthy is True

    def test_it_offers_no_region(self):
        provider = HeuristicProvider()
        got = provider.classify(
            [ClassifyInput(id="a", title="Spanish government approves budget",
                           language="en", excerpt="Madrid")],
            Context(),
        )
        assert got[0].region == ""

    def test_a_provider_with_no_classify_keeps_everything(self):
        """The base default: a new provider degrades to no topic filtering rather
        than to an empty digest."""
        class Minimal(LLMProvider):
            name, model = "minimal", "m"

            def enrich(self, items, context):
                return []

            def write_brief(self, item, context):
                return None

            def same_event(self, pairs, context):
                return {}

        assert Minimal().classify([ClassifyInput("a", "t", "en", "e")], Context()) == []
