"""Feed/DOM position: recorded, deliberately not ranked.

Feed order is the newsroom's own ordering -- measured 80-100% concordant with the
section front page across eight of the configured sources -- and it is now the
digest's primary ranking signal.

What makes that safe is upstream, not here. Position measures PROMOTION, and
Ara's eight advertorial items sat at positions 14-24 of 131 on 2026-09-14,
out-ranking most of its reporting. `max_items: 10` never reaches position 14 and
`exclude_url_patterns` drops those sections, so the advertising never arrives.
Two tests below assert both controls are still in place.
"""

import pytest

from conftest import make_article

from newsdigest import scoring
from newsdigest.config import Preferences, Source
from newsdigest.models import Article
from newsdigest.sources.rss import RSSAdapter


def _wide_window():
    """A window that certainly contains anything just inserted."""
    from datetime import datetime, timedelta, timezone
    from newsdigest.models import utcnow
    return datetime(2000, 1, 1, tzinfo=timezone.utc), utcnow() + timedelta(days=1)


def at(position: int, size: int, **kw) -> Article:
    a = make_article(kw.pop("title", "A story"), **kw)
    a.feed_position, a.feed_size = position, size
    return a


class TestNormalisation:
    def test_leading_item_scores_one(self):
        assert at(0, 20).editorial_rank() == 1.0

    def test_last_item_scores_zero(self):
        assert at(19, 20).editorial_rank() == 0.0

    def test_midway_is_about_half(self):
        assert 0.4 < at(9, 20).editorial_rank() < 0.6

    def test_normalised_within_the_source(self):
        """Position 3 of 10 and of 190 are not the same claim."""
        assert at(3, 10).editorial_rank() < at(3, 190).editorial_rank()

    def test_unknown_position_is_neutral_not_penalised(self):
        """A source giving no order must not read as though all of it trailed."""
        assert at(-1, 0).editorial_rank() == 0.5

    def test_single_item_listing_is_neutral(self):
        """One item carries no ordering information."""
        assert at(0, 1).editorial_rank() == 0.5

    def test_position_beyond_size_is_clamped(self):
        assert at(99, 20).editorial_rank() == 0.0


FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>Fixture Wire</title>
  <item><title>Lead story of the day</title>
        <link>https://fixture.example/one</link>
        <description>First.</description></item>
  <item><title>Second story down</title>
        <link>https://fixture.example/two</link>
        <description>Second.</description></item>
  <item><title>Third story down</title>
        <link>https://fixture.example/three</link>
        <description>Third.</description></item>
</channel></rss>
"""


class FakeResponse:
    def __init__(self, body):
        self.content = body.encode("utf-8")
        self.text = body
        self.url = "https://fixture.example/rss"


class FakeFetcher:
    def __init__(self, body):
        self.body = body

    def get(self, url, **kwargs):
        return FakeResponse(self.body)


class TestRecording:
    def test_the_adapter_records_feed_order(self):
        source = Source(name="Fixture", rss="https://fixture.example/rss")
        articles = RSSAdapter(FakeFetcher(FEED)).fetch(source)
        assert [a.feed_position for a in articles] == [0, 1, 2]
        assert {a.feed_size for a in articles} == {3}
        assert articles[0].editorial_rank() == 1.0

    def test_feed_size_is_what_we_took_not_what_was_offered(self):
        """max_items caps the take, so the rank is relative to the window seen."""
        source = Source(name="Fixture", rss="https://fixture.example/rss", max_items=2)
        articles = RSSAdapter(FakeFetcher(FEED)).fetch(source)
        assert [a.feed_position for a in articles] == [0, 1]
        assert {a.feed_size for a in articles} == {2}

    def test_position_survives_the_store(self, store):
        store.insert_articles([at(3, 25, title="A stored story")])
        got = store.articles_in_window(*_wide_window())
        assert len(got) == 1
        assert (got[0].feed_position, got[0].feed_size) == (3, 25)

    def test_rows_predating_the_column_read_as_unknown(self, store):
        """A v2 database migrated to v3 keeps -1, which is neutral, not last."""
        store.insert_articles([make_article("An older story")])
        store.conn.execute("UPDATE articles SET feed_position = -1, feed_size = 0")
        got = store.articles_in_window(*_wide_window())
        assert got[0].editorial_rank() == 0.5


class TestRanked:
    """Position IS the primary signal now, and what keeps it honest is upstream."""

    def test_it_is_a_computed_signal(self):
        from newsdigest import clustering
        a = at(0, 20, importance=0.5, relevance=0.5)
        story = clustering.build_story([a])
        assert "editorial_position" in scoring.signals(story, [a], Preferences())

    def test_best_position_wins_not_the_mean(self):
        """One outlet leading with it beats several burying it."""
        assert scoring.editorial_position([at(0, 20), at(18, 20), at(19, 20)]) == 1.0
        assert scoring.editorial_position([at(18, 20), at(19, 20)]) < 0.2

    def test_no_articles_is_neutral(self):
        assert scoring.editorial_position([]) == 0.5

    def test_a_lead_story_outranks_a_buried_one(self):
        from newsdigest import clustering
        prefs = Preferences()
        lead, buried = at(0, 25, importance=0.35), at(24, 25, importance=0.35)
        s_lead = clustering.build_story([lead])
        s_buried = clustering.build_story([buried])
        assert (scoring.score_story(s_lead, [lead], prefs)
                > scoring.score_story(s_buried, [buried], prefs))

    def test_terms_still_sum_to_the_total(self):
        from newsdigest import clustering
        a = at(4, 20, importance=0.5, relevance=0.5)
        story = clustering.build_story([a])
        b = scoring.explain(story, [a], Preferences())
        total = b.pop("total")
        assert total == pytest.approx(round(sum(b.values()), 4), abs=1e-4)

    def test_the_cap_is_what_keeps_advertorial_out(self):
        """Ara's advertorial sat at positions 14-24 of 131 on 2026-09-14, ranking
        above most of its reporting. Nothing in the scoring catches that -- only
        `max_items` (top ten) and `exclude_url_patterns` do. This test records the
        dependency so raising the cap is a deliberate act, not an accident."""
        from newsdigest.config import load_sources
        for src in load_sources('config/sources.yaml'):
            if src.enabled and src.method == "rss":
                assert src.max_items <= 10, (
                    f"{src.name} takes {src.max_items} items; advertorial was seen "
                    f"at position 14. Re-check before raising this."
                )

    def test_the_advertorial_blocklist_is_still_configured(self):
        from newsdigest.config import load_sources
        ara = [s for s in load_sources('config/sources.yaml') if s.name == "Ara"]
        assert ara, "Ara not configured"
        assert any("especials" in p for p in ara[0].exclude_url_patterns), (
            "the /especials/ blocklist is gone; position would promote advertising"
        )
