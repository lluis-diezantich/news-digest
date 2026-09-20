"""Geographic spread (section 12).

The failure being prevented: twelve stories about US politics and one about
everything else, in a week that contained more than that. It is a composition
problem, so the fix is applied at selection and never to the score -- these tests
pin both halves of that, including the cases where the cap must NOT bind.
"""

import pytest

from newsdigest import regions
from newsdigest.models import Story

from conftest import make_article


def story(headline, region, score):
    return Story(id=Story.make_id(headline), headline=headline,
                 regions=[region] if region else [], score=score)


def ranked(*specs):
    """(story, articles) pairs in score order, as the digest builds them."""
    return [(story(h, r, s), [make_article(h)]) for h, r, s in specs]


class TestNormalisation:
    @pytest.mark.parametrize("value,expected", [
        ("europe", "europe"),
        ("Europe", "europe"),
        ("EU", "europe"),
        ("united kingdom", "europe"),
        ("spain", "europe"),
        ("US", "north america"),
        ("United States", "north america"),
        ("mexico", "latin america"),
        ("brazil", "latin america"),
        ("gaza", "middle east"),
        ("oriente medio", "middle east"),
        ("nigeria", "africa"),
        ("china", "asia"),
        ("indo-pacific", "asia"),
        ("australia", "oceania"),
        ("world", "global"),
        ("internacional", "global"),
    ])
    def test_known_values_and_aliases_map_in(self, value, expected):
        assert regions.normalize_region(value) == expected

    @pytest.mark.parametrize("value", ["", None, "atlantis", "the moon", "sport"])
    def test_an_unknown_value_becomes_unknown_not_a_guess(self, value):
        """A wrong region is worse than none: it spends another region's cap."""
        assert regions.normalize_region(value) == ""


class TestRegionOf:
    def test_the_commonest_answer_wins(self):
        articles = [
            make_article("A", region="europe"),
            make_article("B", region="europe"),
            make_article("C", region="asia"),
        ]
        assert regions.region_of(articles) == "europe"

    def test_ties_break_deterministically(self):
        """Otherwise the result depends on article order, and so does the digest."""
        one = [make_article("A", region="asia"), make_article("B", region="europe")]
        assert regions.region_of(one) == regions.region_of(list(reversed(one)))

    def test_unknown_regions_do_not_vote(self):
        articles = [
            make_article("A", region=""),
            make_article("B", region=""),
            make_article("C", region="africa"),
        ]
        assert regions.region_of(articles) == "africa"

    def test_no_answers_at_all_is_unknown(self):
        assert regions.region_of([make_article("A")]) == ""

    def test_aliases_are_normalised_before_voting(self):
        articles = [make_article("A", region="US"), make_article("B", region="usa")]
        assert regions.region_of(articles) == "north america"


class TestDiversify:
    def test_it_caps_one_regions_share(self):
        pairs = ranked(
            ("US 1", "north america", 9.0),
            ("US 2", "north america", 8.0),
            ("US 3", "north america", 7.0),
            ("EU 1", "europe", 6.0),
            ("Asia 1", "asia", 5.0),
        )
        got = regions.diversify(pairs, limit=4, max_share=0.5)
        picked = [s.headline for s, _ in got]
        assert sum(1 for h in picked if h.startswith("US")) == 2
        assert "EU 1" in picked and "Asia 1" in picked

    def test_the_order_is_still_by_score(self):
        """This decides WHICH stories are published, never the order. A story does
        not become the week's lead because of where it happened."""
        pairs = ranked(
            ("US 1", "north america", 9.0),
            ("US 2", "north america", 8.0),
            ("EU 1", "europe", 3.0),
        )
        got = regions.diversify(pairs, limit=2, max_share=0.5)
        scores = [s.score for s, _ in got]
        assert scores == sorted(scores, reverse=True)

    def test_it_is_not_a_quota(self):
        """A week genuinely dominated by one region publishes as it is."""
        pairs = ranked(*[(f"US {n}", "north america", 9.0 - n) for n in range(6)])
        got = regions.diversify(pairs, limit=4, max_share=0.5)
        assert len(got) == 4, "with no alternatives the cap must give way"

    def test_deferred_stories_return_in_score_order(self):
        pairs = ranked(
            ("US 1", "north america", 9.0),
            ("US 2", "north america", 8.0),
            ("US 3", "north america", 7.0),
            ("EU 1", "europe", 1.0),
        )
        got = regions.diversify(pairs, limit=4, max_share=0.5)
        scores = [s.score for s, _ in got]
        assert scores == sorted(scores, reverse=True)
        assert len(got) == 4

    def test_unknown_regions_are_never_deferred(self):
        """Deferring them would punish a classification failure rather than a
        real imbalance."""
        pairs = ranked(
            ("Unknown 1", "", 9.0),
            ("Unknown 2", "", 8.0),
            ("Unknown 3", "", 7.0),
            ("EU 1", "europe", 1.0),
        )
        got = regions.diversify(pairs, limit=3, max_share=0.5)
        assert [s.headline for s, _ in got] == ["Unknown 1", "Unknown 2", "Unknown 3"]

    def test_disabled_takes_the_top_by_score(self):
        pairs = ranked(
            ("US 1", "north america", 9.0),
            ("US 2", "north america", 8.0),
            ("EU 1", "europe", 1.0),
        )
        got = regions.diversify(pairs, limit=2, max_share=0.5, enabled=False)
        assert [s.headline for s, _ in got] == ["US 1", "US 2"]

    def test_a_limit_of_zero_returns_nothing(self):
        assert regions.diversify(ranked(("A", "europe", 1.0)), limit=0) == []

    def test_the_cap_is_at_least_one(self):
        """A small digest must not cap a region at zero and publish nothing."""
        pairs = ranked(("A", "europe", 2.0), ("B", "europe", 1.0))
        got = regions.diversify(pairs, limit=1, max_share=0.1)
        assert len(got) == 1

    def test_an_empty_ranking_is_not_an_error(self):
        assert regions.diversify([], limit=5) == []


class TestSpread:
    def test_it_counts_stories_per_region(self):
        stories = [story("A", "europe", 1), story("B", "europe", 1),
                   story("C", "asia", 1)]
        assert regions.spread(stories) == {"europe": 2, "asia": 1}

    def test_unknown_is_reported_as_unknown(self):
        assert regions.spread([story("A", "", 1)]) == {"unknown": 1}
