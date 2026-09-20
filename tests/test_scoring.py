"""Ranking: deterministic, and driven entirely by the config formula."""

from newsdigest import clustering, scoring
from newsdigest.config import KNOWN_RANKING_TERMS, Preferences, load_preferences
from newsdigest.models import utcnow

from conftest import make_article


def story_of(*articles):
    return clustering.build_story(list(articles)), list(articles)


class TestSignals:
    def test_all_known_terms_are_computed(self, config):
        """Every term the config may name has to be computable.

        Compared against KNOWN_RANKING_TERMS rather than the default formula: the
        default deliberately uses only four of the eight, and a term left out of
        it still has to work the moment someone adds the line back."""
        story, arts = story_of(make_article("A story", importance=0.5, relevance=0.5))
        computed = scoring.signals(story, arts, config.preferences)
        assert set(computed) == set(KNOWN_RANKING_TERMS)
        assert set(Preferences().ranking) <= set(computed)

    def test_corroboration_counts_publishers_not_feeds(self, config):
        """Seven El País sections are one outlet, not seven."""
        sections = [
            make_article(f"Story from section {i}", source=f"El País {i}",
                         publisher="El País", importance=0.5)
            for i in range(4)
        ]
        outlets = [
            make_article(f"Story from outlet {i}", source=f"Outlet {i}",
                         publisher=f"Outlet {i}", importance=0.5)
            for i in range(4)
        ]
        assert scoring.corroboration(sections) == 0.0
        assert scoring.corroboration(outlets) > 0.5

    def test_corroboration_does_not_saturate_below_the_range_it_is_given(self):
        """A saturation point inside the range makes every story above it tie.

        The failure this guards is silent: the term still computes, still appears
        in `explain`, and has simply stopped discriminating. With ten newsletters
        the range is narrow, so getting the saturation point right matters more
        here than it did with twenty-two feeds.
        """
        def outlets(n):
            return [make_article(f"Story {i}", source=f"S{i}", publisher=f"P{i}",
                                 importance=0.5) for i in range(n)]

        # At the default of 5, a 5-outlet story is the top of the range and a
        # 4-outlet one must still rank below it.
        assert scoring.corroboration(outlets(4)) < scoring.corroboration(outlets(5))
        assert scoring.corroboration(outlets(5)) == 1.0
        # Monotonic and distinct across the range newsletters actually produce.
        values = [scoring.corroboration(outlets(n)) for n in (2, 3, 4, 5)]
        assert values == sorted(values) and len(set(values)) == len(values)
        # Saturated above it, which is why the point has to be measured.
        assert scoring.corroboration(outlets(9)) == scoring.corroboration(outlets(5))

    def test_saturation_point_is_configurable(self):
        arts = [make_article(f"S{i}", source=f"S{i}", publisher=f"P{i}", importance=0.5)
                for i in range(4)]
        assert scoring.corroboration(arts, saturation=4) == 1.0
        assert scoring.corroboration(arts, saturation=13) < 1.0

    def test_recency_decays_over_the_week(self, config):
        fresh = [make_article("Event", hours_ago=1, importance=0.5)]
        stale = [make_article("Event", hours_ago=140, importance=0.5)]
        s_fresh, _ = story_of(*fresh)
        s_stale, _ = story_of(*stale)
        assert scoring.recency(s_fresh, fresh, config.preferences) > scoring.recency(
            s_stale, stale, config.preferences
        )

    def test_preferred_source_lifts_source_preference(self, config):
        plain = [make_article("Story", source="Other", publisher="Other")]
        preferred = [make_article("Story", source="BBC World", publisher="BBC World")]
        assert scoring.source_preference(preferred, config.preferences) > (
            scoring.source_preference(plain, config.preferences)
        )

    def test_story_size_saturates(self, config):
        few = [make_article(f"S{i}") for i in range(2)]
        many = [make_article(f"S{i}") for i in range(20)]
        assert scoring.story_size(few) < scoring.story_size(many) <= 1.0


class TestScore:
    def test_relevance_changes_the_ranking(self, config):
        # `interest` and `relevance` are not in the shipped formula, so
        # the term under test is named explicitly here.
        config.preferences.ranking = {"relevance": 1.0}
        low, arts_low = story_of(make_article("A", importance=0.6, relevance=0.1,
                                              topics=["world"]))
        high, arts_high = story_of(make_article("B", importance=0.6, relevance=0.9,
                                               topics=["world"]))
        assert scoring.score_story(high, arts_high, config.preferences) > (
            scoring.score_story(low, arts_low, config.preferences)
        )

    def test_preferred_topic_outranks_neutral_topic(self, config):
        # `interest` and `relevance` are not in the shipped formula, so
        # the term under test is named explicitly here.
        config.preferences.ranking = {"interest": 1.0}
        tech, arts_t = story_of(make_article("Chip breakthrough", topics=["technology"],
                                            importance=0.5, relevance=0.5))
        other, arts_o = story_of(make_article("Council repaves road", topics=["local"],
                                             importance=0.5, relevance=0.5))
        assert scoring.score_story(tech, arts_t, config.preferences) > (
            scoring.score_story(other, arts_o, config.preferences)
        )

    def test_keyword_bonus_applies(self, config):
        # `interest` and `relevance` are not in the shipped formula, so
        # the term under test is named explicitly here.
        config.preferences.ranking = {"interest": 1.0}
        plain, a1 = story_of(make_article("A lab released a model", topics=["ai"],
                                         importance=0.5, relevance=0.5))
        named, a2 = story_of(make_article("Anthropic released a model", topics=["ai"],
                                         importance=0.5, relevance=0.5))
        assert scoring.score_story(named, a2, config.preferences) > (
            scoring.score_story(plain, a1, config.preferences)
        )

    def test_excluded_topic_is_penalised(self, config):
        muted, a1 = story_of(make_article("Cup final result", topics=["sports"],
                                         importance=0.5, relevance=0.5))
        normal, a2 = story_of(make_article("Bridge approved", topics=["world"],
                                          importance=0.5, relevance=0.5))
        assert scoring.score_story(muted, a1, config.preferences) < (
            scoring.score_story(normal, a2, config.preferences)
        )

    def test_score_never_goes_negative(self, config):
        config.preferences.excluded_penalty = 99.0
        story, arts = story_of(make_article("Cup final", topics=["sports"], importance=0.1,
                                            relevance=0.1))
        assert scoring.score_story(story, arts, config.preferences) == 0.0

    def test_empty_preferences_still_scores(self):
        story, arts = story_of(make_article("Anything", importance=0.5, relevance=0.5))
        assert scoring.score_story(story, arts, Preferences()) > 0


class TestConfigurableFormula:
    def test_dropping_a_term_removes_its_contribution(self, config):
        arts = [make_article("Fresh story", hours_ago=1, importance=0.5, relevance=0.5)]
        story, _ = story_of(*arts)
        with_recency = scoring.score_story(story, arts, config.preferences)

        config.preferences.ranking = {
            k: v for k, v in config.preferences.ranking.items() if k != "recency"
        }
        without = scoring.score_story(story, arts, config.preferences)
        assert without < with_recency
        assert "recency" not in scoring.explain(story, arts, config.preferences)

    def test_a_single_term_formula_works(self, config):
        config.preferences.ranking = {"importance": 2.0}
        arts = [make_article("Story", importance=0.4, relevance=0.9)]
        story, _ = story_of(*arts)
        assert scoring.score_story(story, arts, config.preferences) == round(
            2.0 * story.importance, 4
        )

    def test_explain_breaks_the_total_down(self, config):
        """The displayed terms must sum to the score used for ranking."""
        arts = [make_article("Story", importance=0.5, relevance=0.5, topics=["technology"])]
        story, _ = story_of(*arts)
        now = utcnow()
        breakdown = scoring.explain(story, arts, config.preferences, now=now)
        assert breakdown["total"] == scoring.score_story(
            story, arts, config.preferences, now=now
        )
        terms = {k: v for k, v in breakdown.items() if k != "total"}
        assert set(terms) == set(config.preferences.ranking)
        assert breakdown["total"] == round(sum(terms.values()), 4)

    def test_explain_shows_the_penalty(self, config):
        arts = [make_article("Cup final", topics=["sports"], importance=0.5, relevance=0.5)]
        story, _ = story_of(*arts)
        assert scoring.explain(story, arts, config.preferences)["excluded_penalty"] < 0


class TestShippedFormula:
    """The formula in config/preferences.yaml, as opposed to the defaults.

    Every other test here builds `Preferences()` from code defaults, so until
    2026-09-17 nothing asserted the file the digest actually runs on. These pin
    the two decisions that are easy to undo by accident.
    """

    @staticmethod
    def shipped():
        return load_preferences()[1]

    def test_importance_is_not_in_the_formula(self):
        """Removed 2026-09-17: qwen3:8b returns 0.80-0.85 for every briefed story,
        so at weight 2.0 it was a constant that could not reorder anything, and it
        inverted -- unbriefed stories kept a higher article-derived value and beat
        briefed ones on it. Restore it only with a measurement. See the note in
        config/preferences.yaml."""
        assert "importance" not in self.shipped().ranking

    def test_the_score_ignores_importance(self):
        """The property that matters, independent of how the term is spelled."""
        prefs = self.shipped()
        dull, arts_dull = story_of(make_article("A", importance=0.05))
        vital, arts_vital = story_of(make_article("B", importance=1.0))
        now = utcnow()
        assert scoring.score_story(dull, arts_dull, prefs, now=now) == (
            scoring.score_story(vital, arts_vital, prefs, now=now)
        )

    def test_more_publishers_outranks_fewer(self):
        """What importance's weight was handed to. Coverage has to be the thing
        that separates two otherwise identical stories."""
        prefs = self.shipped()
        few = [make_article("Event", source=f"S{i}", publisher=f"P{i}") for i in range(3)]
        many = [make_article("Event", source=f"S{i}", publisher=f"P{i}") for i in range(11)]
        story_few, _ = story_of(*few)
        story_many, _ = story_of(*many)
        now = utcnow()
        assert scoring.score_story(story_few, few, prefs, now=now) < (
            scoring.score_story(story_many, many, prefs, now=now)
        )
