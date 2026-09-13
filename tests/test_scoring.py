from newsdigest import clustering, scoring
from newsdigest.config import Interests

from conftest import make_article


def test_preferred_topic_outranks_ignored_topic(config):
    tech = clustering.build_story(
        [make_article("Chip breakthrough", topics=["technology"], importance=0.5)]
    )
    sport = clustering.build_story(
        [make_article("Cup final result", topics=["sports"], importance=0.5)]
    )
    articles_t = [make_article("Chip breakthrough", topics=["technology"], importance=0.5)]
    articles_s = [make_article("Cup final result", topics=["sports"], importance=0.5)]
    assert scoring.score_story(tech, articles_t, config.interests) > scoring.score_story(
        sport, articles_s, config.interests
    )


def test_keyword_bonus_applies(config):
    plain = [make_article("Model released by a lab", topics=["ai"], importance=0.5)]
    named = [make_article("Anthropic releases a model", topics=["ai"], importance=0.5)]
    s_plain = clustering.build_story(plain)
    s_named = clustering.build_story(named)
    assert scoring.score_story(s_named, named, config.interests) > scoring.score_story(
        s_plain, plain, config.interests
    )


def test_mute_penalty_pushes_a_story_down(config):
    muted = [make_article("Your horoscope for today", topics=["culture"], importance=0.5)]
    normal = [make_article("Council approves new bridge", topics=["culture"], importance=0.5)]
    s_muted = clustering.build_story(muted)
    s_normal = clustering.build_story(normal)
    assert scoring.score_story(s_muted, muted, config.interests) < scoring.score_story(
        s_normal, normal, config.interests
    )


def test_recency_decays(config):
    fresh = [make_article("Event happens", hours_ago=0.5, importance=0.5, topics=["world"])]
    stale = [make_article("Event happens", hours_ago=48, importance=0.5, topics=["world"])]
    s_fresh = clustering.build_story(fresh)
    s_stale = clustering.build_story(stale)
    assert scoring.score_story(s_fresh, fresh, config.interests) > scoring.score_story(
        s_stale, stale, config.interests
    )


def test_corroboration_rewards_multiple_sources(config):
    one = [make_article("Treaty signed", source="A", importance=0.5, topics=["world"])]
    three = [
        make_article("Treaty signed", source=name, importance=0.5, topics=["world"])
        for name in ("A", "B", "C")
    ]
    s_one = clustering.build_story(one)
    s_three = clustering.build_story(three)
    assert scoring.score_story(s_three, three, config.interests) > scoring.score_story(
        s_one, one, config.interests
    )


def test_empty_interests_still_scores():
    articles = [make_article("Anything", importance=0.5)]
    story = clustering.build_story(articles)
    assert scoring.score_story(story, articles, Interests()) > 0
