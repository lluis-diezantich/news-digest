from newsdigest import clustering
from newsdigest.models import Story

from conftest import make_article


def test_identical_event_label_clusters_across_sources():
    articles = [
        make_article("Parliament approves the budget", source="BBC",
                     event_label="uk budget vote"),
        make_article("Budget clears final vote", source="Guardian",
                     event_label="uk budget vote"),
        make_article("Rare orchid found in Peru", source="NPR",
                     event_label="peru orchid discovery"),
    ]
    groups = clustering.cluster(articles)
    assert len(groups) == 2
    biggest = groups[0]
    assert {a.source for a in biggest} == {"BBC", "Guardian"}


def test_similar_headlines_cluster_without_labels():
    articles = [
        make_article("Central bank raises interest rates by half a point", source="A"),
        make_article("Interest rates raised half a point by central bank", source="B"),
    ]
    assert len(clustering.cluster(articles)) == 1


def test_entity_overlap_rescues_a_middling_text_match():
    """A pair too weak for TEXT_THRESHOLD but sharing named entities."""
    headlines = ("Central bank raises rates", "Rates lifted by central bank")
    shared = ["Central Bank", "Federal Reserve"]
    with_entities = [
        make_article(headlines[0], source="A", entities=shared, importance=0.6),
        make_article(headlines[1], source="B", entities=shared, importance=0.6),
    ]
    assert len(clustering.cluster(with_entities)) == 1

    without_entities = [
        make_article(headlines[0], source="A", entities=[], importance=0.6),
        make_article(headlines[1], source="B", entities=[], importance=0.6),
    ]
    assert len(clustering.cluster(without_entities)) == 2


def test_reworded_coverage_needs_the_llm_label():
    """Documents a real limit: text alone cannot merge reworded coverage.

    These two headlines describe one event but score 0.10 -- indistinguishable
    from unrelated stories. Only the LLM's event_label merges them, which is why
    clustering treats that label as its primary signal.
    """
    plain = [
        make_article("Parliament approves the budget", source="A"),
        make_article("Budget clears its final vote", source="B"),
    ]
    assert len(clustering.cluster(plain)) == 2

    labelled = [
        make_article("Parliament approves the budget", source="A",
                     event_label="national budget vote"),
        make_article("Budget clears its final vote", source="B",
                     event_label="national budget vote"),
    ]
    assert len(clustering.cluster(labelled)) == 1


def test_unrelated_articles_stay_separate():
    articles = [
        make_article("Flooding closes roads in the north", source="A"),
        make_article("Football club appoints new manager", source="B"),
    ]
    assert len(clustering.cluster(articles)) == 2


def test_empty_input():
    assert clustering.cluster([]) == []


def test_build_story_prefers_the_most_important_article():
    group = [
        make_article("Minor angle on the vote", source="A", importance=0.2),
        make_article("Parliament approves budget", source="B", importance=0.8),
    ]
    story = clustering.build_story(group)
    assert story.headline == "Parliament approves budget"
    # Multi-source coverage nudges importance above the best single article.
    assert story.importance > 0.8
    assert len(story.article_ids) == 2


def test_existing_story_identity_is_reused():
    group = [make_article("Budget vote continues", event_label="uk budget vote")]
    existing = Story(id="sabc", headline="Old headline",
                     keywords=clustering.cluster_keywords(group))
    matched = clustering.match_existing_story(
        clustering.cluster_keywords(group), group, [existing]
    )
    assert matched is existing
    story = clustering.build_story(group, existing=existing)
    assert story.id == "sabc"
    assert story.first_seen == existing.first_seen


def test_assigned_story_id_wins_over_keywords():
    article = make_article("Something happened")
    article.story_id = "sxyz"
    other = Story(id="sxyz", headline="Known story", keywords=["unrelated"])
    matched = clustering.match_existing_story(["nothing", "alike"], [article], [other])
    assert matched is other
