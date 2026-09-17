"""Clustering measured against hand-labelled pairs from two dissected stories.

These are regression guards on real data, not unit tests: `newsdigest.eval`
clusters 27 articles whose true events are known and scores the result. Numbers
below were measured on 2026-09-17 and are floors, deliberately a little slacker
than the measurement, so a genuine improvement does not fail the suite but a
regression does.

Runs offline -- the fixture carries its own cached vectors -- and with no LLM
provider, so this covers the embedding tiers only. Recall is therefore a floor:
pairs the adjudicated band would have merged count as misses here.
"""

import pytest

from newsdigest import eval as cluster_eval

# The owner's tuning as of 2026-09-17, raised from 0.80 the same day: 0.78
# doubles recall on this fixture at no cost in precision.
SIMILARITY = 0.78
AMBIGUOUS = 0.65


@pytest.fixture(scope="module")
def fixture():
    return cluster_eval.load()


def test_fixture_is_intact(fixture):
    """Every article belongs to exactly one event and carries a unit vector."""
    assert len(fixture.articles) == 27
    assert len(fixture.vectors) == 27
    for article in fixture.articles:
        assert fixture.event_of(article.id)
        assert abs(float(fixture.vectors[article.id] @ fixture.vectors[article.id]) - 1.0) < 1e-3


def test_no_false_merges_at_current_tuning(fixture):
    """Precision is what matters most: merging is transitive, so one false
    positive welds two whole clusters. Both stories in the fixture are examples."""
    report = cluster_eval.evaluate(
        fixture, similarity_threshold=SIMILARITY, ambiguous_threshold=AMBIGUOUS
    )
    assert report.false_positives == 0, (
        "auto-merge tier welded unrelated events:\n"
        + "\n".join(f"  {p.left_event} + {p.right_event}" for p in report.welded)
    )


def test_documented_welds_stay_apart(fixture):
    """The specific false positives recorded in doc/pipeline.md.

    A 0.726 edge pulled a Junts piece into the Ceuta story and a 0.705 edge
    pulled in three Podemos articles; the second story welded four unrelated
    court matters. These are the pairs a clustering change must not resurrect.
    """
    report = cluster_eval.evaluate(
        fixture, similarity_threshold=SIMILARITY, ambiguous_threshold=AMBIGUOUS
    )
    welded = {frozenset({p.left_event, p.right_event}) for p in report.welded}
    for pair in [
        {"clavijo_reaction", "junts_puigdemont"},      # the 0.726 edge
        {"morocco_blackmail", "podemos_primaries"},    # the 0.705 edge
        {"jec_session", "alacant_antifascists"},
        {"jec_session", "ceuta_delegate_case"},
        {"jec_session", "caso_lezo"},
    ]:
        assert frozenset(pair) not in welded, f"{sorted(pair)} merged again"


def test_recall_floor(fixture):
    """Measured 0.372 at 0.78 (it was 0.163 at 0.80). Still low, because the
    auto-merge tier is conservative and the adjudicated band -- not exercised
    here -- is meant to catch the rest. Guards against a change that merges
    even less."""
    report = cluster_eval.evaluate(
        fixture, similarity_threshold=SIMILARITY, ambiguous_threshold=AMBIGUOUS
    )
    assert report.recall >= 0.35


def test_cross_language_is_effectively_unreachable_by_auto_merge(fixture):
    """Measured: 8 of the 9 genuine cross-language pairs score below the 0.78
    auto-merge tier, and the 9th only just clears it at 0.782. So nearly every
    merge that spans a language comes from the adjudicated band or from chaining,
    not from the embedding alone.

    Lowering the tier does not fix this: cross-language negatives reach 0.736
    while its positives start at 0.556, so the two overlap almost entirely.

    If this fails because a better embedder lifted cross-language similarity,
    that is good news -- re-measure and consider a lower tier.
    """
    cross = [
        fixture.cosine(p.left, p.right)
        for p in fixture.pairs()
        if p.same and fixture.cross_language(p.left, p.right)
    ]
    assert len(cross) == 9
    assert sum(1 for c in cross if c >= SIMILARITY) <= 1
    assert max(cross) == pytest.approx(0.782, abs=0.01)


def test_bands_overlap_so_no_single_threshold_works(fixture):
    """Why the adjudicated band exists at all.

    Genuine pairs run down to 0.536 and unrelated pairs up to 0.736, so the two
    distributions overlap across a 0.20-wide range. No threshold separates them:
    any value low enough to catch the positives welds unrelated events, which is
    exactly what both fixture stories show. Cross-language is the worse half --
    its negatives reach 0.736 while its positives start at 0.556.
    """
    positives = [p for p in fixture.pairs() if p.same]
    negatives = [p for p in fixture.pairs() if not p.same]
    lowest_positive = min(fixture.cosine(p.left, p.right) for p in positives)
    highest_negative = max(fixture.cosine(p.left, p.right) for p in negatives)
    assert lowest_positive < highest_negative, (
        "the embedding now separates same-event from unrelated pairs outright; "
        "re-tune the thresholds and simplify the tiers"
    )
