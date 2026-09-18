"""Clustering: embeddings first, LLM for the borderline, text as last resort."""

import numpy as np

from newsdigest import clustering
from newsdigest.embeddings.base import normalize
from newsdigest.llm.base import Context, LLMError
from newsdigest.llm.heuristic import HeuristicProvider

from conftest import make_article

SANCTIONS = [
    ("EU announces new sanctions package against Russia", "en", "BBC"),
    ("La UE anuncia un nuevo paquete de sanciones contra Rusia", "es", "El País"),
    ("La UE anuncia un nou paquet de sancions contra Rússia", "ca", "Ara"),
]


def vecs(mapping: dict[str, list[float]]) -> dict[str, np.ndarray]:
    return {k: normalize(np.array(v, dtype=float)) for k, v in mapping.items()}


def sanctions_articles():
    return [make_article(t, source=p, publisher=p, language=l) for t, l, p in SANCTIONS]


class TestEmbeddingClustering:
    def test_merges_the_same_event_across_three_languages(self):
        arts = sanctions_articles()
        vectors = vecs({
            arts[0].id: [1.0, 0.2, 0.0],
            arts[1].id: [0.97, 0.24, 0.02],
            arts[2].id: [0.95, 0.26, 0.01],
        })
        groups = clustering.cluster(arts, vectors, similarity_threshold=0.82)
        assert len(groups) == 1
        assert sorted(a.language for a in groups[0]) == ["ca", "en", "es"]
        assert clustering.cross_language_count(groups) == 1

    def test_keeps_unrelated_events_apart(self):
        arts = sanctions_articles() + [
            make_article("Barcelona metro strike enters third day", source="Verge")
        ]
        vectors = vecs({
            arts[0].id: [1.0, 0.2, 0.0],
            arts[1].id: [0.97, 0.24, 0.02],
            arts[2].id: [0.95, 0.26, 0.01],
            arts[3].id: [0.0, 0.1, 1.0],
        })
        groups = clustering.cluster(arts, vectors, similarity_threshold=0.82)
        assert len(groups) == 2
        assert len(groups[0]) == 3 and len(groups[1]) == 1

    def test_most_covered_cluster_comes_first(self):
        arts = sanctions_articles() + [make_article("Solo story", source="Solo")]
        vectors = vecs({
            arts[0].id: [1.0, 0.0, 0.0], arts[1].id: [0.99, 0.01, 0.0],
            arts[2].id: [0.98, 0.02, 0.0], arts[3].id: [0.0, 0.0, 1.0],
        })
        groups = clustering.cluster(arts, vectors, similarity_threshold=0.82)
        assert len({a.publisher for a in groups[0]}) == 3

    def test_article_without_a_vector_still_lands_somewhere(self):
        arts = sanctions_articles()
        vectors = vecs({arts[0].id: [1.0, 0.0, 0.0], arts[1].id: [0.99, 0.01, 0.0]})
        groups = clustering.cluster(arts, vectors, similarity_threshold=0.82)
        assert sum(len(g) for g in groups) == 3


class TestAmbiguousBand:
    """Pairs between the two thresholds are referred to the LLM."""

    def _setup(self):
        arts = [
            make_article("Ministers debate the budget", source="A", publisher="A"),
            make_article("Budget talks continue in cabinet", source="B", publisher="B"),
        ]
        # Cosine ~0.78: below similarity_threshold, above ambiguous_threshold.
        vectors = vecs({arts[0].id: [1.0, 0.0], arts[1].id: [0.78, 0.63]})
        return arts, vectors

    def test_llm_yes_merges(self):
        arts, vectors = self._setup()

        class Yes(HeuristicProvider):
            def same_event(self, pairs, context):
                return {p.key: True for p in pairs}

        groups = clustering.cluster(
            arts, vectors, similarity_threshold=0.9, ambiguous_threshold=0.7,
            provider=Yes(), context=Context(),
        )
        assert len(groups) == 1

    def test_llm_no_keeps_them_apart(self):
        arts, vectors = self._setup()

        class No(HeuristicProvider):
            def same_event(self, pairs, context):
                return {p.key: False for p in pairs}

        groups = clustering.cluster(
            arts, vectors, similarity_threshold=0.9, ambiguous_threshold=0.7,
            provider=No(), context=Context(),
        )
        assert len(groups) == 2

    def test_silence_leaves_the_embedding_verdict(self):
        arts, vectors = self._setup()
        groups = clustering.cluster(
            arts, vectors, similarity_threshold=0.9, ambiguous_threshold=0.7,
            provider=HeuristicProvider(), context=Context(),
        )
        assert len(groups) == 2

    def test_check_budget_is_respected(self):
        arts = [make_article(f"Story number {i}", source=f"S{i}") for i in range(6)]
        # All mutually in the ambiguous band.
        vectors = vecs({a.id: [1.0, 0.05 * i] for i, a in enumerate(arts)})
        seen: list[int] = []

        class Counting(HeuristicProvider):
            def same_event(self, pairs, context):
                seen.append(len(pairs))
                return {}

        clustering.cluster(
            arts, vectors, similarity_threshold=0.999, ambiguous_threshold=0.5,
            provider=Counting(), context=Context(), max_checks=3,
        )
        assert seen == [3]

    def test_stats_record_the_checks(self):
        arts, vectors = self._setup()

        class Stats:
            llm_checks = 0

        stats = Stats()

        class Yes(HeuristicProvider):
            def same_event(self, pairs, context):
                return {p.key: True for p in pairs}

        clustering.cluster(
            arts, vectors, similarity_threshold=0.9, ambiguous_threshold=0.7,
            provider=Yes(), context=Context(), stats=stats,
        )
        assert stats.llm_checks == 1


class TestAdjudicationBatching:
    """One request per batch. Sending every pair at once overflowed the context
    window, and the unparseable reply made adjudication a silent no-op."""

    def _mutually_ambiguous(self, n):
        arts = [make_article(f"Story number {i}", source=f"S{i}") for i in range(n)]
        return arts, vecs({a.id: [1.0, 0.05 * i] for i, a in enumerate(arts)})

    def test_pairs_are_split_into_batches(self):
        arts, vectors = self._mutually_ambiguous(6)   # 15 pairs
        sizes: list[int] = []

        class Counting(HeuristicProvider):
            def same_event(self, pairs, context):
                sizes.append(len(pairs))
                return {}                             # no opinion, so nothing merges

        clustering.cluster(
            arts, vectors, similarity_threshold=0.999, ambiguous_threshold=0.5,
            provider=Counting(), context=Context(), max_checks=99, check_batch_size=4,
        )
        assert sizes == [4, 4, 4, 3]
        assert sum(sizes) == 15

    def test_budget_is_respected_across_batches(self):
        arts, vectors = self._mutually_ambiguous(6)
        sizes: list[int] = []

        class Counting(HeuristicProvider):
            def same_event(self, pairs, context):
                sizes.append(len(pairs))
                return {}

        clustering.cluster(
            arts, vectors, similarity_threshold=0.999, ambiguous_threshold=0.5,
            provider=Counting(), context=Context(), max_checks=7, check_batch_size=4,
        )
        assert sizes == [4, 3]

    def test_pairs_merged_by_an_earlier_batch_are_not_asked_again(self):
        """The saving that makes a large budget affordable."""
        arts, vectors = self._mutually_ambiguous(3)   # 3 pairs
        sizes: list[int] = []

        class Yes(HeuristicProvider):
            def same_event(self, pairs, context):
                sizes.append(len(pairs))
                return {p.key: True for p in pairs}

        groups = clustering.cluster(
            arts, vectors, similarity_threshold=0.999, ambiguous_threshold=0.5,
            provider=Yes(), context=Context(), max_checks=99, check_batch_size=1,
        )
        # Two merges join all three; the third pair is redundant and never sent.
        assert sizes == [1, 1]
        assert len(groups) == 1

    def test_one_failed_batch_does_not_end_adjudication(self):
        """A single timeout used to abandon the whole phase, which made
        `max_cluster_checks` meaningless -- measured 2026-09-17, one 90s read
        timeout truncated a 600-pair budget to 64 of 274 pairs and cost 32 merges.
        Now it skips that batch and carries on, like enrichment does.

        The call counter is external because `HeuristicProvider` has its own
        `calls` attribute, which shadows a subclass one.
        """
        arts, vectors = self._mutually_ambiguous(4)   # 6 pairs
        attempts: list[int] = []

        class FailsOnceThenWorks(HeuristicProvider):
            def same_event(self, pairs, context):
                attempts.append(len(pairs))
                if len(attempts) == 1:
                    raise LLMError("read timed out")
                return {p.key: True for p in pairs}

        class Stats:
            llm_checks = 0

        stats = Stats()
        groups = clustering.cluster(
            arts, vectors, similarity_threshold=0.999, ambiguous_threshold=0.5,
            provider=FailsOnceThenWorks(), context=Context(), stats=stats,
            max_checks=99, check_batch_size=2,
        )
        # It kept going past the failure instead of stopping at the first batch.
        assert len(attempts) > 1
        # The failed batch is not billed, the later ones are.
        assert stats.llm_checks > 0
        assert len(groups) < 4

    def test_repeated_failures_do_stop_adjudication(self):
        """Two consecutive failures is still the limit, so a provider that is
        simply down does not cost a request per remaining batch."""
        arts, vectors = self._mutually_ambiguous(4)
        attempts: list[int] = []

        class AlwaysFails(HeuristicProvider):
            def same_event(self, pairs, context):
                attempts.append(len(pairs))
                raise LLMError("ollama unreachable")

        groups = clustering.cluster(
            arts, vectors, similarity_threshold=0.999, ambiguous_threshold=0.5,
            provider=AlwaysFails(), context=Context(), max_checks=99,
            check_batch_size=2,
        )
        assert len(attempts) == clustering.MAX_CONSECUTIVE_FAILURES
        # Nothing merged, and nothing crashed: the embedding's verdict stands.
        assert len(groups) == 4

    def test_a_failure_keeps_the_merges_already_made(self):
        arts, vectors = self._mutually_ambiguous(4)   # 6 pairs

        class FailsOnSecondBatch(HeuristicProvider):
            calls = 0

            def same_event(self, pairs, context):
                FailsOnSecondBatch.calls += 1
                if FailsOnSecondBatch.calls > 1:
                    raise LLMError("context overflow")
                return {p.key: True for p in pairs}

        class Stats:
            llm_checks = 0

        stats = Stats()
        groups = clustering.cluster(
            arts, vectors, similarity_threshold=0.999, ambiguous_threshold=0.5,
            provider=FailsOnSecondBatch(), context=Context(), stats=stats,
            max_checks=99, check_batch_size=2,
        )
        # The first batch's two merges survive rather than being rolled back.
        assert len(groups) < 4
        # Only the batch that actually returned verdicts is billed.
        assert stats.llm_checks == 2


class TestTextFallback:
    def test_near_identical_headlines_merge(self):
        arts = [
            make_article("Rússia ataca amb drons la frontera d'Ucraïna amb Polònia",
                         source="VilaWeb", publisher="VilaWeb", language="ca"),
            make_article("Rússia ataca amb drons la frontera d'Ucraïna amb Polònia",
                         source="Nació Digital", publisher="Nació Digital", language="ca"),
        ]
        assert len(clustering.cluster(arts, {})) == 1

    def test_never_merges_across_languages(self):
        """Cross-language text similarity is noise, so it must not be tried."""
        arts = sanctions_articles()
        groups = clustering.cluster(arts, {})
        assert len(groups) == 3

    def test_unrelated_stay_apart(self):
        arts = [
            make_article("Flooding closes roads in the north", source="A"),
            make_article("Football club appoints new manager", source="B"),
        ]
        assert len(clustering.cluster(arts, {})) == 2

    def test_empty_input(self):
        assert clustering.cluster([], {}) == []


class TestBuildStory:
    def test_collects_publishers_languages_and_scores(self):
        arts = [
            make_article(SANCTIONS[0][0], source="BBC", publisher="BBC", language="en",
                         importance=0.7, relevance=0.6, topics=["world"],
                         entities=["European Union"]),
            make_article(SANCTIONS[1][0], source="El País Internacional",
                         publisher="El País", language="es", importance=0.6,
                         relevance=0.8, topics=["world"], entities=["European Union"]),
        ]
        story = clustering.build_story(arts)
        assert story.publishers == ["BBC", "El País"]
        assert story.languages == ["en", "es"]
        assert story.importance > 0.7      # nudged up by a second publisher
        assert story.relevance == 0.8      # best of the cluster
        assert story.topics == ["world"]
        assert len(story.article_ids) == 2

    def test_id_is_stable_across_rebuilds(self):
        arts = [make_article("Same cluster", source="A"), make_article("Other", source="B")]
        assert clustering.build_story(arts).id == clustering.build_story(list(reversed(arts))).id

    def test_reporting_is_preferred_over_opinion_as_lead(self):
        arts = [
            make_article("Opinion: why the budget fails", source="A", publisher="A",
                         importance=0.9, content_type="opinion"),
            make_article("Parliament approves the budget", source="B", publisher="B",
                         importance=0.5, content_type="reporting"),
        ]
        assert clustering.lead_article(arts).content_type == "reporting"
