"""Local embedding provider. No network and no model download.

The encoder is injected, so these cover the contract and the calibration guard
rather than the model's quality -- that was measured separately: on 551 real
articles it produced 442 clusters and 20 cross-language groups against Gemini's
445 and 10, in 8.5s versus roughly five minutes of rate-limited waiting.
"""

import logging

import numpy as np
import pytest

from newsdigest.config import EmbeddingSettings
from newsdigest.embeddings import get_provider
from newsdigest.embeddings.base import EmbeddingError
from newsdigest.embeddings.local import (
    DEFAULT_MODEL, SUGGESTED_AMBIGUOUS, SUGGESTED_SIMILARITY,
    LocalEmbeddingProvider, warn_if_miscalibrated,
)


def fake_encoder(dim=4, scale=1.0):
    """Deterministic vectors: text length decides direction, so equal texts match."""
    def encode(texts):
        for t in texts:
            v = np.zeros(dim, dtype=np.float32)
            v[len(t) % dim] = scale
            v[0] += 0.1
            yield v
    return encode


class TestContract:
    def test_returns_one_vector_per_text(self):
        p = LocalEmbeddingProvider(encoder=fake_encoder())
        assert len(p.embed(["a", "bb", "ccc"])) == 3

    def test_vectors_are_normalized(self):
        p = LocalEmbeddingProvider(encoder=fake_encoder(scale=7.0))
        for v in p.embed(["a", "bb"]):
            assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-5)

    def test_empty_input_makes_no_call(self):
        p = LocalEmbeddingProvider(encoder=fake_encoder())
        assert p.embed([]) == [] and p.calls == 0

    def test_dimensions_are_learned_from_the_output(self):
        """The model dictates the width; a configured value cannot change it."""
        p = LocalEmbeddingProvider(encoder=fake_encoder(dim=8))
        p.embed(["a"])
        assert p.dimensions == 8

    def test_a_short_result_is_an_error_not_a_guess(self):
        def bad(texts):
            yield np.ones(4, dtype=np.float32)      # one vector for three texts
        p = LocalEmbeddingProvider(encoder=bad)
        with pytest.raises(EmbeddingError, match="asked for 3"):
            p.embed(["a", "b", "c"])

    def test_encoder_failure_surfaces_as_an_embedding_error(self):
        def boom(texts):
            raise RuntimeError("onnx exploded")
        p = LocalEmbeddingProvider(encoder=boom)
        with pytest.raises(EmbeddingError, match="local embedding failed"):
            p.embed(["a"])

    def test_cache_key_tracks_the_model(self):
        """Switching model must invalidate cached vectors, not silently mix them."""
        a = LocalEmbeddingProvider(model="m1", encoder=fake_encoder(), dimensions=4)
        b = LocalEmbeddingProvider(model="m2", encoder=fake_encoder(), dimensions=4)
        assert a.cache_key() != b.cache_key()

    def test_it_is_available(self):
        assert LocalEmbeddingProvider(encoder=fake_encoder()).available


class TestRegistry:
    def test_selected_by_name(self):
        p = get_provider(EmbeddingSettings(provider="local"))
        assert p.name == "local"

    def test_a_gemini_model_name_is_not_carried_over(self):
        """EMBEDDING_MODEL left at a gemini value means nothing to this provider."""
        p = get_provider(EmbeddingSettings(provider="local", model="gemini-embedding-2"))
        assert p.model == DEFAULT_MODEL


class TestCalibrationGuard:
    """Cosine scales differ per model; a stale threshold is a silent regression."""

    def test_warns_on_a_gemini_shaped_threshold(self, caplog):
        with caplog.at_level(logging.WARNING):
            warn_if_miscalibrated(0.82, 0.72)
        assert "looks tuned for gemini" in caplog.text

    def test_quiet_at_the_measured_thresholds(self, caplog):
        with caplog.at_level(logging.WARNING):
            warn_if_miscalibrated(SUGGESTED_SIMILARITY, SUGGESTED_AMBIGUOUS)
        assert caplog.text == ""

    def test_warns_when_the_ambiguous_band_is_empty(self, caplog):
        with caplog.at_level(logging.WARNING):
            warn_if_miscalibrated(0.70, 0.70)
        assert "band is empty" in caplog.text

    def test_the_guard_runs_when_the_provider_is_built(self, caplog):
        with caplog.at_level(logging.WARNING):
            get_provider(EmbeddingSettings(provider="local", similarity_threshold=0.82,
                                           ambiguous_threshold=0.72))
        assert "looks tuned for gemini" in caplog.text
