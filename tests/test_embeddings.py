"""Embedding provider tests against a stubbed transport."""

import json

import numpy as np
import pytest

from conftest import PER_DAY_429, PER_MINUTE_429

from newsdigest.config import EmbeddingSettings
from newsdigest.embeddings import get_provider
from newsdigest.embeddings.base import (
    EmbeddingError, EmbeddingQuotaError, cosine_matrix, from_blob, normalize, to_blob,
)
from newsdigest.embeddings.gemini import GeminiEmbeddingProvider
from newsdigest.embeddings.none import NullEmbeddingProvider


class StubResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = payload if isinstance(payload, str) else json.dumps(payload)

    def json(self):
        return self._payload


class StubSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.posts = []
        self.urls = []
        self.headers = {}

    def post(self, url, json=None, timeout=None):
        self.posts.append(json)
        self.urls.append(url)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def provider_with(*responses, **kwargs) -> GeminiEmbeddingProvider:
    # Rate-limit waiting is off unless a test asks for it, so no test sleeps.
    kwargs.setdefault("max_rate_limit_retries", 0)
    provider = GeminiEmbeddingProvider(api_key="k", model="gemini-embedding-2",
                                       max_retries=1, **kwargs)
    provider._session = StubSession(*responses)
    return provider


def embeddings_payload(count, dims=4):
    return {"embeddings": [{"values": [1.0 * (i + 1)] + [0.0] * (dims - 1)}
                           for i in range(count)]}


class TestGeminiEmbeddings:
    def test_uses_the_batch_endpoint_with_one_request_per_text(self):
        """embedContent would return ONE aggregated vector for the whole batch."""
        provider = provider_with(StubResponse(200, embeddings_payload(3)))
        provider.embed(["a", "b", "c"])
        assert provider._session.urls[0].endswith(":batchEmbedContents")
        body = provider._session.posts[0]
        assert len(body["requests"]) == 3
        assert body["requests"][0]["content"]["parts"][0]["text"] == "a"
        assert body["requests"][0]["embedContentConfig"]["outputDimensionality"] == 256

    def test_task_type_is_omitted_by_default(self):
        """Unsupported on gemini-embedding-2; omitting it is valid on both."""
        provider = provider_with(StubResponse(200, embeddings_payload(1)))
        provider.embed(["a"])
        assert "taskType" not in provider._session.posts[0]["requests"][0]["embedContentConfig"]

    def test_task_type_is_sent_when_configured(self):
        provider = provider_with(StubResponse(200, embeddings_payload(1)),
                                 task_type="SEMANTIC_SIMILARITY")
        provider.embed(["a"])
        config = provider._session.posts[0]["requests"][0]["embedContentConfig"]
        assert config["taskType"] == "SEMANTIC_SIMILARITY"

    def test_vectors_are_normalized(self):
        provider = provider_with(StubResponse(200, embeddings_payload(2)))
        for vector in provider.embed(["a", "b"]):
            assert pytest.approx(1.0, abs=1e-5) == float(np.linalg.norm(vector))

    def test_aggregated_response_is_rejected_not_guessed(self):
        """The silent-corruption trap: one vector for many texts."""
        provider = provider_with(StubResponse(200, {"embedding": {"values": [1, 0, 0, 0]}}))
        with pytest.raises(EmbeddingError, match="aggregated"):
            provider.embed(["a", "b", "c"])

    def test_count_mismatch_is_rejected(self):
        provider = provider_with(StubResponse(200, embeddings_payload(2)))
        with pytest.raises(EmbeddingError, match="asked for 3"):
            provider.embed(["a", "b", "c"])

    def test_empty_values_are_rejected(self):
        provider = provider_with(StubResponse(200, {"embeddings": [{"values": []}]}))
        with pytest.raises(EmbeddingError, match="no values"):
            provider.embed(["a"])

    def test_rate_limit_raises_quota_error_with_no_wait_budget(self):
        provider = provider_with(StubResponse(429, {"error": {"message": "quota"}}))
        with pytest.raises(EmbeddingQuotaError):
            provider.embed(["a"])

    def test_per_minute_limit_is_waited_out_then_retried(self, no_sleep):
        provider = provider_with(
            StubResponse(429, PER_MINUTE_429),
            StubResponse(200, embeddings_payload(1)),
            max_rate_limit_retries=2,
        )
        assert len(provider.embed(["a"])) == 1
        assert len(no_sleep) == 1

    def test_per_day_limit_is_terminal_and_never_sleeps(self, no_sleep):
        provider = provider_with(StubResponse(429, PER_DAY_429), max_rate_limit_retries=5)
        with pytest.raises(EmbeddingQuotaError, match="daily quota"):
            provider.embed(["a"])
        assert no_sleep == []

    def test_http_error_raises(self):
        provider = provider_with(StubResponse(400, {"error": {"message": "bad"}}))
        with pytest.raises(EmbeddingError, match="400"):
            provider.embed(["a"])

    def test_large_input_is_split_into_several_calls(self):
        provider = provider_with(StubResponse(200, embeddings_payload(50)))
        provider._session.responses = [StubResponse(200, embeddings_payload(50)),
                                       StubResponse(200, embeddings_payload(10))]
        assert len(provider.embed([f"t{i}" for i in range(60)])) == 60
        assert len(provider._session.posts) == 2

    def test_empty_input_makes_no_call(self):
        provider = provider_with(StubResponse(200, embeddings_payload(1)))
        assert provider.embed([]) == []
        assert provider.calls == 0

    def test_missing_key_is_rejected(self):
        with pytest.raises(EmbeddingError):
            GeminiEmbeddingProvider(api_key="")

    def test_cache_key_includes_model_and_dimensions(self):
        provider = provider_with(StubResponse(200, embeddings_payload(1)))
        assert provider.cache_key() == "gemini:gemini-embedding-2:256"


class TestVectorHelpers:
    def test_blob_roundtrip(self):
        vector = normalize(np.array([1.0, 2.0, 3.0]))
        assert np.allclose(from_blob(to_blob(vector)), vector)

    def test_cosine_matrix_is_dot_product_for_unit_vectors(self):
        a, b = normalize(np.array([1.0, 0.0])), normalize(np.array([0.0, 1.0]))
        matrix = cosine_matrix([a, b, a])
        assert pytest.approx(1.0, abs=1e-6) == matrix[0, 2]
        assert pytest.approx(0.0, abs=1e-6) == matrix[0, 1]

    def test_zero_vector_normalizes_without_dividing_by_zero(self):
        assert not np.any(normalize(np.array([0.0, 0.0])))

    def test_empty_matrix(self):
        assert cosine_matrix([]).shape == (0, 0)


class TestRegistry:
    def test_missing_key_falls_back_to_null(self):
        provider = get_provider(EmbeddingSettings(provider="gemini", api_key=None))
        assert isinstance(provider, NullEmbeddingProvider)
        assert provider.available is False

    def test_unknown_provider_falls_back_to_null(self):
        assert get_provider(EmbeddingSettings(provider="nope")).available is False

    def test_gemini_is_built_when_configured(self):
        provider = get_provider(
            EmbeddingSettings(provider="gemini", api_key="k", dimensions=768)
        )
        assert isinstance(provider, GeminiEmbeddingProvider)
        assert provider.dimensions == 768

    def test_null_provider_returns_nothing(self):
        assert NullEmbeddingProvider().embed(["a", "b"]) == []
