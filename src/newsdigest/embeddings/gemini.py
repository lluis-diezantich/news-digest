"""Google Gemini embedding provider.

Uses `batchEmbedContents`, not `embedContent`. That distinction is load-bearing:
passing several contents to `embedContent` returns ONE aggregated vector for all
of them, which would silently produce plausible-looking nonsense clusters rather
than an error. `batchEmbedContents` takes `requests[]` and returns `embeddings[]`
in the same order, and we assert the counts match so any drift fails loudly.

`taskType` is deliberately not sent by default: it is unsupported on
gemini-embedding-2 (task hints belong in the text there) while older models
accept it. Omitting it is valid on both, and symmetric similarity -- which is
what clustering wants -- is the default behaviour anyway.
"""

from __future__ import annotations

import logging
import random
import time

import numpy as np
import requests

from .base import (
    EmbeddingError,
    EmbeddingProvider,
    EmbeddingQuotaError,
    normalize,
)

log = logging.getLogger(__name__)

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
# Requests per HTTP call. The API caps batch size; 50 stays well inside it and
# keeps a single failure from costing much work.
MAX_BATCH = 50


class GeminiEmbeddingProvider(EmbeddingProvider):
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-2",
        *,
        dimensions: int = 256,
        task_type: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        super().__init__()
        if not api_key:
            raise EmbeddingError("gemini embeddings require GEMINI_API_KEY")
        self.model = model
        self.dimensions = dimensions
        self.task_type = task_type
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update(
            {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        )

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), MAX_BATCH):
            chunk = texts[start : start + MAX_BATCH]
            vectors.extend(self._embed_chunk(chunk))
        if len(vectors) != len(texts):  # belt and braces
            raise EmbeddingError(
                f"expected {len(texts)} vectors, assembled {len(vectors)}"
            )
        return vectors

    def _embed_chunk(self, texts: list[str]) -> list[np.ndarray]:
        config: dict = {"outputDimensionality": self.dimensions}
        if self.task_type:
            config["taskType"] = self.task_type

        body = {
            "requests": [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": text}]},
                    "embedContentConfig": config,
                }
                for text in texts
            ]
        }
        url = f"{API_ROOT}/{self.model}:batchEmbedContents"

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                self.calls += 1
                response = self._session.post(url, json=body, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
                self._backoff(attempt)
                continue

            if response.status_code == 429:
                raise EmbeddingQuotaError(
                    f"gemini embedding quota exhausted: {_short(response.text)}"
                )
            if response.status_code in (500, 502, 503, 504):
                last_error = EmbeddingError(f"gemini {response.status_code}")
                self._backoff(attempt)
                continue
            if response.status_code != 200:
                raise EmbeddingError(
                    f"gemini embeddings {response.status_code}: {_short(response.text)}"
                )
            return self._parse(response.json(), expected=len(texts))

        raise EmbeddingError(
            f"gemini embeddings unreachable after {self.max_retries} attempts: {last_error}"
        )

    def _parse(self, payload: dict, *, expected: int) -> list[np.ndarray]:
        raw = payload.get("embeddings")
        if raw is None:
            # A single "embedding" key means we hit the aggregating endpoint --
            # one vector for the whole batch. Never silently accept that.
            if "embedding" in payload:
                raise EmbeddingError(
                    "got one aggregated embedding instead of a batch; "
                    "batchEmbedContents is required for per-text vectors"
                )
            raise EmbeddingError(f"no embeddings in response: {_short(str(payload))}")
        if not isinstance(raw, list) or len(raw) != expected:
            raise EmbeddingError(
                f"asked for {expected} embeddings, got "
                f"{len(raw) if isinstance(raw, list) else type(raw).__name__} -- "
                "refusing to guess which text each vector belongs to"
            )

        vectors: list[np.ndarray] = []
        for index, entry in enumerate(raw):
            values = (entry or {}).get("values")
            if not values:
                raise EmbeddingError(f"embedding {index} has no values")
            vectors.append(normalize(values))
        return vectors

    @staticmethod
    def _backoff(attempt: int) -> None:
        time.sleep(min(30.0, (2**attempt) + random.uniform(0, 0.5)))


def _short(value: str, limit: int = 300) -> str:
    value = " ".join((value or "").split())
    return value[:limit] + ("…" if len(value) > limit else "")
