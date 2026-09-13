"""Provider-agnostic embedding contract.

Embeddings exist for one job: deciding that an English, a Spanish and a Catalan
article describe the same event. Token overlap cannot do that -- measured on the
same headline in three languages it scores 0.00 (en/es) and 0.06 (en/ca) against
a 0.60 merge threshold -- so this is the component clustering actually rests on.

Vectors are L2-normalized on the way out, which makes cosine similarity a plain
dot product everywhere downstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingError(RuntimeError):
    """Any provider failure. Clustering degrades rather than the run failing."""


class EmbeddingQuotaError(EmbeddingError):
    """Rate limited or out of quota -- stop asking for the rest of this run."""


class EmbeddingProvider(ABC):
    """Implement `embed` to add a provider."""

    name: str = ""
    model: str = ""
    dimensions: int = 0
    #: False means "no vectors from me"; clustering then falls back to
    #: within-language text similarity and says so in the logs.
    available: bool = True

    def __init__(self) -> None:
        self.calls = 0

    @abstractmethod
    def embed(self, texts: list[str]) -> list[np.ndarray]:
        """One L2-normalized vector per input text, in the same order.

        Implementations MUST return exactly as many vectors as they were given.
        Raise EmbeddingQuotaError to stop embedding for this run.
        """

    def cache_key(self) -> str:
        """Vectors from different models are not comparable, so the cache is
        keyed by provider, model and dimensionality as well as content."""
        return f"{self.name}:{self.model}:{self.dimensions}"


def normalize(values: list[float] | np.ndarray) -> np.ndarray:
    """L2-normalize to unit length, as float32."""
    vector = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or not np.isfinite(norm):
        return np.zeros_like(vector)
    return vector / norm


def cosine_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    """Pairwise cosine similarity for already-normalized vectors.

    One matmul rather than a Python double loop: at a week's volume (~2000
    articles, 2M pairs) the loop takes about 12 seconds and grows quadratically,
    this takes milliseconds.
    """
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    matrix = np.vstack(vectors).astype(np.float32)
    return matrix @ matrix.T


def to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
