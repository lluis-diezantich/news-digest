"""No-op embedding provider.

Used when no key is configured and by the tests. It reports itself unavailable,
which makes clustering fall back to within-language text similarity -- the same
degradation the pipeline already applies when the LLM is missing. Cross-language
stories will stay split; that is visible in the logs and in the digest, and is
strictly better than failing the run.
"""

from __future__ import annotations

import numpy as np

from .base import EmbeddingProvider


class NullEmbeddingProvider(EmbeddingProvider):
    name = "none"
    model = "none"
    dimensions = 0
    available = False

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        return []
