"""Local embeddings via fastembed (ONNX), no API and no quota.

Cross-lingual sentence embedding is a small-model problem, and this is the half
of the pipeline where a local model is not a compromise. Measured on 551
articles from one real week, against Gemini on the same set:

    provider                clusters  cross-language  seconds
    gemini-embedding-2 256d      445              10     ~300 (mostly 429 waits)
    MiniLM-L12-v2      384d      442              20        8.5

Same cluster count, twice the cross-language grouping, and no rate limit to
sleep through. Embeddings were the half competing with the LLM for one free-tier
budget; moving them here gives the model calls the whole quota.

`fastembed` is an optional dependency: ONNX runtime rather than PyTorch, so the
CI image stays small. It is imported lazily so the package still works without
it, falling back the same way a missing API key does.

THRESHOLDS ARE MODEL-SPECIFIC. Cosine values are not comparable between
embedding models, so the Gemini-tuned 0.82/0.72 under-merges badly here (522
clusters, 9 cross-language). The measured sweet spot for MiniLM is 0.70/0.60;
below 0.60 it collapses into one 300-article blob. `_warn_if_miscalibrated`
shouts if the config still looks like the Gemini defaults.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable

import numpy as np

from .base import EmbeddingError, EmbeddingProvider, normalize

log = logging.getLogger(__name__)

#: Small enough for a CI runner (0.22 GB) and trained for cross-lingual sentence
#: similarity across 50+ languages, Catalan included.
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

#: Measured on one real week; see the module docstring.
SUGGESTED_SIMILARITY = 0.70
SUGGESTED_AMBIGUOUS = 0.60
#: Above this, the configured threshold is almost certainly a Gemini leftover.
_MISCALIBRATED_ABOVE = 0.78


class LocalEmbeddingProvider(EmbeddingProvider):
    name = "local"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        encoder: Callable[[list[str]], Iterable[np.ndarray]] | None = None,
        dimensions: int = 0,
    ):
        super().__init__()
        self.model = model or DEFAULT_MODEL
        self._encoder = encoder
        # The model dictates the width; a configured `dimensions` cannot change
        # it. Recorded once known so `cache_key` changes if the model does.
        self.dimensions = dimensions

    # -- lazily built so importing the package never needs fastembed ---------

    def _ensure(self) -> Callable[[list[str]], Iterable[np.ndarray]]:
        if self._encoder is not None:
            return self._encoder
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise EmbeddingError(
                "the local embedding provider needs fastembed: "
                "pip install 'news-digest[local]'"
            ) from exc
        log.info("loading %s (first run downloads it, ~0.22 GB)", self.model)
        embedder = TextEmbedding(self.model)
        if not self.dimensions:
            for spec in TextEmbedding.list_supported_models():
                if spec.get("model") == self.model:
                    self.dimensions = int(spec.get("dim") or 0)
                    break
        self._encoder = embedder.embed
        return self._encoder

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        encode = self._ensure()
        self.calls += 1
        try:
            produced = [np.asarray(v, dtype=np.float32) for v in encode(texts)]
        except Exception as exc:  # a local model failing is still a provider error
            raise EmbeddingError(f"local embedding failed: {exc}") from exc
        if len(produced) != len(texts):
            raise EmbeddingError(
                f"asked for {len(texts)} embeddings, got {len(produced)}"
            )
        if not self.dimensions and produced:
            self.dimensions = int(produced[0].size)
        return [normalize(v) for v in produced]


def warn_if_miscalibrated(similarity: float, ambiguous: float) -> None:
    """Say so loudly when the thresholds still look Gemini-shaped.

    Silence here would be a quiet quality regression: at 0.82 this model finds
    9 cross-language clusters where 0.70 finds 20, and nothing in the output
    would say why.
    """
    if similarity >= _MISCALIBRATED_ABOVE:
        log.warning(
            "EMBEDDING_SIMILARITY_THRESHOLD=%.2f looks tuned for gemini; on %s "
            "that under-merges (measured 522 clusters / 9 cross-language, against "
            "442 / 20 at %.2f). Consider %.2f with ambiguous %.2f.",
            similarity, DEFAULT_MODEL.split("/")[-1],
            SUGGESTED_SIMILARITY, SUGGESTED_SIMILARITY, SUGGESTED_AMBIGUOUS,
        )
    if ambiguous >= similarity:
        log.warning(
            "ambiguous threshold %.2f is not below similarity %.2f; the "
            "LLM-adjudicated band is empty", ambiguous, similarity,
        )
