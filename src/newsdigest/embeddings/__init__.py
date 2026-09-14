"""Embedding providers, resolved by name from config/env."""

from __future__ import annotations

import logging

from ..config import EmbeddingSettings
from .base import (
    EmbeddingError,
    EmbeddingProvider,
    EmbeddingQuotaError,
    cosine_matrix,
    from_blob,
    normalize,
    to_blob,
)
from .gemini import GeminiEmbeddingProvider
from .none import NullEmbeddingProvider

log = logging.getLogger(__name__)


def _build_gemini(settings: EmbeddingSettings) -> EmbeddingProvider:
    return GeminiEmbeddingProvider(
        api_key=settings.api_key or "",
        model=settings.model,
        dimensions=settings.dimensions,
        task_type=settings.task_type,
        max_rate_limit_retries=settings.max_rate_limit_retries,
        max_rate_limit_wait=settings.max_rate_limit_wait,
    )


PROVIDERS = {
    "gemini": _build_gemini,
    "none": lambda settings: NullEmbeddingProvider(),
}


def get_provider(settings: EmbeddingSettings) -> EmbeddingProvider:
    """Build the configured provider, falling back to the null one on any problem."""
    factory = PROVIDERS.get(settings.provider)
    if factory is None:
        log.warning(
            "unknown EMBEDDING_PROVIDER=%r (known: %s); clustering will be "
            "within-language only",
            settings.provider,
            ", ".join(sorted(PROVIDERS)),
        )
        return NullEmbeddingProvider()
    try:
        return factory(settings)
    except EmbeddingError as exc:
        log.warning(
            "%s embeddings unavailable (%s); clustering will be within-language only",
            settings.provider, exc,
        )
        return NullEmbeddingProvider()


__all__ = [
    "PROVIDERS",
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingQuotaError",
    "GeminiEmbeddingProvider",
    "NullEmbeddingProvider",
    "cosine_matrix",
    "from_blob",
    "get_provider",
    "normalize",
    "to_blob",
]
