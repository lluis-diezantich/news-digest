"""LLM providers, resolved by name from config/env.

Adding a provider: implement `LLMProvider` in a new module and register it in
`PROVIDERS`. Nothing else in the pipeline changes.
"""

from __future__ import annotations

import logging

from ..config import LLMSettings
from .base import (
    Brief,
    BriefInput,
    Enrichment,
    EnrichInput,
    LLMError,
    LLMProvider,
    LLMQuotaError,
)
from .gemini import GeminiProvider
from .heuristic import HeuristicProvider

log = logging.getLogger(__name__)


def _build_gemini(settings: LLMSettings) -> LLMProvider:
    return GeminiProvider(
        api_key=settings.api_key or "",
        model=settings.model,
        timeout=settings.timeout,
        max_retries=settings.max_retries,
    )


PROVIDERS = {
    "gemini": _build_gemini,
    "none": lambda settings: HeuristicProvider(),
}


def get_provider(settings: LLMSettings) -> LLMProvider:
    """Build the configured provider, falling back to offline on any problem."""
    factory = PROVIDERS.get(settings.provider)
    if factory is None:
        log.warning(
            "unknown LLM_PROVIDER=%r (known: %s); using offline enrichment",
            settings.provider,
            ", ".join(sorted(PROVIDERS)),
        )
        return HeuristicProvider()
    try:
        return factory(settings)
    except LLMError as exc:
        log.warning("%s provider unavailable (%s); using offline enrichment", settings.provider, exc)
        return HeuristicProvider()


__all__ = [
    "PROVIDERS",
    "Brief",
    "BriefInput",
    "Enrichment",
    "EnrichInput",
    "GeminiProvider",
    "HeuristicProvider",
    "LLMError",
    "LLMProvider",
    "LLMQuotaError",
    "get_provider",
]
