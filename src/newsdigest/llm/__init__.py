"""LLM providers, resolved by name from config/env.

Adding a provider: implement `LLMProvider` in a new module and register it in
`PROVIDERS`. Nothing else in the pipeline changes.
"""

from __future__ import annotations

import logging

from ..config import LLMSettings, Preferences, Settings
from .base import (
    Brief,
    BriefInput,
    Context,
    Enrichment,
    EnrichInput,
    LLMError,
    LLMProvider,
    LLMQuotaError,
    PairInput,
)
from .gemini import GeminiProvider
from .heuristic import HeuristicProvider
from .ollama import OllamaProvider

log = logging.getLogger(__name__)


def _build_gemini(settings: LLMSettings) -> LLMProvider:
    return GeminiProvider(
        api_key=settings.api_key or "",
        model=settings.model,
        timeout=settings.timeout,
        max_retries=settings.max_retries,
        max_rate_limit_retries=settings.max_rate_limit_retries,
        max_rate_limit_wait=settings.max_rate_limit_wait,
        thinking_level=settings.thinking_level,
    )


def _build_ollama(settings: LLMSettings) -> LLMProvider:
    from .ollama import DEFAULT_MODEL

    model = settings.model
    if model.startswith("gemini"):
        # Carried over from the other provider; that name means nothing here.
        log.info("LLM_MODEL=%r is a gemini model; using %s", model, DEFAULT_MODEL)
        model = DEFAULT_MODEL
    return OllamaProvider(
        model=model,
        base_url=settings.base_url,
        timeout=settings.timeout,
        num_ctx=settings.num_ctx,
    )


PROVIDERS = {
    "gemini": _build_gemini,
    "ollama": _build_ollama,
    "none": lambda settings: HeuristicProvider(),
}


def get_provider(settings: LLMSettings) -> LLMProvider:
    """Build the configured provider, falling back to offline on any problem."""
    factory = PROVIDERS.get(settings.provider)
    if factory is None:
        log.warning(
            "unknown LLM_PROVIDER=%r (known: %s); using offline enrichment",
            settings.provider, ", ".join(sorted(PROVIDERS)),
        )
        return HeuristicProvider()
    try:
        return factory(settings)
    except LLMError as exc:
        log.warning(
            "%s provider unavailable (%s); using offline enrichment",
            settings.provider, exc,
        )
        return HeuristicProvider()


def build_context(settings: Settings, preferences: Preferences) -> Context:
    """Turn config into the per-run context the providers need."""
    ranked = sorted(preferences.topics.items(), key=lambda kv: -kv[1])
    return Context(
        output_language=settings.output_language,
        interests=[name for name, _ in ranked[:8]],
        excluded_topics=list(preferences.excluded_topics),
    )


__all__ = [
    "PROVIDERS",
    "Brief",
    "BriefInput",
    "Context",
    "Enrichment",
    "EnrichInput",
    "GeminiProvider",
    "HeuristicProvider",
    "LLMError",
    "LLMProvider",
    "LLMQuotaError",
    "PairInput",
    "build_context",
    "get_provider",
]
