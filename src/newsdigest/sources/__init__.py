"""Source adapters, resolved by the `method` key in config/sources.yaml."""

from __future__ import annotations

from ..config import Source
from .base import SourceAdapter
from .http import Fetcher, RobotsDisallowed
from .rss import RSSAdapter
from .scrape import ScrapeAdapter

ADAPTERS: dict[str, type[SourceAdapter]] = {
    RSSAdapter.method: RSSAdapter,
    ScrapeAdapter.method: ScrapeAdapter,
}


def adapter_for(source: Source, fetcher: Fetcher) -> SourceAdapter:
    """Look up the adapter for a source. New methods register in ADAPTERS."""
    try:
        return ADAPTERS[source.method](fetcher)
    except KeyError:
        raise ValueError(
            f"{source.name}: no adapter for method {source.method!r}; "
            f"known methods: {', '.join(sorted(ADAPTERS))}"
        ) from None


__all__ = [
    "ADAPTERS",
    "Fetcher",
    "RSSAdapter",
    "RobotsDisallowed",
    "ScrapeAdapter",
    "SourceAdapter",
    "adapter_for",
]
