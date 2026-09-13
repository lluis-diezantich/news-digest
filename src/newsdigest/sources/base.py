"""Source adapter contract.

An adapter's only job is to turn one configured source into normalized
`Article` objects. It never dedupes, stores, scores or calls an LLM -- and it
raises on failure so the pipeline can record that one source broke and carry on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import Source
from ..models import Article
from .http import Fetcher


class SourceAdapter(ABC):
    """Base for every fetch strategy."""

    method: str = ""

    def __init__(self, fetcher: Fetcher):
        self.fetcher = fetcher

    @abstractmethod
    def fetch(self, source: Source) -> list[Article]:
        """Return normalized articles, newest-first where the source allows."""
        raise NotImplementedError
