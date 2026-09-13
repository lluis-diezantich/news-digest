from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from newsdigest.config import (  # noqa: E402
    Config, DigestSettings, EmbeddingSettings, LLMSettings, Preferences,
    Settings, Source, StorageSettings,
)
from newsdigest.embeddings.base import EmbeddingProvider, normalize  # noqa: E402
from newsdigest.llm.base import Context  # noqa: E402
from newsdigest.llm.heuristic import HeuristicProvider  # noqa: E402
from newsdigest.models import Article, utcnow  # noqa: E402

# A fixed "now" keeps recency-sensitive assertions stable.
NOW = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)


def make_article(
    title: str,
    source: str = "Example",
    url: str | None = None,
    *,
    language: str | None = "en",
    publisher: str | None = None,
    hours_ago: float = 1.0,
    description: str = "",
    topics: list[str] | None = None,
    entities: list[str] | None = None,
    importance: float | None = None,
    relevance: float | None = None,
    content_type: str | None = None,
    weight: float = 1.0,
    published: datetime | None = None,
) -> Article:
    slug = "".join(c if c.isalnum() else "-" for c in title.lower())[:50]
    article = Article(
        title=title,
        source=source,
        publisher=publisher or source,
        url=url or f"https://{source.lower().replace(' ', '')}.example/{slug}",
        published_at=published or (utcnow() - timedelta(hours=hours_ago)),
        # Long enough to clear text.MIN_EXCERPT_CHARS, so tests exercise the
        # blended similarity path rather than the title-only shortcut.
        description=description
        or f"{title}. Reported at length with enough background detail that the "
           f"excerpt carries comparable signal for similarity purposes.",
        language=language,
        source_weight=weight,
    )
    if topics is not None:
        article.topics = topics
    if entities is not None:
        article.entities = entities
    if importance is not None:
        article.importance = importance
    if relevance is not None:
        article.relevance = relevance
    if content_type is not None:
        article.content_type = content_type
    if importance is not None or topics is not None:
        article.summary = article.description
        article.enriched_at = utcnow()
    return article


class StubEmbedder(EmbeddingProvider):
    """Deterministic embeddings from a caller-supplied event map.

    Articles mapped to the same event get near-identical vectors, so tests can
    exercise the cross-language clustering path without an API key. It tests the
    mechanism, never embedding quality.
    """

    name = "stub"
    model = "stub-v1"
    dimensions = 8

    def __init__(self, events: dict[str, str] | None = None, *, jitter: float = 0.02):
        super().__init__()
        self.events = events or {}
        self.jitter = jitter
        self.embedded: list[str] = []

    def _vector(self, text: str) -> np.ndarray:
        event = next((e for key, e in self.events.items() if key.lower() in text.lower()), None)
        seed = abs(hash(event or text)) % (2**32)
        # Normal, not uniform. Two random all-positive vectors have a cosine
        # around 0.8 in low dimensions, so a uniform stub made every pair look
        # like the same event and cross-language tests passed for the wrong
        # reason. Zero-centred vectors are near-orthogonal as intended.
        base = np.random.default_rng(seed).standard_normal(self.dimensions)
        if event is not None:
            # Same event -> same base, plus a nudge so vectors are not identical.
            base = base + np.random.default_rng(
                abs(hash(text)) % 1000
            ).standard_normal(self.dimensions) * self.jitter
        return normalize(base)

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        self.calls += 1
        self.embedded.extend(texts)
        return [self._vector(t) for t in texts]


@pytest.fixture
def context() -> Context:
    return Context(
        output_language="en",
        interests=["technology", "ai", "economics"],
        excluded_topics=["sports"],
    )


@pytest.fixture
def config() -> Config:
    return Config(
        sources=[Source(name="Example", rss="https://example.invalid/rss", languages=["en"])],
        settings=Settings(supported_languages=["en", "es", "ca"], output_language="en"),
        preferences=Preferences(
            topics={"technology": 1.5, "ai": 1.6, "economics": 1.3},
            excluded_topics=["sports"],
            preferred_sources=["BBC World"],
            keywords=["anthropic"],
        ),
        digest=DigestSettings(max_stories=10, archive=True),
        storage=StorageSettings(),
        llm=LLMSettings(provider="none", articles_per_run=100),
        embeddings=EmbeddingSettings(provider="none"),
    )


@pytest.fixture
def store(tmp_path):
    from newsdigest.store import Store

    with Store(tmp_path / "test.db") as s:
        yield s


@pytest.fixture
def heuristic() -> HeuristicProvider:
    return HeuristicProvider()
