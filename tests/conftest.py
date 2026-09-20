from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from newsdigest.config import (  # noqa: E402
    Config, DigestSettings, EmbeddingSettings, FilterSettings, LLMSettings,
    NewsletterSource, Preferences, RegionSettings, Settings, StorageSettings,
)
from newsdigest.embeddings.base import EmbeddingProvider, normalize  # noqa: E402
from newsdigest.llm.base import Context  # noqa: E402
from newsdigest.llm.heuristic import HeuristicProvider  # noqa: E402
from newsdigest.models import Article, Email, utcnow  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """No test may see, or leak, process environment.

    `load_config` calls `load_dotenv`, which copies the repository's `.env` into
    `os.environ` -- by design, so a local run matches Actions. The side effect is
    that one test touching `load_config` hands the developer's own settings to
    every test that runs after it. That is how `LLM_MAX_CLUSTER_CHECKS=600` in a
    local `.env` came to fail an assertion about the default of 40, in a test that
    passed perfectly well on its own.

    Autouse, because the hazard is invisible at the call site: the polluting test
    and the failing test are in different files.
    """
    for name in list(os.environ):
        if name.startswith((
            "LLM_", "EMBEDDING_", "NEWS_EMAIL_", "NEWS_DIGEST_", "GEMINI_",
            "MAILBOX_", "FASTEMBED_", "SITE_URL",
        )):
            monkeypatch.delenv(name, raising=False)
    # A test that wants .env behaviour can call load_dotenv itself.
    monkeypatch.setattr("newsdigest.config.load_dotenv", lambda *a, **k: None)

# A fixed "now" keeps recency-sensitive assertions stable.
NOW = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)

# Google answers both a per-minute and a per-day limit with 429 and tells them
# apart only inside `error.details`. These are the two shapes, trimmed.
PER_MINUTE_429 = {
    "error": {
        "code": 429,
        "message": "Quota exceeded for quota metric 'Generate requests per minute'.",
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [{
                    "quotaMetric": "generativelanguage.googleapis.com/generate_requests",
                    "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                }],
            },
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "31s"},
        ],
    }
}

PER_DAY_429 = {
    "error": {
        "code": 429,
        "message": "Quota exceeded for quota metric 'Generate requests per day'.",
        "status": "RESOURCE_EXHAUSTED",
        "details": [{
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{
                "quotaMetric": "generativelanguage.googleapis.com/generate_requests",
                "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
            }],
        }],
    }
}


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
    newsletter: str = "",
    email_id: str = "",
    position: int = -1,
    of: int = 0,
    region: str = "",
    newsworthy: bool | None = None,
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
        newsletter=newsletter or f"{source} newsletter",
        email_id=email_id,
        item_position=position,
        item_count=of,
        region=region,
        newsworthy=newsworthy,
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


def make_email(
    source: str = "Example",
    *,
    subject: str = "The week in review",
    sender: str = "news@example.invalid",
    html: str = "",
    text: str = "",
    hours_ago: float = 1.0,
    message_id: str | None = None,
) -> Email:
    return Email(
        message_id=message_id or f"<{abs(hash(subject)) % 10**12}@example.invalid>",
        source=source,
        newsletter=f"{source} newsletter",
        subject=subject,
        sender=sender,
        sender_name=source,
        received_at=utcnow() - timedelta(hours=hours_ago),
        html_body=html,
        text_body=text,
    )


@pytest.fixture
def source() -> NewsletterSource:
    return NewsletterSource(
        name="Example",
        senders=["news@example.invalid"],
        publisher="Example",
        languages=["en"],
    )


@pytest.fixture
def config() -> Config:
    return Config(
        sources=[
            NewsletterSource(
                name="Example",
                senders=["news@example.invalid"],
                languages=["en"],
            )
        ],
        settings=Settings(supported_languages=["en", "es"], output_language="en"),
        preferences=Preferences(
            topics={"technology": 1.5, "ai": 1.6, "economics": 1.3},
            excluded_topics=["sports"],
            preferred_sources=["BBC World"],
            keywords=["anthropic"],
        ),
        # classify off by default: most tests are about a later stage, and a
        # classification pass in between would make their article sets depend on
        # the offline keyword matcher.
        filters=FilterSettings(classify=False),
        regions=RegionSettings(),
        digest=DigestSettings(max_stories=10, minor_stories=3, min_articles=1),
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


@pytest.fixture
def no_sleep(monkeypatch) -> list[float]:
    """Record rate-limit waits instead of taking them.

    Returns the list of requested delays, so a test can assert both how many
    waits happened and how long they were without spending that long.
    """
    from newsdigest import ratelimit

    delays: list[float] = []
    monkeypatch.setattr(ratelimit, "sleep", delays.append)
    return delays
