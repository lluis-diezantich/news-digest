from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from newsdigest.config import Config, Interests, LLMSettings, Source  # noqa: E402
from newsdigest.llm.heuristic import HeuristicProvider  # noqa: E402
from newsdigest.models import Article, utcnow  # noqa: E402


def make_article(
    title: str,
    source: str = "Example",
    url: str | None = None,
    *,
    hours_ago: float = 1.0,
    description: str = "",
    topics: list[str] | None = None,
    entities: list[str] | None = None,
    importance: float | None = None,
    event_label: str | None = None,
    weight: float = 1.0,
) -> Article:
    slug = "".join(c if c.isalnum() else "-" for c in title.lower())[:50]
    article = Article(
        title=title,
        source=source,
        url=url or f"https://{source.lower().replace(' ', '')}.example/{slug}",
        published_at=utcnow() - timedelta(hours=hours_ago),
        description=description or f"{title}. Some background detail about it.",
        source_weight=weight,
    )
    if topics is not None:
        article.topics = topics
    if entities is not None:
        article.entities = entities
    if importance is not None:
        article.importance = importance
    if event_label is not None:
        article.event_label = event_label
    if importance is not None or topics is not None:
        article.summary = article.summary or article.description
        article.enriched_at = utcnow()
    return article


@pytest.fixture
def config(tmp_path) -> Config:
    return Config(
        sources=[Source(name="Example", rss="https://example.invalid/rss")],
        interests=Interests(
            topics={"technology": 1.5, "sports": 0.2},
            keywords=["anthropic"],
            mute=["horoscope"],
        ),
        llm=LLMSettings(provider="none"),
    )


@pytest.fixture
def store(tmp_path):
    from newsdigest.store import Store

    with Store(tmp_path / "test.db") as s:
        yield s


class LabelingProvider(HeuristicProvider):
    """Offline enrichment plus the one thing only a real LLM supplies.

    `event_label` is the primary clustering signal, and heuristic labels derived
    from a headline cannot agree across differently-worded coverage. This stub
    assigns the label a real model would, so tests exercise the clustering path
    the production pipeline actually takes.
    """

    name = "labeling"

    #: keyword found in the title -> event label the model would return
    RULES = {
        "budget": "national budget vote",
        "parliament": "national budget vote",
        "chip": "chipmaker product launch",
        "processor": "chipmaker product launch",
    }

    def enrich(self, items):
        results = super().enrich(items)
        by_id = {item.id: item for item in items}
        for enrichment in results:
            title = by_id[enrichment.id].title.lower()
            for keyword, label in self.RULES.items():
                if keyword in title:
                    enrichment.event_label = label
                    break
        return results


@pytest.fixture
def labeling_provider():
    return LabelingProvider()
