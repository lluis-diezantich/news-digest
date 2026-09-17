"""Score clustering against hand-labelled pairs.

`tests/fixtures/cluster_eval.json` holds 27 articles from two stories the owner
dissected by hand, grouped into the real events they cover. Two articles in the
same event are a positive, two in different events are a negative, and event
pairs listed as unsure are excluded rather than guessed at.

This exists so a clustering change can be answered with a number instead of an
argument. It runs offline: the fixture carries its own cached vectors, so there
is no embedder, no ollama and no digest run involved, and
`python -m newsdigest.eval` takes about a second.

**It measures the embedding tiers only.** With no `provider` the ambiguous band
is left to the embedding's own verdict, exactly as `cluster()` does when the LLM
is disabled, so recall here is a floor: pairs the LLM would have merged are
counted as misses. That is the honest thing to regression-test, since it is the
part that is deterministic.

`report_thresholds` is the other half. Precision and recall tell you whether a
setting is good; the cosine distribution, split by same-language against
cross-language, tells you *where to put the threshold* -- and it is the only way
to see that a value tuned on Spanish-to-Spanish pairs is far too high for
Spanish-to-Catalan ones.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np

from . import clustering
from .models import Article

FIXTURE = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "cluster_eval.json"


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


@dataclass(frozen=True)
class Pair:
    left: str
    right: str
    same: bool
    #: The two events, for a readable failure message.
    left_event: str
    right_event: str

    def label(self) -> str:
        return "same" if self.same else "different"


@dataclass
class Fixture:
    articles: list[Article]
    vectors: dict[str, np.ndarray]
    events: dict[str, list[str]]
    unsure: set[frozenset[str]]
    titles: dict[str, str] = field(default_factory=dict)
    languages: dict[str, str | None] = field(default_factory=dict)

    def event_of(self, article_id: str) -> str:
        for event, ids in self.events.items():
            if article_id in ids:
                return event
        raise KeyError(article_id)

    def pairs(self) -> list[Pair]:
        """Every labelled pair. Unsure event pairs are dropped, not defaulted."""
        out: list[Pair] = []
        for left, right in combinations(sorted(self.vectors), 2):
            le, re_ = self.event_of(left), self.event_of(right)
            if le != re_ and frozenset({le, re_}) in self.unsure:
                continue
            out.append(Pair(left, right, le == re_, le, re_))
        return out

    def cosine(self, left: str, right: str) -> float:
        return float(self.vectors[left] @ self.vectors[right])

    def cross_language(self, left: str, right: str) -> bool:
        return self.languages[left] != self.languages[right]


def load(path: Path | None = None) -> Fixture:
    raw = json.loads((path or FIXTURE).read_text())

    articles, vectors, titles, languages = [], {}, {}, {}
    for item in raw["articles"]:
        articles.append(Article(
            id=item["id"],
            title=item["title"],
            source=item["source"],
            url=item["url"],
            description=item["description"],
            summary=item["summary"],
            language=item["language"],
            publisher=item["publisher"],
            published_at=_dt(item["published_at"]),
            collected_at=_dt(item["collected_at"]) or datetime.now(),
            source_weight=item["source_weight"],
            content_type=item["content_type"],
        ))
        vectors[item["id"]] = np.frombuffer(
            base64.b64decode(item["vector"]), dtype=np.float32
        )
        titles[item["id"]] = item["title"]
        languages[item["id"]] = item["language"]

    return Fixture(
        articles=articles,
        vectors=vectors,
        events=raw["events"],
        unsure={frozenset(p) for p in raw["unsure_event_pairs"]},
        titles=titles,
        languages=languages,
    )


@dataclass
class Report:
    similarity_threshold: float
    ambiguous_threshold: float
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    #: Pairs merged that should not have been -- the expensive kind, since
    #: merging is transitive and one welds two clusters.
    welded: list[Pair] = field(default_factory=list)
    #: Pairs left apart that belong together.
    split: list[Pair] = field(default_factory=list)
    clusters: int = 0

    @property
    def precision(self) -> float:
        merged = self.true_positives + self.false_positives
        return self.true_positives / merged if merged else 1.0

    @property
    def recall(self) -> float:
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def summary(self) -> str:
        return (
            f"similarity {self.similarity_threshold:.2f} / "
            f"ambiguous {self.ambiguous_threshold:.2f}: "
            f"precision {self.precision:.3f} recall {self.recall:.3f} "
            f"f1 {self.f1:.3f} "
            f"(tp {self.true_positives} fp {self.false_positives} "
            f"fn {self.false_negatives} tn {self.true_negatives}) "
            f"-> {self.clusters} clusters"
        )


def evaluate(
    fixture: Fixture,
    *,
    similarity_threshold: float = 0.80,
    ambiguous_threshold: float = 0.65,
    provider=None,
    context=None,
    max_checks: int = 0,
    check_batch_size: int = 8,
) -> Report:
    """Cluster the fixture and score the result against the labels.

    `max_checks` defaults to 0 so this is deterministic and offline. Pass a real
    provider to measure what LLM adjudication adds on top.
    """
    groups = clustering.cluster(
        fixture.articles,
        fixture.vectors,
        similarity_threshold=similarity_threshold,
        ambiguous_threshold=ambiguous_threshold,
        provider=provider,
        context=context,
        max_checks=max_checks,
        check_batch_size=check_batch_size,
    )
    where = {a.id: n for n, group in enumerate(groups) for a in group}

    report = Report(similarity_threshold, ambiguous_threshold, clusters=len(groups))
    for pair in fixture.pairs():
        merged = where[pair.left] == where[pair.right]
        if pair.same and merged:
            report.true_positives += 1
        elif pair.same:
            report.false_negatives += 1
            report.split.append(pair)
        elif merged:
            report.false_positives += 1
            report.welded.append(pair)
        else:
            report.true_negatives += 1
    return report


def report_thresholds(fixture: Fixture) -> str:
    """Cosine distribution per label, split by within- and cross-language.

    This is the threshold-picking view. A gap between the negatives' ceiling and
    the positives' floor is a threshold that works; an overlap means no single
    number separates them and the band has to be adjudicated instead.
    """
    buckets: dict[tuple[str, str], list[float]] = {}
    for pair in fixture.pairs():
        key = (pair.label(), "cross-language" if fixture.cross_language(pair.left, pair.right) else "same-language")
        buckets.setdefault(key, []).append(fixture.cosine(pair.left, pair.right))

    lines = [f"{'label':<10} {'languages':<15} {'n':>4} {'min':>7} {'p50':>7} {'max':>7}"]
    for key in sorted(buckets):
        values = sorted(buckets[key])
        lines.append(
            f"{key[0]:<10} {key[1]:<15} {len(values):>4} "
            f"{values[0]:>7.3f} {values[len(values) // 2]:>7.3f} {values[-1]:>7.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    fixture = load()
    pairs = fixture.pairs()
    positives = sum(1 for p in pairs if p.same)
    cross = sum(1 for p in pairs if p.same and fixture.cross_language(p.left, p.right))

    print(f"{len(fixture.articles)} articles, {len(fixture.events)} events, "
          f"{len(pairs)} labelled pairs "
          f"({positives} same / {len(pairs) - positives} different, "
          f"{cross} of the positives cross-language)\n")

    print(report_thresholds(fixture), "\n")

    for similarity in (0.90, 0.85, 0.82, 0.80, 0.75, 0.70, 0.65):
        print(evaluate(fixture, similarity_threshold=similarity).summary())

    print()
    report = evaluate(fixture)
    for pair in report.welded:
        print(f"  WELDED  {pair.left_event} + {pair.right_event}  "
              f"cos {fixture.cosine(pair.left, pair.right):.3f}\n"
              f"          {fixture.titles[pair.left][:70]}\n"
              f"          {fixture.titles[pair.right][:70]}")
    for pair in report.split:
        print(f"  SPLIT   {pair.left_event}  "
              f"cos {fixture.cosine(pair.left, pair.right):.3f}"
              f"{'  [cross-language]' if fixture.cross_language(pair.left, pair.right) else ''}\n"
              f"          {fixture.titles[pair.left][:70]}\n"
              f"          {fixture.titles[pair.right][:70]}")


if __name__ == "__main__":
    main()
