"""Themes: what the week was *about*, keyed on the names in the headlines.

A deliberately different unit from a story. `clustering.py` groups articles that
report the SAME EVENT; this groups articles that concern the SAME SUBJECT across
the whole window, however different the events.

Both are needed because the digest could not see half of one measured week.
On 2026-W38 the Iran war's 39 headlines sat in 31 separate event clusters, 27 of
them a single outlet, the largest reaching two publishers -- so nothing about it
could ever clear the digest's ranking, and the week's biggest international story
was invisible. Keyed on `iran`, the same articles are one subject with 7 outlets
across three languages. Ceuta went the other way: 161 articles in 67 clusters
(11, 9, 8 and 8 publishers at the top), which is four digest entries for one
subject.

NO MODEL AND NO EMBEDDINGS. This is a proper-noun regex over stored headlines and
some set arithmetic, so it runs in about a second, cannot be degraded by a missing
API key or an exhausted quota, and reads the same whether or not the week has been
enriched. That is the whole appeal: it answers "what was this week about" from
data collection alone.

WHAT WAS TRIED AND REJECTED, measured on 2026-W38:

  * Clustering the CLUSTERS by embedding centroid, at every threshold from 0.50
    to 0.70. It does not work, in both directions at once: at 0.50 the largest
    theme swallows 576 of 1365 articles and mixes Ceuta with AI regulation, while
    Iran is STILL split across 13 themes. Its articles -- Hormuz shipping, visa
    bans, oil futures, Yemen -- are not semantically one thing, which is exactly
    why the event clusterer was right to separate them and why a name, not a
    vector, is the thing that unites them.

  * A coherence floor to detect container words automatically, on the theory that
    a place has diverse articles and a subject does not. The numbers refute it:
    `catalunya` scores 0.588 and `trump` 0.566, `madrid` 0.632 and `iran` 0.624.
    Any floor that removes the places removes the subjects too. Hence
    `containers` is a hand-maintained list in config -- judgement, openly, rather
    than a metric that does not exist.

  * Symmetric Jaccard for merging keys, to stop a big key absorbing a small one.
    It fixed that (`oasis barcelona` stopped being swallowed by `barcelona`) and
    broke something worse: Ceuta split into three themes of 161, 31 and 14. The
    min-overlap rule below keeps it as one. No single threshold does both.

  * Feeding themes back into the digest as a one-story-per-theme diversity cap.
    Measured: it removed the Trump tariff threat, the Fed's rate rise AND
    Carney's EU visit from the top eight -- three unrelated international stories
    that entity-keying all files under `trump` -- and replaced them with a schools
    announcement and a UK-independence opinion piece. `trump` is a container in
    exactly the way `catalunya` is, and unlike a place you cannot enumerate those
    in advance. So this module is a reading tool, not a ranking input.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .models import Article
from .text import normalize

#: Words that are never a theme on their own: articles, prepositions, and the
#: job titles and time words that show up capitalized mid-headline. Kept small --
#: the `>= min_outlets` floor removes most noise without a vocabulary.
STOPWORDS = frozenset("""
el la los las un una y de del en con por para que se su sus al lo als dels les i amb per que uns
unes the a an of in on at to for and or with from by as is are was were be new first after over
says say against amid into out about not what who will would can may mas ya no aquest aquesta este
esta ser fer tras sobre entre sin como segons segun desde hasta cuando quan donde on tot tota tots
totes seva seus gobierno govern president presidenta ministro ministra millones milions anys anos
years dia dies dias hoy avui ahir ayer nou nueva nuevo gran grans major
""".split())

#: A capitalized word, or two in a row, is the theme key candidate. Two-word keys
#: catch the names that only mean something together ("audiencia nacional"), and
#: the merge step folds them into the one-word key when they travel together.
#:
#: KNOWN WEAKNESS: a headline's first word is capitalized because it starts the
#: sentence, not because it is a name, so "Dimiteix", "Comienza" and "Notícies"
#: are all candidate keys. `min_publishers` is what keeps them out in practice --
#: a sentence-initial verb rarely reaches three outlets -- and skipping the first
#: token was rejected because plenty of real headlines open with the subject
#: ("Ceuta pide...", "Iran expels..."), which is exactly the key worth having.
_TOKEN = re.compile(r"[A-ZÀ-ÖØ-Þ][\wÀ-ÿ'·-]{2,}")

#: Overlap above which two keys are the same theme, as a share of the SMALLER
#: key's articles. 0.6 measured best on 2026-W38: it folds `hormuz` into `iran`
#: and `llarena` into `puigdemont`, where symmetric Jaccard left Ceuta split into
#: three themes of 161, 31 and 14 articles.
#:
#: IT OVER-MERGES A DOMINANT KEY, and badly. On 2026-W38 `ceuta` -- 194 articles,
#: a seventh of the window -- absorbed 34 keys including `psoe`, `feijoo`, `cis`,
#: `marruecos`, `melilla` and `zapatero`, because at that size almost any Spanish
#: political key clears 60% overlap with it. That is why the biggest theme is also
#: always the most dispersed one, and why `marruecos` can appear both inside
#: `ceuta` and as its own theme. `dispersion` below is what makes the damage
#: visible rather than something a reader has to guess at; a size-aware threshold
#: is the obvious next thing to try and has not been measured.
MERGE_OVERLAP = 0.6

#: Publishers a sub-cluster needs before it counts as a story in its own right
#: rather than one desk's stray piece.
SUBCLUSTER_MIN_PUBLISHERS = 2
#: Sub-clusters below which dispersion says nothing: a theme holding one or two
#: stories is a subject however unlike they are.
DISPERSED_MIN_SUBCLUSTERS = 3
#: Mean similarity between a theme's sub-clusters below which summarizing any ONE
#: of them would misrepresent the theme. 0.55 from the 2026-W3x measurements:
#: `iran` and `suecia` hold a single story and score 1.000, `openai` holds 3-5
#: related ones at 0.57, while `trump` holds 11 unrelated ones at 0.331 and
#: `ceuta` 18 at 0.464.
DISPERSED_BELOW = 0.55


@dataclass
class Theme:
    """One subject the week was about."""

    key: str
    #: Every key that merged into this one, most-covered first. Shown so a reader
    #: can see WHY two names are one theme and disagree with it.
    keys: list[str] = field(default_factory=list)
    articles: list[Article] = field(default_factory=list)

    @property
    def publishers(self) -> list[str]:
        return sorted({a.publisher for a in self.articles})

    @property
    def languages(self) -> list[str]:
        return sorted({a.language for a in self.articles if a.language})

    @property
    def days(self) -> int:
        return len({(a.published_at or a.collected_at).date() for a in self.articles})

    def by_publisher(self) -> list[tuple[str, int]]:
        counts = Counter(a.publisher for a in self.articles)
        return counts.most_common()

    def newest_first(self) -> list[Article]:
        return sorted(
            self.articles,
            key=lambda a: (a.published_at or a.collected_at),
            reverse=True,
        )

    def to_json(self) -> dict:
        return {
            "key": self.key,
            "keys": self.keys,
            "publishers": self.publishers,
            "publisher_count": len(self.publishers),
            "article_count": len(self.articles),
            "languages": self.languages,
            "days": self.days,
            "articles": [
                {
                    "title": a.title,
                    "url": a.url,
                    "publisher": a.publisher,
                    "language": a.language,
                    "published_at": (a.published_at or a.collected_at).isoformat(),
                }
                for a in self.newest_first()
            ],
        }


def _keys_of(article: Article) -> list[str]:
    caps = [
        token
        for token in _TOKEN.findall(article.title or "")
        if normalize(token) not in STOPWORDS
    ]
    singles = [normalize(token) for token in caps]
    pairs = [normalize(f"{a} {b}") for a, b in zip(caps, caps[1:])]
    return singles + pairs


def group(
    articles: list[Article],
    *,
    containers: list[str] | tuple[str, ...] = (),
    min_publishers: int = 3,
) -> list[Theme]:
    """Group articles into themes, most-covered first.

    `containers` are dropped as keys: names that locate a story rather than being
    one. On a Catalan and Spanish source list that is `catalunya`, `barcelona`,
    `madrid`, `espanya`, `europa` -- they appear in everything, so as themes they
    are grab-bags. It is a config list and not a metric because no metric
    separates them (see the module docstring).

    `min_publishers` is the noise floor. At 3 it is also the claim the ranking
    makes elsewhere: one outlet writing alone is not yet a subject.

    Ordered by publishers, then articles -- the same ordering `clustering._groups`
    uses, so the two views sort alike.
    """
    excluded = {normalize(c) for c in containers}
    by_id = {a.id: a for a in articles}

    candidates: dict[str, set[str]] = {}
    for article in articles:
        for key in _keys_of(article):
            if key and key not in excluded:
                candidates.setdefault(key, set()).add(article.id)

    def publishers_of(ids: set[str]) -> set[str]:
        return {by_id[i].publisher for i in ids}

    viable = {
        key: ids
        for key, ids in candidates.items()
        if len(publishers_of(ids)) >= min_publishers
    }

    # Largest key first, absorbing the smaller keys that travel with it. Order
    # matters: `iran` has to be the one that takes `hormuz`, not the reverse, or
    # the theme ends up named after its smallest part.
    ordered = sorted(viable, key=lambda k: (-len(viable[k]), k))
    themes: list[Theme] = []
    used: set[str] = set()
    for key in ordered:
        if key in used:
            continue
        merged = {key}
        ids = set(viable[key])
        for other in ordered:
            if other in used or other in merged:
                continue
            shared = len(viable[other] & ids)
            if shared and shared / min(len(viable[other]), len(ids)) >= MERGE_OVERLAP:
                merged.add(other)
                ids |= viable[other]
                used.add(other)
        used.add(key)
        themes.append(
            Theme(
                key=key,
                keys=sorted(merged, key=lambda k: (-len(viable[k]), k)),
                articles=[by_id[i] for i in ids],
            )
        )

    themes.sort(key=lambda t: (len(t.publishers), len(t.articles)), reverse=True)
    return themes


def dispersion(
    theme: Theme,
    events: list[list[Article]],
    vectors: dict[str, np.ndarray],
) -> tuple[int, float | None]:
    """How many separate stories a theme holds, and how unlike each other they are.

    Returns `(sub_clusters, mean_similarity)`, the second being None when there is
    nothing to compare. A theme with one sub-cluster is a story. A theme with
    eleven that average 0.33 is a filing cabinet.

    This is the number that decides whether a theme could be summarized at all,
    and it is why themes are not fed to the LLM. Measured across six windows of
    2026-W3x, 18 of 48 top-eight theme slots held three or more mutually
    dissimilar stories -- 38% -- so briefing "the theme" would have produced one
    arbitrary sub-story 38% of the time. `trump` reached 11 sub-clusters at 0.331
    mean similarity and `ceuta` 18 at 0.464, while `iran` and `suecia` held one
    each. It also gets WORSE as a window fills: the quiet week had none, the busy
    week four.

    Needs the embedding cache, and is the only part of this module that does. The
    caller passes whatever vectors it already has; missing ones simply leave
    clusters out of the count rather than triggering a model call.
    """
    ids = {a.id for a in theme.articles}
    centroids = []
    for group_ in events:
        if not any(a.id in ids for a in group_):
            continue
        if len({a.publisher for a in group_}) < SUBCLUSTER_MIN_PUBLISHERS:
            continue
        members = [vectors[a.id] for a in group_ if a.id in vectors]
        if not members:
            continue
        mean = np.mean(members, axis=0)
        norm = np.linalg.norm(mean)
        if norm:
            centroids.append(mean / norm)

    if len(centroids) < 2:
        return len(centroids), None
    sims = [
        float(centroids[i] @ centroids[j])
        for i in range(len(centroids))
        for j in range(i + 1, len(centroids))
    ]
    return len(centroids), sum(sims) / len(sims)


def is_dispersed(sub_clusters: int, mean_similarity: float | None) -> bool:
    """Whether summarizing one of a theme's stories would misrepresent the rest."""
    return (
        sub_clusters >= DISPERSED_MIN_SUBCLUSTERS
        and mean_similarity is not None
        and mean_similarity < DISPERSED_BELOW
    )


def find(themes: list[Theme], wanted: str) -> Theme | None:
    """Look a theme up by any of the keys that merged into it.

    Exact match first, then a prefix, so `news-digest themes puig` finds the
    Puigdemont theme without the reader having to know which key won the naming.
    """
    target = normalize(wanted)
    for theme in themes:
        if target == theme.key or target in theme.keys:
            return theme
    for theme in themes:
        if any(key.startswith(target) for key in theme.keys):
            return theme
    return None
