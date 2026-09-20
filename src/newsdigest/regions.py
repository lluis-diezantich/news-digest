"""Geographic spread (section 12).

The failure this prevents is specific: twelve stories about US politics and one
about everything else, in a week that contained more than that. It is a
composition problem, not a ranking one -- every one of those twelve may well have
outranked the alternatives on the formula.

So the fix is applied at SELECTION, not to the score. Stories are still ordered
by score, and the diversity pass only decides which ones make the cut, capping
how many slots one region may hold. It is explicitly not a quota: when a region
genuinely holds most of the week's major news, the cap runs out of alternatives
and the stories are published anyway.
"""

from __future__ import annotations

import logging
from collections import Counter

from .text import normalize

log = logging.getLogger(__name__)

#: The buckets. Flat and few on purpose: this is a spread control, not a
#: geography, and a taxonomy fine enough to be accurate would never bind.
REGIONS = (
    "europe",
    "north america",
    "latin america",
    "middle east",
    "africa",
    "asia",
    "oceania",
    "global",
)
_REGION_SET = frozenset(REGIONS)

#: What models and feeds actually emit in place of the eight names above. A value
#: that maps to nothing becomes "" -- unknown -- rather than being forced into a
#: bucket, because a wrong region is worse than no region: it spends another
#: region's cap.
ALIASES = {
    "eu": "europe", "european union": "europe", "e.u.": "europe",
    "uk": "europe", "united kingdom": "europe", "britain": "europe",
    "spain": "europe", "espana": "europe", "france": "europe",
    "germany": "europe", "italy": "europe", "ukraine": "europe",
    "russia": "europe", "eastern europe": "europe", "western europe": "europe",
    "balkans": "europe", "scandinavia": "europe", "nordics": "europe",
    "europa": "europe",
    "us": "north america", "usa": "north america", "u.s.": "north america",
    "united states": "north america", "america": "north america",
    "canada": "north america", "mexico": "latin america",
    "norteamerica": "north america", "estados unidos": "north america",
    "south america": "latin america", "central america": "latin america",
    "caribbean": "latin america", "americas": "latin america",
    "latam": "latin america", "latinoamerica": "latin america",
    "sudamerica": "latin america", "brazil": "latin america",
    "argentina": "latin america", "venezuela": "latin america",
    "mena": "middle east", "gulf": "middle east", "israel": "middle east",
    "palestine": "middle east", "gaza": "middle east", "iran": "middle east",
    "syria": "middle east", "lebanon": "middle east", "yemen": "middle east",
    "oriente medio": "middle east", "medio oriente": "middle east",
    "north africa": "africa", "sahel": "africa", "sub-saharan africa": "africa",
    "subsaharan africa": "africa", "nigeria": "africa", "sudan": "africa",
    "ethiopia": "africa", "south africa": "africa", "egypt": "africa",
    "china": "asia", "india": "asia", "japan": "asia", "korea": "asia",
    "south korea": "asia", "north korea": "asia", "taiwan": "asia",
    "pakistan": "asia", "afghanistan": "asia", "southeast asia": "asia",
    "south asia": "asia", "east asia": "asia", "central asia": "asia",
    "indo-pacific": "asia", "asia-pacific": "asia", "asia pacific": "asia",
    "australia": "oceania", "new zealand": "oceania", "pacific": "oceania",
    "pacific islands": "oceania",
    "world": "global", "international": "global", "worldwide": "global",
    "transnational": "global", "multiple": "global", "various": "global",
    "mundo": "global", "internacional": "global",
}


def normalize_region(value: str | None) -> str:
    """Map a free-text region onto `REGIONS`, or "" if it maps to nothing."""
    key = normalize(str(value or ""))
    if not key:
        return ""
    if key in _REGION_SET:
        return key
    return ALIASES.get(key, "")


def region_of(articles: list) -> str:
    """The region a cluster is about, by vote of its articles.

    A vote rather than the lead article's answer, because the classifier sees one
    item at a time and a cluster that spans languages usually spans framings too:
    a Spanish outlet files a Brussels decision under Europe and a US outlet files
    the same decision under trade. The commonest non-empty answer wins, and ties
    break alphabetically so the result does not depend on article order.
    """
    votes = Counter(
        region for region in (normalize_region(getattr(a, "region", "")) for a in articles)
        if region
    )
    if not votes:
        return ""
    best = max(votes.values())
    return sorted(name for name, count in votes.items() if count == best)[0]


def spread(stories: list) -> dict[str, int]:
    """region -> story count, for the run summary and `inspect`."""
    counts: Counter[str] = Counter()
    for story in stories:
        counts[(story.regions or [""])[0] or "unknown"] += 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def diversify(
    ranked: list,
    *,
    limit: int,
    max_share: float = 0.5,
    enabled: bool = True,
) -> list:
    """Pick `limit` stories from `ranked`, capping any one region's share.

    `ranked` is (story, articles) pairs in score order, and the return value is a
    subset of it still in score order: this changes WHICH stories are published,
    never the order they are published in. Reordering would be the wrong lever --
    a story does not become the week's lead because of where it happened.

    The cap is soft in both directions. A story whose region is unknown is never
    deferred, since deferring it would punish a classification failure rather
    than a real imbalance. And a deferred story is put back if the slots cannot be
    filled any other way, so a week genuinely dominated by one region publishes
    as it is.
    """
    if limit <= 0:
        return []
    if not enabled:
        return ranked[:limit]

    cap = max(1, int(limit * max_share))
    chosen: list = []
    deferred: list = []
    counts: Counter[str] = Counter()

    for pair in ranked:
        story = pair[0]
        region = (story.regions or [""])[0]
        if region and counts[region] >= cap:
            deferred.append(pair)
            continue
        chosen.append(pair)
        counts[region] += 1
        if len(chosen) >= limit:
            break

    if len(chosen) < limit and deferred:
        filling = deferred[: limit - len(chosen)]
        log.info(
            "diversity: %d slot(s) filled from over-represented regions "
            "(no alternatives left)", len(filling),
        )
        chosen.extend(filling)
        # Back into score order: `deferred` outranked some of what is in `chosen`.
        chosen.sort(key=lambda pair: pair[0].score, reverse=True)

    return chosen[:limit]
