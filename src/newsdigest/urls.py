"""URL handling. Three forms, for three different jobs.

    a.url             what the newsletter linked. Always stored, never altered.
    canonical_url()   an identity key. Never shown, never followed.
    display_url()     what the digest publishes: the original, minus tracking.

Keeping them apart matters. `canonical_url` normalises aggressively -- scheme,
`www.`, trailing slashes, `/amp` -- which is right for deciding whether two links
are the same article and wrong for a link a reader clicks. `display_url` touches
only the query string, because a newsletter appends its own campaign parameters to
every link and those, unlike a feed's, were not put there by the publisher.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .text import normalize

_TRACKING_PREFIXES = ("utm_", "at_", "mc_", "pk_", "hsa_", "_hs")
_TRACKING_PARAMS = frozenset(
    """
    fbclid gclid dclid msclkid igshid twclid ttclid yclid
    ref ref_src ref_url referrer source src cmpid cmp campaign_id
    ito smid smtyp partner sh spm scrolla guccounter
    ns_campaign ns_mchannel ns_source ns_linkname
    """.split()
)


def _is_tracking(key: str) -> bool:
    lowered = key.lower()
    return lowered in _TRACKING_PARAMS or lowered.startswith(_TRACKING_PREFIXES)


def canonical_url(url: str) -> str:
    """Normalize scheme/host/query/fragment so equivalent URLs compare equal."""
    if not url:
        return ""
    parts = urlsplit(url.strip())
    scheme = (parts.scheme or "https").lower()
    if scheme == "http":
        scheme = "https"

    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if parts.port and parts.port not in (80, 443):
        host = f"{host}:{parts.port}"

    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/") or "/"
    if path.endswith(("/index.html", "/index.htm", "/amp")):
        path = path.rsplit("/", 1)[0] or "/"

    kept = sorted(
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=False)
        if not _is_tracking(k)
    )
    return urlunsplit((scheme, host, path, urlencode(kept), ""))


def display_url(url: str) -> str:
    """The URL to PUBLISH: the original, minus the tracking parameters.

    Distinct from `canonical_url`, which is an identity key and normalises the
    scheme, host and path as well. Those normalisations are safe for comparison
    and not worth risking on a link a reader clicks -- dropping `www.` or a
    trailing `/amp` can break the odd site.

    Needed because a newsletter appends its own campaign parameters to every
    link, and unlike a feed's URL those were not put there by the publisher:

        ...?utm_source=Los%20peligros%20de%20la%20IA%20y%20un%20PSOE%20catat...
        &utm_medium=email&utm_campaign=bol2079

    Publishing that is ugly, and it forwards the name of the newsletter issue to
    anyone reading the digest.
    """
    if not url:
        return ""
    parts = urlsplit(url.strip())
    kept = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not _is_tracking(k)
    ]
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(kept), parts.fragment)
    )


def exclude_by_url(articles: list, patterns: list[str]) -> list:
    """Drop articles whose URL matches any pattern.

    For structural junk, not taste: service content and ad slots sit at
    predictable paths (`/loterias/`, `/horoscopo/`, `/ir_anuncio/`), and no ranking
    signal catches them reliably -- advertorial is written to match whatever topics
    score well, so a topic match actively promotes it.

    Matched against the original URL, not the canonical one, since the section
    path is what identifies the junk and canonicalization may rewrite it.
    Patterns are validated at config load, so they compile here.

    This only works on a RESOLVED url. A newsletter's links go through a click
    tracker, which has no section path at all, so every pattern here is inert
    until `extract/links.py` has unwrapped them -- see `extract.links.resolve`.
    """
    if not patterns:
        return list(articles)
    compiled = [re.compile(p) for p in patterns]
    return [a for a in articles
            if not any(c.search(a.url or "") for c in compiled)]


def exclude_by_title(articles: list, patterns: list[str]) -> list:
    """Drop articles whose HEADLINE matches any pattern.

    The companion to `exclude_by_url`, for junk that has no path of its own.
    Two kinds needed it:

    * Service content an outlet files under its main news path. RAC1 publishes
      "la previsio del temps d'avui" every single day under /politica/ and
      elsewhere, six times in the 2026-W38 window; VilaWeb files football under
      /noticies/ and 3Cat under /3catinfo/, which is why `/esports?/` never sees
      it.
    * One recurring subject. A name is not a section, so no URL pattern reaches
      it.

    TITLE ONLY, deliberately. Matching the excerpt as well would drop any article
    whose background paragraph happens to mention football, and the excerpt is
    not what the article is about.

    Matched against `normalize`d text -- lowercased, accents stripped -- and the
    patterns are normalized too, so a pattern may be written "previsió" and still
    match the "prevision" it becomes. Validated at config load, so they compile
    here.
    """
    if not patterns:
        return list(articles)
    compiled = [re.compile(normalize(p)) for p in patterns]
    return [a for a in articles
            if not any(c.search(normalize(a.title or "")) for c in compiled)]


def domain(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def same_domain(a: str, b: str) -> bool:
    return bool(domain(a)) and domain(a) == domain(b)
