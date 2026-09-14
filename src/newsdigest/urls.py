"""URL canonicalization.

The canonical form is an identity key only -- it is never shown to the user and
never followed. The original URL is always stored and always what the feed
links to, so stripping parameters here cannot break attribution.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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


def exclude_by_url(articles: list, patterns: list[str]) -> list:
    """Drop articles whose URL matches any pattern.

    For structural junk, not taste: native advertising and service content sit
    at predictable paths (`/especials/`, `/loterias/`, `/horoscopo/`), and no
    ranking signal catches them reliably -- advertorial is written to match
    whatever topics score well, so a topic match actively promotes it.

    Matched against the original URL, not the canonical one, since the section
    path is what identifies the junk and canonicalization may rewrite it.
    Patterns are validated at config load, so they compile here.
    """
    if not patterns:
        return list(articles)
    compiled = [re.compile(p) for p in patterns]
    return [a for a in articles
            if not any(c.search(a.url or "") for c in compiled)]


def domain(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def same_domain(a: str, b: str) -> bool:
    return bool(domain(a)) and domain(a) == domain(b)
