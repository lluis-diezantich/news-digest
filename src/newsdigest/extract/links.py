"""Tracking links -> real article URLs.

This is load-bearing, not cosmetic. Newsletter links are rewritten through a
click tracker, so the URL in the HTML looks like

    https://link.mail.elpais.com/c/eJx1kM...

and three separate things break as long as it stays that way:

  * attribution, because the digest would link to a tracker that expires;
  * deduplication, because the same article carries a different opaque token in
    every newsletter, so `canonical_url` cannot tell they are one article;
  * topic filtering, because `exclude_url_patterns` matches section paths --
    `/deportes/`, `/horoscopo/` -- and a tracker has no section path at all.

Two mechanisms, cheapest first. Most trackers put the destination in a query
parameter, which costs nothing to unwrap. Only an opaque token needs the network.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import parse_qsl, unquote, urlsplit

from ..urls import domain

log = logging.getLogger(__name__)

#: Query parameters that hold a destination URL. Ordered by how often they are
#: the real thing rather than a metrics echo of it.
_URL_PARAMS = ("url", "u", "redirect", "redirect_url", "redirect_uri", "target",
               "dest", "destination", "link", "to", "r", "out", "ref_url")

#: Hosts whose whole business is redirecting mail. A link here is always opaque.
#: `piano.io` was added 2026-09-20 after a real Público issue: every one of its 27
#: links went through `api-esp-eu.piano.io` and NONE was detected, so `resolve`
#: reported them as real article URLs. That is the worst failure this module has --
#: the tracker is stored as permanent attribution, section blocklists match
#: nothing, and the run does not warn, because it only counts links it knows it
#: failed to resolve. `_opaque_path` below is the general answer; this list is the
#: cheap one.
_MAILER_HOSTS = frozenset(
    """
    list-manage.com mailchimp.com mailchi.mp sendgrid.net sparkpostmail.com
    mailgun.org cmail1.com cmail2.com cmail19.com createsend.com
    mandrillapp.com sendinblue.com brevo.com klaviyomail.com
    piano.io tinypass.com exct.net exacttarget.com mkt.com
    salesforce-communities.com sailthru.com email.mailgun.net
    substack.com beehiiv.com ghost.io convertkit-mail.com
    bit.ly ow.ly tinyurl.com buff.ly trib.al dlvr.it
    """.split()
)

#: A path segment needs a run of at least this many letters to look like words
#: rather than an identifier. "internacional" and "floods" clear it; "2090179c18",
#: "30394" and "-c" do not.
_WORD_RUN = re.compile(r"[^\W\d_]{4,}")

#: Subdomain labels a publisher uses for its own click tracker.
_TRACKER_LABELS = ("link", "links", "click", "clicks", "track", "tracking",
                   "email", "mail", "e", "em", "url", "go", "r", "redirect",
                   "newsletter", "nl", "boletin")


def _opaque_path(url: str) -> bool:
    """True when no path segment contains anything word-like.

    This is the general test, and it is what catches the next unknown vendor.
    An article URL always carries words somewhere -- a section, a slug, a month
    name -- because it is meant to be read:

        /internacional/2026-09-18/inundaciones-nigeria.html   -> wordy
        /world/2026/sep/18/floods-displace-thousands          -> wordy
        /-c/117/30394/671878/20018313/860429/2090179c18       -> opaque

    A date-only archive URL (`/2026/09/18/`) reads as opaque too. That is the
    safe direction: it is treated as unresolved, which keeps the article and
    counts it, rather than being silently trusted.
    """
    path = urlsplit(url).path
    return not _WORD_RUN.search(path)


def looks_like_tracker(url: str) -> bool:
    """True when a URL is probably a redirect rather than an article.

    Three shapes, in order of confidence:

    1. A known mail vendor's host.
    2. A publisher's own tracker subdomain -- `link.mail.elpais.com` -- recognised
       by its leading label. Only when the path carries no section, because
       `www.elpais.com/deportes/...` and `link.elpais.com/xyz` must not be treated
       alike and some outlets do serve real articles from a `go.` host.
    3. Any SUBDOMAIN whose path is entirely identifiers. This is the general case
       and the one that matters: the label test only recognises vendors we have
       already met, and every ESP invents its own hostname. Restricted to
       subdomains so a publisher's own date-style archive URL on its bare domain
       is not swept up.
    """
    host = domain(url)
    if not host:
        return False
    if host in _MAILER_HOSTS or any(host.endswith("." + h) for h in _MAILER_HOSTS):
        return True

    labels = host.split(".")
    if labels[0] in _TRACKER_LABELS:
        # Checked at any label count, not only on subdomains: a domain literally
        # named `track.example` or `links.example` is a tracker in its own right,
        # and requiring a third label let one straight through.
        path = urlsplit(url).path.strip("/")
        if _opaque_path(url):
            return True
        # One segment and no file extension: a token, not a section. This is the
        # case a base64 blob falls into -- `/c/eJx1kMtuAyE` contains a long letter
        # run, so it reads as "wordy" and `_opaque_path` cannot see it.
        return path.count("/") <= 1 and "." not in path.rpartition("/")[2]
    if len(labels) > 2:
        # Subdomain with an all-identifier path. Restricted to subdomains so a
        # publisher's own date-style archive URL on its bare domain is not swept
        # up, and it is the rule that catches the next unknown vendor.
        return _opaque_path(url)
    return False


def unwrap(url: str) -> str | None:
    """The destination encoded in this URL's query string, if there is one.

    No network access. Recursive, because a link is sometimes wrapped twice --
    a publisher's tracker inside a mail vendor's -- and unwrapping one layer
    would leave the other in place.
    """
    if not url:
        return None
    try:
        query = parse_qsl(urlsplit(url).query, keep_blank_values=False)
    except ValueError:
        return None

    found = {key.lower(): value for key, value in query}
    for name in _URL_PARAMS:
        candidate = unquote(found.get(name, "") or "")
        if candidate.startswith(("http://", "https://")) and domain(candidate):
            return unwrap(candidate) or candidate
    return None


def resolve(url: str, resolver=None) -> tuple[str, bool]:
    """(best known URL, whether it is the real article).

    The flag is what the caller acts on: an unresolved tracker should not be
    stored as an article's permanent URL, and it must not be matched against
    section blocklists, because a tracker matches none of them and would sail
    through a filter it should have been caught by.

    `resolver` is optional so the whole extraction path stays testable and
    runnable with no network at all.
    """
    if not url:
        return url, False

    static = unwrap(url)
    if static:
        return static, True
    if not looks_like_tracker(url):
        return url, True
    if resolver is None:
        log.debug("unresolved tracker (no resolver): %s", url)
        return url, False

    final = resolver.resolve(url)
    if final and not looks_like_tracker(final):
        return final, True
    # A tracker that redirects to another tracker, or did not answer.
    return (final or url), False
