"""The little HTTP we still need: resolving tracking links to real articles.

This project fetches no article pages and crawls nothing -- newsletters are the
only input -- so what remains of the old HTTP layer is one job: follow a mailer's
redirect to the URL it is hiding. robots.txt and crawl-delay have gone with the
crawling; a redirect we were emailed is not a crawl.

Still polite: one connection reused, a floor between two requests to the same
host, a short timeout, and HEAD before GET.
"""

from __future__ import annotations

import logging
import threading
import time
from urllib.parse import urlsplit

import requests

log = logging.getLogger(__name__)

#: Seconds between two requests to the same host.
MIN_INTERVAL = 0.5
#: A redirect chain longer than this is a loop or a bounce farm.
MAX_REDIRECTS = 8


class Resolver:
    """Follows redirects and nothing else. One instance per run."""

    def __init__(self, user_agent: str, *, timeout: float = 10.0):
        self.timeout = timeout
        self._session = requests.Session()
        self._session.max_redirects = MAX_REDIRECTS
        self._session.headers.update({"User-Agent": user_agent, "Accept": "*/*"})
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def _wait(self, host: str) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last.get(host, 0.0)
            if elapsed < MIN_INTERVAL:
                time.sleep(MIN_INTERVAL - elapsed)
            self._last[host] = time.monotonic()

    def resolve(self, url: str) -> str | None:
        """The URL this one redirects to, or None if it could not be resolved.

        Returns None rather than raising or returning the input: the caller has
        to be able to tell "this is the real URL" from "we never found out", and
        conflating them would store a tracking URL as an article's permanent
        attribution.

        HEAD first, because a redirector answers it in a few hundred bytes. Some
        mailers answer HEAD with 405 while redirecting a GET perfectly well, so a
        method-level rejection falls through to GET with the body streamed and
        discarded.
        """
        host = (urlsplit(url).hostname or "").lower()
        if not host:
            return None
        self._wait(host)
        try:
            response = self._session.head(url, allow_redirects=True, timeout=self.timeout)
            if response.status_code in (400, 403, 405, 501) or not response.url:
                response = self._session.get(
                    url, allow_redirects=True, timeout=self.timeout, stream=True
                )
                response.close()
        except requests.TooManyRedirects:
            log.debug("redirect loop resolving %s", url)
            return None
        except requests.RequestException as exc:
            log.debug("could not resolve %s (%s)", url, type(exc).__name__)
            return None

        final = response.url or ""
        return final if final and final != url else (final or None)

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "Resolver":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
