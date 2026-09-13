"""Shared HTTP access: one session, per-domain rate limiting, robots.txt."""

from __future__ import annotations

import logging
import threading
import time
import urllib.robotparser
from urllib.parse import urlsplit, urlunsplit

import requests

log = logging.getLogger(__name__)

# Floor between two requests to the same host, in seconds. robots.txt
# Crawl-delay overrides this upward, never downward.
MIN_INTERVAL = 1.5
ROBOTS_TIMEOUT = 10.0


class RobotsDisallowed(RuntimeError):
    """Raised when robots.txt forbids the URL we were asked to fetch."""


class Fetcher:
    """Polite HTTP client. One instance per run, shared by all sources."""

    def __init__(self, user_agent: str, *, timeout: float = 20.0, respect_robots: bool = True):
        self.user_agent = user_agent
        self.timeout = timeout
        self.respect_robots = respect_robots
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml,"
                          "application/rss+xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en",
            }
        )
        self._lock = threading.Lock()
        self._last_request: dict[str, float] = {}
        self._delays: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    # -- politeness ---------------------------------------------------------

    def _wait(self, host: str) -> None:
        with self._lock:
            interval = max(MIN_INTERVAL, self._delays.get(host, 0.0))
            last = self._last_request.get(host)
            now = time.monotonic()
            if last is not None:
                remaining = interval - (now - last)
                if remaining > 0:
                    time.sleep(remaining)
                    now = time.monotonic()
            self._last_request[host] = now

    def _robots_for(self, url: str) -> urllib.robotparser.RobotFileParser | None:
        parts = urlsplit(url)
        host = parts.netloc
        if host in self._robots:
            return self._robots[host]

        parser: urllib.robotparser.RobotFileParser | None = None
        robots_url = urlunsplit((parts.scheme or "https", host, "/robots.txt", "", ""))
        try:
            response = self._session.get(robots_url, timeout=ROBOTS_TIMEOUT)
            if response.status_code == 200:
                parser = urllib.robotparser.RobotFileParser()
                parser.parse(response.text.splitlines())
                delay = parser.crawl_delay(self.user_agent)
                if delay:
                    self._delays[host] = float(delay)
            elif response.status_code in (401, 403):
                # Explicitly protected: treat the whole site as disallowed.
                parser = urllib.robotparser.RobotFileParser()
                parser.parse(["User-agent: *", "Disallow: /"])
            # 404 and friends mean "no restrictions"; leave parser as None.
        except requests.RequestException as exc:
            log.debug("robots.txt unavailable for %s (%s); proceeding", host, exc)

        self._robots[host] = parser
        return parser

    def allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parser = self._robots_for(url)
        return True if parser is None else parser.can_fetch(self.user_agent, url)

    # -- requests -----------------------------------------------------------

    def get(self, url: str, *, check_robots: bool = True) -> requests.Response:
        """GET with rate limiting. Raises on HTTP error or robots denial."""
        if check_robots and not self.allowed(url):
            raise RobotsDisallowed(f"robots.txt disallows {url}")
        self._wait(urlsplit(url).netloc)
        response = self._session.get(url, timeout=self.timeout, allow_redirects=True)
        response.raise_for_status()
        return response

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "Fetcher":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
