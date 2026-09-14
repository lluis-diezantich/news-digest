"""Telling Google's two kinds of 429 apart.

A 429 means two different things on the free tier and Google uses one status
code for both:

  * a per-MINUTE limit -- transient. A weekly run fires its ~80 requests in one
    burst, which outruns the free-tier allowance for a few seconds. Waiting
    clears it, and on a batch job with no reader waiting, waiting costs nothing.
  * a per-DAY limit -- terminal for this run. No amount of waiting helps.

Treating every 429 as terminal publishes a digest with unenriched stories
whenever a burst outruns the per-minute allowance. Treating every 429 as
transient would stall a run against an exhausted daily quota until its timeout.
So we read the body and tell them apart.

Google reports the distinction in `error.details[]`: a `QuotaFailure` names the
violated `quotaId` (`...PerMinute...` or `...PerDay...`) and a `RetryInfo`
carries the delay the server wants. Both are advisory -- when they are absent or
unparseable we fall back to a fixed schedule and to scanning the message text,
because a missing detail must not turn into a crash on the one path whose whole
job is to survive a bad response.

Waiting is also bounded per run, not just per request. A low per-minute
allowance against 80 queued requests could otherwise spend longer in `sleep`
than the workflow's timeout allows, so `cap` gives the caller a total wait
budget; past it a 429 becomes terminal and the digest publishes with whatever
succeeded.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass

#: Indirection so tests can run the retry paths without really sleeping.
sleep = time.sleep

#: No single wait is longer than this. Per-minute windows are 60s wide, so
#: sleeping much past one window cannot buy anything.
MAX_DELAY = 75.0
#: Waits used when the response carries no usable RetryInfo, in seconds.
FALLBACK_DELAYS = (20.0, 40.0, 60.0)

#: A protobuf duration, e.g. "31s" or "1.5s".
_DURATION = re.compile(r"([0-9]+(?:\.[0-9]+)?)s")
#: Quota ids and messages that mean "come back tomorrow".
_DAILY = re.compile(r"per[\s_-]*day|daily", re.IGNORECASE)


@dataclass(frozen=True)
class Decision:
    """What to do about one 429."""

    retry: bool
    delay: float = 0.0
    reason: str = ""


def plan(
    body: str,
    *,
    attempt: int,
    budget: int,
    spent: float = 0.0,
    cap: float = float("inf"),
) -> Decision:
    """Decide whether to wait out a 429, and for how long.

    `attempt` is how many times this request has already been rate limited and
    `budget` how many waits it is allowed; `spent` is how long this run has
    already spent waiting on rate limits and `cap` its total allowance. A daily
    limit is refused however much budget is left, since waiting cannot fix it.
    """
    details = _details(body)
    if _is_daily(body, details):
        return Decision(False, reason="daily quota exhausted")
    if attempt >= budget:
        return Decision(False, reason=f"still rate limited after {budget} waits")

    server = _server_delay(details)
    delay = min(MAX_DELAY, server if server is not None else _fallback(attempt))
    # Jitter so several rate-limited callers do not resynchronise on one window.
    delay += random.uniform(0.0, 1.0)

    if spent + delay > cap:
        return Decision(False, reason=f"run's {cap:.0f}s rate-limit wait budget spent")
    return Decision(
        True,
        delay=delay,
        reason="per-minute limit, server asked" if server is not None
        else "per-minute limit, no RetryInfo",
    )


def _fallback(attempt: int) -> float:
    return FALLBACK_DELAYS[min(max(attempt, 0), len(FALLBACK_DELAYS) - 1)]


def _details(body: str) -> list[dict]:
    """`error.details[]`, or an empty list for anything unparseable."""
    try:
        payload = json.loads(body or "")
    except (TypeError, ValueError):
        return []
    error = payload.get("error") if isinstance(payload, dict) else None
    details = error.get("details") if isinstance(error, dict) else None
    return [d for d in details if isinstance(d, dict)] if isinstance(details, list) else []


def _is_daily(body: str, details: list[dict]) -> bool:
    quota_failures = [
        d for d in details if str(d.get("@type", "")).endswith("QuotaFailure")
    ]
    for detail in quota_failures:
        violations = detail.get("violations")
        if not isinstance(violations, list):
            continue
        for violation in violations:
            if isinstance(violation, dict) and _DAILY.search(
                " ".join(
                    str(violation.get(key, ""))
                    for key in ("quotaId", "quotaMetric", "description")
                )
            ):
                return True
    # A QuotaFailure that named no daily violation is a per-minute limit. Only
    # when Google sent no structured violation at all do we read the prose.
    return False if quota_failures else bool(_DAILY.search(body or ""))


def _server_delay(details: list[dict]) -> float | None:
    for detail in details:
        if str(detail.get("@type", "")).endswith("RetryInfo"):
            match = _DURATION.search(str(detail.get("retryDelay", "")))
            if match:
                return float(match.group(1))
    return None
