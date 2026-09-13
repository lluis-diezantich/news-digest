"""Language detection for collected articles.

Detection runs during daily collection and never calls an LLM. Two things make
it reliable enough to key clustering off:

  * The candidate set is restricted to the languages you actually configured.
    Asking "is this en, es or ca?" is a far easier question than asking which of
    97 languages it is, and it is what lifts accuracy on the es/ca pair.
  * A source's declared `languages` acts as both a filter and a fallback.
    Publishers know what language they publish in; the detector only has a
    headline and an excerpt.

Measured on 195 real headlines from the configured feeds, this agrees with the
source's own declared language 99.5% of the time.
"""

from __future__ import annotations

import logging
import math
import threading

import py3langid

from .text import strip_html

log = logging.getLogger(__name__)

DEFAULT_SUPPORTED = ("en", "es", "ca")

# Below this many characters a verdict is noise -- a headline like "Wegovy"
# ranks languages in an essentially arbitrary order.
MIN_CHARS = 20
# Softmax confidence below which we prefer the source's declared language.
MIN_CONFIDENCE = 0.65

_lock = threading.Lock()
_configured: tuple[str, ...] = ()


def configure(supported: list[str] | tuple[str, ...] = DEFAULT_SUPPORTED) -> None:
    """Restrict the detector to the configured languages. Idempotent."""
    global _configured
    wanted = tuple(dict.fromkeys(str(code).lower() for code in supported if code))
    if not wanted:
        wanted = DEFAULT_SUPPORTED
    with _lock:
        if wanted == _configured:
            return
        # py3langid keeps one process-wide identifier, so this is global state.
        # We set it from the config once per run rather than per article.
        py3langid.set_languages(list(wanted))
        _configured = wanted
    log.debug("language detection restricted to %s", ", ".join(wanted))


def _confidence(ranked: list[tuple[str, float]]) -> float:
    """Softmax over py3langid's log probabilities.

    The raw scores are unnormalized and scale with text length, so comparing
    them to a fixed threshold would make long articles look more confident than
    short ones. Softmax gives a length-independent share instead.
    """
    if not ranked:
        return 0.0
    top = ranked[0][1]
    weights = [math.exp(score - top) for _, score in ranked]
    return weights[0] / sum(weights)


def detect(
    text: str,
    *,
    allowed: list[str] | tuple[str, ...] | None = None,
    fallback: str | None = None,
) -> str | None:
    """Best language code for `text`, or `fallback` when unsure.

    `allowed` restricts the answer to a source's declared languages; a source
    publishing only Catalan cannot yield a Spanish verdict from one cognate.
    """
    if not _configured:
        configure()

    cleaned = strip_html(text or "").strip()
    permitted = tuple(c.lower() for c in allowed) if allowed else None

    if len(cleaned) < MIN_CHARS:
        return _only(permitted) or fallback

    try:
        ranked = py3langid.rank(cleaned)
    except Exception as exc:  # a detector failure must not lose the article
        log.debug("language detection failed (%s)", exc)
        return _only(permitted) or fallback

    if permitted:
        ranked = [pair for pair in ranked if pair[0] in permitted] or ranked

    if not ranked:
        return fallback
    if _confidence(ranked) < MIN_CONFIDENCE:
        return _only(permitted) or fallback or ranked[0][0]
    return ranked[0][0]


def _only(permitted: tuple[str, ...] | None) -> str | None:
    """A source declaring exactly one language settles the question itself."""
    return permitted[0] if permitted and len(permitted) == 1 else None


def detect_article(
    title: str,
    description: str = "",
    *,
    allowed: list[str] | None = None,
    fallback: str | None = None,
) -> str | None:
    """Detect from title plus excerpt -- more text means a firmer verdict."""
    return detect(f"{title}. {description}", allowed=allowed, fallback=fallback)
