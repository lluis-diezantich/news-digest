"""Newsletter parsing: one email in, several news items out."""

from __future__ import annotations

from .boilerplate import (
    is_boilerplate_link,
    is_boilerplate_text,
    is_housekeeping,
    is_never_article,
    is_sponsored,
    strip_chrome,
)
from .links import looks_like_tracker, resolve, unwrap
from .newsletter import (
    ExtractedItem,
    extract,
    extract_html,
    extract_text,
    is_masthead,
    looks_like_caption,
    looks_like_fragment,
    plausible_title,
    to_articles,
)

__all__ = [
    "ExtractedItem",
    "extract",
    "extract_html",
    "extract_text",
    "is_boilerplate_link",
    "is_boilerplate_text",
    "is_housekeeping",
    "is_masthead",
    "is_never_article",
    "is_sponsored",
    "looks_like_caption",
    "looks_like_fragment",
    "looks_like_tracker",
    "plausible_title",
    "resolve",
    "strip_chrome",
    "to_articles",
    "unwrap",
]
