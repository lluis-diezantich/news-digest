"""Prompts live in files, not in Python.

Section 18: a prompt is the thing most often changed and least often reviewed as
code, so it belongs in `prompts/` where a diff shows the wording change on its
own line and `git log -p prompts/` is a readable history of the digest's voice.

Placeholders are `$name`, not `{name}`. Prompts are full of prose that may
legitimately contain braces (JSON examples, most obviously), and a stray brace in
a `.format()` template raises at render time -- which is to say mid-run, after the
mailbox has already been read.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from string import Template

from .config import PROMPT_DIR

log = logging.getLogger(__name__)


class PromptError(RuntimeError):
    """A prompt file is missing or empty."""


@lru_cache(maxsize=None)
def _template(name: str, directory: str) -> Template:
    path = Path(directory) / f"{name}.txt"
    if not path.exists():
        raise PromptError(
            f"prompt file not found: {path}. Prompts are versioned in {directory}/; "
            f"restore it from git rather than inlining the text."
        )
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise PromptError(f"prompt file is empty: {path}")
    return Template(text)


def render(name: str, *, directory: Path | str = PROMPT_DIR, **values: object) -> str:
    """One prompt with its placeholders filled in.

    `safe_substitute`, so an unknown `$word` in the prose is left alone instead of
    raising. A MISSING value is the dangerous case -- it would silently send the
    model the literal text `$output_language` -- so those are checked for and
    named.
    """
    template = _template(name, str(directory))
    rendered = template.safe_substitute(
        {key: str(value) for key, value in values.items()}
    )
    missing = sorted(
        set(template.get_identifiers()) - set(values)
        if hasattr(template, "get_identifiers")
        else []
    )
    if missing:
        raise PromptError(
            f"prompt {name!r} needs values for {missing}; got {sorted(values)}"
        )
    return rendered
