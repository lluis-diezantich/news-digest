"""RFC822 bytes -> `Email`.

Newsletters are the least well-formed mail there is: mangled charsets, headers
folded in the wrong places, `multipart/alternative` inside `multipart/related`
inside `multipart/mixed`, and the occasional missing Message-ID. Nothing here may
raise on a malformed message, because one bad newsletter must not cost the week.
"""

from __future__ import annotations

import email
import email.policy
import hashlib
import logging
import re
from email.header import decode_header, make_header
from email.message import Message
from email.parser import BytesHeaderParser
from email.utils import parseaddr

from ..models import Email, parse_date, utcnow

log = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")

#: Hard cap per body, characters. A newsletter is tens of kilobytes; anything
#: past this is a mail client's quoted history or an embedded image gone wrong,
#: and the extractor gains nothing from it.
MAX_BODY_CHARS = 500_000


#: The only headers this project reads. Kept explicit so the strict parse below
#: touches as little as possible.
_WANTED_HEADERS = ("Message-ID", "From", "Subject", "Date")


def _collapse(value: str) -> str:
    """Undo header folding.

    A long unencoded Subject arrives split across lines and `decode_header` does
    not rejoin it -- a real Público issue produced `"...consecuencias que\\r\\n
    retratan el..."`, newline and all. Subject MATCHING survived that, because it
    runs through `normalize`, which collapses whitespace; the stored and displayed
    subject did not.
    """
    return _WS_RE.sub(" ", value or "").strip()


def _lenient_header(message: Message, name: str) -> str:
    """One header from a compat32 message, decoding encoded words by hand."""
    raw = message.get(name)
    if raw is None:
        return ""
    try:
        return _collapse(str(make_header(decode_header(str(raw)))))
    except (UnicodeDecodeError, LookupError, ValueError):
        return _collapse(str(raw))


def _read_headers(raw: bytes, fallback: Message) -> dict[str, str]:
    """The headers we need, preferring the strict parser for each one.

    Two parsers, because neither is sufficient alone.

    `policy.compat32` never raises on a malformed message, which is why the BODY
    walk uses it -- and it decodes headers as ASCII, replacing anything else. A
    sender who puts raw UTF-8 in a header instead of RFC 2047 encoded words (not
    conformant, and common) therefore arrives as "P\ufffd\ufffdblico", with the
    original bytes gone rather than recoverable. That is not cosmetic:
    `sender_name_patterns` matches on the display name, and on a shared bulk-mail
    domain it is the only thing telling one publisher from another, so a mangled
    name silently matches nothing.

    `policy.default` decodes that header correctly but is stricter about the rest
    of the message, so it is used for headers only, per header, with compat32 as
    the fallback. Header-only parsing is cheap.
    """
    strict: Message | None
    try:
        strict = BytesHeaderParser(policy=email.policy.default).parsebytes(raw)
    except Exception:  # the strict parser is the optional half
        strict = None

    headers: dict[str, str] = {}
    for name in _WANTED_HEADERS:
        value = ""
        if strict is not None:
            try:
                # Already decoded by the policy, so only folding needs undoing.
                value = _collapse(str(strict.get(name) or ""))
            except Exception:
                value = ""
        headers[name] = value or _lenient_header(fallback, name)
    return headers


def _decode(part: Message) -> str:
    """One part's text, whatever it claims about its own encoding."""
    payload = part.get_payload(decode=True)
    if payload is None:
        content = part.get_payload()
        return content if isinstance(content, str) else ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        # A charset the stdlib has never heard of. Latin-1 decodes any byte
        # sequence, so this cannot fail again.
        return payload.decode("latin-1", errors="replace")


def _bodies(message: Message) -> tuple[str, str]:
    """(html, plain), taking the LONGEST candidate part of each type.

    Longest rather than first: a `multipart/related` newsletter often carries a
    short HTML preamble alongside the real body, and picking the first part gets
    the preamble. Attachments are skipped -- a text attachment is not the body,
    however plausible its content type.
    """
    html_parts: list[str] = []
    text_parts: list[str] = []
    for part in message.walk():
        if part.get_content_maintype() == "multipart":
            continue
        disposition = (part.get("Content-Disposition") or "").lower()
        if "attachment" in disposition:
            continue
        subtype = part.get_content_subtype()
        if subtype == "html":
            html_parts.append(_decode(part))
        elif subtype == "plain":
            text_parts.append(_decode(part))

    html = max(html_parts, key=len, default="")
    text = max(text_parts, key=len, default="")
    return html[:MAX_BODY_CHARS], text[:MAX_BODY_CHARS]


def _message_id(value: str, raw: bytes) -> str:
    """The Message-ID, or a deterministic stand-in derived from the bytes.

    A synthesised id has to be stable across runs or the same newsletter is
    ingested every week, so it hashes the message rather than using the clock or
    the server's UID -- neither of which survives a re-download.
    """
    if value:
        return value.strip("<> ")
    digest = hashlib.sha1(raw).hexdigest()[:24]
    log.debug("message has no Message-ID; synthesised %s", digest)
    return f"synthetic-{digest}"


def parse_message(raw: bytes) -> Email:
    """Turn one raw message into an `Email`. Never raises on malformed input."""
    try:
        message = email.message_from_bytes(raw, policy=email.policy.compat32)
    except Exception as exc:  # the stdlib parser is not exhaustively documented
        raise ValueError(f"unparseable message: {type(exc).__name__}: {exc}") from None

    headers = _read_headers(raw, message)
    sender_name, sender = parseaddr(headers["From"])
    html, text = _bodies(message)
    received = parse_date(headers["Date"]) or utcnow()

    # The stdlib parser does not raise on input that is not a message at all --
    # `message_from_bytes(b"")` returns an empty Message quite happily. Without
    # this check that empty shell would be stored, matched against no source, and
    # marked parsed, so the problem would never surface anywhere. A real
    # newsletter has at least a sender or a body.
    if not sender and not html and not text:
        raise ValueError("not a message: no sender and no body")

    return Email(
        message_id=_message_id(headers["Message-ID"], raw),
        source="",                      # assigned by match.py
        subject=headers["Subject"],
        sender=sender.lower(),
        sender_name=sender_name,
        received_at=received,
        html_body=html,
        text_body=text,
    )
