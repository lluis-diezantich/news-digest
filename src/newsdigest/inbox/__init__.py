"""Mailbox access, resolved by name from config/env.

Adding a provider -- a Gmail or Graph API client, say -- means implementing
`Mailbox` in a new module and registering it here. Nothing else changes.
"""

from __future__ import annotations

import logging

from ..config import MailboxSettings
from .base import Mailbox, MailboxError, RawMessage
from .imap import IMAPMailbox
from .match import Matcher, assign
from .message import parse_message

log = logging.getLogger(__name__)


def _build_imap(settings: MailboxSettings) -> Mailbox:
    return IMAPMailbox(
        host=settings.host,
        username=settings.username,
        password=settings.password,
        port=settings.port,
        folder=settings.folder,
        timeout=settings.timeout,
    )


PROVIDERS = {
    "imap": _build_imap,
}


def get_mailbox(settings: MailboxSettings) -> Mailbox:
    """Build the configured mailbox. Raises rather than degrading: with no mail
    there is no digest, so this is the one stage that has nothing to fall back
    on."""
    factory = PROVIDERS.get(settings.provider)
    if factory is None:
        raise MailboxError(
            f"unknown MAILBOX_PROVIDER={settings.provider!r} "
            f"(known: {', '.join(sorted(PROVIDERS))})"
        )
    return factory(settings)


__all__ = [
    "IMAPMailbox",
    "Mailbox",
    "MailboxError",
    "Matcher",
    "PROVIDERS",
    "RawMessage",
    "assign",
    "get_mailbox",
    "parse_message",
]
