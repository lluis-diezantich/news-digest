"""Mailbox contract.

IMAP is the only implementation today, but it is deliberately behind this
interface: section 3 of the specification wants room for a provider whose OAuth
or HTTP API turns out cleaner than IMAP, and that should be one new module and
one registry entry rather than a change to the pipeline.

A mailbox returns RAW messages and knows nothing about newsletters, sources or
articles. Deciding which source a message belongs to is `match.py`; turning it
into an `Email` is `message.py`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


class MailboxError(RuntimeError):
    """Any mailbox failure. Never carries the password in its message."""


@dataclass
class RawMessage:
    """One message as the server gave it to us."""

    #: Server-assigned identifier, for logging only. Not stable across sessions,
    #: so it is never used as a key -- that is the Message-ID header's job.
    uid: str
    #: Complete RFC822 bytes, headers included.
    raw: bytes


class Mailbox(ABC):
    """A source of raw newsletter messages."""

    name: str = ""

    @abstractmethod
    def fetch(
        self, since: datetime | None = None, until: datetime | None = None
    ) -> list[RawMessage]:
        """Messages whose server-side date falls in [since, until).

        Implementations may over-return -- IMAP's own date search has day
        granularity and no notion of a timezone -- so callers filter precisely on
        the parsed `received_at`. Under-returning would silently lose a
        newsletter, which is why the contract is one-sided.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release the connection. Safe to call twice."""

    def __enter__(self) -> "Mailbox":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
