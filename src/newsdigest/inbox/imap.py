"""IMAP mailbox.

Read-only by construction: the connection is opened with `readonly=True`, so a
run can neither delete a newsletter nor mark one seen. Re-running a week is
therefore always safe, which is the whole point of storing message ids.

Credentials come from the environment and are never logged. The one place they
could leak is an exception message from imaplib, so every call that can carry
them is wrapped.
"""

from __future__ import annotations

import imaplib
import logging
from datetime import datetime, timedelta

from ..models import to_utc
from .base import Mailbox, MailboxError, RawMessage

log = logging.getLogger(__name__)

#: IMAP's own date format. Day granularity, and the server decides the timezone.
IMAP_DATE = "%d-%b-%Y"

#: Messages fetched per round trip. One FETCH per message is a round trip each
#: and a weekly run has tens of them, not thousands, so this is about politeness
#: to the server rather than throughput.
FETCH_BATCH = 25


class IMAPMailbox(Mailbox):
    """A dedicated newsletter inbox, read over IMAP4 with TLS."""

    name = "imap"

    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        *,
        port: int = 993,
        folder: str = "INBOX",
        timeout: float = 30.0,
    ):
        if not host or not username or not password:
            raise MailboxError(
                "incomplete mailbox credentials: set NEWS_EMAIL_HOST, "
                "NEWS_EMAIL_USERNAME and NEWS_EMAIL_PASSWORD"
            )
        self.host = host
        self.port = port
        self.username = username
        self.folder = folder
        self.timeout = timeout
        self._password = password
        self._conn: imaplib.IMAP4_SSL | None = None

    # -- connection ------------------------------------------------------- #

    def _connect(self) -> imaplib.IMAP4_SSL:
        if self._conn is not None:
            return self._conn
        try:
            conn = imaplib.IMAP4_SSL(self.host, self.port, timeout=self.timeout)
        except OSError as exc:
            raise MailboxError(f"cannot reach {self.host}:{self.port} ({exc})") from None
        try:
            conn.login(self.username, self._password)
        except imaplib.IMAP4.error as exc:
            # The server's own reason IS included, and scrubbed.
            #
            # It used to be dropped entirely, because some servers echo the login
            # line back in the error with the password in it. That made this the
            # least diagnosable failure in the project: "login rejected" cannot
            # tell a wrong password from an app password belonging to a different
            # Google account, and the server's response is the only thing that
            # can. So it is relayed with every form of the credential replaced.
            # Still not chained, so no traceback can carry it into a log either.
            raise MailboxError(
                f"login rejected for {self.username} on {self.host}: "
                f"{self._scrub(str(exc))}\n"
                f"  Check in this order: the app password was created on "
                f"{self.username} and not another account; it has not been "
                f"revoked; 2-Step Verification is still on."
            ) from None
        try:
            # readonly: a digest run must never mutate the mailbox.
            status, _ = conn.select(self.folder, readonly=True)
        except imaplib.IMAP4.error as exc:
            raise MailboxError(f"cannot open folder {self.folder!r} ({exc})") from None
        if status != "OK":
            raise MailboxError(f"cannot open folder {self.folder!r}: {status}")
        log.info("mailbox %s/%s open (read-only)", self.host, self.folder)
        self._conn = conn
        return conn

    def _scrub(self, text: str) -> str:
        """Remove every form of the credential from a server message.

        Both the spaced and unspaced forms: a Gmail app password is copied as
        "abcd efgh ijkl mnop" and may be sent either way, so a server echoing one
        back must not be relayed.
        """
        cleaned = text
        for secret in {self._password, self._password.replace(" ", "")}:
            if secret and len(secret) > 3:
                cleaned = cleaned.replace(secret, "[redacted]")
        return cleaned

    def close(self) -> None:
        if self._conn is None:
            return
        try:
            self._conn.close()
            self._conn.logout()
        except (imaplib.IMAP4.error, OSError):
            pass  # closing a connection the server already dropped is not news
        finally:
            self._conn = None

    # -- fetching --------------------------------------------------------- #

    def _criteria(self, since: datetime | None, until: datetime | None) -> str:
        """IMAP search terms for a date range, widened by a day at each end.

        SINCE and BEFORE compare against the server's own INTERNALDATE at day
        granularity in a timezone we do not know. A message that arrived at
        23:40 UTC on the last day of the window can sit on the wrong side of the
        server's midnight, so both ends are widened and the caller filters
        precisely on the parsed timestamp. Over-fetching costs a few messages;
        under-fetching loses a newsletter for good.
        """
        terms: list[str] = []
        if since is not None:
            start = to_utc(since) - timedelta(days=1)
            terms.append(f"SINCE {start.strftime(IMAP_DATE)}")
        if until is not None:
            end = to_utc(until) + timedelta(days=1)
            terms.append(f"BEFORE {end.strftime(IMAP_DATE)}")
        return " ".join(terms) or "ALL"

    def fetch(
        self, since: datetime | None = None, until: datetime | None = None
    ) -> list[RawMessage]:
        conn = self._connect()
        criteria = self._criteria(since, until)
        try:
            status, data = conn.search(None, criteria)
        except imaplib.IMAP4.error as exc:
            raise MailboxError(f"search failed ({criteria}): {exc}") from None
        if status != "OK":
            raise MailboxError(f"search failed ({criteria}): {status}")

        uids = (data[0] or b"").split()
        log.info("%d messages match %s", len(uids), criteria)
        if not uids:
            return []

        messages: list[RawMessage] = []
        for batch in _chunks(uids, FETCH_BATCH):
            for uid in batch:
                raw = self._fetch_one(conn, uid)
                if raw is not None:
                    messages.append(RawMessage(uid=uid.decode("ascii", "replace"), raw=raw))
        return messages

    def _fetch_one(self, conn: imaplib.IMAP4_SSL, uid: bytes) -> bytes | None:
        """One message, or None if the server would not give it to us.

        A single unreadable message must not cost us the rest of the week, so
        this logs and returns None rather than raising.
        """
        try:
            status, data = conn.fetch(uid, "(RFC822)")
        except imaplib.IMAP4.error as exc:
            log.warning("message %s: fetch failed (%s)", uid.decode("ascii", "replace"), exc)
            return None
        if status != "OK" or not data:
            log.warning("message %s: fetch returned %s", uid.decode("ascii", "replace"), status)
            return None
        for part in data:
            # A successful FETCH is a list of (descriptor, payload) tuples mixed
            # with bare closing parens; the payload is the tuple's second item.
            if isinstance(part, tuple) and len(part) > 1 and isinstance(part[1], bytes):
                return part[1]
        log.warning("message %s: no RFC822 payload in response", uid.decode("ascii", "replace"))
        return None


def _chunks(items: list, size: int):
    for index in range(0, len(items), size):
        yield items[index:index + size]
