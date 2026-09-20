"""Which configured source does this message belong to?

Sender address alone is not enough, for two separate reasons.

One publisher commonly sends every newsletter it has from a single address -- EL
PAÍS and The Guardian both do -- so a source may narrow by SUBJECT, and a message
matching the sender but not the subject belongs to a newsletter we do not collect.

And the sender address is often not the publisher's at all. Público arrives from
`news@publisher-news.com`, a shared bulk-mail domain that could serve any number
of publishers, so matching on it alone risks claiming another outlet's newsletter
as Público's. There the From DISPLAY NAME is the discriminator, which is what
`sender_name_patterns` is for.

Unmatched messages are counted and logged, never guessed at. A digest built from
"probably The Economist" is worse than one built from nine sources.
"""

from __future__ import annotations

import logging
import re

from ..config import NewsletterSource
from ..models import Email
from ..text import normalize

log = logging.getLogger(__name__)


class Matcher:
    """Precompiled sender and subject rules for one run."""

    def __init__(self, sources: list[NewsletterSource]):
        self.sources = sources
        self._subjects: dict[str, list[re.Pattern]] = {
            s.name: [re.compile(normalize(p)) for p in s.subject_patterns]
            for s in sources
        }
        self._names: dict[str, list[re.Pattern]] = {
            s.name: [re.compile(normalize(p)) for p in s.sender_name_patterns]
            for s in sources
        }

    @staticmethod
    def _narrows(patterns: list[re.Pattern], text: str) -> tuple[bool, bool]:
        """(does this rule allow the message, did it actively match).

        No patterns means "allow anything", and that is NOT a match -- the
        distinction is what makes a named newsletter beat a catch-all.
        """
        if not patterns:
            return True, False
        hit = any(p.search(text) for p in patterns)
        return hit, hit

    @staticmethod
    def _sender_matches(email_address: str, rules: list[str]) -> bool:
        """True when the From address satisfies any rule.

        A rule starting with `@` is a domain rule and matches any address at that
        domain OR a subdomain of it, since bulk mailers move between
        `mail.example.com` and `news.example.com` without telling anyone. A rule
        with no `@` is treated the same way, so `aljazeera.net` and
        `@aljazeera.net` both work and neither is a silent no-match.
        Anything else must match the whole address.
        """
        address = (email_address or "").lower().strip()
        if not address:
            return False
        host = address.rpartition("@")[2]
        for rule in rules:
            rule = (rule or "").lower().strip()
            if not rule:
                continue
            if "@" not in rule or rule.startswith("@"):
                domain = rule.lstrip("@")
                if host == domain or host.endswith("." + domain):
                    return True
            elif address == rule:
                return True
        return False

    def match(self, message: Email) -> NewsletterSource | None:
        """The source this message belongs to, or None.

        Where several sources match, the most SPECIFIC wins: a source that named
        subject or display-name patterns AND matched one beats a source that
        accepts everything from the address. Without that rule a catch-all entry
        for a publisher would swallow its own named newsletters, and which one won
        would depend on config file order.

        Every rule a source names must pass. They narrow rather than accumulate,
        so a source with both a subject and a display-name pattern matches only a
        message satisfying both -- which is what makes a shared bulk-mail domain
        safe to list at all.
        """
        specific: NewsletterSource | None = None
        catch_all: NewsletterSource | None = None

        for source in self.sources:
            if not self._sender_matches(message.sender, source.senders):
                continue
            subject_ok, by_subject = self._narrows(
                self._subjects.get(source.name) or [], normalize(message.subject)
            )
            name_ok, by_name = self._narrows(
                self._names.get(source.name) or [], normalize(message.sender_name)
            )
            if not (subject_ok and name_ok):
                continue
            if by_subject or by_name:
                specific = specific or source
            else:
                catch_all = catch_all or source

        chosen = specific or catch_all
        if chosen is None:
            log.info(
                "unmatched: %r from %s", message.subject[:70] or "(no subject)",
                message.sender or "(no sender)",
            )
        return chosen


def assign(message: Email, source: NewsletterSource) -> Email:
    """Stamp a matched source onto a message, in place."""
    message.source = source.name
    message.newsletter = source.newsletter or source.name
    return message
