"""Mail: parsing a message, and matching it to a configured source."""

from datetime import timezone
from pathlib import Path

import pytest

from newsdigest.config import NewsletterSource
from newsdigest.inbox import Matcher, assign, parse_message
from newsdigest.inbox.imap import IMAPMailbox
from newsdigest.inbox.base import MailboxError

FIXTURES = Path(__file__).parent / "fixtures" / "emails"


def build(
    *,
    sender="Example <news@example.invalid>",
    subject="The week in review",
    date="Wed, 16 Sep 2026 09:00:00 +0000",
    message_id="<m1@example.invalid>",
    headers="",
    body='Content-Type: text/html; charset="utf-8"\r\n\r\n<p>Hello</p>',
) -> bytes:
    lines = [f"From: {sender}", "To: digest@example.com", f"Subject: {subject}",
             f"Date: {date}"]
    if message_id is not None:
        lines.append(f"Message-ID: {message_id}")
    lines.append("MIME-Version: 1.0")
    if headers:
        lines.append(headers)
    return ("\r\n".join(lines) + "\r\n" + body + "\r\n").encode("utf-8")


class TestParseMessage:
    def test_reads_the_headers(self):
        message = parse_message(build())
        assert message.subject == "The week in review"
        assert message.sender == "news@example.invalid"
        assert message.sender_name == "Example"
        assert message.received_at.tzinfo is not None
        assert message.received_at.astimezone(timezone.utc).hour == 9

    def test_decodes_encoded_words_in_the_subject(self):
        """Every Spanish newsletter arrives this way."""
        message = parse_message(
            build(subject="=?utf-8?q?Lo_m=C3=A1s_importante_de_la_semana?=")
        )
        assert message.subject == "Lo más importante de la semana"

    def test_decodes_an_encoded_sender_name(self):
        message = parse_message(
            build(sender="=?utf-8?q?EL_PA=C3=8DS?= <news@elpais.invalid>")
        )
        assert message.sender_name == "EL PAÍS"
        assert message.sender == "news@elpais.invalid"

    def test_the_sender_is_lowercased(self):
        message = parse_message(build(sender="News@Example.Invalid"))
        assert message.sender == "news@example.invalid"

    def test_prefers_html_over_the_plain_alternative(self):
        """A newsletter's plain-text part is a courtesy copy with the links
        flattened out, and links are most of what we are here for."""
        raw = build(body=(
            'Content-Type: multipart/alternative; boundary="b"\r\n\r\n'
            "--b\r\nContent-Type: text/plain\r\n\r\nplain version\r\n"
            '--b\r\nContent-Type: text/html\r\n\r\n<p>html version</p>\r\n--b--'
        ))
        message = parse_message(raw)
        assert "html version" in message.html_body
        assert "plain version" in message.text_body
        assert "html" in message.body

    def test_takes_the_longest_part_of_each_type(self):
        """A multipart/related newsletter often carries a short HTML preamble
        alongside the real body, and the first part is the preamble."""
        raw = build(body=(
            'Content-Type: multipart/mixed; boundary="b"\r\n\r\n'
            "--b\r\nContent-Type: text/html\r\n\r\n<p>short</p>\r\n"
            "--b\r\nContent-Type: text/html\r\n\r\n"
            + "<p>the actual body, much longer than the preamble</p>" * 3
            + "\r\n--b--"
        ))
        assert "actual body" in parse_message(raw).html_body

    def test_skips_attachments(self):
        raw = build(body=(
            'Content-Type: multipart/mixed; boundary="b"\r\n\r\n'
            "--b\r\nContent-Type: text/html\r\n\r\n<p>body</p>\r\n"
            "--b\r\nContent-Type: text/plain\r\n"
            'Content-Disposition: attachment; filename="notes.txt"\r\n\r\n'
            "an attachment that is not the body\r\n--b--"
        ))
        message = parse_message(raw)
        assert "attachment that is not" not in message.text_body

    def test_an_unknown_charset_still_decodes(self):
        """Latin-1 decodes any byte sequence, so this cannot fail twice."""
        raw = build(body='Content-Type: text/html; charset="definitely-not-a-charset"'
                         "\r\n\r\n<p>body</p>")
        assert "body" in parse_message(raw).html_body

    def test_a_missing_message_id_gets_a_stable_stand_in(self):
        """Synthesised from the bytes, not the clock: an id that changed between
        runs would re-ingest the same newsletter every week."""
        raw = build(message_id=None)
        first, second = parse_message(raw), parse_message(raw)
        assert first.message_id.startswith("synthetic-")
        assert first.id == second.id

    def test_the_angle_brackets_are_stripped_from_the_id(self):
        assert parse_message(build()).message_id == "m1@example.invalid"

    def test_a_message_with_no_date_still_parses(self):
        message = parse_message(build(date=""))
        assert message.received_at is not None

    def test_garbage_raises_rather_than_returning_an_empty_message(self):
        """An empty Email would be stored and marked parsed, hiding the problem."""
        with pytest.raises(ValueError):
            parse_message(b"")

    def test_the_real_fixtures_parse(self):
        for path in sorted(FIXTURES.glob("*.eml")):
            message = parse_message(path.read_bytes())
            assert message.subject
            assert message.sender
            assert message.html_body

    def test_the_body_hash_keys_the_parse_cache(self):
        a = parse_message(build())
        b = parse_message(build(subject="A different subject"))
        assert a.content_hash() == b.content_hash(), "same body, same hash"


class TestMatching:
    def source(self, **kw):
        kw.setdefault("name", "Example")
        kw.setdefault("senders", ["news@example.invalid"])
        return NewsletterSource(**kw)

    def test_an_exact_address_matches(self):
        matcher = Matcher([self.source()])
        assert matcher.match(parse_message(build())).name == "Example"

    def test_a_different_address_does_not(self):
        matcher = Matcher([self.source(senders=["other@example.invalid"])])
        assert matcher.match(parse_message(build())) is None

    @pytest.mark.parametrize("rule", ["@example.invalid", "example.invalid"])
    def test_a_domain_rule_matches_with_or_without_the_at(self, rule):
        """Writing it without the @ is the obvious mistake, and a silent
        no-match is the worst possible response to it."""
        matcher = Matcher([self.source(senders=[rule])])
        assert matcher.match(parse_message(build())) is not None

    def test_a_domain_rule_matches_a_subdomain(self):
        """Bulk mailers move between mail. and news. hosts without notice."""
        matcher = Matcher([self.source(senders=["@example.invalid"])])
        message = parse_message(build(sender="news@mail.example.invalid"))
        assert matcher.match(message) is not None

    def test_a_domain_rule_does_not_match_a_lookalike(self):
        matcher = Matcher([self.source(senders=["@example.invalid"])])
        message = parse_message(build(sender="news@notexample.invalid"))
        assert matcher.match(message) is None

    def test_a_subject_pattern_narrows_one_address(self):
        """Publishers send every newsletter they have from one address."""
        matcher = Matcher([
            self.source(name="Saturday", subject_patterns=["saturday edition"])
        ])
        assert matcher.match(parse_message(build(subject="The Saturday Edition"))) is not None
        assert matcher.match(parse_message(build(subject="Morning Briefing"))) is None

    def test_subject_matching_ignores_accents_and_case(self):
        matcher = Matcher([self.source(subject_patterns=["lo mas importante"])])
        message = parse_message(
            build(subject="=?utf-8?q?Lo_M=C3=A1s_Importante_de_la_semana?=")
        )
        assert matcher.match(message) is not None

    def test_a_named_newsletter_beats_a_catch_all(self):
        """Otherwise a catch-all entry swallows the publisher's own newsletters,
        and which one won would depend on config file order."""
        catch_all = self.source(name="AnyGuardian", senders=["@example.invalid"])
        named = self.source(name="Saturday", senders=["@example.invalid"],
                            subject_patterns=["saturday edition"])
        for order in ([catch_all, named], [named, catch_all]):
            matcher = Matcher(order)
            got = matcher.match(parse_message(build(subject="The Saturday Edition")))
            assert got.name == "Saturday"

    def test_a_catch_all_still_takes_what_no_pattern_claims(self):
        matcher = Matcher([
            self.source(name="AnyGuardian", senders=["@example.invalid"]),
            self.source(name="Saturday", senders=["@example.invalid"],
                        subject_patterns=["saturday edition"]),
        ])
        got = matcher.match(parse_message(build(subject="Morning Briefing")))
        assert got.name == "AnyGuardian"

    def test_assign_stamps_the_source_and_newsletter(self):
        message = parse_message(build())
        source = self.source(newsletter="Weekly Roundup")
        assign(message, source)
        assert (message.source, message.newsletter) == ("Example", "Weekly Roundup")

    def test_a_message_with_no_sender_matches_nothing(self):
        matcher = Matcher([self.source(senders=["@example.invalid"])])
        assert matcher.match(parse_message(build(sender=""))) is None


class TestIMAPSettings:
    def test_incomplete_credentials_are_refused_before_connecting(self):
        with pytest.raises(MailboxError, match="NEWS_EMAIL"):
            IMAPMailbox(host="", username="u", password="p")

    def test_the_date_range_is_widened_at_both_ends(self):
        """IMAP compares against the server's own date, at day granularity, in a
        timezone we do not know -- so over-fetching is the safe direction."""
        from datetime import datetime

        mailbox = IMAPMailbox(host="h", username="u", password="p")
        criteria = mailbox._criteria(
            datetime(2026, 9, 14, tzinfo=timezone.utc),
            datetime(2026, 9, 21, tzinfo=timezone.utc),
        )
        assert "SINCE 13-Sep-2026" in criteria
        assert "BEFORE 22-Sep-2026" in criteria

    def test_no_dates_searches_everything(self):
        mailbox = IMAPMailbox(host="h", username="u", password="p")
        assert mailbox._criteria(None, None) == "ALL"

    def test_the_description_never_carries_the_password(self):
        """Section 28: this string goes into log output."""
        mailbox_settings_description = IMAPMailbox(
            host="imap.example.com", username="digest@example.com", password="sekrit"
        )
        assert "sekrit" not in repr(mailbox_settings_description.__dict__.get("host"))
        from newsdigest.config import MailboxSettings

        described = MailboxSettings(
            host="imap.example.com", username="digest@example.com", password="sekrit"
        ).describe()
        assert "sekrit" not in described
        assert "example.com" not in described.split("@")[0]


class TestSenderNameMatching:
    """A publisher's mail often comes from a SHARED bulk-mail domain rather than
    its own. Público arrives from `news@publisher-news.com`, which identifies
    nothing and could serve any number of publishers, so matching on it alone
    risks claiming another outlet's newsletter. The display name is what
    distinguishes them.
    """

    def source(self, **kw):
        kw.setdefault("name", "Shared")
        kw.setdefault("senders", ["@shared-mailer.example"])
        return NewsletterSource(**kw)

    def message(self, name, subject="Weekly roundup"):
        return parse_message(build(
            sender=f"{name} <news@shared-mailer.example>", subject=subject
        ))

    def test_a_display_name_pattern_matches(self):
        matcher = Matcher([self.source(sender_name_patterns=["publico"])])
        assert matcher.match(self.message("Público - En pocas palabras")) is not None

    def test_a_different_publisher_on_the_same_mailer_is_rejected(self):
        """The case that makes a shared domain safe to list at all."""
        matcher = Matcher([self.source(sender_name_patterns=["publico"])])
        assert matcher.match(self.message("Some Other Outlet")) is None

    def test_display_name_matching_ignores_accents_and_case(self):
        matcher = Matcher([self.source(sender_name_patterns=["publico"])])
        assert matcher.match(self.message("PÚBLICO")) is not None

    def test_subject_and_name_rules_both_have_to_pass(self):
        """They narrow rather than accumulate, which is the whole point: the same
        sender and display name also carry the publisher's DAILY newsletter."""
        matcher = Matcher([self.source(
            sender_name_patterns=["publico"],
            subject_patterns=["en pocas palabras"],
        )])
        assert matcher.match(
            self.message("Público", subject="En pocas palabras: la semana")
        ) is not None
        assert matcher.match(
            self.message("Público", subject="Hoy en Público: martes")
        ) is None
        assert matcher.match(
            self.message("Otro medio", subject="En pocas palabras")
        ) is None

    def test_a_name_pattern_counts_as_specific_against_a_catch_all(self):
        catch_all = self.source(name="Anything")
        named = self.source(name="Named", sender_name_patterns=["publico"])
        for order in ([catch_all, named], [named, catch_all]):
            got = Matcher(order).match(self.message("Público"))
            assert got.name == "Named"

    def test_no_name_pattern_still_accepts_any_name(self):
        matcher = Matcher([self.source()])
        assert matcher.match(self.message("Anyone At All")) is not None


class TestShippedSourceRules:
    """The rules in config/sources.yaml, against the senders really observed."""

    def _match(self, name, sender, subject):
        # load_sources, not load_config: the latter reads `.env` into os.environ,
        # which is a side effect no test should cause for the ones after it.
        from newsdigest.config import load_sources

        enabled = [s for s in load_sources() if s.enabled]
        message = parse_message(build(sender=f"{name} <{sender}>", subject=subject))
        got = Matcher(enabled).match(message)
        return got.name if got else None

    def test_el_salto_matches_its_weekly(self):
        assert self._match(
            "El Salto boletines", "redes@elsaltodiario.com",
            "Los peligros de la IA y un PSOE catatónico, entre otros temas de la semana",
        ) == "el-salto"

    def test_publico_matches_its_weekly_through_the_shared_mailer(self):
        assert self._match(
            "Público - En pocas palabras", "news@publisher-news.com",
            "En pocas palabras: lo que deja la semana",
        ) == "publico-pocas-palabras"

    def test_publicos_daily_is_deliberately_not_collected(self):
        """This is a weekly digest. The daily arrives from the same sender and
        display name, so only the subject rule separates them."""
        assert self._match(
            "Público - Hoy en Público", "news@publisher-news.com",
            "Togas contra el Gobierno: diez ejemplos de ataques sin consecuencias",
        ) is None

    def test_an_unrelated_publisher_on_the_shared_mailer_is_not_claimed(self):
        assert self._match(
            "Unrelated Daily", "news@publisher-news.com",
            "En pocas palabras about something else",
        ) is None


class TestLoginErrors:
    """The server's reason is relayed, and the credential never is.

    Swallowing the reason entirely made this the least diagnosable failure in the
    project: "login rejected" cannot distinguish a wrong password from an app
    password created on a different Google account, and only the server can.
    """

    def _failing(self, monkeypatch, server_message):
        import imaplib

        mailbox = IMAPMailbox(
            host="imap.gmail.com", username="someone@gmail.com",
            password="tjfa gmgp cgch dnfq",
        )

        class FakeConn:
            def login(self, username, password):
                raise imaplib.IMAP4.error(server_message)

        monkeypatch.setattr(imaplib, "IMAP4_SSL", lambda *a, **k: FakeConn())
        return mailbox

    def test_the_servers_reason_is_relayed(self, monkeypatch):
        mailbox = self._failing(
            monkeypatch, b"[AUTHENTICATIONFAILED] Invalid credentials (Failure)"
        )
        with pytest.raises(MailboxError, match="Invalid credentials"):
            mailbox.fetch()

    def test_the_message_says_what_to_check(self, monkeypatch):
        mailbox = self._failing(monkeypatch, "nope")
        with pytest.raises(MailboxError, match="created on someone@gmail.com"):
            mailbox.fetch()

    @pytest.mark.parametrize("echoed", [
        "tjfa gmgp cgch dnfq",          # as pasted, with spaces
        "tjfagmgpcgchdnfq",             # as sent, without
    ])
    def test_a_server_that_echoes_the_password_is_scrubbed(self, monkeypatch, echoed):
        """Some servers include the whole login line in the error."""
        mailbox = self._failing(monkeypatch, f"LOGIN someone@gmail.com {echoed} failed")
        with pytest.raises(MailboxError) as caught:
            mailbox.fetch()
        message = str(caught.value)
        assert echoed not in message
        assert "[redacted]" in message

    def test_an_unreachable_host_is_a_different_message(self, monkeypatch):
        import imaplib

        def boom(*a, **k):
            raise OSError("nodename nor servname provided")

        monkeypatch.setattr(imaplib, "IMAP4_SSL", boom)
        mailbox = IMAPMailbox(host="imap.bogus", username="u", password="p")
        with pytest.raises(MailboxError, match="cannot reach imap.bogus"):
            mailbox.fetch()
