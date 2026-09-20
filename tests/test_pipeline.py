"""The pipeline end to end, with no network, no mailbox and no API keys."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from newsdigest import pipeline
from newsdigest.config import NewsletterSource
from newsdigest.inbox.base import Mailbox, MailboxError, RawMessage
from newsdigest.models import RunStats

from conftest import StubEmbedder

MONDAY = datetime(2026, 9, 14, tzinfo=timezone.utc)
WEEK = (MONDAY, MONDAY + timedelta(days=7))
FIXTURES = Path(__file__).parent / "fixtures" / "emails"


def message(
    source_domain="example.invalid",
    *,
    subject="The week in review",
    date="Wed, 16 Sep 2026 09:00:00 +0000",
    message_id="<m1@example.invalid>",
    body="<html><body>{}</body></html>",
    items=(("A significant thing happened in the world", "https://news.example/a"),),
):
    """One RFC822 message, built from a template rather than a fixture file."""
    blocks = "".join(
        f'<table><tr><td><h2><a href="{url}">{title}</a></h2>'
        f"<p>Reported at length, with enough detail to summarize from.</p>"
        f"</td></tr></table>"
        for title, url in items
    )
    raw = (
        f"From: Example <news@{source_domain}>\r\n"
        f"To: digest@example.com\r\n"
        f"Subject: {subject}\r\n"
        f"Date: {date}\r\n"
        f"Message-ID: {message_id}\r\n"
        f"MIME-Version: 1.0\r\n"
        f'Content-Type: text/html; charset="utf-8"\r\n'
        f"\r\n"
        f"{body.format(blocks)}\r\n"
    )
    return RawMessage(uid="1", raw=raw.encode("utf-8"))


class FakeMailbox(Mailbox):
    """A mailbox that hands back fixed messages, or raises."""

    name = "fake"

    def __init__(self, messages, error=None):
        self.messages = list(messages)
        self.error = error
        self.closed = False
        self.ranges = []

    def fetch(self, since=None, until=None):
        self.ranges.append((since, until))
        if self.error:
            raise self.error
        return list(self.messages)

    def close(self):
        self.closed = True


def wire(monkeypatch, mailbox):
    monkeypatch.setattr(pipeline, "get_mailbox", lambda settings: mailbox)
    return mailbox


@pytest.fixture
def options(tmp_path):
    return pipeline.Options(
        db=tmp_path / "test.db",
        out=tmp_path / "digests",
        readme=None,
        # No network in tests: link resolution is the only stage that would want
        # one, and the static unwrapping it falls back to is what we test.
        resolve_links=False,
    )


class TestFetch:
    def test_stores_matching_messages(self, config, store, options, monkeypatch):
        wire(monkeypatch, FakeMailbox([message()]))
        stats = RunStats()
        pipeline.fetch(config, store, stats, since=WEEK[0], until=WEEK[1],
                       options=options)
        assert (stats.emails_fetched, stats.emails_new) == (1, 1)
        assert len(store.emails_in_window(*WEEK)) == 1

    def test_unmatched_senders_are_counted_not_guessed(
        self, config, store, options, monkeypatch
    ):
        """A digest built from "probably The Economist" is worse than one built
        from nine sources."""
        wire(monkeypatch, FakeMailbox([
            message(),
            message(source_domain="stranger.invalid", message_id="<m2@x>"),
        ]))
        stats = RunStats()
        pipeline.fetch(config, store, stats, since=WEEK[0], until=WEEK[1],
                       options=options)
        assert stats.emails_fetched == 1
        assert stats.emails_unmatched == 1

    def test_the_window_is_enforced_on_the_parsed_header(
        self, config, store, options, monkeypatch
    ):
        """IMAP's own date search is day-granular in an unknown timezone, so it
        over-returns on purpose and the window is applied here."""
        wire(monkeypatch, FakeMailbox([
            message(date="Wed, 16 Sep 2026 09:00:00 +0000", message_id="<in@x>"),
            message(date="Wed, 30 Sep 2026 09:00:00 +0000", message_id="<out@x>"),
        ]))
        stats = RunStats()
        pipeline.fetch(config, store, stats, since=WEEK[0], until=WEEK[1],
                       options=options)
        assert stats.emails_fetched == 1

    def test_a_message_is_never_stored_twice(
        self, config, store, options, monkeypatch
    ):
        """Re-running a week must be free, which is what keys the whole design."""
        wire(monkeypatch, FakeMailbox([message()]))
        for _ in range(3):
            stats = RunStats()
            pipeline.fetch(config, store, stats, since=WEEK[0], until=WEEK[1],
                           options=options)
        assert stats.emails_fetched == 1
        assert stats.emails_new == 0
        assert len(store.emails_in_window(*WEEK)) == 1

    def test_an_unparseable_message_does_not_cost_the_week(
        self, config, store, options, monkeypatch
    ):
        wire(monkeypatch, FakeMailbox([
            RawMessage(uid="bad", raw=b"\xff\xfe not a message at all"),
            message(),
        ]))
        stats = RunStats()
        pipeline.fetch(config, store, stats, since=WEEK[0], until=WEEK[1],
                       options=options)
        assert stats.emails_fetched == 1

    def test_dry_run_stores_nothing(self, config, store, options, monkeypatch):
        wire(monkeypatch, FakeMailbox([message()]))
        options.dry_run = True
        stats = RunStats()
        pipeline.fetch(config, store, stats, since=WEEK[0], until=WEEK[1],
                       options=options)
        assert stats.emails_new == 1
        assert store.emails_in_window(*WEEK) == []

    def test_a_mailbox_that_will_not_open_is_fatal(
        self, config, store, options, monkeypatch
    ):
        """The one stage with nothing to fall back on: no mail, no digest."""
        wire(monkeypatch, FakeMailbox([], error=MailboxError("login rejected")))
        with pytest.raises(MailboxError):
            pipeline.fetch(config, store, RunStats(), since=WEEK[0], until=WEEK[1],
                           options=options)


class TestParse:
    def _stored(self, config, store, options, messages):
        stats = RunStats()
        pipeline.fetch(config, store, stats, since=WEEK[0], until=WEEK[1],
                       options=options)
        return stats

    def test_extracts_articles_and_marks_the_message_parsed(
        self, config, store, options, monkeypatch
    ):
        wire(monkeypatch, FakeMailbox([message(items=(
            ("A significant thing happened in the world", "https://news.example/a"),
            ("Another significant thing happened elsewhere", "https://news.example/b"),
        ))]))
        self._stored(config, store, options, None)

        stats = RunStats()
        pipeline.parse(config, store, stats, window=WEEK, options=options)
        assert stats.articles_new == 2
        assert all(m.parsed for m in store.emails_in_window(*WEEK))

    def test_reparsing_is_skipped_until_forced(
        self, config, store, options, monkeypatch
    ):
        wire(monkeypatch, FakeMailbox([message()]))
        self._stored(config, store, options, None)

        first = RunStats()
        pipeline.parse(config, store, first, window=WEEK, options=options)
        assert first.emails_parsed == 1

        again = RunStats()
        pipeline.parse(config, store, again, window=WEEK, options=options)
        assert again.emails_parsed == 0, "a parsed message must not be parsed again"

        options.force = True
        forced = RunStats()
        pipeline.parse(config, store, forced, window=WEEK, options=options)
        assert forced.emails_parsed == 1

    def test_language_is_detected_per_source(
        self, config, store, options, monkeypatch
    ):
        config.sources = [
            NewsletterSource(name="English", senders=["news@en.invalid"],
                             languages=["en"]),
            NewsletterSource(name="Spanish", senders=["news@es.invalid"],
                             languages=["es"]),
        ]
        config.settings.supported_languages = ["en", "es"]
        wire(monkeypatch, FakeMailbox([
            message(source_domain="en.invalid", message_id="<en@x>", items=(
                ("The government approves the annual budget bill",
                 "https://news.example/en"),)),
            message(source_domain="es.invalid", message_id="<es@x>", items=(
                ("El Gobierno aprueba el presupuesto anual del país",
                 "https://news.example/es"),)),
        ]))
        self._stored(config, store, options, None)

        stats = RunStats()
        pipeline.parse(config, store, stats, window=WEEK, options=options)
        assert stats.languages == {"en": 1, "es": 1}

    def test_excluded_sections_never_reach_the_database(
        self, config, store, options, monkeypatch
    ):
        config.sources[0].exclude_url_patterns = ["/deportes/"]
        wire(monkeypatch, FakeMailbox([message(items=(
            ("A significant thing happened in the world", "https://news.example/world/a"),
            ("Resumen de la jornada con goles en el descuento",
             "https://news.example/deportes/b"),
        ))]))
        self._stored(config, store, options, None)

        stats = RunStats()
        pipeline.parse(config, store, stats, window=WEEK, options=options)
        assert stats.excluded == 1
        assert stats.articles_new == 1

    def test_one_broken_newsletter_does_not_stop_the_others(
        self, config, store, options, monkeypatch
    ):
        """A source that throws is recorded and the run carries on."""
        wire(monkeypatch, FakeMailbox([message(message_id="<ok@x>")]))
        self._stored(config, store, options, None)

        import newsdigest.pipeline as mod

        real = mod.extract
        calls = {"n": 0}

        def flaky(email, **kw):
            calls["n"] += 1
            raise RuntimeError("extractor blew up")

        monkeypatch.setattr(mod, "extract", flaky)
        stats = RunStats()
        pipeline.parse(config, store, stats, window=WEEK, options=options)
        assert calls["n"] == 1
        assert stats.sources_failed
        assert "extractor blew up" in stats.sources_failed[0].error
        monkeypatch.setattr(mod, "extract", real)

    def test_only_the_named_source_is_parsed(
        self, config, store, options, monkeypatch
    ):
        config.sources = [
            NewsletterSource(name="A", senders=["news@a.invalid"], languages=["en"]),
            NewsletterSource(name="B", senders=["news@b.invalid"], languages=["en"]),
        ]
        wire(monkeypatch, FakeMailbox([
            message(source_domain="a.invalid", message_id="<a@x>"),
            message(source_domain="b.invalid", message_id="<b@x>"),
        ]))
        self._stored(config, store, options, None)

        options.only = ["A"]
        stats = RunStats()
        pipeline.parse(config, store, stats, window=WEEK, options=options)
        assert stats.emails_parsed == 1


class TestRunAll:
    def test_fetch_parse_and_publish_in_one_go(self, config, options, monkeypatch):
        wire(monkeypatch, FakeMailbox([message(items=(
            ("EU leaders agree a new sanctions package after the summit",
             "https://news.example/world/sanctions"),
            ("Floods displace thousands in the north of the country",
             "https://news.example/world/floods"),
        ))]))
        stats = pipeline.run_all(config, options, window=WEEK)
        assert stats.emails_new == 1
        assert stats.articles_new == 2
        assert stats.stories_published == 2
        assert (options.out / "2026" / "2026-W38.md").exists()

    def test_a_mailbox_failure_does_not_publish_a_stale_digest(
        self, config, options, monkeypatch
    ):
        """Publishing last week's digest as though it were this week's is the one
        silent failure worth being loud about."""
        wire(monkeypatch, FakeMailbox([], error=MailboxError("host unreachable")))
        with pytest.raises(MailboxError):
            pipeline.run_all(config, options, window=WEEK)
        assert not (options.out / "2026" / "2026-W38.md").exists()

    def test_cross_language_coverage_merges_into_one_story(
        self, config, options, monkeypatch
    ):
        """The whole reason embeddings are not optional."""
        config.sources = [
            NewsletterSource(name="English", senders=["news@en.invalid"],
                             languages=["en"], publisher="English Outlet"),
            NewsletterSource(name="Spanish", senders=["news@es.invalid"],
                             languages=["es"], publisher="Spanish Outlet"),
        ]
        wire(monkeypatch, FakeMailbox([
            message(source_domain="en.invalid", message_id="<en@x>", items=(
                ("EU leaders agree new sanctions against Russia",
                 "https://news.example/en/sanctions"),)),
            message(source_domain="es.invalid", message_id="<es@x>", items=(
                ("La UE acuerda nuevas sanciones contra Rusia",
                 "https://news.example/es/sanciones"),)),
        ]))
        embedder = StubEmbedder({"sanctions": "eu", "sanciones": "eu"})
        monkeypatch.setattr(pipeline, "get_embedder", lambda settings: embedder)

        stats = pipeline.run_all(config, options, window=WEEK)
        assert stats.articles_new == 2
        assert stats.stories_published == 1
        assert stats.cross_language_clusters == 1
