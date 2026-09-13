"""Both pipelines end to end, with no network and no API keys."""

import json
from datetime import datetime, timedelta, timezone

from newsdigest import pipeline
from newsdigest.config import Source
from newsdigest.models import RunStats
from newsdigest.store import Store

from conftest import StubEmbedder, make_article

MONDAY = datetime(2026, 9, 7, tzinfo=timezone.utc)


class NullFetcher:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


def fake_sources(monkeypatch, mapping: dict[str, list]):
    """Wire each source name to a fixed list of articles, or an exception."""

    class FakeAdapter:
        def __init__(self, source):
            self.source = source

        def fetch(self, source):
            result = mapping[source.name]
            if isinstance(result, Exception):
                raise result
            return list(result)

    monkeypatch.setattr(pipeline, "adapter_for", lambda source, fetcher: FakeAdapter(source))
    monkeypatch.setattr(pipeline, "Fetcher", lambda *a, **k: NullFetcher())


class TestCollect:
    def test_detects_language_per_source(self, config, store, monkeypatch):
        config.sources = [
            Source(name="BBC", rss="https://a.invalid/rss", languages=["en"]),
            Source(name="Ara", rss="https://b.invalid/rss", languages=["ca"]),
            Source(name="El País Economía", rss="https://c.invalid/rss",
                   languages=["es"], publisher="El País"),
        ]
        fake_sources(monkeypatch, {
            "BBC": [make_article("Government approves the annual budget bill",
                                 source="BBC", language=None)],
            "Ara": [make_article("El Govern aprova el pressupost anual del país",
                                 source="Ara", language=None)],
            "El País Economía": [
                make_article("El Gobierno aprueba el presupuesto anual del país",
                             source="El País Economía", language=None)],
        })
        stats = RunStats()
        pipeline.collect(config, store, stats)
        assert stats.languages == {"en": 1, "ca": 1, "es": 1}
        assert store.language_counts() == {"ca": 1, "en": 1, "es": 1}

    def test_publisher_and_source_url_are_recorded(self, config, store, monkeypatch):
        config.sources = [
            Source(name="El País Economía", rss="https://feed.invalid/eco",
                   languages=["es"], publisher="El País")
        ]
        fake_sources(monkeypatch, {
            "El País Economía": [
                make_article("La inflación se moderó en agosto según el instituto",
                             source="El País Economía", language=None)]
        })
        pipeline.collect(config, store, RunStats())
        article = store.articles_in_window(
            MONDAY, datetime.now(timezone.utc) + timedelta(days=1))[0]
        assert article.publisher == "El País"
        assert article.source_url == "https://feed.invalid/eco"

    def test_one_broken_source_does_not_stop_the_run(self, config, store, monkeypatch):
        config.sources = [
            Source(name="Good", rss="https://a.invalid/rss", languages=["en"]),
            Source(name="Bad", rss="https://b.invalid/rss", languages=["en"]),
        ]
        fake_sources(monkeypatch, {
            "Good": [make_article("A working source story about the harbour", source="Good")],
            "Bad": ConnectionError("host unreachable"),
        })
        stats = RunStats()
        pipeline.collect(config, store, stats)
        assert stats.articles_new == 1
        assert stats.sources_ok == 1
        assert [s.name for s in stats.sources_failed] == ["Bad"]
        assert "ConnectionError" in stats.sources_failed[0].error

    def test_collection_calls_no_model(self, config, store, monkeypatch):
        """The daily pipeline must never touch an LLM or embedding API."""
        called: list[str] = []
        monkeypatch.setattr(pipeline, "get_llm",
                            lambda *a, **k: called.append("llm"))
        monkeypatch.setattr(pipeline, "get_embedder",
                            lambda *a, **k: called.append("embed"))
        config.sources = [Source(name="Good", rss="https://a.invalid/rss", languages=["en"])]
        fake_sources(monkeypatch, {
            "Good": [make_article("A story about the new harbour wall", source="Good")]
        })
        pipeline.run_collect(config, pipeline.Options(db=":memory:", out=None,
                                                     dry_run=True, prune=False))
        assert called == []

    def test_rerunning_collection_adds_nothing(self, config, store, monkeypatch):
        config.sources = [Source(name="Good", rss="https://a.invalid/rss", languages=["en"])]
        articles = [make_article("A story about the new harbour wall", source="Good")]
        fake_sources(monkeypatch, {"Good": articles})
        first, second = RunStats(), RunStats()
        pipeline.collect(config, store, first)
        pipeline.collect(config, store, second)
        assert first.articles_new == 1
        assert second.articles_new == 0
        assert second.duplicates == 1


class TestWeekly:
    def _collect(self, config, monkeypatch, db):
        config.sources = [
            Source(name="BBC", rss="https://a.invalid/rss", languages=["en"]),
            Source(name="Ara", rss="https://b.invalid/rss", languages=["ca"]),
            Source(name="El País", rss="https://c.invalid/rss", languages=["es"]),
        ]
        published = MONDAY + timedelta(days=1)
        fake_sources(monkeypatch, {
            "BBC": [
                make_article("EU announces new sanctions against Russia", source="BBC",
                             language=None, published=published),
                make_article("Wildfires force evacuations on the northern coast",
                             source="BBC", language=None, published=published),
            ],
            "Ara": [make_article("La UE anuncia noves sancions contra Rússia",
                                 source="Ara", language=None, published=published)],
            "El País": [make_article("La UE anuncia nuevas sanciones contra Rusia",
                                          source="El País", language=None,
                                          published=published)],
        })
        with Store(db) as store:
            pipeline.collect(config, store, RunStats())

    def test_full_two_stage_run_offline(self, tmp_path, config, monkeypatch):
        db, out = tmp_path / "run.db", tmp_path / "site"
        self._collect(config, monkeypatch, db)

        options = pipeline.Options(db=db, out=out)
        stats = pipeline.run_weekly(
            config, options, window=(MONDAY, MONDAY + timedelta(days=7))
        )
        assert stats.digest_id == "2026-W37"
        assert stats.stories_published >= 1
        assert (out / "index.json").exists()
        assert (out / "digests" / "2026-W37.json").exists()

        index = json.loads((out / "index.json").read_text())
        assert index["digests"][0]["id"] == "2026-W37"

    def test_cross_language_merge_with_embeddings(self, tmp_path, config, monkeypatch):
        db, out = tmp_path / "run.db", tmp_path / "site"
        self._collect(config, monkeypatch, db)

        embedder = StubEmbedder({"sanctions": "eu", "sanciones": "eu", "sancions": "eu"})
        monkeypatch.setattr(pipeline, "get_embedder", lambda settings: embedder)

        stats = pipeline.run_weekly(
            config, pipeline.Options(db=db, out=out),
            window=(MONDAY, MONDAY + timedelta(days=7)),
        )
        assert stats.cross_language_clusters == 1

        payload = json.loads((out / "digests" / "2026-W37.json").read_text())
        merged = next(s for s in payload["stories"] if s["publisher_count"] == 3)
        assert sorted(merged["languages"]) == ["ca", "en", "es"]
        assert sorted(merged["publishers"]) == ["Ara", "BBC", "El País"]

    def test_without_embeddings_the_same_event_stays_split(self, tmp_path, config, monkeypatch):
        """The documented degradation, asserted so it cannot regress silently."""
        db, out = tmp_path / "run.db", tmp_path / "site"
        self._collect(config, monkeypatch, db)
        stats = pipeline.run_weekly(
            config, pipeline.Options(db=db, out=out),
            window=(MONDAY, MONDAY + timedelta(days=7)),
        )
        assert stats.cross_language_clusters == 0
        payload = json.loads((out / "digests" / "2026-W37.json").read_text())
        assert all(s["publisher_count"] == 1 for s in payload["stories"])

    def test_dry_run_writes_nothing(self, tmp_path, config, monkeypatch):
        db, out = tmp_path / "run.db", tmp_path / "site"
        self._collect(config, monkeypatch, db)
        pipeline.run_weekly(
            config, pipeline.Options(db=db, out=out, dry_run=True),
            window=(MONDAY, MONDAY + timedelta(days=7)),
        )
        assert not out.exists()
        with Store(db) as store:
            # No weekly run recorded, and no digest persisted.
            kinds = [r["kind"] for r in store.conn.execute("SELECT kind FROM runs")]
            assert "weekly" not in kinds
            assert store.list_digests() == []

    def test_rerun_is_idempotent(self, tmp_path, config, monkeypatch):
        db, out = tmp_path / "run.db", tmp_path / "site"
        self._collect(config, monkeypatch, db)
        window = (MONDAY, MONDAY + timedelta(days=7))
        options = pipeline.Options(db=db, out=out)
        first = pipeline.run_weekly(config, options, window=window)
        second = pipeline.run_weekly(config, options, window=window)
        assert first.digest_id == second.digest_id
        assert second.enriched == 0            # everything already enriched
        with Store(db) as store:
            assert len(store.list_digests()) == 1


class TestCli:
    def test_collect_and_digest_via_the_cli(self, tmp_path, monkeypatch):
        from newsdigest import cli

        db, out = tmp_path / "cli.db", tmp_path / "site"
        monkeypatch.setattr(
            cli.pipeline, "adapter_for",
            lambda source, fetcher: type("A", (), {
                "fetch": lambda self, s: [
                    make_article("Council approves the new harbour development",
                                 source=s.name, language=None)]
            })(),
        )
        monkeypatch.setattr(cli.pipeline, "Fetcher", lambda *a, **k: NullFetcher())

        assert cli.main(["--db", str(db), "--out", str(out), "collect"]) == 0
        assert cli.main(["--db", str(db), "--out", str(out), "digest",
                         "--no-llm", "--no-embeddings", "--week",
                         f"{datetime.now(timezone.utc).isocalendar()[0]}-W"
                         f"{datetime.now(timezone.utc).isocalendar()[1]:02d}"]) == 0
        assert (out / "index.json").exists()
        assert cli.main(["--db", str(db), "--out", str(out), "build"]) == 0
        assert cli.main(["--db", str(db), "stats"]) == 0

    def test_bad_week_is_reported_not_crashed(self, tmp_path):
        from newsdigest import cli

        assert cli.main(["--db", str(tmp_path / "x.db"), "digest", "--week", "junk"]) == 2

    def test_global_flags_work_on_either_side(self, tmp_path):
        from newsdigest import cli

        assert cli.main(["--db", str(tmp_path / "a.db"), "stats"]) == 0
        assert cli.main(["stats", "--db", str(tmp_path / "b.db")]) == 0
