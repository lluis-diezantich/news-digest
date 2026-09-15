"""Static output: the archive, the index, and RSS."""

import json
import re
from datetime import datetime, timedelta, timezone

from newsdigest import clustering, render
from newsdigest.digest import DigestResult
from newsdigest.models import Digest

from conftest import make_article

MONDAY = datetime(2026, 9, 7, tzinfo=timezone.utc)


def seed_digest(store, week="2026-W37", *, offset_weeks=0, titles=None):
    titles = titles or ["Parliament approves the budget", "Wildfires close the coast road"]
    start = MONDAY + timedelta(weeks=offset_weeks)
    articles = [
        make_article(title, source=f"Outlet {i}", publisher=f"Outlet {i}",
                     importance=0.5 + 0.1 * i, relevance=0.5,
                     topics=["world"], entities=["Parliament"],
                     published=start + timedelta(days=1, hours=i))
        for i, title in enumerate(titles)
    ]
    store.insert_articles(articles)
    pairs = []
    for article in articles:
        story = clustering.build_story([article])
        story.score = 1.0 + articles.index(article) * 0.1
        store.replace_story(story, [article.id])
        pairs.append((story, [article]))
    pairs.sort(key=lambda p: -p[0].score)
    digest = Digest(id=week, period_start=start, period_end=start + timedelta(days=7),
                    story_ids=[s.id for s, _ in pairs], article_count=len(articles))
    store.save_digest(digest)
    return DigestResult(digest=digest, stories=pairs)


class TestWriteSite:
    def test_writes_the_expected_files(self, tmp_path, config, store):
        result = seed_digest(store)
        render.write_site(tmp_path, config, store, result)
        assert (tmp_path / "index.html").exists()
        assert (tmp_path / "index.json").exists()
        assert (tmp_path / "digests" / "2026-W37.json").exists()
        assert (tmp_path / "feed.xml").exists()
        assert (tmp_path / ".nojekyll").exists()

    def test_digest_payload_shape(self, tmp_path, config, store):
        result = seed_digest(store)
        render.write_site(tmp_path, config, store, result)
        payload = json.loads((tmp_path / "digests" / "2026-W37.json").read_text())
        assert payload["id"] == "2026-W37"
        assert payload["label"]
        assert payload["story_count"] == 2
        assert payload["output_language"] == "en"
        assert {p["name"] for p in payload["publishers"]} == {"Outlet 0", "Outlet 1"}
        assert payload["languages"] == [{"code": "en", "articles": 2}]

        story = payload["stories"][0]
        for field in ("headline", "summary", "why_it_matters", "key_facts", "topics",
                      "entities", "importance", "relevance", "score",
                      "publishers", "publisher_count", "languages", "articles"):
            assert field in story, field
        article = story["articles"][0]
        for field in ("title", "url", "source", "publisher", "language", "published_at"):
            assert field in article, field

    def test_original_title_and_url_are_preserved(self, tmp_path, config, store):
        """Attribution: the link and the outlet's own headline must survive."""
        result = seed_digest(store)
        render.write_site(tmp_path, config, store, result)
        payload = json.loads((tmp_path / "digests" / "2026-W37.json").read_text())
        titles = {a["title"] for s in payload["stories"] for a in s["articles"]}
        assert "Parliament approves the budget" in titles
        urls = {a["url"] for s in payload["stories"] for a in s["articles"]}
        assert all(u.startswith("https://") for u in urls)

    def test_story_order_follows_the_digest_ranking(self, tmp_path, config, store):
        result = seed_digest(store)
        render.write_site(tmp_path, config, store, result)
        payload = json.loads((tmp_path / "digests" / "2026-W37.json").read_text())
        assert [s["id"] for s in payload["stories"]] == result.digest.story_ids


class TestArchive:
    def test_every_week_gets_its_own_file_and_index_entry(self, tmp_path, config, store):
        seed_digest(store, "2026-W36", offset_weeks=-1,
                    titles=["Older story about a ferry"])
        result = seed_digest(store, "2026-W37")
        render.write_site(tmp_path, config, store, result)

        assert (tmp_path / "digests" / "2026-W36.json").exists()
        assert (tmp_path / "digests" / "2026-W37.json").exists()
        index = json.loads((tmp_path / "index.json").read_text())
        assert [d["id"] for d in index["digests"]] == ["2026-W37", "2026-W36"]
        assert index["archive"] is True
        assert index["digests"][0]["story_count"] == 2

    def test_archive_disabled_writes_only_the_latest(self, tmp_path, config, store):
        seed_digest(store, "2026-W36", offset_weeks=-1, titles=["Older story"])
        result = seed_digest(store, "2026-W37")
        config.digest.archive = False
        render.write_site(tmp_path, config, store, result)
        index = json.loads((tmp_path / "index.json").read_text())
        assert [d["id"] for d in index["digests"]] == ["2026-W37"]
        assert not (tmp_path / "digests" / "2026-W36.json").exists()

    def test_archive_limit_is_respected(self, tmp_path, config, store):
        for week in range(30, 38):
            seed_digest(store, f"2026-W{week}", offset_weeks=week - 37,
                        titles=[f"Story from week {week} about local matters"])
        config.digest.archive_limit = 3
        render.write_site(tmp_path, config, store, None)
        index = json.loads((tmp_path / "index.json").read_text())
        assert len(index["digests"]) == 3

    def test_output_is_reproducible_from_the_database_alone(self, tmp_path, config, store):
        """Delete docs/ and one `build` restores it."""
        seed_digest(store, "2026-W36", offset_weeks=-1, titles=["Older story"])
        seed_digest(store, "2026-W37")
        render.write_site(tmp_path, config, store, None)
        first = json.loads((tmp_path / "digests" / "2026-W37.json").read_text())

        for path in tmp_path.rglob("*"):
            if path.is_file():
                path.unlink()
        render.write_site(tmp_path, config, store, None)
        second = json.loads((tmp_path / "digests" / "2026-W37.json").read_text())
        assert first["stories"] == second["stories"]

    def test_no_digest_yet_still_writes_a_valid_index(self, tmp_path, config, store):
        render.write_site(tmp_path, config, store, None)
        index = json.loads((tmp_path / "index.json").read_text())
        assert index["digests"] == []
        assert (tmp_path / "index.html").exists()


class TestRss:
    def test_contains_one_item_per_story_with_attribution(self, tmp_path, config, store):
        result = seed_digest(store)
        render.write_site(tmp_path, config, store, result, site_url="https://me.example/")
        xml = (tmp_path / "feed.xml").read_text()
        assert xml.count("<item>") == 2
        assert "https://me.example/" in xml
        assert "Sources:" in xml
        assert "Outlet 0" in xml

    def test_escapes_markup_in_headlines(self, tmp_path, config, store):
        result = seed_digest(store, titles=["Rates rise & <b>markets</b> react"])
        render.write_site(tmp_path, config, store, result)
        xml = (tmp_path / "feed.xml").read_text()
        assert "&amp;" in xml and "<b>" not in xml

    def test_is_well_formed(self, tmp_path, config, store):
        from xml.etree import ElementTree

        result = seed_digest(store)
        render.write_site(tmp_path, config, store, result)
        ElementTree.parse(tmp_path / "feed.xml")  # raises if malformed


class TestPageShell:
    def test_shell_reads_only_fields_the_payload_provides(self, tmp_path, config, store):
        """Guards against the page referencing a field the renderer dropped."""
        result = seed_digest(store)
        render.write_site(tmp_path, config, store, result)
        html = (tmp_path / "index.html").read_text()
        payload = json.loads((tmp_path / "digests" / "2026-W37.json").read_text())
        index = json.loads((tmp_path / "index.json").read_text())

        referenced = set(re.findall(r"\b(?:story|a|d|entry)\.([a-z_]+)\b", html))
        available = (
            set(payload) | set(payload["stories"][0]) | set(payload["stories"][0]["articles"][0])
            | set(index) | set(index["digests"][0])
            # JS builtins and locals that the regex also catches.
            | {"length", "map", "filter", "join", "toFixed", "split", "every",
               "includes", "indexOf", "toUpperCase", "value", "code", "articles", "name"}
        )
        assert referenced <= available, referenced - available
