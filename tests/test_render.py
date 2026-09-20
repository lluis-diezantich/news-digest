"""Markdown output: the digest file, the archive index, and the README block."""

from datetime import datetime, timedelta, timezone

from newsdigest import render
from newsdigest.digest import DigestResult
from newsdigest.models import Digest, Story

from conftest import make_article

MONDAY = datetime(2026, 9, 14, tzinfo=timezone.utc)


def story(headline, *, summary="What happened, briefly.", **kw):
    return Story(id=Story.make_id(headline), headline=headline, summary=summary, **kw)


def result(pairs, minor=(), week="2026-W38"):
    digest = Digest(
        id=week,
        period_start=MONDAY,
        period_end=MONDAY + timedelta(days=7),
        story_ids=[s.id for s, _ in pairs],
        minor_story_ids=[s.id for s, _ in minor],
        article_count=sum(len(a) for _, a in pairs),
    )
    return DigestResult(digest=digest, stories=list(pairs), minor=list(minor))


def one(headline, *, publishers=("BBC",), **kw):
    articles = [
        make_article(headline, source=p, publisher=p) for p in publishers
    ]
    return story(headline, **kw), articles


class TestDigestMarkdown:
    def test_the_heading_carries_the_title_and_the_dates(self, config):
        text = render.digest_markdown(config, result([one("A thing happened")]))
        assert text.startswith(f"# {config.digest.title}")
        # period_end is exclusive, so the label stops at the last day covered.
        assert "14–20 September 2026" in text

    def test_stories_are_numbered_in_order(self, config):
        text = render.digest_markdown(
            config, result([one("First thing"), one("Second thing")])
        )
        assert "## 1. First thing" in text
        assert "## 2. Second thing" in text
        assert text.index("## 1.") < text.index("## 2.")

    def test_each_story_links_every_publisher_once(self, config):
        text = render.digest_markdown(
            config, result([one("A thing", publishers=("BBC", "Reuters", "BBC"))])
        )
        assert text.count("[BBC]") == 1
        assert "[Reuters]" in text

    def test_why_it_matters_appears_when_there_is_one(self, config):
        s, a = one("A thing", why_it_matters="It changes the budget.")
        text = render.digest_markdown(config, result([(s, a)]))
        assert "**Why it matters:** It changes the budget." in text

    def test_an_empty_section_is_omitted_entirely(self, config):
        """The target is five to ten minutes of reading, so nothing is padded."""
        text = render.digest_markdown(config, result([one("A thing")]))
        assert "Why it matters" not in text
        assert "Where sources differ" not in text

    def test_a_week_with_no_stories_says_so(self, config):
        text = render.digest_markdown(config, result([]))
        assert "No stories were published" in text

    def test_the_footer_records_what_it_was_built_from(self, config):
        text = render.digest_markdown(
            config, result([one("A thing", publishers=("BBC", "EL PAÍS"))])
        )
        assert "2 publishers" in text
        assert "Sources: BBC, EL PAÍS" in text

    def test_brackets_in_a_headline_cannot_break_a_link(self, config):
        text = render.digest_markdown(config, result([one("A [bracketed] thing")]))
        assert "\\[bracketed\\]" in text

    def test_a_headline_cannot_smuggle_in_a_heading(self, config):
        text = render.digest_markdown(config, result([one("### Not a heading")]))
        assert "## 1. Not a heading" in text


class TestDisagreements:
    """Section 15: where sources differ, the digest says so, and never resolves
    it silently into the summary."""

    def test_they_get_their_own_section(self, config):
        s, a = one(
            "A thing", publishers=("Reuters", "EL PAÍS"),
            disagreements=["Reuters reported 12 dead, while EL PAÍS reported 14."],
        )
        text = render.digest_markdown(config, result([(s, a)]))
        assert "**Where sources differ:**" in text
        assert "Reuters reported 12 dead" in text

    def test_they_are_not_folded_into_the_summary(self, config):
        s, a = one("A thing", summary="Agreed facts only.",
                   disagreements=["One outlet said X, another said Y."])
        text = render.digest_markdown(config, result([(s, a)]))
        summary_part = text.split("**Where sources differ:**")[0]
        assert "another said Y" not in summary_part

    def test_no_disagreement_means_no_section(self, config):
        text = render.digest_markdown(config, result([one("A thing")]))
        assert "Where sources differ" not in text


class TestMinorStories:
    """Section 14's "Also worth knowing": listed, not written up."""

    def test_they_are_listed_under_their_own_heading(self, config):
        text = render.digest_markdown(
            config, result([one("Main thing")], minor=[one("Smaller thing")])
        )
        assert "## Also worth knowing" in text
        assert "Smaller thing" in text

    def test_they_get_no_write_up(self, config):
        s, a = one("Smaller thing", summary="A summary nobody should see here.")
        text = render.digest_markdown(config, result([one("Main")], minor=[(s, a)]))
        assert "A summary nobody should see here" not in text

    def test_none_means_no_heading(self, config):
        text = render.digest_markdown(config, result([one("Main thing")]))
        assert "Also worth knowing" not in text


class TestFiles:
    def test_the_path_is_year_then_week(self, tmp_path):
        digest = Digest(id="2026-W38", period_start=MONDAY,
                        period_end=MONDAY + timedelta(days=7))
        assert render.digest_path(tmp_path, digest) == tmp_path / "2026" / "2026-W38.md"

    def test_writing_creates_the_year_directory(self, tmp_path, config):
        path = render.write_digest(tmp_path, config, result([one("A thing")]))
        assert path.exists()
        assert path.read_text(encoding="utf-8").startswith("#")

    def test_the_index_lists_every_week_newest_first(self, tmp_path, config, store):
        for week, day in (("2026-W37", 7), ("2026-W38", 14)):
            store.save_digest(Digest(
                id=week,
                period_start=datetime(2026, 9, day, tzinfo=timezone.utc),
                period_end=datetime(2026, 9, day + 7, tzinfo=timezone.utc),
            ))
        render.write_all(tmp_path, config, store, result([one("A thing")]))
        index = (tmp_path / "README.md").read_text(encoding="utf-8")
        assert index.index("2026-W38") < index.index("2026-W37")

    def test_rebuild_restores_every_file_from_the_database(
        self, tmp_path, config, store
    ):
        """`digests/` is an artefact of the database, not a second copy of it."""
        s, articles = one("A stored thing", publishers=("BBC",))
        store.insert_articles(articles)
        store.replace_story(s, [a.id for a in articles])
        store.save_digest(Digest(
            id="2026-W38", period_start=MONDAY, period_end=MONDAY + timedelta(days=7),
            story_ids=[s.id],
        ))
        written = render.rebuild(tmp_path, config, store)
        assert written == 1
        text = (tmp_path / "2026" / "2026-W38.md").read_text(encoding="utf-8")
        assert "A stored thing" in text

    def test_rebuild_keeps_the_two_tiers_apart(self, tmp_path, config, store):
        main, main_articles = one("Main thing")
        minor, minor_articles = one("Smaller thing")
        store.insert_articles(main_articles + minor_articles)
        store.replace_story(main, [a.id for a in main_articles])
        store.replace_story(minor, [a.id for a in minor_articles])
        store.save_digest(Digest(
            id="2026-W38", period_start=MONDAY, period_end=MONDAY + timedelta(days=7),
            story_ids=[main.id], minor_story_ids=[minor.id],
        ))
        render.rebuild(tmp_path, config, store)
        text = (tmp_path / "2026" / "2026-W38.md").read_text(encoding="utf-8")
        assert "## 1. Main thing" in text
        assert "## Also worth knowing" in text
        assert "## 2." not in text


class TestReadme:
    def test_the_digest_goes_between_the_markers(self, tmp_path, config):
        readme = tmp_path / "README.md"
        readme.write_text(
            f"# My project\n\nSome prose.\n\n{render.README_START}\nold\n"
            f"{render.README_END}\n\nMore prose.\n",
            encoding="utf-8",
        )
        render.update_readme(readme, config, result([one("A new thing")]))
        text = readme.read_text(encoding="utf-8")
        assert "A new thing" in text
        assert "old" not in text
        assert text.startswith("# My project")
        assert text.rstrip().endswith("More prose.")

    def test_a_readme_without_markers_is_left_alone(self, tmp_path, config):
        """This runs unattended every Monday. A workflow that rewrites
        hand-written documentation is worse than one that does nothing."""
        readme = tmp_path / "README.md"
        readme.write_text("# My project\n\nAll hand written.\n", encoding="utf-8")
        render.update_readme(readme, config, result([one("A new thing")]))
        assert readme.read_text(encoding="utf-8") == "# My project\n\nAll hand written.\n"

    def test_a_missing_readme_is_not_created(self, tmp_path, config):
        readme = tmp_path / "README.md"
        render.update_readme(readme, config, result([one("A thing")]))
        assert not readme.exists()
