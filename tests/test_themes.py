"""The layer above clusters: stories grouped into narratives."""

from __future__ import annotations

import pytest

from newsdigest import render, themes
from newsdigest.extract.newsletter import editorial_summary
from newsdigest.llm.base import (
    Context, LLMProvider, ThemeBrief, ThemeGroup, ThemeGroupInput, ThemeInput,
)
from newsdigest.models import Article, Story, Theme


class Base(LLMProvider):
    name, model = "stub", "stub"

    def enrich(self, items, context):
        return []

    def write_brief(self, item, context):
        return None

    def same_event(self, pairs, context):
        return {}


def story(sid, headline="A headline of a plausible length", publisher="Público"):
    st = Story(id=sid, headline=headline, summary=f"Summary of {sid}.", score=1.0)
    ar = [Article(id=f"a-{sid}", title=headline, source="src", publisher=publisher,
                  url=f"https://example.com/news/{sid}", description=f"excerpt {sid}")]
    return (st, ar)


def ranked(n=5):
    return [story(f"s{i}", publisher=f"Pub{i}") for i in range(n)]


class TestGrouping:
    def test_a_provider_that_cannot_group_publishes_flat(self):
        """The default on LLMProvider. Themes are optional by construction, so
        the offline path and every existing provider keep working."""
        assert themes.group(ranked(), Base(), Context()) == []

    def test_disabled_costs_nothing(self):
        class Boom(Base):
            def group_themes(self, items, context, *, editorial=None):
                raise AssertionError("must not be called")

        assert themes.group(ranked(), Boom(), Context(), enabled=False) == []

    def test_a_grouping_failure_does_not_lose_the_digest(self):
        class Broken(Base):
            def group_themes(self, items, context, *, editorial=None):
                raise RuntimeError("502")

        assert themes.group(ranked(), Broken(), Context()) == []

    def test_a_theme_write_failure_keeps_the_group(self):
        """The grouping was already paid for, and the label plus the stories is
        no worse than the flat output."""
        class HalfBroken(Base):
            def group_themes(self, items, context, *, editorial=None):
                return [ThemeGroup(label="A real connection",
                                   story_ids=[i.story_id for i in items[:2]])]

            def write_theme(self, item, context):
                raise RuntimeError("quota")

        out = themes.group(ranked(), HalfBroken(), Context())
        assert len(out) == 1
        assert out[0].story_ids == ["s0", "s1"]
        assert not out[0].written

    def test_a_story_cannot_appear_in_two_themes(self):
        """It would be published twice."""
        class Greedy(Base):
            def group_themes(self, items, context, *, editorial=None):
                ids = [i.story_id for i in items]
                return [ThemeGroup(label="One", story_ids=ids[:3]),
                        ThemeGroup(label="Two", story_ids=ids[1:4])]

            def write_theme(self, item, context):
                return ThemeBrief(headline="h", narrative="n")

        out = themes.group(ranked(), Greedy(), Context())
        seen = [sid for theme in out for sid in theme.story_ids]
        assert len(seen) == len(set(seen))

    def test_a_single_story_is_not_a_theme(self):
        class Thin(Base):
            def group_themes(self, items, context, *, editorial=None):
                return [ThemeGroup(label="Lonely", story_ids=["s0"])]

            def write_theme(self, item, context):
                return ThemeBrief(headline="h", narrative="n")

        assert themes.group(ranked(), Thin(), Context()) == []

    def test_the_writer_is_given_an_excerpt_per_story(self):
        seen = {}

        class Recorder(Base):
            def group_themes(self, items, context, *, editorial=None):
                seen["editorial"] = editorial
                return [ThemeGroup(label="L", story_ids=["s0", "s1"])]

            def write_theme(self, item, context):
                seen["coverage"] = item.coverage
                return ThemeBrief(headline="h", narrative="n")

        themes.group(ranked(), Recorder(), Context(), editorial=["Esta semana..."])
        assert seen["editorial"] == ["Esta semana..."]
        assert [c[1] for c in seen["coverage"]] == ["Pub0", "Pub1"]
        assert all(c[2] for c in seen["coverage"]), "every story needs text"


class TestRendering:
    def _result(self, themes_list):
        from newsdigest.digest import DigestResult
        from newsdigest.models import Digest
        from datetime import datetime, timezone
        stories = ranked(4)
        digest = Digest(id="2026-W40",
                        period_start=datetime(2026, 9, 28, tzinfo=timezone.utc),
                        period_end=datetime(2026, 10, 5, tzinfo=timezone.utc))
        return DigestResult(digest=digest, stories=stories, themes=themes_list)

    def test_ungrouped_stories_are_still_published(self):
        from newsdigest.config import load_config
        theme = Theme(label="L", story_ids=["s0", "s1"],
                      headline="Themed", narrative="Connected.")
        out = render.digest_markdown(load_config(), self._result([theme]))
        assert "## 1. Themed" in out
        # s2 and s3 belong to no narrative and must not vanish.
        assert "## 2." in out and "## 3." in out
        assert out.count("## ") >= 3

    def test_a_story_is_not_published_twice(self):
        from newsdigest.config import load_config
        theme = Theme(label="L", story_ids=["s0", "s1"],
                      headline="Themed", narrative="Connected.")
        out = render.digest_markdown(load_config(), self._result([theme]))
        assert out.count("https://example.com/news/s0") == 1

    def test_a_per_story_disagreement_survives_inside_a_theme(self):
        """Section 15 applies whether a story publishes alone or in a theme."""
        from newsdigest.config import load_config
        result = self._result([Theme(label="L", story_ids=["s0", "s1"],
                                     headline="T", narrative="N")])
        result.stories[0][0].disagreements = ["A said 12, B said 14."]
        out = render.digest_markdown(load_config(), result)
        assert "A said 12, B said 14." in out

    def test_an_unwritten_theme_still_renders_its_stories(self):
        from newsdigest.config import load_config
        theme = Theme(label="Only a label", story_ids=["s0", "s1"])
        out = render.digest_markdown(load_config(), self._result([theme]))
        assert "Only a label" in out
        assert "https://example.com/news/s1" in out


class TestThemePersistence:
    def test_a_theme_round_trips_through_the_stats_blob(self):
        """`weekly_digests.stats` is free-form JSON, which is what lets a themed
        digest survive `news-digest build` with no schema change."""
        import json
        theme = Theme(label="L", story_ids=["a", "b"], headline="H",
                      narrative="N", why_it_matters="W", open_questions=["Q?"])
        assert Theme.from_dict(json.loads(json.dumps(theme.as_dict()))) == theme


class TestEditorialSummary:
    def test_it_finds_a_contents_sentence(self):
        html = (
            "<html><body><table><tr><td>"
            "<p>El Orden Mundial</p><p>2 DE OCTUBRE DE 2026</p>"
            "<p>Hola. <span>Esta semana analizamos si la vivienda puede revivir "
            "a la izquierda, el resurgir de la guerra en Etiopia y el acuerdo "
            "entre Dinamarca y Estados Unidos.</span></p>"
            "</td></tr></table></body></html>"
        )
        got = editorial_summary(html)
        assert "Esta semana analizamos" in got
        assert "Etiopia" in got

    def test_ordinary_prose_mentioning_the_week_is_not_a_summary(self):
        """The first version matched "the UN debate is under way this week" and
        returned a paragraph of article copy."""
        html = (
            "<html><body><p>In New York, the United Nations General Assembly's "
            "high-level debate is under way this week with world leaders "
            "arriving to speak.</p></body></html>"
        )
        assert editorial_summary(html) == ""

    def test_no_summary_is_not_an_error(self):
        assert editorial_summary("") == ""
        assert editorial_summary("<html><body><p>Short.</p></body></html>") == ""
