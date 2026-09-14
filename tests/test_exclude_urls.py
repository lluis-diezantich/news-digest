"""Tests for exclude_url_patterns: structural junk filtering at collection."""

import pytest

from conftest import make_article

from newsdigest.config import ConfigError, load_sources
from newsdigest.urls import exclude_by_url


class TestExcludeByUrl:
    def articles(self):
        return [
            make_article("Real reporting", url="https://ara.cat/internacional/story_1_1.html"),
            make_article("Advertorial", url="https://ara.cat/especials/masters/ia-pal_1_2.html"),
            make_article("Branded", url="https://www.20minutos.es/lainformacion/economia/x.html"),
            make_article("Lottery", url="https://example.es/loterias/sorteo-hoy.html"),
        ]

    def test_no_patterns_keeps_everything(self):
        arts = self.articles()
        assert len(exclude_by_url(arts, [])) == 4

    def test_matching_pattern_drops_the_article(self):
        kept = exclude_by_url(self.articles(), ["/especials/"])
        assert [a.title for a in kept] == ["Real reporting", "Branded", "Lottery"]

    def test_several_patterns_combine(self):
        kept = exclude_by_url(self.articles(), ["/especials/", "/lainformacion/", "/loterias/"])
        assert [a.title for a in kept] == ["Real reporting"]

    def test_a_pattern_that_matches_nothing_is_harmless(self):
        assert len(exclude_by_url(self.articles(), ["/horoscopo/"])) == 4

    def test_regex_not_just_substring(self):
        kept = exclude_by_url(self.articles(), [r"/(especials|loterias)/"])
        assert len(kept) == 2

    def test_the_input_list_is_not_mutated(self):
        arts = self.articles()
        exclude_by_url(arts, ["/especials/"])
        assert len(arts) == 4

    def test_an_article_with_no_url_is_kept(self):
        """Never silently drop something for lacking the field we match on."""
        a = make_article("No url")
        a.url = ""
        assert len(exclude_by_url([a], ["/especials/"])) == 1


class TestConfigLoading:
    def write(self, tmp_path, body):
        p = tmp_path / "sources.yaml"
        p.write_text(body, encoding="utf-8")
        return p

    BASE = """
sources:
  - name: Ara
    rss: https://www.ara.cat/rss/
    languages: [ca]
"""

    def test_absent_by_default(self, tmp_path):
        src = load_sources(self.write(tmp_path, self.BASE))[0]
        assert src.exclude_url_patterns == []

    def test_per_source_patterns_are_read(self, tmp_path):
        body = self.BASE + '    exclude_url_patterns: ["/especials/"]\n'
        assert load_sources(self.write(tmp_path, body))[0].exclude_url_patterns == ["/especials/"]

    def test_global_patterns_apply_to_every_source(self, tmp_path):
        body = 'exclude_url_patterns: ["/loterias/"]\n' + self.BASE
        assert load_sources(self.write(tmp_path, body))[0].exclude_url_patterns == ["/loterias/"]

    def test_global_and_per_source_are_merged(self, tmp_path):
        body = ('exclude_url_patterns: ["/loterias/"]\n' + self.BASE
                + '    exclude_url_patterns: ["/especials/"]\n')
        got = load_sources(self.write(tmp_path, body))[0].exclude_url_patterns
        assert got == ["/loterias/", "/especials/"]

    def test_duplicates_are_collapsed(self, tmp_path):
        body = ('exclude_url_patterns: ["/especials/"]\n' + self.BASE
                + '    exclude_url_patterns: ["/especials/"]\n')
        assert load_sources(self.write(tmp_path, body))[0].exclude_url_patterns == ["/especials/"]

    def test_a_string_is_accepted_as_one_pattern(self, tmp_path):
        body = self.BASE + '    exclude_url_patterns: "/especials/"\n'
        assert load_sources(self.write(tmp_path, body))[0].exclude_url_patterns == ["/especials/"]

    def test_a_broken_regex_is_a_startup_error(self, tmp_path):
        """It would otherwise throw once per article, mid-run."""
        body = self.BASE + '    exclude_url_patterns: ["/especials(/"]\n'
        with pytest.raises(ConfigError, match="not a valid regex"):
            load_sources(self.write(tmp_path, body))

    def test_the_error_names_the_source_and_pattern(self, tmp_path):
        body = self.BASE + '    exclude_url_patterns: ["["]\n'
        with pytest.raises(ConfigError, match=r"Ara.*\["):
            load_sources(self.write(tmp_path, body))
