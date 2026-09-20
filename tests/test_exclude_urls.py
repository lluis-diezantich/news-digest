"""Tests for exclude_url_patterns and exclude_title_patterns: junk filtering at
collection, before anything reaches the database."""

import pytest

from conftest import make_article

from newsdigest.config import ConfigError, load_sources
from newsdigest.urls import exclude_by_title, exclude_by_url


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
    senders: ["@ara.cat"]
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


class TestExcludeByTitle:
    """Headline filtering, for junk with no section path of its own."""

    WEATHER = ["previsi[oó] del temps", "previsi[oó] d'avui", "el tiempo hoy"]
    FOOTBALL = ["lamine yamal", r"\byamal\b", r"\bbar[çc]a\b", r"\bfutbol\b"]

    def test_no_patterns_keeps_everything(self):
        arts = [make_article("Anything at all")]
        assert len(exclude_by_title(arts, [])) == 1

    @pytest.mark.parametrize("title", [
        "La setmana comença amb calor: la previsió del temps d'avui, 14 de setembre",
        "Més clarianes i menys calor: la previsió d'avui, 17 de setembre, a Catalunya",
        "El tiempo hoy 17 de septiembre en España: avisos por tormentas en seis comunidades",
    ])
    def test_daily_forecasts_are_dropped(self, title):
        """RAC1 files one every day under its news path, so no URL pattern sees
        them; six landed in the 2026-W38 window and one topped the prominence
        signal outright."""
        assert exclude_by_title([make_article(title)], self.WEATHER) == []

    @pytest.mark.parametrize("title", [
        "Un muerto en Barcelona y grave caos de transporte por las lluvias torrenciales",
        "Activat l'Inuncat per pluja intensa demà a la meitat est del país",
        "Murcia, Valencia y Almería suspenden las clases ante la alerta por lluvias",
        "Protecció Civil envia un ES-Alert a Eivissa i Formentera pel temporal de pluja",
        "Alerta per la previsió de pluges: el Govern demana limitar els desplaçaments",
    ])
    def test_weather_NEWS_survives(self, title):
        """The line is forecasts, not weather. A death, a shut-down metro, an
        ES-Alert and closed schools are news; matching `lluvia`/`pluja` would take
        all of them, and the last one is the trap -- it contains "previsió" but is
        a government instruction, not a forecast.
        """
        assert len(exclude_by_title([make_article(title)], self.WEATHER)) == 1

    @pytest.mark.parametrize("title", [
        "Lamine Yamal: 'Merezco el Balón de Oro por lo que he ganado'",
        "Barcelona beat Levante as Yamal scores twice to maintain perfect start",
        "Así ha sido el cumpleaños de Keyne, el hermano de Lamine Yamal",
        "Vendaval del Barça contra el Llevant per consolidar el lideratge",
    ])
    def test_football_is_dropped_including_off_section_filings(self, title):
        """The last two are the point: a family piece filed under /gente/ and a
        match report under /noticies/, neither reachable by `/esports?/`."""
        assert exclude_by_title([make_article(title)], self.FOOTBALL) == []

    def test_accents_do_not_matter_either_side(self):
        """Titles and patterns both pass through `normalize`, so an unaccented
        pattern matches an accented headline and vice versa."""
        assert exclude_by_title([make_article("Entre el fútbol i la política")],
                                [r"\bfutbol\b"]) == []
        assert exclude_by_title([make_article("La previsio del temps d'avui")],
                                ["previsió del temps"]) == []

    def test_the_excerpt_is_not_matched(self):
        """Title only: an article whose background mentions football is not a
        football article."""
        art = make_article("El Govern aprova els pressupostos",
                           description="Un acord tancat mentre el Barça jugava.")
        assert len(exclude_by_title([art], self.FOOTBALL)) == 1


class TestTitlePatternConfig:
    def test_a_broken_title_regex_is_a_startup_error(self, tmp_path):
        path = tmp_path / "s.yaml"
        path.write_text(
            "exclude_title_patterns:\n  - 'unclosed ('\n"
            "sources:\n  - name: X\n    senders: ['@x.example']\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="exclude_title_patterns"):
            load_sources(path)

    def test_global_and_per_source_patterns_are_merged(self, tmp_path):
        path = tmp_path / "s.yaml"
        path.write_text(
            "exclude_title_patterns:\n  - lamine yamal\n"
            "sources:\n  - name: X\n    senders: ['@x.example']\n"
            "    exclude_title_patterns:\n      - el tiempo hoy\n",
            encoding="utf-8",
        )
        patterns = load_sources(path)[0].exclude_title_patterns
        assert patterns == ["lamine yamal", "el tiempo hoy"]
