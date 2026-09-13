"""Language detection, including the es/ca pair that motivated the design."""

import pytest

from newsdigest import lang


@pytest.fixture(autouse=True)
def configured():
    lang.configure(["en", "es", "ca"])


class TestDetect:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Government approves the 2027 budget after a long debate", "en"),
            ("El Gobierno aprueba el presupuesto de 2027 tras un largo debate", "es"),
            ("El Govern aprova el pressupost del 2027 després d'un llarg debat", "ca"),
            ("La UE anuncia noves sancions contra Rússia aquesta setmana", "ca"),
            ("La UE anuncia nuevas sanciones contra Rusia esta semana", "es"),
        ],
    )
    def test_distinguishes_the_three_languages(self, text, expected):
        assert lang.detect(text) == expected

    def test_short_text_yields_the_declared_language(self):
        """A one-word headline ranks languages arbitrarily, so trust the source."""
        assert lang.detect("Wegovy", allowed=["ca"]) == "ca"

    def test_short_text_with_nothing_declared_is_undetermined(self):
        assert lang.detect("Wegovy") is None

    def test_declared_languages_constrain_the_answer(self):
        """A Catalan-only source cannot yield Spanish from a cognate-heavy line."""
        spanish = "El Gobierno aprueba el presupuesto de 2027 tras un largo debate"
        assert lang.detect(spanish) == "es"
        assert lang.detect(spanish, allowed=["ca"]) == "ca"

    def test_multi_language_source_still_detects(self):
        """La Vanguardia publishes both, so both are declared and detection decides."""
        catalan = "El Govern aprova el pressupost del 2027 després d'un llarg debat"
        assert lang.detect(catalan, allowed=["es", "ca"]) == "ca"

    def test_empty_input_uses_the_fallback(self):
        assert lang.detect("", fallback="es") == "es"
        assert lang.detect(None) is None

    def test_detect_article_uses_title_and_excerpt(self):
        assert lang.detect_article(
            "Acord",
            "El Parlament ha aprovat aquest dimarts la nova llei amb una majoria àmplia.",
            allowed=["es", "ca"],
        ) == "ca"


class TestConfigure:
    def test_configure_is_idempotent_and_restricts(self):
        lang.configure(["en"])
        # With only English available, Spanish text can only come back as English.
        assert lang.detect("El Gobierno aprueba el presupuesto de 2027") == "en"
        lang.configure(["en", "es", "ca"])
        assert lang.detect("El Gobierno aprueba el presupuesto de 2027") == "es"

    def test_empty_supported_list_falls_back_to_defaults(self):
        lang.configure([])
        assert lang.detect("Government approves the 2027 budget") == "en"
