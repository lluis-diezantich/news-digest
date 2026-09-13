from newsdigest import text, urls


class TestCanonicalUrl:
    def test_strips_tracking_and_normalizes(self):
        assert urls.canonical_url(
            "HTTP://WWW.Example.com/story/?utm_source=x&id=7&fbclid=abc#top"
        ) == "https://example.com/story?id=7"

    def test_trailing_slash_and_index_collapse(self):
        assert urls.canonical_url("https://a.example/news/") == urls.canonical_url(
            "https://a.example/news/index.html"
        )

    def test_root_path_survives(self):
        assert urls.canonical_url("https://a.example") == "https://a.example/"

    def test_query_order_is_irrelevant(self):
        assert urls.canonical_url("https://a.example/x?b=2&a=1") == urls.canonical_url(
            "https://a.example/x?a=1&b=2"
        )

    def test_domain_drops_www(self):
        assert urls.domain("https://www.bbc.co.uk/news") == "bbc.co.uk"


class TestText:
    def test_strip_html_unescapes(self):
        assert text.strip_html("<p>Tom &amp; Jerry</p>") == "Tom & Jerry"

    def test_truncate_on_word_boundary(self):
        out = text.truncate("the quick brown fox jumps over the lazy dog", 20)
        assert len(out) <= 21 and out.endswith("…")

    def test_title_key_is_order_insensitive(self):
        assert text.title_key("US and China agree trade deal") == text.title_key(
            "China and US agree trade deal"
        )

    def test_stem_makes_word_forms_agree(self):
        assert text.stem("raises") == text.stem("raised") == text.stem("raise")

    def test_similarity_high_for_reworded_headline(self):
        assert text.similarity(
            "Central bank raises interest rates by half a point",
            "Interest rates raised half a point by central bank",
        ) > 0.7

    def test_similarity_is_useless_across_languages(self):
        """The measurement that makes embeddings non-optional."""
        en = "EU announces new sanctions package against Russia"
        es = "La UE anuncia un nuevo paquete de sanciones contra Rusia"
        ca = "La UE anuncia un nou paquet de sancions contra Rússia"
        assert text.similarity(en, es) < 0.1
        assert text.similarity(en, ca) < 0.1


class TestArticleSimilarity:
    LONG_A = ("Rescuers searched through the night for survivors after the vessel "
              "went down in bad weather, officials said on Sunday morning.")
    LONG_B = ("Search teams deployed helicopters and boats as families waited for "
              "news at the port, according to the transport ministry.")

    def test_headline_is_not_drowned_by_differing_excerpts(self):
        """The bug this function exists to fix.

        Weighting title and excerpt equally scored this real pair at 0.25 --
        below any usable threshold -- because the two excerpts differ.
        """
        blended = text.article_similarity(
            "Six dead, 130 missing after Indonesian ferry capsizes in Java Sea",
            self.LONG_A,
            "At least six dead and 130 missing after Indonesian ferry capsizes",
            self.LONG_B,
        )
        naive = text.similarity(
            f"Six dead, 130 missing after Indonesian ferry capsizes in Java Sea. {self.LONG_A}",
            f"At least six dead and 130 missing after Indonesian ferry capsizes. {self.LONG_B}",
        )
        assert blended > naive
        assert blended >= 0.45

    def test_short_excerpts_fall_back_to_title_only(self):
        """Two empty excerpts look identical, which once beat real matches.

        Hacker News items carry no description; unrelated pairs scored 0.28-0.34
        purely on excerpt emptiness, above genuinely matching articles.
        """
        score = text.article_similarity(
            "Making Startups Powerful", "",
            "Make your first edit to OpenStreetMap", "",
        )
        assert score == text.similarity(
            "Making Startups Powerful", "Make your first edit to OpenStreetMap"
        )
        assert score < 0.2

    def test_one_short_excerpt_also_uses_title_only(self):
        assert text.article_similarity("A title here", "tiny", "A title here", self.LONG_A) == 1.0

    def test_unrelated_articles_stay_low(self):
        assert text.article_similarity(
            "Council approves new bridge", self.LONG_A,
            "Rare orchid discovered in cloud forest", self.LONG_B,
        ) < 0.2


class TestStripBoilerplate:
    GUARDIAN = ("'Red lines' legislation would forbid defence equipment being used, "
                "includes parts for F-35 fighter jets Get our breaking news email , "
                "free app or daily news podcast Crossbench MPs will put pressure on "
                "the government")

    def test_removes_the_newsletter_pitch(self):
        out = text.strip_boilerplate(self.GUARDIAN)
        assert "breaking news email" not in out
        assert "podcast" not in out
        assert out.startswith("'Red lines' legislation")

    def test_removes_continue_reading(self):
        out = text.strip_boilerplate("The yen hit a six-month high against the dollar "
                                     "on Friday. Continue reading...")
        assert out == "The yen hit a six-month high against the dollar on Friday."

    def test_handles_spanish_and_catalan_pitches(self):
        assert "Suscr" not in text.strip_boilerplate(
            "El Gobierno aprueba el presupuesto anual tras un largo debate. "
            "Suscr\u00edbete para seguir leyendo")
        assert "Segueix" not in text.strip_boilerplate(
            "El Govern aprova el pressupost anual despr\u00e9s d'un llarg debat. "
            "Segueix-nos a les xarxes")

    def test_clean_text_is_untouched(self):
        clean = "A description with no promotional trailer of any kind in it."
        assert text.strip_boilerplate(clean) == clean

    def test_keeps_the_original_when_stripping_would_empty_it(self):
        """Better a promo-laden excerpt than no excerpt at all."""
        mostly_promo = "Sign up for our daily newsletter and never miss a story"
        assert text.strip_boilerplate(mostly_promo) == mostly_promo

    def test_empty_input(self):
        assert text.strip_boilerplate("") == ""
        assert text.strip_boilerplate(None) == ""

    def test_shared_boilerplate_no_longer_links_unrelated_articles(self):
        """It inflated unrelated same-publisher similarity from 0.000 to 0.099."""
        promo = " Get our breaking news email , free app or daily news podcast"
        a = "The yen reached its highest level against the dollar in six months."
        b = "Crossbench MPs want to restrict defence equipment exports."
        assert text.similarity(a + promo, b + promo) > text.similarity(
            text.strip_boilerplate(a + promo), text.strip_boilerplate(b + promo)
        )
