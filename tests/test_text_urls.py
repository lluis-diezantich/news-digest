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
        assert len(out) <= 21 and out.endswith("…") and " f" not in out[-3:]

    def test_truncate_leaves_short_text_alone(self):
        assert text.truncate("short", 20) == "short"

    def test_title_key_is_order_insensitive(self):
        assert text.title_key("US and China agree trade deal") == text.title_key(
            "China and US agree trade deal"
        )

    def test_similarity_high_for_reworded_headline(self):
        score = text.similarity(
            "Central bank raises interest rates by half a point",
            "Interest rates raised half a point by central bank",
        )
        assert score > 0.7

    def test_similarity_low_for_unrelated(self):
        score = text.similarity(
            "Central bank raises interest rates",
            "Rare orchid discovered in Peruvian cloud forest",
        )
        assert score < 0.15

    def test_capitalized_phrases_finds_entities(self):
        found = text.capitalized_phrases("Anthropic and Google announced a deal in London.")
        # "and" must not glue two separate names into one entity.
        assert "Anthropic" in found and "Google" in found

    def test_capitalized_phrases_keeps_multiword_names_whole(self):
        found = text.capitalized_phrases("Ursula von der Leyen met the Bank of England.")
        assert "Ursula von der Leyen" in found

    def test_stem_makes_word_forms_agree(self):
        assert text.stem("raises") == text.stem("raised") == text.stem("raise")
        assert text.stem("approves") == text.stem("approved")
