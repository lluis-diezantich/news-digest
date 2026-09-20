"""Newsletter body -> news items.

The stage with no schema on its input, so these tests are mostly about the shapes
real newsletters take: nested tables, one article linked three times, a sponsor
slot built to look like editorial, and a footer full of links.
"""

from pathlib import Path

import pytest

from newsdigest.extract import (
    extract,
    extract_html,
    extract_text,
    is_boilerplate_link,
    is_sponsored,
    looks_like_tracker,
    plausible_title,
    resolve,
    to_articles,
    unwrap,
)

from conftest import make_email

FIXTURES = Path(__file__).parent / "fixtures" / "emails"


def item(title, url, blurb="Something happened, at enough length to summarize."):
    return f'<table><tr><td><h2><a href="{url}">{title}</a></h2><p>{blurb}</p></td></tr></table>'


def page(*blocks):
    return "<html><body>" + "".join(blocks) + "</body></html>"


class TestTitles:
    @pytest.mark.parametrize("text", [
        "EU leaders agree new sanctions package",
        "Floods displace thousands in northern Nigeria",
    ])
    def test_a_headline_is_plausible(self, text):
        assert plausible_title(text)

    @pytest.mark.parametrize("text", [
        "", "WORLD", "Middle East", "Read more", "Sport",
        "THE WEEK IN NUMBERS",            # a standing all-caps banner
        "Short one",                       # too few words and too short
    ])
    def test_furniture_is_not(self, text):
        assert not plausible_title(text)

    def test_a_whole_paragraph_is_not_a_headline(self):
        assert not plausible_title("word " * 100)


class TestExtraction:
    def test_finds_each_item_in_document_order(self):
        html = page(
            item("The first story of the week", "https://x.example/a"),
            item("The second story of the week", "https://x.example/b"),
        )
        items = extract_html(html)
        assert [i.title for i in items] == [
            "The first story of the week", "The second story of the week"
        ]
        assert [i.position for i in items] == [0, 1]

    def test_the_blurb_comes_from_the_block_not_the_link(self):
        items = extract_html(item(
            "A story about something", "https://x.example/a",
            blurb="Three people were appointed and two resigned.",
        ))
        assert "Three people were appointed" in items[0].blurb
        assert "A story about something" not in items[0].blurb

    def test_an_image_only_link_takes_its_headline_from_the_block(self):
        """Every newsletter wraps its LEAD story in an image link, so the
        fallbacks matter most for the item that matters most."""
        html = page(
            '<table><tr><td>'
            '<a href="https://x.example/lead"><img src="i.jpg" alt="ignored alt"></a>'
            '<h2>EU leaders agree a new sanctions package</h2>'
            '<p>Agreed after nine hours of negotiation.</p>'
            '</td></tr></table>'
        )
        items = extract_html(html)
        assert items[0].title == "EU leaders agree a new sanctions package"

    def test_alt_text_is_used_when_there_is_nothing_else(self):
        html = page(
            '<table><tr><td>'
            '<a href="https://x.example/lead">'
            '<img src="i.jpg" alt="Floods displace thousands in the north"></a>'
            '</td></tr></table>'
        )
        assert extract_html(html)[0].title == "Floods displace thousands in the north"

    def test_two_adjacent_items_do_not_merge(self):
        """The block grows until another ARTICLE appears, which is what keeps one
        table cell holding two stories from becoming one."""
        html = page(
            '<table><tr><td>'
            '<h2><a href="https://x.example/a">The first story of the week</a></h2>'
            '<p>First blurb about the first thing.</p>'
            '<h2><a href="https://x.example/b">The second story of the week</a></h2>'
            '<p>Second blurb about the second thing.</p>'
            '</td></tr></table>'
        )
        items = extract_html(html)
        assert len(items) == 2
        assert "Second blurb" not in items[0].blurb

    def test_one_article_linked_three_times_is_one_item(self):
        html = page(
            '<table><tr><td>'
            '<a href="https://x.example/a"><img src="i.jpg" alt="x"></a>'
            '<h2><a href="https://x.example/a">A story linked more than once</a></h2>'
            '<p>The blurb that belongs to it.</p>'
            '<a href="https://x.example/a">Read more</a>'
            '</td></tr></table>'
        )
        items = extract_html(html)
        assert len(items) == 1
        assert "The blurb that belongs to it" in items[0].blurb

    def test_tracking_parameters_do_not_split_one_article_in_two(self):
        html = page(
            '<table><tr><td>'
            '<h2><a href="https://x.example/a?utm_source=news">A story about something</a></h2>'
            '<p>A blurb.</p>'
            '<a href="https://x.example/a?utm_source=footer">A story about something</a>'
            '</td></tr></table>'
        )
        assert len(extract_html(html)) == 1

    def test_an_empty_body_yields_nothing(self):
        assert extract_html("") == []

    def test_a_body_with_no_links_yields_nothing(self):
        assert extract_html("<html><body><p>Just prose.</p></body></html>") == []


class TestBoilerplate:
    @pytest.mark.parametrize("href,text", [
        ("https://x.example/unsubscribe", "Unsubscribe"),
        ("https://x.example/p", "Darse de baja"),
        ("https://x.example/p", "View in browser"),
        ("https://x.example/p", "Ver en el navegador"),
        ("https://x.example/p", "Privacy policy"),
        ("https://x.example/p", "Follow us"),
        ("https://x.example/p", "Read more"),
        ("https://x.example/p", "Suscríbete"),
        ("https://www.facebook.com/outlet", "Facebook"),
        ("https://twitter.com/outlet", "Twitter"),
        ("https://apps.apple.com/app/id1", "Download the app"),
        ("mailto:editor@x.example", "Email us"),
        ("#top", "Back to top"),
        ("https://x.example", "The Outlet"),      # a masthead, no path
    ])
    def test_furniture_links_are_rejected(self, href, text):
        assert is_boilerplate_link(href, text)

    def test_a_real_article_link_is_kept(self):
        assert not is_boilerplate_link(
            "https://x.example/world/2026/sep/18/floods", "Floods displace thousands"
        )

    @pytest.mark.parametrize("text", [
        "Sponsored by Acme Bank",
        "Paid post",
        "In partnership with Acme",
        "Contenido patrocinado",
        "Publicidad",
        "Brought to you by Acme",
    ])
    def test_paid_placement_is_detected(self, text):
        assert is_sponsored(text)

    def test_a_sponsored_block_is_dropped_whole(self):
        html = page(
            item("A real news story this week", "https://x.example/real"),
            '<table><tr><td><p><strong>Sponsored</strong></p>'
            '<h2><a href="https://acme.example/o">Five ways to save more money</a></h2>'
            '<p>Our experts explain.</p></td></tr></table>',
        )
        assert [i.title for i in extract_html(html)] == ["A real news story this week"]

    def test_the_unsubscribe_footer_is_removed(self):
        html = page(
            item("A real news story this week", "https://x.example/real"),
            '<table><tr><td><p>You are receiving this because you subscribed.'
            '<a href="https://x.example/legal/terms">Terms of service apply here</a>'
            "</p></td></tr></table>",
        )
        assert len(extract_html(html)) == 1

    def test_the_publishers_own_masthead_is_not_a_story(self):
        html = page(
            '<table><tr><td><a href="https://x.example/">The Example Times</a></td></tr></table>',
            item("A real news story this week", "https://x.example/real"),
        )
        assert len(extract_html(html)) == 1


class TestPlainText:
    def test_a_headline_above_its_url(self):
        text = (
            "The Saturday Edition\n\n"
            "EU leaders agree new sanctions package\n"
            "https://x.example/world/sanctions\n"
            "Agreed after nine hours of talks.\n"
        )
        items = extract_text(text)
        assert len(items) == 1
        assert items[0].title == "EU leaders agree new sanctions package"
        assert items[0].url == "https://x.example/world/sanctions"

    def test_trailing_punctuation_is_not_part_of_the_url(self):
        items = extract_text(
            "A story about something happening\nhttps://x.example/a.\n"
        )
        assert items[0].url == "https://x.example/a"

    def test_html_is_preferred_when_both_exist(self):
        email = make_email(
            "X",
            html=item("The HTML version of the story", "https://x.example/a"),
            text="The plain version of the story\nhttps://x.example/a\n",
        )
        assert extract(email)[0].title == "The HTML version of the story"

    def test_plain_text_is_the_fallback_when_html_yields_nothing(self):
        email = make_email(
            "X",
            html="<html><body><p>No links here at all.</p></body></html>",
            text="A story about something happening\nhttps://x.example/a\n",
        )
        assert extract(email)[0].title == "A story about something happening"


class TestLinks:
    @pytest.mark.parametrize("url,expected", [
        ("https://x.example/c?url=https%3A%2F%2Freal.example%2Fworld%2Fa",
         "https://real.example/world/a"),
        ("https://x.example/click?u=https://real.example/a", "https://real.example/a"),
        ("https://x.example/r?redirect=https%3A%2F%2Freal.example%2Fa",
         "https://real.example/a"),
    ])
    def test_a_destination_in_the_query_is_unwrapped_for_free(self, url, expected):
        assert unwrap(url) == expected

    def test_two_layers_are_both_unwrapped(self):
        """A publisher's tracker inside a mail vendor's. Unwrapping one layer
        would leave the other in place."""
        inner = "https://link.elpais.example/c?url=https%3A%2F%2Felpais.example%2Fa"
        outer = f"https://list-manage.com/track?u={inner}"
        assert unwrap(outer) == "https://elpais.example/a"

    def test_a_plain_article_url_unwraps_to_nothing(self):
        assert unwrap("https://real.example/world/a") is None

    @pytest.mark.parametrize("url", [
        "https://x.list-manage.com/track/click?id=1",
        "https://link.mail.elpais.example/c/eJx1kMtuAyE",
        "https://click.example.com/abcdef",
        "https://bit.ly/abc",
    ])
    def test_trackers_are_recognised(self, url):
        assert looks_like_tracker(url)

    @pytest.mark.parametrize("url", [
        "https://www.theguardian.com/world/2026/sep/18/floods",
        "https://elpais.com/internacional/2026-09-18/inundaciones.html",
        # A `go.` host that serves real articles with a real section path.
        "https://go.example.com/world/2026/a-real-article",
    ])
    def test_real_article_urls_are_not(self, url):
        assert not looks_like_tracker(url)

    def test_resolve_reports_whether_the_url_is_real(self):
        """The flag is what the caller acts on: an unresolved tracker must not be
        stored as an article's permanent URL, and must not be matched against
        section blocklists -- it matches none of them and would sail through a
        filter it should have been caught by."""
        url, real = resolve("https://real.example/world/a")
        assert (url, real) == ("https://real.example/world/a", True)

        url, real = resolve("https://link.mail.example.com/c/opaque", resolver=None)
        assert real is False

    def test_a_static_unwrap_needs_no_resolver(self):
        url, real = resolve(
            "https://list-manage.com/t?url=https%3A%2F%2Freal.example%2Fa", resolver=None
        )
        assert (url, real) == ("https://real.example/a", True)

    def test_a_resolver_is_used_only_when_it_has_to_be(self):
        class Counting:
            def __init__(self):
                self.calls = 0

            def resolve(self, url):
                self.calls += 1
                return "https://real.example/world/a"

        resolver = Counting()
        resolve("https://real.example/world/b", resolver)
        assert resolver.calls == 0, "a real URL must not cost a request"
        resolve("https://link.mail.example.com/c/opaque", resolver)
        assert resolver.calls == 1


class TestToArticles:
    def test_carries_the_newsletter_provenance(self):
        email = make_email("Guardian", html=item("A story this week", "https://x.example/a"))
        email.newsletter = "Saturday Edition"
        articles, _ = to_articles(
            email, extract(email), publisher="The Guardian", resolver=None
        )
        assert articles[0].publisher == "The Guardian"
        assert articles[0].newsletter == "Saturday Edition"
        assert articles[0].email_id == email.id

    def test_collected_at_is_the_emails_own_time(self):
        """The honest answer -- the item reached us when the newsletter did -- and
        it makes every window and recency calculation work without pretending to
        know a publication time nobody sent."""
        email = make_email("X", html=item("A story this week", "https://x.example/a"))
        articles, _ = to_articles(email, extract(email), resolver=None)
        assert articles[0].collected_at == email.received_at
        assert articles[0].published_at is None

    def test_unresolved_links_are_kept_and_counted(self):
        """Dropping them would silently shrink the digest whenever a publisher
        changed mailers."""
        email = make_email("X", html=item(
            "A story behind a tracker", "https://link.mail.example.com/c/opaque"
        ))
        articles, unresolved = to_articles(email, extract(email), resolver=None)
        assert len(articles) == 1
        assert unresolved == 1


class TestRealFixtures:
    """The two fixtures are real newsletter structure, one per language."""

    def _items(self, name):
        from newsdigest.inbox import parse_message

        email = parse_message((FIXTURES / name).read_bytes())
        return extract(email)

    def test_the_guardian_fixture_yields_its_stories(self):
        titles = [i.title for i in self._items("guardian_saturday.eml")]
        assert "EU leaders agree new sanctions package after Brussels summit" in titles
        assert "Floods displace thousands in northern Nigeria" in titles
        assert "Antarctic ice loss faster than models predicted, study finds" in titles

    def test_the_guardian_sponsor_slot_is_not_among_them(self):
        titles = [i.title for i in self._items("guardian_saturday.eml")]
        assert not any("savings" in t for t in titles)

    def test_the_lead_story_keeps_its_blurb(self):
        items = self._items("guardian_saturday.eml")
        assert items[0].blurb, "the lead story is the one worth getting right"

    def test_the_spanish_fixture_yields_its_stories(self):
        titles = [i.title for i in self._items("elpais_semana.eml")]
        assert any("sanciones" in t for t in titles)
        assert any("Nigeria" in t for t in titles)

    def test_neither_fixture_yields_footer_furniture(self):
        for name in ("guardian_saturday.eml", "elpais_semana.eml"):
            for i in self._items(name):
                assert "unsubscribe" not in i.url.lower()
                assert "facebook" not in i.url.lower()


class TestTrackedNewsletter:
    """Everything a real Piano-delivered issue got wrong, 2026-09-20.

    The `tracked_newsletter.eml` fixture reproduces its shape: a masthead logo, a
    membership appeal, three links to one article, one article repeated with fresh
    tokens, per-item bylines with no standfirst, and a Spanish footer. None of the
    real newsletter's data is in it -- the real issues carry the mailbox address and
    a per-subscriber id, so they stay out of git.
    """

    def _items(self):
        from newsdigest.inbox import parse_message

        email = parse_message((FIXTURES / "tracked_newsletter.eml").read_bytes())
        return email, extract(email, excerpt_chars=300)

    def test_only_the_real_stories_survive(self):
        _, items = self._items()
        assert [i.title for i in items] == [
            "El Congreso tumba una propuesta para prohibir la venta de pisos",
            "La inflación vuelve a comerse los salarios de los trabajadores",
        ]

    def test_every_link_is_reported_unresolved(self):
        """All of them are trackers. Reporting them as real was the original bug:
        they would be stored as permanent attribution, section blocklists would
        match nothing, and the run would not warn."""
        email, items = self._items()
        _, unresolved = to_articles(email, items, resolver=None)
        assert unresolved == len(items) == 2

    def test_one_article_with_two_tokens_is_one_story(self):
        """Piano mints a fresh token per link, so URL-keyed dedupe cannot see it."""
        _, items = self._items()
        titles = [i.title for i in items]
        assert len(titles) == len(set(titles))

    def test_a_masthead_logo_is_not_a_headline(self):
        _, items = self._items()
        assert not any("Logo" in i.title for i in items)

    def test_a_membership_appeal_arriving_as_alt_text_is_dropped(self):
        """The link wraps an image, so there is no link text to reject and the
        appeal reaches the title through the alt attribute instead."""
        _, items = self._items()
        assert not any("Únete" in i.title for i in items)

    def test_the_spanish_footer_is_not_a_story(self):
        _, items = self._items()
        # Substring tests only: "darte de baja" would match "trabajadores".
        assert not any("pulsa aquí" in i.title.lower() for i in items)
        assert not any("darte de baja" in i.title.lower() for i in items)
        assert not any("no quieres seguir" in i.title.lower() for i in items)

    def test_a_folded_subject_is_unfolded(self):
        """`decode_header` does not rejoin a split header, so the stored subject
        kept a literal CRLF."""
        email, _ = self._items()
        assert "\n" not in email.subject and "\r" not in email.subject
        assert email.subject == "Resumen de la semana: diez ejemplos que retratan el asunto"


class TestBlurbCleaning:
    def test_the_headline_is_removed_from_every_position(self):
        """It commonly appears twice in one block -- as link text and as the
        heading -- and removing one copy left the other. 14 of 27 blurbs in a real
        issue were their own headline repeated back."""
        html = page(
            '<table><tr><td>'
            '<h2><a href="https://x.example/a">A significant thing happened today</a></h2>'
            '<p>A significant thing happened today. The detail follows here at length.</p>'
            '</td></tr></table>'
        )
        blurb = extract_html(html)[0].blurb
        assert "A significant thing happened today" not in blurb
        assert "The detail follows" in blurb

    def test_a_call_to_action_is_not_a_summary(self):
        html = page(
            '<table><tr><td>'
            '<h2><a href="https://x.example/a">A significant thing happened today</a></h2>'
            "<p>Leer artículo completo »</p>"
            '</td></tr></table>'
        )
        assert extract_html(html)[0].blurb == ""

    def test_a_bare_byline_is_not_a_summary(self):
        html = page(
            '<table><tr><td>'
            '<h2><a href="https://x.example/a">A significant thing happened today</a></h2>'
            "<p>Pilar Araque Conde</p><p>Leer artículo completo »</p>"
            '</td></tr></table>'
        )
        assert extract_html(html)[0].blurb == ""

    def test_a_real_summary_beginning_with_a_proper_noun_is_kept(self):
        """A byline and the subject of a Spanish sentence are the same shape, so
        stripping a leading "byline" turned "El Movimiento Regularización Ya,
        junto a..." into "Ya, junto a...". The heuristic was removed; this is the
        regression guard."""
        html = page(
            '<table><tr><td>'
            '<h2><a href="https://x.example/a">Un asunto importante de la semana</a></h2>'
            "<p>El Movimiento Regularización Ya, junto a otras entidades y "
            "colectivos, hacen balance de los primeros meses.</p>"
            '</td></tr></table>'
        )
        blurb = extract_html(html)[0].blurb
        assert blurb.startswith("El Movimiento Regularización Ya")

    def test_a_fragment_is_dropped_rather_than_published(self):
        html = page(
            '<table><tr><td>'
            '<h2><a href="https://x.example/a">A significant thing happened today</a></h2>'
            "<p>Por</p>"
            '</td></tr></table>'
        )
        assert extract_html(html)[0].blurb == ""


class TestSpanishBoilerplate:
    @pytest.mark.parametrize("text", [
        "Si no ves correctamente este correo pulsa aquí",
        "Si no visualizas bien este boletín",
        "Apoya el periodismo crítico",
        "Únete al periodismo valiente",
        "Leernos es totalmente gratis",
        "Escucha el último episodio",
        "Leer artículo completo",
        "Seguir leyendo",
        "Suscribirse a la newsletter",
    ])
    def test_appeals_and_calls_to_action_are_furniture(self, text):
        from newsdigest.extract import is_boilerplate_text

        assert is_boilerplate_text(text)

    @pytest.mark.parametrize("text", [
        "Si no quieres seguir recibiendo este boletín",
        "Para dejar de recibir estos correos",
        "Recibes este correo porque estás suscrito",
    ])
    def test_spanish_housekeeping_is_detected(self, text):
        from newsdigest.extract import is_housekeeping

        assert is_housekeeping(text)

    @pytest.mark.parametrize("text", [
        "El Congreso tumba una propuesta para prohibir la venta de pisos",
        "La inflación vuelve a comerse los salarios",
        "Comuneros, la revuelta del siglo XVI que perdura en Villalar",
    ])
    def test_real_headlines_are_not(self, text):
        from newsdigest.extract import is_boilerplate_text, is_housekeeping

        assert not is_boilerplate_text(text)
        assert not is_housekeeping(text)


class TestOpaquePaths:
    """The general tracker test, which is what catches the next unknown vendor.

    The vendor list only recognises the ones already met, and a real Público issue
    went through `api-esp-eu.piano.io` -- an unknown host with an unknown label --
    so all 27 of its links were reported as real article URLs.
    """

    @pytest.mark.parametrize("url", [
        "https://api-esp-eu.piano.io/-c/117/30394/671878/20018313/860429/2090179c18",
        "https://unknown-vendor.example.net/1/2/3/4/5/6",
        "https://track.example/-c/117/30394/860429/ff00",
    ])
    def test_identifier_only_paths_are_trackers(self, url):
        assert looks_like_tracker(url)

    @pytest.mark.parametrize("url", [
        "https://www.theguardian.com/world/2026/sep/18/floods-displace-thousands",
        "https://elpais.com/internacional/2026-09-18/inundaciones-nigeria.html",
        "https://www.elsaltodiario.com/culturas/allianz-retira-patrocinio",
        # A `go.` or `mail.` host that serves real articles with a real section.
        "https://go.example.com/world/2026/a-real-article",
        "https://mail.example.com/noticias/2026/un-articulo-real",
        "https://news.bbc.co.uk/world/middle-east/report",
    ])
    def test_urls_with_words_in_them_are_not(self, url):
        assert not looks_like_tracker(url)

    def test_a_bare_domain_date_archive_is_left_alone(self):
        """Opaque, but on the publisher's own bare domain. Restricting the general
        rule to subdomains is what keeps it from sweeping these up -- and if one
        does slip through it is merely counted as unresolved, never dropped."""
        assert not looks_like_tracker("https://elpais.com/2026/09/18/")


class TestCallToActionTails:
    """A call to action cut off mid-phrase, which is what is left when the link
    text sat in its own element. El Salto's blocks end "...alemana. Leer"."""

    def _blurb(self, tail):
        html = page(
            '<table><tr><td>'
            '<h2><a href="https://x.example/a">Un asunto importante de la semana</a></h2>'
            f"<p>El anuncio ha precipitado la ruptura del evento con la "
            f"aseguradora alemana.{tail}</p>"
            '</td></tr></table>'
        )
        return extract_html(html)[0].blurb

    @pytest.mark.parametrize("tail", [" Leer", " Leer »", " Read", " Más", " ver"])
    def test_a_trailing_fragment_is_removed(self, tail):
        assert self._blurb(tail).endswith("aseguradora alemana")

    def test_the_same_word_mid_sentence_is_left_alone(self):
        """Only its position makes it junk: "leer es importante" is prose."""
        html = page(
            '<table><tr><td>'
            '<h2><a href="https://x.example/a">Un asunto importante de la semana</a></h2>'
            "<p>Leer es importante para la democracia, sostiene el informe "
            "presentado esta semana.</p>"
            '</td></tr></table>'
        )
        assert extract_html(html)[0].blurb.startswith("Leer es importante")
