"""Position in the newsletter: recorded, normalised, and ranked first.

A curated newsletter is a stronger version of the signal feed order gave the RSS
version of this project. An editor chose these items out of the day's hundreds
AND chose which one opens, so position 0 is two judgements rather than one, and it
is available in every language for free. It is the highest-weighted term in the
shipped formula.

What makes that safe is upstream, not here. Position measures PROMOTION, and a
newsletter's sponsor slot is bought promotion that can sit anywhere, including
first. Two extraction-side controls keep it out of reach: the sponsored-block
filter and topic classification. The last tests in this file assert both are
still in place, so removing one is a deliberate act rather than an accident.
"""

import pytest

from conftest import make_article, make_email

from newsdigest import scoring
from newsdigest.config import Preferences
from newsdigest.extract import extract_html
from newsdigest.models import Article


def _wide_window():
    """A window that certainly contains anything just inserted."""
    from datetime import datetime, timedelta, timezone
    from newsdigest.models import utcnow
    return datetime(2000, 1, 1, tzinfo=timezone.utc), utcnow() + timedelta(days=1)


def at(position: int, size: int, **kw) -> Article:
    a = make_article(kw.pop("title", "A story"), **kw)
    a.item_position, a.item_count = position, size
    return a


class TestNormalisation:
    def test_leading_item_scores_one(self):
        assert at(0, 20).editorial_rank() == 1.0

    def test_last_item_scores_zero(self):
        assert at(19, 20).editorial_rank() == 0.0

    def test_midway_is_about_half(self):
        assert 0.4 < at(9, 20).editorial_rank() < 0.6

    def test_normalised_within_the_newsletter(self):
        """Position 3 of 8 items and of 40 are not the same claim."""
        assert at(3, 8).editorial_rank() < at(3, 40).editorial_rank()

    def test_unknown_position_is_neutral_not_penalised(self):
        """A newsletter we could not order must not read as though all of it
        trailed."""
        assert at(-1, 0).editorial_rank() == 0.5

    def test_single_item_newsletter_is_neutral(self):
        """One item carries no ordering information."""
        assert at(0, 1).editorial_rank() == 0.5

    def test_position_beyond_size_is_clamped(self):
        assert at(99, 20).editorial_rank() == 0.0


NEWSLETTER = """
<html><body>
  <table>
    <tr><td><h2><a href="https://x.example/lead">The lead story of the week</a></h2>
        <p>What happened, at enough length to summarize.</p></td></tr>
    <tr><td><h2><a href="https://x.example/second">The second story down the page</a></h2>
        <p>Also what happened, at similar length.</p></td></tr>
    <tr><td><h2><a href="https://x.example/third">The third story down the page</a></h2>
        <p>And a third thing that happened this week.</p></td></tr>
  </table>
</body></html>
"""


class TestRecording:
    def test_the_extractor_records_newsletter_order(self):
        items = extract_html(NEWSLETTER)
        assert [i.position for i in items] == [0, 1, 2]

    def test_position_is_relative_to_what_the_newsletter_held(self):
        articles, _ = _articles(NEWSLETTER)
        assert {a.item_count for a in articles} == {3}
        assert articles[0].editorial_rank() == 1.0
        assert articles[-1].editorial_rank() == 0.0

    def test_repeated_links_to_one_article_do_not_shift_the_positions(self):
        """Every newsletter links its lead story from the image, the headline and
        a "read more". Counting those as three items would push the second story
        to position 3 and make the lead's block look like three items -- which is
        how the lead once ended up published with no summary at all."""
        thrice = """
        <html><body><table>
          <tr><td>
            <a href="https://x.example/lead"><img src="i.jpg" alt="ignored"></a>
            <h2><a href="https://x.example/lead">The lead story of the week</a></h2>
            <p>What happened, at enough length to summarize from.</p>
            <a href="https://x.example/lead">Read more</a>
          </td></tr>
          <tr><td><h2><a href="https://x.example/second">The second story down the page</a></h2>
              <p>Also what happened.</p></td></tr>
        </table></body></html>
        """
        items = extract_html(thrice)
        assert [i.position for i in items] == [0, 1]
        assert items[0].blurb, "the lead story must keep its summary"

    def test_position_survives_the_store(self, store):
        store.insert_articles([at(3, 25, title="A stored story")])
        got = store.articles_in_window(*_wide_window())
        assert len(got) == 1
        assert (got[0].item_position, got[0].item_count) == (3, 25)

    def test_an_unordered_row_reads_as_unknown(self, store):
        store.insert_articles([make_article("An older story")])
        store.conn.execute("UPDATE articles SET item_position = -1, item_count = 0")
        got = store.articles_in_window(*_wide_window())
        assert got[0].editorial_rank() == 0.5


def _articles(html):
    from newsdigest.extract import to_articles

    email = make_email("X", html=html)
    return to_articles(email, extract_html(html), publisher="X", resolver=None)


class TestRanked:
    def test_it_is_a_computed_signal(self):
        from newsdigest import clustering
        a = at(0, 20, importance=0.5, relevance=0.5)
        story = clustering.build_story([a])
        assert "editorial_position" in scoring.signals(story, [a], Preferences())

    def test_it_is_in_the_shipped_formula(self):
        """It is the primary term by intent; losing it would be silent."""
        assert Preferences().ranking.get("editorial_position", 0) > 0

    def test_best_position_wins_not_the_mean(self):
        """One outlet leading with it beats several burying it."""
        assert scoring.editorial_position([at(0, 20), at(18, 20), at(19, 20)]) == 1.0
        assert scoring.editorial_position([at(18, 20), at(19, 20)]) < 0.2

    def test_no_articles_is_neutral(self):
        assert scoring.editorial_position([]) == 0.5

    def test_a_lead_story_outranks_a_buried_one(self):
        from newsdigest import clustering
        prefs = Preferences()
        lead, buried = at(0, 25, importance=0.35), at(24, 25, importance=0.35)
        s_lead = clustering.build_story([lead])
        s_buried = clustering.build_story([buried])
        assert (scoring.score_story(s_lead, [lead], prefs)
                > scoring.score_story(s_buried, [buried], prefs))

    def test_terms_still_sum_to_the_total(self):
        from newsdigest import clustering
        a = at(4, 20, importance=0.5, relevance=0.5)
        story = clustering.build_story([a])
        b = scoring.explain(story, [a], Preferences())
        total = b.pop("total")
        assert total == pytest.approx(round(sum(b.values()), 4), abs=1e-4)


class TestWhatKeepsItHonest:
    """Position promotes whatever the editor promoted, including what was paid
    for. These are the two controls that stand between a sponsor slot and the top
    of the digest."""

    SPONSORED = """
    <html><body><table>
      <tr><td><h2><a href="https://x.example/real">A real news story this week</a></h2>
          <p>Something that actually happened.</p></td></tr>
      <tr><td><p><strong>Sponsored by Acme Bank</strong></p>
          <h2><a href="https://acme.example/offer">Five ways to make your savings work harder</a></h2>
          <p>Our experts explain how to get more from your money.</p></td></tr>
    </table></body></html>
    """

    def test_a_sponsored_block_never_becomes_an_article(self):
        titles = [i.title for i in extract_html(self.SPONSORED)]
        assert titles == ["A real news story this week"]

    def test_a_sponsored_block_in_first_position_is_still_dropped(self):
        """Position 0 is where a sponsor slot is worth most, so that is the case
        that has to work."""
        first = self.SPONSORED.replace(
            '<tr><td><h2><a href="https://x.example/real">A real news story this week</a></h2>\n'
            '          <p>Something that actually happened.</p></td></tr>\n      ', ""
        ) + ""
        items = extract_html(first)
        assert all("savings" not in i.title for i in items)

    def test_the_shipped_filters_still_drop_promotional_topics(self):
        """Classification is the second line, for a slot with no label. If this
        list is emptied, position promotes advertising."""
        from newsdigest.config import load_filters
        assert load_filters().excluded_topics, (
            "no excluded topics: nothing stands between promoted copy and the top "
            "of the digest except the sponsored-block filter"
        )

    def test_classification_is_on_by_default(self):
        from newsdigest.config import load_filters
        assert load_filters().classify, (
            "with classify off, filtering is regex only -- and a tracking link has "
            "no section path to match"
        )
