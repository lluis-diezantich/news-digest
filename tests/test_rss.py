"""RSS adapter tests against a fixture feed -- no network."""

from newsdigest.config import Source
from newsdigest.sources.rss import RSSAdapter

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>Fixture Wire</title>
  <item>
    <title>Central bank raises rates &amp; signals more</title>
    <link>https://fixture.example/rates?utm_source=rss</link>
    <description><![CDATA[<p>The bank moved by half a point.</p>]]></description>
    <pubDate>Tue, 10 Sep 2026 08:30:00 GMT</pubDate>
    <author>ada@fixture.example (Ada Reporter)</author>
    <category>economics</category>
  </item>
  <item>
    <title>Second story</title>
    <link>https://fixture.example/second</link>
    <description>Short blurb.</description>
  </item>
  <item>
    <title></title>
    <link>https://fixture.example/untitled</link>
  </item>
</channel></rss>
"""


class FakeResponse:
    def __init__(self, body: str):
        self.content = body.encode("utf-8")
        self.text = body
        self.url = "https://fixture.example/rss"


class FakeFetcher:
    def __init__(self, body: str):
        self.body = body
        self.requested: list[str] = []

    def get(self, url, **kwargs):
        self.requested.append(url)
        return FakeResponse(self.body)


def parse(body=FEED, **overrides):
    source = Source(name="Fixture", rss="https://fixture.example/rss",
                    topics=["world"], weight=1.3, **overrides)
    fetcher = FakeFetcher(body)
    return RSSAdapter(fetcher).fetch(source), fetcher


def test_normalizes_into_the_common_schema():
    articles, fetcher = parse()
    assert fetcher.requested == ["https://fixture.example/rss"]
    # The untitled item is skipped.
    assert len(articles) == 2

    first = articles[0]
    assert first.title == "Central bank raises rates & signals more"
    assert first.source == "Fixture"
    assert first.url == "https://fixture.example/rates?utm_source=rss"
    # Tracking params are stripped from the identity, not from the link.
    assert first.canonical == "https://fixture.example/rates"
    assert first.published_at.year == 2026
    assert first.published_at.hour == 8
    assert "Ada Reporter" in first.author
    assert first.description == "The bank moved by half a point."
    assert "world" in first.source_topics and "economics" in first.source_topics
    assert first.source_weight == 1.3
    assert first.id and len(first.id) == 16


def test_missing_date_is_allowed():
    articles, _ = parse()
    assert articles[1].published_at is None


def test_max_items_is_honoured():
    articles, _ = parse(max_items=1)
    assert len(articles) == 1


def test_excerpt_is_truncated():
    body = FEED.replace("The bank moved by half a point.", "word " * 400)
    articles, _ = parse(body, excerpt_chars=100)
    assert len(articles[0].description) <= 101


def test_unparseable_feed_raises():
    import pytest

    with pytest.raises(ValueError):
        parse("this is not xml at all")
