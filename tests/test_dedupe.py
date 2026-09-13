from newsdigest.dedupe import dedupe
from newsdigest.text import title_key

from conftest import make_article


def test_same_canonical_url_is_dropped():
    a = make_article("Rates rise", url="https://x.example/a?utm_source=rss")
    b = make_article("Rates rise", url="https://www.x.example/a")
    keep, dropped = dedupe([a, b])
    assert len(keep) == 1 and len(dropped) == 1


def test_known_id_is_dropped():
    a = make_article("Rates rise")
    keep, dropped = dedupe([a], known_ids={a.id})
    assert keep == [] and dropped[0].reason == "already stored"


def test_same_source_rerun_is_dropped():
    a = make_article("Storm hits the coast overnight", url="https://x.example/1")
    b = make_article("Storm hits the coast overnight again", url="https://x.example/2")
    keep, dropped = dedupe([a, b])
    assert len(keep) == 1
    assert "re-run" in dropped[0].reason
    assert keep[0].url == "https://x.example/1"  # earliest copy kept


def test_distinct_same_source_articles_survive():
    a = make_article("Storm hits the coast overnight", url="https://x.example/1")
    b = make_article("Storm death toll rises to twelve", url="https://x.example/2")
    keep, _ = dedupe([a, b])
    assert len(keep) == 2


def test_stored_headline_from_same_source_is_dropped():
    a = make_article("Budget passes parliament", source="Wire")
    keep, dropped = dedupe(
        [a], known_fingerprints=[("old", title_key("Parliament passes budget"), "Wire")]
    )
    assert keep == [] and "stored headline" in dropped[0].reason


def test_other_outlets_are_not_duplicates():
    """Cross-source coverage is corroboration; clustering handles it."""
    a = make_article("Budget passes parliament", source="BBC")
    b = make_article("Budget passes parliament", source="Guardian")
    keep, dropped = dedupe([a, b])
    assert len(keep) == 2 and dropped == []


def test_translations_are_not_duplicates():
    """One publisher's Spanish and English editions are separate articles."""
    es = make_article("La UE anuncia nuevas sanciones", source="El País",
                      publisher="El País", language="es")
    en = make_article("EU announces new sanctions", source="El País (English)",
                      publisher="El País", language="en")
    keep, _ = dedupe([es, en])
    assert len(keep) == 2


def test_untitled_article_is_dropped():
    a = make_article("Real one")
    b = make_article("x")
    b.title = ""
    keep, dropped = dedupe([a, b])
    assert len(keep) == 1 and dropped[0].reason == "incomplete"
