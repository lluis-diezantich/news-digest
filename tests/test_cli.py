"""The command layer.

Thin on purpose: this file covers `headlines` only. The rest of the CLI was
untested when it was added, and widening that gap quietly seemed worse than
saying so here.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import timedelta

from newsdigest import cli
from newsdigest.models import utcnow

from conftest import make_article

LOG = logging.getLogger("test")


def _args(store, **overrides):
    """The namespace `_headlines` reads, as `build_parser` would produce it."""
    ns = argparse.Namespace(
        db=store.path, only=[], urls=False, json=False,
        week=None, days=30, from_date=None, to_date=None,
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def _stored(store, *articles):
    store.insert_articles(list(articles))


class TestHeadlines:
    def test_it_groups_by_source(self, store, config, capsys):
        _stored(
            store,
            make_article("A Spanish headline about housing", source="publico"),
            make_article("Another one from the same source", source="publico"),
            make_article("An English headline about Ukraine", source="guardian"),
        )
        assert cli._headlines(_args(store), config, LOG) == 0
        out = capsys.readouterr().out
        assert "--- publico (2)" in out
        assert "--- guardian (1)" in out
        assert "3 headlines across 2 sources" in out

    def test_one_source_is_not_pluralized(self, store, config, capsys):
        _stored(store, make_article("A headline about something", source="publico"))
        cli._headlines(_args(store), config, LOG)
        out = capsys.readouterr().out
        assert "across 1 source" in out
        assert "sources" not in out

    def test_source_filters_on_a_substring(self, store, config, capsys):
        _stored(
            store,
            make_article("A headline from one feed", source="publico-temas-semana"),
            make_article("A headline from another", source="guardian-this-is-europe"),
        )
        cli._headlines(_args(store, only=["publico"]), config, LOG)
        out = capsys.readouterr().out
        assert "publico-temas-semana" in out
        assert "guardian" not in out

    def test_the_classifier_verdict_is_marked(self, store, config, capsys):
        """Only the stored view can report this: classification runs after
        parse, so a fresh extraction has no verdict.

        Written through the classification path rather than by setting the field
        and inserting -- `insert_articles` does not persist a verdict, because
        a freshly parsed article has not been classified yet.
        """
        from newsdigest.llm.base import Classification

        kept = make_article("A headline that is news", source="publico")
        dropped = make_article("A headline that is not news", source="publico")
        _stored(store, kept, dropped)
        store.save_classification(
            dropped.id, Classification(id=dropped.id, newsworthy=False), "test"
        )
        cli._headlines(_args(store), config, LOG)
        out = capsys.readouterr().out
        assert " x A headline that is not news" in out
        assert "   A headline that is news" in out
        assert "1 filtered as not news" in out

    def test_an_empty_window_is_an_error_not_an_empty_list(self, store, config, capsys):
        """Exit 1, so a script can tell "nothing stored" from "nothing matched"."""
        assert cli._headlines(_args(store), config, LOG) == 1
        assert "no stored headlines" in capsys.readouterr().out

    def test_json_is_machine_readable(self, store, config, capsys):
        _stored(store, make_article("A headline about housing", source="publico"))
        cli._headlines(_args(store, json=True), config, LOG)
        payload = json.loads(capsys.readouterr().out)
        assert list(payload) == ["publico"]
        assert payload["publico"][0]["title"] == "A headline about housing"

    def test_urls_are_opt_in(self, store, config, capsys):
        _stored(store, make_article("A headline", source="publico",
                                    url="https://example.com/a-story"))
        cli._headlines(_args(store), config, LOG)
        assert "https://example.com/a-story" not in capsys.readouterr().out
        cli._headlines(_args(store, urls=True), config, LOG)
        assert "https://example.com/a-story" in capsys.readouterr().out
