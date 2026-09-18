"""Themes: subject-level grouping over a window, keyed on headline names.

A different unit from a story, and the tests that matter are the two failures it
exists to fix -- a subject scattered across many single-outlet event clusters, and
a subject broken into several competing ones.
"""

import pytest

from conftest import make_article

from newsdigest import themes
from newsdigest.config import ThemeSettings, load_preferences


def art(title, publisher, **kw):
    return make_article(title, source=publisher, publisher=publisher, **kw)


class TestGrouping:
    def test_a_scattered_subject_becomes_one_theme(self):
        """The Iran case: five unrelated events, one subject, no two of them
        clusterable as the same event."""
        arts = [
            art("Iran war live: Trump weighs big decision on Iran", "Al Jazeera"),
            art("Strait of Hormuz talks postponed, says Iran", "BBC World"),
            art("Iran expels Swedish diplomat in retaliatory move", "The Guardian"),
            art("L'ONU acusa els EUA de crims de guerra a l'Iran", "Ara", language="ca"),
            art("China y Rusia vetan una resolución sobre Iran", "El País", language="es"),
        ]
        found = themes.group(arts, min_publishers=3)
        assert found[0].key == "iran"
        assert len(found[0].articles) == 5
        assert len(found[0].publishers) == 5

    def test_a_key_needs_enough_outlets(self):
        """One outlet writing alone is not a subject."""
        arts = [art(f"Badalona council story {n}", "Ara") for n in range(5)]
        assert themes.group(arts, min_publishers=3) == []

    def test_containers_are_excluded(self):
        """`catalunya` locates a story rather than being one. Measured: coherence
        cannot tell it from a real subject, hence a config list."""
        arts = [
            art("La criminalitat baixa a Catalunya", "Ara", language="ca"),
            art("Catalunya aprova el pressupost", "VilaWeb", language="ca"),
            art("El turisme creix a Catalunya", "RAC1", language="ca"),
        ]
        assert themes.group(arts, containers=["catalunya"], min_publishers=3) == []
        assert themes.group(arts, min_publishers=3)[0].key == "catalunya"

    def test_containers_are_matched_accent_and_case_insensitively(self):
        """Checked against every key of every theme, not just the naming one: keys
        with identical article sets merge, and which of them names the result is an
        alphabetical tie-break that this behaviour must not depend on."""
        arts = [art(f"Cimera d'Espanya amb el Marroc {n}", f"P{n}") for n in range(3)]
        excluded = themes.group(arts, containers=["ESPANYA"], min_publishers=3)
        included = themes.group(arts, min_publishers=3)
        assert not any("espanya" in theme.keys for theme in excluded)
        assert any("espanya" in theme.keys for theme in included)

    def test_a_travelling_key_merges_into_the_bigger_one(self):
        """`hormuz` only ever appears with `iran`, so it is not its own theme --
        and `iran` must be the one that names the result, not `hormuz`."""
        arts = [
            art("Iran and Hormuz shipping disrupted", "A"),
            art("Iran talks on Hormuz postponed", "B"),
            art("Hormuz tanker hit as Iran responds", "C"),
            art("Iran expels a diplomat", "D"),
            art("Iran sanctions vetoed at the UN", "E"),
        ]
        found = themes.group(arts, min_publishers=3)
        assert len(found) == 1
        assert found[0].key == "iran"
        assert "hormuz" in found[0].keys

    def test_a_key_with_little_overlap_stays_its_own_theme(self):
        """A key that mostly appears alone is its own theme.

        Named for the CIS poll, but NOT a claim about it: on the real 2026-W38 data
        `cis` is absorbed by `ceuta`, because at 194 articles `ceuta` over-merges
        anything that clears 60% overlap with it. See `MERGE_OVERLAP`. What this
        pins is the rule, not that outcome.
        """
        poll = [art(f"El CIS mide la intención de voto {n}", f"P{n}", language="es")
                for n in range(6)]
        poll += [art("El CIS y la crisis de Ceuta", "P9", language="es")]
        other = [art(f"Ceuta court ruling day {n}", f"Q{n}") for n in range(6)]
        keys = {t.key for t in themes.group(poll + other, min_publishers=3)}
        assert "cis" in keys, "a one-day poll release must stay visible as itself"
        assert "ceuta" in keys

    def test_ordering_is_publishers_then_articles(self):
        wide = [art(f"Wide subject {n}", f"W{n}") for n in range(6)]
        deep = [art(f"Deep subject {n}", f"D{n % 3}") for n in range(12)]
        found = themes.group(wide + deep, min_publishers=3)
        assert [len(t.publishers) for t in found] == sorted(
            [len(t.publishers) for t in found], reverse=True
        )

    def test_no_articles_is_not_an_error(self):
        assert themes.group([]) == []


class TestThemeFields:
    def theme(self):
        arts = [
            art("Iran war live", "Al Jazeera", language="en", hours_ago=2),
            art("Iran talks resume", "Ara", language="ca", hours_ago=30),
            art("Irán y el Estrecho", "El País", language="es", hours_ago=54),
            art("Iran sanctions vetoed", "Al Jazeera", language="en", hours_ago=4),
        ]
        return themes.group(arts, min_publishers=3)[0]

    def test_counts_and_languages(self):
        t = self.theme()
        assert len(t.publishers) == 3
        assert t.languages == ["ca", "en", "es"]
        assert t.days >= 2

    def test_by_publisher_is_most_prolific_first(self):
        assert self.theme().by_publisher()[0] == ("Al Jazeera", 2)

    def test_newest_first(self):
        ordered = self.theme().newest_first()
        stamps = [(a.published_at or a.collected_at) for a in ordered]
        assert stamps == sorted(stamps, reverse=True)

    def test_json_is_serializable_and_complete(self):
        import json
        payload = self.theme().to_json()
        assert json.loads(json.dumps(payload))["publisher_count"] == 3
        assert len(payload["articles"]) == 4
        assert all(a["url"] for a in payload["articles"])


class TestDispersion:
    """How many separate stories a theme holds -- the number that decided themes
    are not fed to the LLM. Measured across six windows, 38% of top-eight theme
    slots held three or more mutually dissimilar stories."""

    def events_and_vectors(self, groups):
        """`groups` is a list of (n_publishers, vector) -- one fake event cluster
        each, with every member sharing that cluster's vector."""
        import numpy as np
        events, vectors = [], {}
        for index, (n_pub, vec) in enumerate(groups):
            members = [art(f"Story {index} take {p}", f"P{index}_{p}")
                       for p in range(n_pub)]
            for member in members:
                vectors[member.id] = np.array(vec, dtype=np.float32)
            events.append(members)
        return events, vectors

    def theme_of(self, events):
        flat = [a for group_ in events for a in group_]
        return themes.Theme(key="k", keys=["k"], articles=flat)

    def test_one_story_has_nothing_to_compare(self):
        events, vectors = self.events_and_vectors([(3, [1.0, 0.0, 0.0])])
        subs, mean = themes.dispersion(self.theme_of(events), events, vectors)
        assert (subs, mean) == (1, None)
        assert not themes.is_dispersed(subs, mean)

    def test_unlike_stories_are_dispersed(self):
        """The `trump` shape: several substantial stories with nothing in common."""
        events, vectors = self.events_and_vectors(
            [(3, [1.0, 0.0, 0.0]), (3, [0.0, 1.0, 0.0]), (3, [0.0, 0.0, 1.0])])
        subs, mean = themes.dispersion(self.theme_of(events), events, vectors)
        assert subs == 3 and mean == pytest.approx(0.0)
        assert themes.is_dispersed(subs, mean)

    def test_related_stories_are_not_dispersed(self):
        """The `openai` shape: several stories, all about the same thing."""
        events, vectors = self.events_and_vectors(
            [(3, [1.0, 0.1, 0.0]), (3, [0.98, 0.15, 0.0]), (3, [0.95, 0.2, 0.0])])
        subs, mean = themes.dispersion(self.theme_of(events), events, vectors)
        assert subs == 3 and mean > themes.DISPERSED_BELOW
        assert not themes.is_dispersed(subs, mean)

    def test_two_unlike_stories_are_not_enough(self):
        """A theme holding two stories is still a subject, however unlike."""
        events, vectors = self.events_and_vectors(
            [(3, [1.0, 0.0, 0.0]), (3, [0.0, 1.0, 0.0])])
        subs, mean = themes.dispersion(self.theme_of(events), events, vectors)
        assert subs == 2 and not themes.is_dispersed(subs, mean)

    def test_single_outlet_clusters_do_not_count(self):
        """One desk's stray piece is not a story in its own right."""
        events, vectors = self.events_and_vectors(
            [(3, [1.0, 0.0, 0.0]), (1, [0.0, 1.0, 0.0]), (1, [0.0, 0.0, 1.0])])
        subs, _ = themes.dispersion(self.theme_of(events), events, vectors)
        assert subs == 1

    def test_missing_vectors_are_skipped_not_fatal(self):
        events, vectors = self.events_and_vectors(
            [(3, [1.0, 0.0, 0.0]), (3, [0.0, 1.0, 0.0])])
        subs, mean = themes.dispersion(self.theme_of(events), events, {})
        assert (subs, mean) == (0, None)


class TestFind:
    def themes_list(self):
        """Puigdemont is the bigger key, so it names the theme; `llarena` travels
        with it. Equal-sized keys tie-break alphabetically, which is arbitrary but
        deterministic -- do not build a test on that case."""
        arts = [art(f"Puigdemont and Llarena {n}", f"P{n}") for n in range(4)]
        arts += [art("Puigdemont returns to court", "P9")]
        return themes.group(arts, min_publishers=3)

    def test_exact_key(self):
        assert themes.find(self.themes_list(), "puigdemont") is not None

    def test_a_merged_key_finds_the_theme(self):
        found = themes.find(self.themes_list(), "llarena")
        assert found is not None and found.key == "puigdemont"

    def test_a_prefix_finds_the_theme(self):
        assert themes.find(self.themes_list(), "puig") is not None

    def test_an_unknown_key_is_none(self):
        assert themes.find(self.themes_list(), "nothing at all") is None


class TestConfig:
    def test_defaults_exclude_the_obvious_containers(self):
        assert "catalunya" in ThemeSettings().containers
        assert ThemeSettings().min_publishers == 3

    def test_the_shipped_config_defines_themes(self):
        shipped = load_preferences()[4]
        assert shipped.min_publishers >= 1
        assert "catalunya" in [c.lower() for c in shipped.containers]

    def test_an_empty_container_list_is_honoured_not_defaulted(self, tmp_path):
        """`containers: []` means "no exclusions", not "use the defaults"."""
        path = tmp_path / "p.yaml"
        path.write_text("themes:\n  containers: []\n", encoding="utf-8")
        assert load_preferences(path)[4].containers == []
