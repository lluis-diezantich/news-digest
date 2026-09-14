import pytest

from newsdigest.config import (
    ConfigError, load_embedding_settings, load_llm_settings, load_preferences,
    load_sources,
)


def write(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


class TestShippedConfig:
    def test_the_repo_config_is_valid(self):
        """What a first run actually loads."""
        sources = load_sources()
        settings, prefs, digest, storage = load_preferences()
        assert len(sources) >= 10
        assert any(s.enabled for s in sources)
        assert all(s.rss for s in sources if s.method == "rss")
        assert settings.supported_languages == ["en", "es", "ca"]
        assert digest.max_stories > 0
        assert storage.embedding_retention_days <= storage.retention_days

    def test_every_enabled_source_declares_a_language(self):
        """Undeclared languages make detection guess on short headlines."""
        undeclared = [s.name for s in load_sources() if s.enabled and not s.languages]
        assert undeclared == []

    def test_el_pais_sections_share_one_publisher(self):
        """Otherwise seven feeds would read as seven corroborating outlets."""
        sections = [s for s in load_sources() if s.name.startswith("El País")]
        assert len(sections) > 1
        assert {s.publisher for s in sections} == {"El País"}


class TestSources:
    def test_defaults_apply_per_source(self, tmp_path):
        path = write(tmp_path, "s.yaml", """
defaults:
  max_items_per_source: 5
  excerpt_chars: 300
sources:
  - name: A
    rss: https://a.invalid/rss
  - name: B
    rss: https://b.invalid/rss
    max_items: 50
""")
        sources = load_sources(path)
        assert (sources[0].max_items, sources[0].excerpt_chars) == (5, 300)
        assert sources[1].max_items == 50

    def test_publisher_defaults_to_the_name(self, tmp_path):
        path = write(tmp_path, "s.yaml",
                     "sources:\n  - name: Solo\n    rss: https://a.invalid/rss\n")
        assert load_sources(path)[0].publisher == "Solo"

    def test_languages_are_lowercased(self, tmp_path):
        path = write(tmp_path, "s.yaml",
                     "sources:\n  - name: A\n    rss: https://a.invalid/rss\n"
                     "    languages: [EN, Ca]\n")
        assert load_sources(path)[0].languages == ["en", "ca"]

    @pytest.mark.parametrize("body,message", [
        ("sources:\n  - name: NoUrl\n    method: scrape\n", "requires 'url'"),
        ("sources:\n  - name: NoFeed\n", "requires 'rss'"),
        ("sources:\n  - name: A\n    rss: x\n  - name: A\n    rss: y\n", "duplicate"),
        ("sources:\n  - name: W\n    method: telepathy\n    rss: x\n", "unknown method"),
        ("sources: []\n", "non-empty list"),
        ("sources:\n  - rss: https://a.invalid/rss\n", "missing 'name'"),
    ])
    def test_invalid_configs_are_rejected_with_the_reason(self, tmp_path, body, message):
        with pytest.raises(ConfigError, match=message):
            load_sources(write(tmp_path, "s.yaml", body))

    def test_missing_file_is_reported(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_sources(tmp_path / "absent.yaml")


class TestPreferences:
    def test_topics_as_a_plain_list(self, tmp_path):
        """The shape the spec's example uses."""
        path = write(tmp_path, "p.yaml",
                     "preferences:\n  topics:\n    - technology\n    - Economics\n")
        _, prefs, _, _ = load_preferences(path)
        assert prefs.topics == {"technology": 1.5, "economics": 1.5}

    def test_topics_as_a_weights_mapping(self, tmp_path):
        path = write(tmp_path, "p.yaml",
                     "preferences:\n  topics:\n    technology: 1.8\n    sports: 0.2\n")
        _, prefs, _, _ = load_preferences(path)
        assert prefs.topics == {"technology": 1.8, "sports": 0.2}

    def test_topics_of_a_wrong_type_is_rejected(self, tmp_path):
        path = write(tmp_path, "p.yaml", "preferences:\n  topics: 12\n")
        with pytest.raises(ConfigError, match="list or a mapping"):
            load_preferences(path)

    def test_ranking_terms_replace_the_defaults(self, tmp_path):
        path = write(tmp_path, "p.yaml",
                     "ranking:\n  terms:\n    importance: 2.0\n    recency: 0.1\n")
        _, prefs, _, _ = load_preferences(path)
        assert prefs.ranking == {"importance": 2.0, "recency": 0.1}

    def test_unknown_ranking_term_is_an_error_not_a_no_op(self, tmp_path):
        """A typo'd term would otherwise silently do nothing."""
        path = write(tmp_path, "p.yaml",
                     "ranking:\n  terms:\n    importance: 1.0\n    recencyy: 0.5\n")
        with pytest.raises(ConfigError, match="unknown ranking term"):
            load_preferences(path)

    def test_empty_ranking_terms_is_an_error(self, tmp_path):
        path = write(tmp_path, "p.yaml", "ranking:\n  terms: {}\n")
        with pytest.raises(ConfigError, match="non-empty mapping"):
            load_preferences(path)

    def test_week_ends_on_is_validated(self, tmp_path):
        path = write(tmp_path, "p.yaml", "digest:\n  week_ends_on: 9\n")
        with pytest.raises(ConfigError, match="0-6"):
            load_preferences(path)

    def test_missing_file_uses_defaults(self, tmp_path):
        settings, prefs, digest, storage = load_preferences(tmp_path / "absent.yaml")
        assert settings.supported_languages == ["en", "es", "ca"]
        assert prefs.topics == {}
        assert digest.archive is True

    def test_output_language_outside_supported_is_allowed(self, tmp_path):
        """You may read Spanish sources and want an English digest."""
        path = write(tmp_path, "p.yaml",
                     "settings:\n  supported_languages: [es, ca]\n  output_language: en\n")
        settings, _, _, _ = load_preferences(path)
        assert settings.output_language == "en"


class TestEnvSettings:
    def test_gemini_without_a_key_degrades_to_offline(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        assert load_llm_settings().provider == "none"
        assert load_embedding_settings().provider == "none"

    def test_one_key_serves_both_llm_and_embeddings(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        assert load_llm_settings().api_key == "k"
        assert load_embedding_settings().api_key == "k"

    def test_embeddings_can_use_a_separate_key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "llm-key")
        monkeypatch.setenv("EMBEDDING_API_KEY", "embed-key")
        assert load_embedding_settings().api_key == "embed-key"

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        monkeypatch.setenv("LLM_MODEL", "gemini-custom")
        monkeypatch.setenv("LLM_BATCH_SIZE", "3")
        monkeypatch.setenv("EMBEDDING_DIMENSIONS", "768")
        monkeypatch.setenv("EMBEDDING_SIMILARITY_THRESHOLD", "0.9")
        assert load_llm_settings().model == "gemini-custom"
        assert load_llm_settings().batch_size == 3
        assert load_embedding_settings().dimensions == 768
        assert load_embedding_settings().similarity_threshold == 0.9

    def test_rate_limit_env_overrides(self, monkeypatch):
        monkeypatch.setenv("LLM_RATE_LIMIT_RETRIES", "0")
        monkeypatch.setenv("LLM_RATE_LIMIT_WAIT", "120")
        monkeypatch.setenv("EMBEDDING_RATE_LIMIT_RETRIES", "5")
        assert load_llm_settings().max_rate_limit_retries == 0
        assert load_llm_settings().max_rate_limit_wait == 120.0
        assert load_embedding_settings().max_rate_limit_retries == 5

    def test_rate_limit_defaults_wait_out_a_per_minute_limit(self):
        assert load_llm_settings().max_rate_limit_retries == 3
        assert load_embedding_settings().max_rate_limit_wait == 600.0

    def test_bad_env_numbers_fall_back(self, monkeypatch):
        monkeypatch.setenv("LLM_BATCH_SIZE", "not-a-number")
        monkeypatch.setenv("EMBEDDING_SIMILARITY_THRESHOLD", "nope")
        assert load_llm_settings().batch_size == 8
        assert load_embedding_settings().similarity_threshold == 0.82

    def test_output_language_reaches_the_llm_settings(self):
        assert load_llm_settings("ca").output_language == "ca"
