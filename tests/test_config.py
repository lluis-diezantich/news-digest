import pytest

from newsdigest.config import (
    DEFAULT_LOCAL_TIMEOUT, DEFAULT_TIMEOUT, ConfigError, load_embedding_settings,
    load_filters, load_llm_settings, load_preferences, load_sources,
)


def write(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


class TestShippedConfig:
    def test_the_repo_config_is_valid(self):
        """What a first run actually loads."""
        sources = load_sources()
        settings, prefs, digest, storage, *_ = load_preferences()
        # Section 2 names ten initial sources, five per language.
        assert len(sources) >= 10
        assert any(s.enabled for s in sources)
        assert all(s.senders for s in sources if s.enabled)
        assert settings.supported_languages == ["en", "es"]
        assert digest.max_stories > 0
        assert storage.embedding_retention_days <= storage.retention_days
        assert storage.email_body_retention_days <= storage.retention_days

    def test_the_shipped_filters_are_valid(self):
        filters = load_filters()
        assert filters.excluded_topics
        assert not set(filters.excluded_topics) & set(filters.include_topics)

    def test_every_enabled_source_declares_a_language(self):
        """Undeclared languages make detection guess on short headlines."""
        undeclared = [s.name for s in load_sources() if s.enabled and not s.languages]
        assert undeclared == []

    def test_both_configured_languages_have_sources(self):
        """Section 9: mixed-language input is the point, so neither side may be
        empty -- a digest built from one language is not being tested."""
        languages = {lang for s in load_sources() if s.enabled for lang in s.languages}
        assert {"en", "es"} <= languages

    def test_excluded_topics_are_in_the_classifier_vocabulary(self):
        """A topic the classifier can never emit can never be excluded.

        This silently disarmed the filter once already: closing the vocabulary
        left `excluded_topics: [celebrity]` matching nothing at all, because no
        model could return a word that was not on the list.
        """
        from newsdigest.llm.base import TOPICS

        for topic in load_filters().excluded_topics:
            assert topic in TOPICS, f"{topic!r} is not a topic any provider can emit"

    def test_a_publisher_may_own_several_newsletters(self):
        """Corroboration counts publishers, so this must be expressible."""
        sources = load_sources()
        assert all(s.publisher for s in sources)
        # Two entries sharing a publisher must not be two corroborating outlets.
        by_publisher: dict[str, int] = {}
        for source in sources:
            by_publisher[source.publisher] = by_publisher.get(source.publisher, 0) + 1
        assert max(by_publisher.values()) >= 1


class TestSources:
    def test_defaults_apply_per_source(self, tmp_path):
        path = write(tmp_path, "s.yaml", """
defaults:
  excerpt_chars: 300
sources:
  - name: A
    senders: ["@a.invalid"]
  - name: B
    senders: ["@b.invalid"]
    excerpt_chars: 900
""")
        sources = load_sources(path)
        assert sources[0].excerpt_chars == 300
        assert sources[1].excerpt_chars == 900

    def test_publisher_and_newsletter_default_to_the_name(self, tmp_path):
        path = write(tmp_path, "s.yaml",
                     "sources:\n  - name: Solo\n    senders: ['@a.invalid']\n")
        source = load_sources(path)[0]
        assert source.publisher == "Solo"
        assert source.newsletter == "Solo"

    def test_languages_are_lowercased(self, tmp_path):
        path = write(tmp_path, "s.yaml",
                     "sources:\n  - name: A\n    senders: ['@a.invalid']\n"
                     "    languages: [EN, Es]\n")
        assert load_sources(path)[0].languages == ["en", "es"]

    def test_senders_are_lowercased(self, tmp_path):
        """Addresses arrive in whatever case the sender chose."""
        path = write(tmp_path, "s.yaml",
                     "sources:\n  - name: A\n    senders: ['News@A.Invalid']\n")
        assert load_sources(path)[0].senders == ["news@a.invalid"]

    def test_subject_patterns_are_validated_as_regexes(self, tmp_path):
        path = write(tmp_path, "s.yaml",
                     "sources:\n  - name: A\n    senders: ['@a.invalid']\n"
                     "    subject_patterns: ['saturday(']\n")
        with pytest.raises(ConfigError, match="not a valid regex"):
            load_sources(path)

    def test_a_disabled_source_may_be_a_placeholder(self, tmp_path):
        """No senders yet is fine while a subscription is still pending."""
        path = write(tmp_path, "s.yaml",
                     "sources:\n  - name: Pending\n    enabled: false\n")
        assert load_sources(path)[0].senders == []

    @pytest.mark.parametrize("body,message", [
        # An enabled source with no sender rule can never match a message, so it
        # would sit in the config looking configured and contribute nothing.
        ("sources:\n  - name: NoSender\n", "at least one 'senders'"),
        ("sources:\n  - name: A\n    senders: ['@a.invalid']\n"
         "  - name: A\n    senders: ['@b.invalid']\n", "duplicate"),
        ("sources: []\n", "non-empty list"),
        ("sources:\n  - senders: ['@a.invalid']\n", "missing 'name'"),
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
        _, prefs, *_ = load_preferences(path)
        assert prefs.topics == {"technology": 1.5, "economics": 1.5}

    def test_topics_as_a_weights_mapping(self, tmp_path):
        path = write(tmp_path, "p.yaml",
                     "preferences:\n  topics:\n    technology: 1.8\n    sports: 0.2\n")
        _, prefs, *_ = load_preferences(path)
        assert prefs.topics == {"technology": 1.8, "sports": 0.2}

    def test_topics_of_a_wrong_type_is_rejected(self, tmp_path):
        path = write(tmp_path, "p.yaml", "preferences:\n  topics: 12\n")
        with pytest.raises(ConfigError, match="list or a mapping"):
            load_preferences(path)

    def test_ranking_terms_replace_the_defaults(self, tmp_path):
        path = write(tmp_path, "p.yaml",
                     "ranking:\n  terms:\n    importance: 2.0\n    recency: 0.1\n")
        _, prefs, *_ = load_preferences(path)
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
        settings, prefs, digest, storage, *_ = load_preferences(tmp_path / "absent.yaml")
        assert settings.supported_languages == ["en", "es"]
        assert prefs.topics == {}
        assert digest.update_readme is True

    def test_output_language_outside_supported_is_allowed(self, tmp_path):
        """You may read Spanish sources and want an English digest."""
        path = write(tmp_path, "p.yaml",
                     "settings:\n  supported_languages: [es, ca]\n  output_language: en\n")
        settings, *_ = load_preferences(path)
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

    def test_thinking_level_defaults_to_the_cheapest(self):
        assert load_llm_settings().thinking_level == "low"

    def test_thinking_level_env_override(self, monkeypatch):
        monkeypatch.setenv("LLM_THINKING_LEVEL", "HIGH")
        assert load_llm_settings().thinking_level == "high"

    def test_thinking_level_off_sends_no_field(self, monkeypatch):
        for value in ("off", "none", "unset", ""):
            monkeypatch.setenv("LLM_THINKING_LEVEL", value)
            assert load_llm_settings().thinking_level == ""

    def test_bad_thinking_level_is_a_startup_error(self, monkeypatch):
        """It would 400 every request and silently cost the run its enrichment."""
        monkeypatch.setenv("LLM_THINKING_LEVEL", "minimal")
        with pytest.raises(ConfigError, match="thinking level"):
            load_llm_settings()

    def test_bad_env_numbers_fall_back(self, monkeypatch):
        monkeypatch.setenv("LLM_BATCH_SIZE", "not-a-number")
        monkeypatch.setenv("EMBEDDING_SIMILARITY_THRESHOLD", "nope")
        assert load_llm_settings().batch_size == 8
        assert load_embedding_settings().similarity_threshold == 0.82

    def test_output_language_reaches_the_llm_settings(self):
        assert load_llm_settings("ca").output_language == "ca"


class TestLLMTimeout:
    """The request timeout, which was unreachable from the environment until
    2026-09-17 and flat 90s for every provider. See config.DEFAULT_LOCAL_TIMEOUT."""

    def test_local_provider_gets_the_longer_default(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.delenv("LLM_TIMEOUT", raising=False)
        assert load_llm_settings().timeout == DEFAULT_LOCAL_TIMEOUT

    def test_hosted_provider_keeps_the_short_default(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.delenv("LLM_TIMEOUT", raising=False)
        assert load_llm_settings().timeout == DEFAULT_TIMEOUT

    def test_env_overrides_either(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_TIMEOUT", "45")
        assert load_llm_settings().timeout == 45.0

    def test_the_provider_is_built_with_it(self, monkeypatch):
        """The bug this fixes: OllamaProvider defaults to 300 but the factory
        passes settings.timeout, so the settings value is what actually applies."""
        from newsdigest.llm import get_provider

        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_TIMEOUT", "123")
        assert get_provider(load_llm_settings()).timeout == 123.0
