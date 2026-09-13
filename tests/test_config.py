import pytest

from newsdigest.config import (
    ConfigError,
    load_interests,
    load_llm_settings,
    load_sources,
)


def write(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_loads_the_repo_config():
    """The shipped config must be valid -- it is what a first run uses."""
    sources = load_sources()
    interests = load_interests()
    assert len(sources) >= 3
    assert all(s.rss for s in sources if s.method == "rss")
    assert any(s.enabled for s in sources)
    assert interests.max_stories > 0


def test_defaults_apply_to_each_source(tmp_path):
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
    assert sources[0].max_items == 5
    assert sources[0].excerpt_chars == 300
    assert sources[1].max_items == 50


def test_scrape_source_requires_url(tmp_path):
    path = write(tmp_path, "s.yaml", """
sources:
  - name: NoUrl
    method: scrape
""")
    with pytest.raises(ConfigError, match="requires 'url'"):
        load_sources(path)


def test_rss_source_requires_feed(tmp_path):
    path = write(tmp_path, "s.yaml", "sources:\n  - name: NoFeed\n")
    with pytest.raises(ConfigError, match="requires 'rss'"):
        load_sources(path)


def test_duplicate_names_rejected(tmp_path):
    path = write(tmp_path, "s.yaml", """
sources:
  - name: Same
    rss: https://a.invalid/rss
  - name: Same
    rss: https://b.invalid/rss
""")
    with pytest.raises(ConfigError, match="duplicate"):
        load_sources(path)


def test_unknown_method_rejected(tmp_path):
    path = write(tmp_path, "s.yaml", """
sources:
  - name: Weird
    method: telepathy
    rss: https://a.invalid/rss
""")
    with pytest.raises(ConfigError, match="unknown method"):
        load_sources(path)


def test_missing_interests_file_uses_defaults(tmp_path):
    interests = load_interests(tmp_path / "absent.yaml")
    assert interests.topics == {}
    assert interests.max_stories == 60


def test_gemini_without_a_key_degrades_to_offline(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert load_llm_settings().provider == "none"


def test_env_overrides_model_and_budget(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "gemini-custom")
    monkeypatch.setenv("LLM_BATCH_SIZE", "3")
    monkeypatch.setenv("LLM_ARTICLES_PER_RUN", "12")
    settings = load_llm_settings()
    assert (settings.model, settings.batch_size, settings.articles_per_run) == (
        "gemini-custom", 3, 12
    )


def test_bad_env_numbers_fall_back(monkeypatch):
    monkeypatch.setenv("LLM_BATCH_SIZE", "not-a-number")
    assert load_llm_settings().batch_size == 8
