"""Config loading: YAML files for the durable stuff, env vars for secrets."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES = REPO_ROOT / "config" / "sources.yaml"
DEFAULT_INTERESTS = REPO_ROOT / "config" / "interests.yaml"
DEFAULT_DB = REPO_ROOT / "data" / "news.db"
DEFAULT_OUT = REPO_ROOT / "docs"
WEB_DIR = REPO_ROOT / "web"


class ConfigError(RuntimeError):
    """Raised for a malformed config file -- always names the offending entry."""


@dataclass
class Source:
    name: str
    method: str = "rss"
    rss: str | None = None
    url: str | None = None
    enabled: bool = True
    weight: float = 1.0
    topics: list[str] = field(default_factory=list)
    max_items: int = 25
    excerpt_chars: int = 1200
    # scrape only
    link_selector: str = "a"
    link_pattern: str | None = None
    max_links: int = 8

    @property
    def target(self) -> str:
        return self.rss if self.method == "rss" else (self.url or "")


@dataclass
class Interests:
    topics: dict[str, float] = field(default_factory=dict)
    keywords: list[str] = field(default_factory=list)
    mute: list[str] = field(default_factory=list)
    weight_importance: float = 1.0
    weight_interest: float = 0.8
    weight_recency: float = 0.6
    weight_corroboration: float = 0.3
    keyword_bonus: float = 0.15
    mute_penalty: float = 0.9
    recency_half_life_hours: float = 18.0
    max_stories: int = 60
    max_age_hours: float = 72.0
    title: str = "Daily Digest"
    subtitle: str = ""
    retention_days: int = 30


@dataclass
class LLMSettings:
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    api_key: str | None = None
    batch_size: int = 8
    # Hard ceiling on articles sent to the LLM per run. The free tier has daily
    # request limits; this is what keeps a busy news day from exhausting them.
    articles_per_run: int = 60
    max_retries: int = 3
    timeout: float = 90.0
    # Write merged headlines for multi-source clusters (one extra call each).
    write_story_briefs: bool = True


@dataclass
class Config:
    sources: list[Source]
    interests: Interests
    llm: LLMSettings
    user_agent: str = "news-digest/0.1 (+https://github.com/)"

    @property
    def enabled_sources(self) -> list[Source]:
        return [s for s in self.sources if s.enabled]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return data


def load_sources(path: Path | str = DEFAULT_SOURCES) -> list[Source]:
    data = _load_yaml(Path(path))
    defaults = data.get("defaults") or {}
    raw_sources = data.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ConfigError(f"{path}: 'sources' must be a non-empty list")

    sources: list[Source] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw_sources):
        if not isinstance(entry, dict):
            raise ConfigError(f"{path}: sources[{index}] must be a mapping")
        name = str(entry.get("name") or "").strip()
        if not name:
            raise ConfigError(f"{path}: sources[{index}] is missing 'name'")
        if name in seen:
            raise ConfigError(f"{path}: duplicate source name {name!r}")
        seen.add(name)

        method = str(entry.get("method") or "rss").lower()
        if method not in ("rss", "scrape"):
            raise ConfigError(f"{path}: {name}: unknown method {method!r} (rss|scrape)")
        rss = entry.get("rss")
        url = entry.get("url")
        if method == "rss" and not rss:
            raise ConfigError(f"{path}: {name}: method 'rss' requires 'rss'")
        if method == "scrape" and not url:
            raise ConfigError(f"{path}: {name}: method 'scrape' requires 'url'")

        topics = entry.get("topics") or []
        if isinstance(topics, str):
            topics = [topics]

        sources.append(
            Source(
                name=name,
                method=method,
                rss=str(rss) if rss else None,
                url=str(url) if url else None,
                enabled=bool(entry.get("enabled", True)),
                weight=float(entry.get("weight", 1.0)),
                topics=[str(t).lower() for t in topics],
                max_items=int(entry.get("max_items", defaults.get("max_items_per_source", 25))),
                excerpt_chars=int(entry.get("excerpt_chars", defaults.get("excerpt_chars", 1200))),
                link_selector=str(entry.get("link_selector", "a")),
                link_pattern=entry.get("link_pattern"),
                max_links=int(entry.get("max_links", 8)),
            )
        )
    return sources


def load_interests(path: Path | str = DEFAULT_INTERESTS) -> Interests:
    p = Path(path)
    if not p.exists():
        return Interests()
    data = _load_yaml(p)
    weights = data.get("weights") or {}
    feed = data.get("feed") or {}
    storage = data.get("storage") or {}
    topics = {str(k).lower(): float(v) for k, v in (data.get("topics") or {}).items()}
    return Interests(
        topics=topics,
        keywords=[str(k).lower() for k in (data.get("keywords") or [])],
        mute=[str(k).lower() for k in (data.get("mute") or [])],
        weight_importance=float(weights.get("importance", 1.0)),
        weight_interest=float(weights.get("interest", 0.8)),
        weight_recency=float(weights.get("recency", 0.6)),
        weight_corroboration=float(weights.get("corroboration", 0.3)),
        keyword_bonus=float(weights.get("keyword_bonus", 0.15)),
        mute_penalty=float(weights.get("mute_penalty", 0.9)),
        recency_half_life_hours=float(data.get("recency_half_life_hours", 18.0)),
        max_stories=int(feed.get("max_stories", 60)),
        max_age_hours=float(feed.get("max_age_hours", 72.0)),
        title=str(feed.get("title", "Daily Digest")),
        subtitle=str(feed.get("subtitle", "")),
        retention_days=int(storage.get("retention_days", 30)),
    )


def load_dotenv(path: Path | str = REPO_ROOT / ".env") -> None:
    """Minimal .env support so local runs match Actions without extra deps."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def load_llm_settings() -> LLMSettings:
    provider = (os.environ.get("LLM_PROVIDER") or "gemini").strip().lower()
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY")
    settings = LLMSettings(
        provider=provider,
        model=os.environ.get("LLM_MODEL") or "gemini-2.5-flash",
        api_key=key.strip() if key else None,
        batch_size=max(1, _env_int("LLM_BATCH_SIZE", 8)),
        articles_per_run=max(0, _env_int("LLM_ARTICLES_PER_RUN", 60)),
    )
    # Falling back rather than failing keeps the pipeline useful with no key --
    # the run still produces a feed, just with heuristic summaries.
    if settings.provider == "gemini" and not settings.api_key:
        settings.provider = "none"
    return settings


def load_config(
    sources_path: Path | str = DEFAULT_SOURCES,
    interests_path: Path | str = DEFAULT_INTERESTS,
) -> Config:
    load_dotenv()
    return Config(
        sources=load_sources(sources_path),
        interests=load_interests(interests_path),
        llm=load_llm_settings(),
        user_agent=os.environ.get(
            "NEWS_DIGEST_USER_AGENT",
            "news-digest/0.1 (+https://github.com/; personal news aggregator)",
        ),
    )
