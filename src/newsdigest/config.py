"""Config loading: YAML for the durable stuff, env vars for secrets.

Two files, both editable without touching code:
  config/sources.yaml      -- what to read
  config/preferences.yaml  -- language settings, interests, and the ranking formula
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES = REPO_ROOT / "config" / "sources.yaml"
DEFAULT_PREFERENCES = REPO_ROOT / "config" / "preferences.yaml"
DEFAULT_DB = REPO_ROOT / "data" / "news.db"
DEFAULT_OUT = REPO_ROOT / "docs"
WEB_DIR = REPO_ROOT / "web"

#: Ranking signals `scoring.py` knows how to compute. Anything else in the
#: ranking config is a typo, and we say so rather than silently ignoring it.
KNOWN_RANKING_TERMS = frozenset(
    {
        "importance",
        "relevance",
        "interest",
        "recency",
        "corroboration",
        "source_preference",
        "story_size",
    }
)


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
    #: Languages this source publishes in. Used to constrain detection and as
    #: the fallback when a headline is too short to judge.
    languages: list[str] = field(default_factory=list)
    #: Outlet this feed belongs to. Several feeds can share one publisher, and
    #: corroboration counts publishers -- seven El País section feeds covering
    #: one story are one outlet's view of it, not seven independent ones.
    publisher: str = ""
    max_items: int = 25
    excerpt_chars: int = 1200
    # scrape only
    link_selector: str = "a"
    link_pattern: str | None = None
    max_links: int = 8

    def __post_init__(self) -> None:
        if not self.publisher:
            self.publisher = self.name

    @property
    def target(self) -> str:
        return (self.rss if self.method == "rss" else self.url) or ""


@dataclass
class Settings:
    supported_languages: list[str] = field(default_factory=lambda: ["en", "es", "ca"])
    #: Language the LLM writes summaries in, independent of the sources'.
    output_language: str = "en"


@dataclass
class Preferences:
    topics: dict[str, float] = field(default_factory=dict)
    excluded_topics: list[str] = field(default_factory=list)
    preferred_sources: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    #: signal name -> weight. Drop a term to remove it from the formula.
    ranking: dict[str, float] = field(
        default_factory=lambda: {
            "importance": 1.0,
            "relevance": 1.0,
            "interest": 0.8,
            "recency": 0.6,
            "corroboration": 0.4,
            "source_preference": 0.3,
            "story_size": 0.2,
        }
    )
    keyword_bonus: float = 0.15
    excluded_penalty: float = 1.5
    preferred_source_bonus: float = 0.5
    recency_half_life_hours: float = 72.0


@dataclass
class DigestSettings:
    title: str = "Weekly Digest"
    subtitle: str = ""
    max_stories: int = 30
    #: Stories needing at least this many articles are kept even if they rank
    #: low; 1 means no filtering.
    min_articles: int = 1
    #: Weekday the digest window ends on (0 = Monday, 6 = Sunday).
    week_ends_on: int = 6
    #: Keep past digests browsable on the site.
    archive: bool = True
    archive_limit: int = 52


@dataclass
class StorageSettings:
    retention_days: int = 45
    #: Embeddings are working data for the weekly run, and the biggest thing in
    #: the committed database, so they go sooner than the articles.
    embedding_retention_days: int = 21


@dataclass
class LLMSettings:
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    api_key: str | None = None
    batch_size: int = 8
    articles_per_run: int = 400
    max_retries: int = 3
    timeout: float = 90.0
    write_story_briefs: bool = True
    output_language: str = "en"
    #: Ask the LLM to break ties on ambiguous embedding clusters.
    resolve_clusters: bool = True
    max_cluster_checks: int = 40


@dataclass
class EmbeddingSettings:
    provider: str = "gemini"
    model: str = "gemini-embedding-2"
    api_key: str | None = None
    #: 256 keeps the committed database small (~1 KB/article; 768 would be 3x).
    #: Gemini embeddings degrade gracefully when truncated.
    dimensions: int = 256
    task_type: str | None = None
    #: Cosine similarity at or above which two articles are the same event.
    similarity_threshold: float = 0.82
    #: Between this and the threshold, ask the LLM (if enabled) to decide.
    ambiguous_threshold: float = 0.72


@dataclass
class Config:
    sources: list[Source]
    settings: Settings = field(default_factory=Settings)
    preferences: Preferences = field(default_factory=Preferences)
    digest: DigestSettings = field(default_factory=DigestSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    embeddings: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    user_agent: str = "news-digest/0.2 (+https://github.com/)"

    @property
    def enabled_sources(self) -> list[Source]:
        return [s for s in self.sources if s.enabled]

    def publisher_of(self, source_name: str) -> str:
        for source in self.sources:
            if source.name == source_name:
                return source.publisher
        return source_name


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return data


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


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
        rss, url = entry.get("rss"), entry.get("url")
        if method == "rss" and not rss:
            raise ConfigError(f"{path}: {name}: method 'rss' requires 'rss'")
        if method == "scrape" and not url:
            raise ConfigError(f"{path}: {name}: method 'scrape' requires 'url'")

        sources.append(
            Source(
                name=name,
                method=method,
                rss=str(rss) if rss else None,
                url=str(url) if url else None,
                enabled=bool(entry.get("enabled", True)),
                weight=float(entry.get("weight", 1.0)),
                topics=[t.lower() for t in _as_list(entry.get("topics"))],
                languages=[l.lower() for l in _as_list(entry.get("languages"))],
                publisher=str(entry.get("publisher") or "").strip(),
                max_items=int(entry.get("max_items", defaults.get("max_items_per_source", 25))),
                excerpt_chars=int(entry.get("excerpt_chars", defaults.get("excerpt_chars", 1200))),
                link_selector=str(entry.get("link_selector", "a")),
                link_pattern=entry.get("link_pattern"),
                max_links=int(entry.get("max_links", 8)),
            )
        )
    return sources


def load_preferences(path: Path | str = DEFAULT_PREFERENCES) -> tuple[
    Settings, Preferences, DigestSettings, StorageSettings
]:
    p = Path(path)
    if not p.exists():
        log.warning("no preferences file at %s; using defaults", p)
        return Settings(), Preferences(), DigestSettings(), StorageSettings()

    data = _load_yaml(p)
    raw_settings = data.get("settings") or {}
    raw_prefs = data.get("preferences") or {}
    raw_ranking = data.get("ranking") or {}
    raw_digest = data.get("digest") or {}
    raw_storage = data.get("storage") or {}

    supported = [l.lower() for l in _as_list(raw_settings.get("supported_languages"))]
    settings = Settings(
        supported_languages=supported or ["en", "es", "ca"],
        output_language=str(raw_settings.get("output_language", "en")).lower(),
    )
    if settings.output_language not in settings.supported_languages:
        # Not fatal: the LLM can write a language you do not collect.
        log.info(
            "output_language %r is not in supported_languages %s",
            settings.output_language, settings.supported_languages,
        )

    terms = raw_ranking.get("terms")
    ranking = Preferences().ranking
    if terms is not None:
        if not isinstance(terms, dict) or not terms:
            raise ConfigError(f"{p}: ranking.terms must be a non-empty mapping")
        unknown = set(terms) - KNOWN_RANKING_TERMS
        if unknown:
            raise ConfigError(
                f"{p}: unknown ranking term(s) {sorted(unknown)}; "
                f"known terms: {sorted(KNOWN_RANKING_TERMS)}"
            )
        ranking = {str(k): float(v) for k, v in terms.items()}

    preferences = Preferences(
        topics=_topic_weights(raw_prefs.get("topics"), p),
        excluded_topics=[t.lower() for t in _as_list(raw_prefs.get("excluded_topics"))],
        preferred_sources=_as_list(raw_prefs.get("preferred_sources")),
        keywords=[k.lower() for k in _as_list(raw_prefs.get("keywords"))],
        ranking=ranking,
        keyword_bonus=float(raw_ranking.get("keyword_bonus", 0.15)),
        excluded_penalty=float(raw_ranking.get("excluded_penalty", 1.5)),
        preferred_source_bonus=float(raw_ranking.get("preferred_source_bonus", 0.5)),
        recency_half_life_hours=float(raw_ranking.get("recency_half_life_hours", 72.0)),
    )

    week_ends_on = int(raw_digest.get("week_ends_on", 6))
    if not 0 <= week_ends_on <= 6:
        raise ConfigError(f"{p}: digest.week_ends_on must be 0-6 (Monday-Sunday)")

    digest = DigestSettings(
        title=str(raw_digest.get("title", "Weekly Digest")),
        subtitle=str(raw_digest.get("subtitle", "")),
        max_stories=int(raw_digest.get("max_stories", 30)),
        min_articles=int(raw_digest.get("min_articles", 1)),
        week_ends_on=week_ends_on,
        archive=bool(raw_digest.get("archive", True)),
        archive_limit=int(raw_digest.get("archive_limit", 52)),
    )
    storage = StorageSettings(
        retention_days=int(raw_storage.get("retention_days", 45)),
        embedding_retention_days=int(raw_storage.get("embedding_retention_days", 21)),
    )
    return settings, preferences, digest, storage


#: Weight given to a topic listed without one, as the spec's example does.
LISTED_TOPIC_WEIGHT = 1.5


def _topic_weights(raw: Any, path: Path) -> dict[str, float]:
    """Accept either shape: a weights mapping, or a plain list of topics.

    The list form (`topics: [technology, economics]`) is what most people write
    first; it means "these matter", so each entry gets a weight above the 1.0
    default rather than being silently ignored.
    """
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {str(k).lower(): float(v) for k, v in raw.items()}
    if isinstance(raw, list):
        return {str(t).lower(): LISTED_TOPIC_WEIGHT for t in raw}
    raise ConfigError(
        f"{path}: preferences.topics must be a list or a mapping of topic -> weight"
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
        log.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        log.warning("%s=%r is not a number; using %s", name, raw, default)
        return default


def _gemini_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY")
    return key.strip() if key and key.strip() else None


def load_llm_settings(output_language: str = "en") -> LLMSettings:
    key = _gemini_key()
    settings = LLMSettings(
        provider=(os.environ.get("LLM_PROVIDER") or "gemini").strip().lower(),
        model=os.environ.get("LLM_MODEL") or "gemini-2.5-flash",
        api_key=key,
        batch_size=max(1, _env_int("LLM_BATCH_SIZE", 8)),
        articles_per_run=max(0, _env_int("LLM_ARTICLES_PER_RUN", 400)),
        output_language=output_language,
    )
    # Degrade rather than fail: the digest still builds without a key.
    if settings.provider == "gemini" and not settings.api_key:
        settings.provider = "none"
    return settings


def load_embedding_settings() -> EmbeddingSettings:
    key = os.environ.get("EMBEDDING_API_KEY") or _gemini_key()
    settings = EmbeddingSettings(
        provider=(os.environ.get("EMBEDDING_PROVIDER") or "gemini").strip().lower(),
        model=os.environ.get("EMBEDDING_MODEL") or "gemini-embedding-2",
        api_key=key.strip() if key else None,
        dimensions=max(64, _env_int("EMBEDDING_DIMENSIONS", 256)),
        task_type=os.environ.get("EMBEDDING_TASK_TYPE") or None,
        similarity_threshold=_env_float("EMBEDDING_SIMILARITY_THRESHOLD", 0.82),
        ambiguous_threshold=_env_float("EMBEDDING_AMBIGUOUS_THRESHOLD", 0.72),
    )
    if settings.provider == "gemini" and not settings.api_key:
        settings.provider = "none"
    return settings


def load_config(
    sources_path: Path | str = DEFAULT_SOURCES,
    preferences_path: Path | str = DEFAULT_PREFERENCES,
) -> Config:
    load_dotenv()
    settings, preferences, digest, storage = load_preferences(preferences_path)
    return Config(
        sources=load_sources(sources_path),
        settings=settings,
        preferences=preferences,
        digest=digest,
        storage=storage,
        llm=load_llm_settings(settings.output_language),
        embeddings=load_embedding_settings(),
        user_agent=os.environ.get(
            "NEWS_DIGEST_USER_AGENT",
            "news-digest/0.2 (+https://github.com/; personal news aggregator)",
        ),
    )
