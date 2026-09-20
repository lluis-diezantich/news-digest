"""Config loading: YAML for the durable stuff, env vars for secrets.

Three files, all editable without touching code:
  config/sources.yaml      -- which newsletters, and how to recognise them
  config/filters.yaml      -- what is not news (section 8)
  config/preferences.yaml  -- languages, the ranking formula, digest size

Credentials are never in any of them. The mailbox password and the LLM key come
from the environment, which is `.env` locally and Actions secrets in CI.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES = REPO_ROOT / "config" / "sources.yaml"
DEFAULT_PREFERENCES = REPO_ROOT / "config" / "preferences.yaml"
DEFAULT_FILTERS = REPO_ROOT / "config" / "filters.yaml"
DEFAULT_DB = REPO_ROOT / "data" / "news.db"
DEFAULT_OUT = REPO_ROOT / "digests"
DEFAULT_DEBUG = REPO_ROOT / "debug"
PROMPT_DIR = REPO_ROOT / "prompts"

#: Ranking signals `scoring.py` knows how to compute. Anything else in the
#: ranking config is a typo, and we say so rather than silently ignoring it.
KNOWN_RANKING_TERMS = frozenset(
    {
        "importance",
        "relevance",
        "interest",
        "recency",
        "corroboration",
        "editorial_position",
        "source_preference",
        "story_size",
    }
)

#: Values Google documents for `generation_config.thinking_level`. The empty
#: string means "send no thinking field at all", which leaves the model on its
#: own default -- the right choice for models where thinking is already off.
THINKING_LEVELS = ("", "low", "medium", "high")

#: Output language values. `auto` picks the commonest input language at run time.
OUTPUT_LANGUAGES = ("auto", "en", "es")


class ConfigError(RuntimeError):
    """Raised for a malformed config file -- always names the offending entry."""


@dataclass
class NewsletterSource:
    """One newsletter, and the rules that recognise its mail.

    Recognition is two-part on purpose. `senders` is necessary but rarely
    sufficient: a publisher usually sends every newsletter it has from one
    address, so `subject_patterns` is what separates the weekly digest we want
    from the daily briefing we do not.
    """

    name: str
    #: Addresses or domains this newsletter arrives from. A bare domain, with or
    #: without a leading `@`, matches that domain and its subdomains -- bulk
    #: mailers move between `mail.` and `news.` hosts without notice.
    senders: list[str] = field(default_factory=list)
    #: Regexes matched against the SUBJECT, accent-stripped and lowercased. Empty
    #: means "every subject from these senders".
    subject_patterns: list[str] = field(default_factory=list)
    #: Regexes matched against the From DISPLAY NAME, same normalisation.
    #:
    #: Needed because a publisher's mail often comes from a SHARED bulk-mail
    #: domain rather than its own: Público arrives from `news@publisher-news.com`,
    #: which tells you nothing about the publisher and could serve any number of
    #: them. The display name -- "Público - Hoy en Público" -- is the only reliable
    #: discriminator there, so a sender rule on a shared domain should always be
    #: paired with one of these.
    sender_name_patterns: list[str] = field(default_factory=list)
    #: Human name of the newsletter, e.g. "Saturday Edition".
    newsletter: str = ""
    #: Outlet this newsletter belongs to. Several newsletters can share one, and
    #: corroboration counts PUBLISHERS -- two EL PAÍS newsletters covering one
    #: story are one outlet's view of it, not two independent ones.
    publisher: str = ""
    enabled: bool = True
    weight: float = 1.0
    topics: list[str] = field(default_factory=list)
    #: Languages this newsletter publishes in. Constrains language detection and
    #: is the fallback when a headline is too short to judge.
    languages: list[str] = field(default_factory=list)
    excerpt_chars: int = 1200
    #: Regexes matched against an item's URL, dropping it before it reaches the
    #: database. Still useful with newsletters, because an extracted item links
    #: the publisher's own article: `/deportes/`, `/horoscopo/`, `/loterias/`.
    #: It only works on a RESOLVED url, though -- a tracking link has no section
    #: path at all, which is why `extract/links.py` exists.
    #: The global list in sources.yaml applies to every source; these are extra.
    exclude_url_patterns: list[str] = field(default_factory=list)
    #: Regexes matched against an item's HEADLINE, dropping it the same way. For
    #: junk with no section path: a weekly weather round-up, a football result at
    #: an outlet that files sport under its main path.
    exclude_title_patterns: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.publisher:
            self.publisher = self.name
        if not self.newsletter:
            self.newsletter = self.name


@dataclass
class MailboxSettings:
    """Where the newsletters arrive. Every value comes from the environment."""

    provider: str = "imap"
    host: str = ""
    port: int = 993
    username: str = ""
    #: Never logged, never written to disk, never included in debug output.
    password: str = ""
    folder: str = "INBOX"
    timeout: float = 30.0

    @property
    def configured(self) -> bool:
        return bool(self.host and self.username and self.password)

    def describe(self) -> str:
        """A safe one-line description. Used in logs, so it omits the password
        and shows only the local part of the address."""
        who = self.username.partition("@")[0] or "?"
        return f"{self.provider}://{who}@{self.host}:{self.port}/{self.folder}"


@dataclass
class Settings:
    supported_languages: list[str] = field(default_factory=lambda: ["en", "es"])
    #: Language the digest is written in, independent of what it reads.
    #: `auto` resolves at run time -- see `resolve_output_language`.
    output_language: str = "en"

    def resolve_output_language(self, counts: dict[str, int] | None = None) -> str:
        """The language to write in, resolving `auto` against the week's input.

        `auto` means "whichever language most of this week's newsletters were
        in", which keeps a digest built mostly from Spanish sources in Spanish
        without anyone having to switch a setting. Ties and an empty week fall
        back to the first supported language, so the result is never empty and
        never depends on dict ordering.
        """
        if self.output_language != "auto":
            return self.output_language
        fallback = (self.supported_languages or ["en"])[0]
        if not counts:
            return fallback
        best = max(
            (lang for lang in counts if lang in self.supported_languages),
            key=lambda lang: (counts[lang], lang == fallback),
            default=fallback,
        )
        return best


#: Publisher count at which corroboration tops out.
#:
#: UNCALIBRATED for newsletters. The RSS version of this project measured 13 over
#: a real week and used it, but that was 22 feeds; ten hand-curated newsletters
#: cannot exceed 10 and will rarely pass 5, so 13 would saturate above the range
#: that occurs and flatten the term to near-zero for every story. 5 is a guess at
#: the observed maximum. Measure it -- `news-digest inspect --week` prints the
#: publisher spread -- and set it to what the weeks actually produce.
DEFAULT_CORROBORATION_SATURATION = 5.0


@dataclass
class Preferences:
    topics: dict[str, float] = field(default_factory=dict)
    excluded_topics: list[str] = field(default_factory=list)
    preferred_sources: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    #: signal name -> weight. Drop a term to remove it from the formula.
    #:
    #: UNCALIBRATED. The weights are carried over from the RSS project, where
    #: each was measured against a published top ten, and the source set they
    #: were measured on no longer exists. The shape should still be right --
    #: position in a curated newsletter is a stronger editorial signal than
    #: position in a feed, not a weaker one -- but treat the numbers as a
    #: starting point, not a finding.
    ranking: dict[str, float] = field(
        default_factory=lambda: {
            "editorial_position": 2.0,
            "corroboration": 1.5,
            "recency": 0.3,
            "story_size": 0.2,
        }
    )
    keyword_bonus: float = 0.15
    excluded_penalty: float = 1.5
    preferred_source_bonus: float = 0.5
    #: Over a weekly window this should be long: Monday's news is still part of
    #: the week on Sunday.
    recency_half_life_hours: float = 72.0
    corroboration_saturation: float = DEFAULT_CORROBORATION_SATURATION


@dataclass
class FilterSettings:
    """Section 8: what is unlikely to belong in a global weekly digest.

    Filtering happens at the STORY level, never the newsletter level. A source
    that runs one sports item has not disqualified its front page.
    """

    #: Topics dropped outright. Matched against the classifier's own vocabulary.
    excluded_topics: list[str] = field(
        default_factory=lambda: [
            "sports", "celebrity", "entertainment", "lifestyle", "fashion",
            "food", "shopping", "advertising",
        ]
    )
    #: Topics this digest is for. Advisory rather than a whitelist: an item
    #: classified into none of them is kept but ranked below one that matches,
    #: because the vocabulary will always lag the news.
    include_topics: list[str] = field(
        default_factory=lambda: [
            "politics", "world", "economics", "science", "technology",
            "climate", "health", "conflict", "society",
        ]
    )
    #: Content types dropped. Empty by default: section 8 excludes opinion that
    #: carries no news, which is a judgement the classifier makes per item, not a
    #: property of the type -- and a blanket `opinion` drop would lose
    #: elDiario.es's Boletín del director, which is a director's column.
    drop_content_types: list[str] = field(default_factory=list)
    #: Ask the LLM to classify items. With this off, filtering is the regex lists
    #: in sources.yaml alone, which newsletters defeat more easily than feeds do:
    #: there is no section path in a tracking link.
    classify: bool = True
    #: Items sent to the classifier per run. Classification is one cheap call per
    #: batch and runs before clustering, so it sees more items than enrichment.
    max_items: int = 600


@dataclass
class RegionSettings:
    """Section 12: keep one country from taking the whole digest."""

    #: The buckets we try to spread across. Order is display order.
    regions: list[str] = field(
        default_factory=lambda: [
            "europe", "north america", "latin america", "middle east",
            "africa", "asia", "oceania", "global",
        ]
    )
    #: Largest share of the main stories one region may hold before the
    #: diversity pass starts promoting others. Not a quota: the pass only ever
    #: reorders stories that already qualified, and it stops as soon as it runs
    #: out of alternatives, so a week in which one region genuinely holds every
    #: major story still publishes them all.
    max_share: float = 0.5
    #: Off means rank on score alone.
    enabled: bool = True


@dataclass
class DigestSettings:
    title: str = "The Week in Global News"
    subtitle: str = ""
    #: Section 11 asks for 10-15 major stories. 12 sits in the middle.
    max_stories: int = 12
    #: Section 14's "Also worth knowing": stories just below the cut, as a list.
    minor_stories: int = 5
    #: Stories needing at least this many articles. 1, not the RSS project's 3:
    #: ten newsletters rarely triple-cover anything, and at 3 most weeks would
    #: publish nothing at all.
    min_articles: int = 1
    #: Weekday the digest window ends on (0 = Monday, 6 = Sunday).
    week_ends_on: int = 6
    #: Rewrite README.md with the newest digest.
    update_readme: bool = True


@dataclass
class StorageSettings:
    retention_days: int = 120
    #: Raw email bodies are the bulkiest and most sensitive thing stored, and
    #: nothing downstream needs them once articles are extracted. They go first.
    email_body_retention_days: int = 30
    #: Embeddings are working data for one run, and big (~1 KB per article).
    embedding_retention_days: int = 60


@dataclass
class LLMSettings:
    provider: str = "gemini"
    model: str = "gemini-3.6-flash"
    api_key: str | None = None
    batch_size: int = 8
    articles_per_run: int = 400
    max_retries: int = 3
    #: Seconds per request. 90 suits a hosted model; a local one needs far more,
    #: and `load_llm_settings` raises the default accordingly.
    timeout: float = 90.0
    #: Waits allowed per request against a per-minute 429, and the total time one
    #: run may spend waiting on rate limits. A per-day 429 is never waited out.
    max_rate_limit_retries: int = 3
    max_rate_limit_wait: float = 600.0
    #: Where a local provider listens. Ignored by hosted ones.
    base_url: str = ""
    #: Context window for a local model. A batch of 16 articles with 900-char
    #: excerpts overflows 4k, and an overflowed prompt is truncated silently --
    #: which surfaces as missing ids rather than an error.
    num_ctx: int = 8192
    #: `low` is the cheapest level the 3.x Flash models accept. Thought tokens
    #: are billed as output and come out of `maxOutputTokens`, so a
    #: thinking-heavy reply can exhaust the budget before writing any JSON.
    #: Empty sends no field, taking the model default.
    thinking_level: str = "low"
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
    dimensions: int = 256
    task_type: str | None = None
    max_rate_limit_retries: int = 3
    max_rate_limit_wait: float = 600.0
    #: Cosine similarity at or above which two articles are the same event.
    #: MODEL-SPECIFIC: 0.82/0.72 for gemini-embedding-2, 0.80/0.65 for the local
    #: MiniLM. Cosine values do not transfer between embedding models.
    similarity_threshold: float = 0.82
    #: Between this and the threshold, ask the LLM (if enabled) to decide.
    ambiguous_threshold: float = 0.72


@dataclass
class Config:
    sources: list[NewsletterSource]
    settings: Settings = field(default_factory=Settings)
    preferences: Preferences = field(default_factory=Preferences)
    filters: FilterSettings = field(default_factory=FilterSettings)
    regions: RegionSettings = field(default_factory=RegionSettings)
    digest: DigestSettings = field(default_factory=DigestSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    mailbox: MailboxSettings = field(default_factory=MailboxSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    embeddings: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    user_agent: str = "news-digest/0.3 (+https://github.com/)"

    @property
    def enabled_sources(self) -> list[NewsletterSource]:
        return [s for s in self.sources if s.enabled]

    def source(self, name: str) -> NewsletterSource | None:
        for candidate in self.sources:
            if candidate.name == name:
                return candidate
        return None

    def publisher_of(self, source_name: str) -> str:
        found = self.source(source_name)
        return found.publisher if found else source_name


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


def load_sources(path: Path | str = DEFAULT_SOURCES) -> list[NewsletterSource]:
    data = _load_yaml(Path(path))
    defaults = data.get("defaults") or {}
    global_excludes = _as_list(data.get("exclude_url_patterns"))
    global_title_excludes = _as_list(data.get("exclude_title_patterns"))
    raw_sources = data.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ConfigError(f"{path}: 'sources' must be a non-empty list")

    sources: list[NewsletterSource] = []
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

        senders = [s.strip().lower() for s in _as_list(entry.get("senders")) if s.strip()]
        if not senders and bool(entry.get("enabled", True)):
            # An enabled source with no sender rule can never match a message, so
            # it would sit in the config looking configured while contributing
            # nothing. A disabled one is allowed to be a placeholder.
            raise ConfigError(
                f"{path}: {name}: an enabled source needs at least one 'senders' "
                f"entry (an address, or a domain like '@aljazeera.net')"
            )

        sources.append(
            NewsletterSource(
                name=name,
                senders=senders,
                subject_patterns=_regex_patterns(
                    _as_list(entry.get("subject_patterns")),
                    path, name, "subject_patterns",
                ),
                sender_name_patterns=_regex_patterns(
                    _as_list(entry.get("sender_name_patterns")),
                    path, name, "sender_name_patterns",
                ),
                newsletter=str(entry.get("newsletter") or "").strip(),
                publisher=str(entry.get("publisher") or "").strip(),
                enabled=bool(entry.get("enabled", True)),
                weight=float(entry.get("weight", 1.0)),
                topics=[t.lower() for t in _as_list(entry.get("topics"))],
                languages=[l.lower() for l in _as_list(entry.get("languages"))],
                excerpt_chars=int(
                    entry.get("excerpt_chars", defaults.get("excerpt_chars", 1200))
                ),
                exclude_url_patterns=_regex_patterns(
                    global_excludes + _as_list(entry.get("exclude_url_patterns")),
                    path, name, "exclude_url_patterns",
                ),
                exclude_title_patterns=_regex_patterns(
                    global_title_excludes
                    + _as_list(entry.get("exclude_title_patterns")),
                    path, name, "exclude_title_patterns",
                ),
            )
        )
    return sources


def _regex_patterns(
    raw: list[str], path: Path | str, name: str, field_name: str
) -> list[str]:
    """Validate a list of regexes at load time.

    A broken pattern would silently stop excluding -- or throw once per item
    mid-run -- so it is a startup error, as a misspelled ranking term is.
    `field_name` is there so the error names the list the reader has to fix.
    """
    out: list[str] = []
    for pattern in raw:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ConfigError(
                f"{path}: {name}: {field_name} entry {pattern!r} "
                f"is not a valid regex: {exc}"
            ) from exc
        if pattern not in out:
            out.append(pattern)
    return out


def load_filters(path: Path | str = DEFAULT_FILTERS) -> FilterSettings:
    p = Path(path)
    if not p.exists():
        log.info("no filters file at %s; using defaults", p)
        return FilterSettings()

    data = _load_yaml(p)
    raw = data.get("filters") or {}
    defaults = FilterSettings()
    settings = FilterSettings(
        excluded_topics=(
            [t.lower() for t in _as_list(raw.get("excluded_topics"))]
            if raw.get("excluded_topics") is not None
            else defaults.excluded_topics
        ),
        include_topics=(
            [t.lower() for t in _as_list(raw.get("include_topics"))]
            if raw.get("include_topics") is not None
            else defaults.include_topics
        ),
        drop_content_types=[
            t.lower() for t in _as_list(raw.get("drop_content_types"))
        ],
        classify=bool(raw.get("classify", True)),
        max_items=max(0, int(raw.get("max_items", defaults.max_items))),
    )
    overlap = set(settings.excluded_topics) & set(settings.include_topics)
    if overlap:
        raise ConfigError(
            f"{p}: {sorted(overlap)} appear in both include_topics and "
            f"excluded_topics; a topic cannot be both"
        )
    return settings


def load_preferences(path: Path | str = DEFAULT_PREFERENCES) -> tuple[
    Settings, Preferences, DigestSettings, StorageSettings, RegionSettings
]:
    p = Path(path)
    if not p.exists():
        log.warning("no preferences file at %s; using defaults", p)
        return (Settings(), Preferences(), DigestSettings(), StorageSettings(),
                RegionSettings())

    data = _load_yaml(p)
    raw_settings = data.get("settings") or {}
    raw_prefs = data.get("preferences") or {}
    raw_ranking = data.get("ranking") or {}
    raw_digest = data.get("digest") or {}
    raw_storage = data.get("storage") or {}
    raw_regions = data.get("regions") or {}

    supported = [l.lower() for l in _as_list(raw_settings.get("supported_languages"))]
    output_language = str(raw_settings.get("output_language", "en")).lower()
    if output_language not in OUTPUT_LANGUAGES:
        raise ConfigError(
            f"{p}: settings.output_language must be one of "
            f"{list(OUTPUT_LANGUAGES)}, not {output_language!r}"
        )
    settings = Settings(
        supported_languages=supported or ["en", "es"],
        output_language=output_language,
    )
    if (
        settings.output_language != "auto"
        and settings.output_language not in settings.supported_languages
    ):
        # Not fatal: the digest can be written in a language you do not collect.
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
        corroboration_saturation=max(
            2.0,
            float(raw_ranking.get(
                "corroboration_saturation", DEFAULT_CORROBORATION_SATURATION
            )),
        ),
    )

    week_ends_on = int(raw_digest.get("week_ends_on", 6))
    if not 0 <= week_ends_on <= 6:
        raise ConfigError(f"{p}: digest.week_ends_on must be 0-6 (Monday-Sunday)")

    digest_defaults = DigestSettings()
    digest = DigestSettings(
        title=str(raw_digest.get("title", digest_defaults.title)),
        subtitle=str(raw_digest.get("subtitle", "")),
        max_stories=int(raw_digest.get("max_stories", digest_defaults.max_stories)),
        minor_stories=int(raw_digest.get("minor_stories", digest_defaults.minor_stories)),
        min_articles=int(raw_digest.get("min_articles", digest_defaults.min_articles)),
        week_ends_on=week_ends_on,
        update_readme=bool(raw_digest.get("update_readme", True)),
    )
    storage_defaults = StorageSettings()
    storage = StorageSettings(
        retention_days=int(
            raw_storage.get("retention_days", storage_defaults.retention_days)
        ),
        email_body_retention_days=int(
            raw_storage.get(
                "email_body_retention_days", storage_defaults.email_body_retention_days
            )
        ),
        embedding_retention_days=int(
            raw_storage.get(
                "embedding_retention_days", storage_defaults.embedding_retention_days
            )
        ),
    )
    region_defaults = RegionSettings()
    regions = RegionSettings(
        regions=(
            [r.lower() for r in _as_list(raw_regions.get("regions"))]
            if raw_regions.get("regions") is not None
            else region_defaults.regions
        ),
        max_share=float(raw_regions.get("max_share", region_defaults.max_share)),
        enabled=bool(raw_regions.get("enabled", True)),
    )
    if not 0.0 < regions.max_share <= 1.0:
        raise ConfigError(f"{p}: regions.max_share must be between 0 and 1")

    return settings, preferences, digest, storage, regions


#: Weight given to a topic listed without one, as the spec's example does.
LISTED_TOPIC_WEIGHT = 1.5


def _topic_weights(raw: Any, path: Path) -> dict[str, float]:
    """Accept either shape: a weights mapping, or a plain list of topics."""
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


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean env var. Unset means the default, not False."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    log.warning("%s=%r is not a boolean; using %s", name, raw, default)
    return default


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


def _thinking_level(raw: str | None) -> str:
    """Validate LLM_THINKING_LEVEL, defaulting to the cheapest level.

    A bad value would 400 every request and, after two consecutive batch
    failures, silently cost the run all of its enrichment -- so it is a startup
    error. "off" and "none" are accepted as the obvious ways to ask for no
    thinking field, since the API itself has no "off" level.
    """
    if raw is None:
        return "low"
    level = raw.strip().lower()
    if level in ("off", "none", "unset"):
        return ""
    if level not in THINKING_LEVELS:
        raise ConfigError(
            f"LLM_THINKING_LEVEL={raw!r} is not a thinking level; "
            f"use one of {[lvl for lvl in THINKING_LEVELS if lvl]}, "
            f"or 'off' to send no thinking field"
        )
    return level


def _gemini_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY")
    return key.strip() if key and key.strip() else None


#: Request timeout default, per provider kind. A local model generates every
#: token on your machine, so the same batch a hosted model answers in seconds can
#: take minutes.
DEFAULT_TIMEOUT = 90.0
DEFAULT_LOCAL_TIMEOUT = 300.0


def load_mailbox_settings() -> MailboxSettings:
    """Mailbox credentials, from the environment only.

    Section 28: these never reach a config file, a log line or the debug export.
    The variable names are the ones the specification names, so a reader can move
    between the two without translating.
    """
    return MailboxSettings(
        provider=(os.environ.get("MAILBOX_PROVIDER") or "imap").strip().lower(),
        host=(os.environ.get("NEWS_EMAIL_HOST") or "").strip(),
        port=_env_int("NEWS_EMAIL_PORT", 993),
        username=(os.environ.get("NEWS_EMAIL_USERNAME") or "").strip(),
        password=os.environ.get("NEWS_EMAIL_PASSWORD") or "",
        folder=(os.environ.get("NEWS_EMAIL_FOLDER") or "INBOX").strip(),
        timeout=max(1.0, _env_float("NEWS_EMAIL_TIMEOUT", 30.0)),
    )


def load_llm_settings(output_language: str = "en") -> LLMSettings:
    key = _gemini_key()
    provider = (os.environ.get("LLM_PROVIDER") or "gemini").strip().lower()
    settings = LLMSettings(
        provider=provider,
        model=os.environ.get("LLM_MODEL") or "gemini-3.6-flash",
        api_key=key,
        batch_size=max(1, _env_int("LLM_BATCH_SIZE", 8)),
        resolve_clusters=_env_bool("LLM_RESOLVE_CLUSTERS", True),
        max_cluster_checks=max(0, _env_int("LLM_MAX_CLUSTER_CHECKS", 40)),
        articles_per_run=max(0, _env_int("LLM_ARTICLES_PER_RUN", 400)),
        max_rate_limit_retries=max(0, _env_int("LLM_RATE_LIMIT_RETRIES", 3)),
        max_rate_limit_wait=max(0.0, _env_float("LLM_RATE_LIMIT_WAIT", 600.0)),
        thinking_level=_thinking_level(os.environ.get("LLM_THINKING_LEVEL")),
        base_url=os.environ.get("LLM_BASE_URL") or "",
        num_ctx=max(2048, _env_int("LLM_NUM_CTX", 8192)),
        timeout=max(1.0, _env_float(
            "LLM_TIMEOUT",
            DEFAULT_LOCAL_TIMEOUT if provider == "ollama" else DEFAULT_TIMEOUT,
        )),
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
        max_rate_limit_retries=max(0, _env_int("EMBEDDING_RATE_LIMIT_RETRIES", 3)),
        max_rate_limit_wait=max(0.0, _env_float("EMBEDDING_RATE_LIMIT_WAIT", 600.0)),
        similarity_threshold=_env_float("EMBEDDING_SIMILARITY_THRESHOLD", 0.82),
        ambiguous_threshold=_env_float("EMBEDDING_AMBIGUOUS_THRESHOLD", 0.72),
    )
    if settings.provider == "gemini" and not settings.api_key:
        settings.provider = "none"
    return settings


def load_config(
    sources_path: Path | str = DEFAULT_SOURCES,
    preferences_path: Path | str = DEFAULT_PREFERENCES,
    filters_path: Path | str = DEFAULT_FILTERS,
) -> Config:
    load_dotenv()
    settings, preferences, digest, storage, regions = load_preferences(preferences_path)
    return Config(
        sources=load_sources(sources_path),
        settings=settings,
        preferences=preferences,
        filters=load_filters(filters_path),
        regions=regions,
        digest=digest,
        storage=storage,
        mailbox=load_mailbox_settings(),
        llm=load_llm_settings(settings.output_language),
        embeddings=load_embedding_settings(),
        user_agent=os.environ.get(
            "NEWS_DIGEST_USER_AGENT",
            "news-digest/0.3 (+https://github.com/; personal newsletter digest)",
        ),
    )
