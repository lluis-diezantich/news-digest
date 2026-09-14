"""Google Gemini provider (free tier).

Uses the REST endpoint directly with `requests` -- one fewer dependency than the
SDK, and `responseSchema` makes the JSON contract explicit and server-enforced.

Free-tier quotas are per-minute and per-day, so calls are batched and every
result is cached by content hash. A per-minute 429 is waited out and retried --
see `ratelimit` for how the two are told apart. A per-day 429, or a run that has
spent its wait budget, raises LLMQuotaError, which stops LLM work for the run
without failing it.

Thinking is set to the cheapest level the model allows rather than left on its
default. Thought tokens are billed as output, counted against the per-minute
token allowance, and drawn from `maxOutputTokens` -- so on schema-enforced
extraction they cost money, invite 429s, and can starve the reply of room for
the JSON itself. `thinking_level` is a documented field only on recent models,
so a model that rejects it is retried once without it rather than failing.
"""

from __future__ import annotations

import json
import logging
import random
import time

import requests

from .. import ratelimit
from .base import (
    SAME_EVENT_SYSTEM_PROMPT,
    Brief,
    BriefInput,
    Context,
    Enrichment,
    EnrichInput,
    LLMError,
    LLMProvider,
    LLMQuotaError,
    PairInput,
    brief_system_prompt,
    enrich_system_prompt,
)

log = logging.getLogger(__name__)

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

_STRINGS = {"type": "array", "items": {"type": "string"}}

ENRICH_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "summary": {"type": "string"},
            "why_it_matters": {"type": "string"},
            "key_facts": _STRINGS,
            "topics": _STRINGS,
            "entities": _STRINGS,
            "importance": {"type": "number"},
            "relevance": {"type": "number"},
            "content_type": {
                "type": "string",
                "enum": ["reporting", "analysis", "opinion", "other"],
            },
        },
        "required": [
            "id", "summary", "why_it_matters", "key_facts", "topics",
            "entities", "importance", "relevance", "content_type",
        ],
    },
}

BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "summary": {"type": "string"},
        "why_it_matters": {"type": "string"},
        "key_facts": _STRINGS,
        "topics": _STRINGS,
        "importance": {"type": "number"},
        "relevance": {"type": "number"},
    },
    "required": ["headline", "summary", "why_it_matters", "key_facts", "topics"],
}

SAME_EVENT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "same_event": {"type": "boolean"},
        },
        "required": ["key", "same_event"],
    },
}


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        *,
        timeout: float = 90.0,
        max_retries: int = 3,
        max_rate_limit_retries: int = 3,
        max_rate_limit_wait: float = 600.0,
        thinking_level: str = "low",
    ):
        super().__init__()
        if not api_key:
            raise LLMError("gemini provider requires GEMINI_API_KEY")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        #: Waits allowed per request when rate limited, and for the run overall.
        #: The run-level cap keeps a low per-minute allowance from spending the
        #: workflow's whole timeout in `sleep`.
        self.max_rate_limit_retries = max_rate_limit_retries
        self.max_rate_limit_wait = max_rate_limit_wait
        self._rate_limit_waited = 0.0
        #: Emptied for the rest of the run if the model turns out to reject it,
        #: so one 400 costs one retry rather than one per request.
        self.thinking_level = thinking_level
        self._session = requests.Session()
        self._session.headers.update(
            {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        )

    # -- transport ----------------------------------------------------------

    def _generate(self, system: str, prompt: str, schema: dict) -> str:
        config: dict = {
            "responseMimeType": "application/json",
            "responseSchema": schema,
            "temperature": 0.2,
            "maxOutputTokens": 8192,
        }
        if self.thinking_level:
            config["thinkingLevel"] = self.thinking_level
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": config,
        }
        url = f"{API_ROOT}/{self.model}:generateContent"

        last_error: Exception | None = None
        attempt = 0  # transport errors and 5xx
        waits = 0    # 429s, budgeted separately: for those, waiting is the fix
        while attempt < self.max_retries:
            try:
                self.calls += 1
                response = self._session.post(url, json=body, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
                self._backoff(attempt)
                attempt += 1
                continue

            if response.status_code == 429:
                decision = ratelimit.plan(
                    response.text,
                    attempt=waits,
                    budget=self.max_rate_limit_retries,
                    spent=self._rate_limit_waited,
                    cap=self.max_rate_limit_wait,
                )
                if not decision.retry:
                    raise LLMQuotaError(
                        f"gemini rate limited, giving up ({decision.reason}): "
                        f"{_short(response.text)}"
                    )
                log.warning(
                    "gemini rate limited (%s); waiting %.0fs then retrying",
                    decision.reason, decision.delay,
                )
                ratelimit.sleep(decision.delay)
                self._rate_limit_waited += decision.delay
                waits += 1
                continue

            if response.status_code in (500, 502, 503, 504):
                last_error = LLMError(f"gemini {response.status_code}")
                self._backoff(attempt)
                attempt += 1
                continue
            if response.status_code == 400 and self._drop_thinking(response.text, config):
                continue  # same attempt, one field lighter
            if response.status_code != 200:
                raise LLMError(f"gemini {response.status_code}: {_short(response.text)}")
            return self._extract_text(response.json())

        raise LLMError(f"gemini unreachable after {self.max_retries} attempts: {last_error}")

    def _drop_thinking(self, body: str, config: dict) -> bool:
        """Retry without `thinkingLevel` when the model does not accept it.

        Only recent models document the field. Rather than make the digest's
        quality depend on the reader having pinned a compatible `LLM_MODEL`, an
        unrecognised-field 400 drops it for the rest of the run.
        """
        if "thinkingLevel" not in config or "thinking" not in body.lower():
            return False
        log.warning(
            "%s rejected thinkingLevel=%r; retrying without it for the rest of "
            "the run (thought tokens will be billed at the model's default)",
            self.model, config["thinkingLevel"],
        )
        config.pop("thinkingLevel")
        self.thinking_level = ""
        return True

    @staticmethod
    def _backoff(attempt: int) -> None:
        time.sleep(min(20.0, (2**attempt) + random.uniform(0, 0.5)))

    @staticmethod
    def _extract_text(payload: dict) -> str:
        feedback = payload.get("promptFeedback") or {}
        if feedback.get("blockReason"):
            raise LLMError(f"gemini blocked the prompt: {feedback['blockReason']}")

        candidates = payload.get("candidates") or []
        if not candidates:
            raise LLMError("gemini returned no candidates")

        candidate = candidates[0]
        finish = candidate.get("finishReason")
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts).strip()
        if finish == "MAX_TOKENS" and not text:
            raise LLMError("gemini hit maxOutputTokens with no output; reduce batch size")
        if not text:
            raise LLMError(f"gemini returned empty output (finishReason={finish})")
        return text

    # -- LLMProvider --------------------------------------------------------

    def enrich(self, items: list[EnrichInput], context: Context) -> list[Enrichment]:
        if not items:
            return []
        prompt = json.dumps(
            [
                {
                    "id": item.id,
                    "title": item.title,
                    "source": item.source,
                    "language": item.language,
                    "published": item.published,
                    "excerpt": item.excerpt,
                }
                for item in items
            ],
            ensure_ascii=False,
            indent=1,
        )
        data = _parse_json(self._generate(enrich_system_prompt(context), prompt, ENRICH_SCHEMA))
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            raise LLMError(f"expected a JSON array, got {type(data).__name__}")

        valid_ids = {item.id for item in items}
        results: list[Enrichment] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            article_id = str(entry.get("id", ""))
            if article_id not in valid_ids:
                log.debug("dropping enrichment for unknown id %r", article_id)
                continue
            summary = str(entry.get("summary", "")).strip()
            if not summary:
                continue
            results.append(
                Enrichment(
                    id=article_id,
                    summary=summary,
                    why_it_matters=str(entry.get("why_it_matters", "")).strip(),
                    key_facts=list(entry.get("key_facts") or []),
                    topics=list(entry.get("topics") or []),
                    entities=list(entry.get("entities") or []),
                    importance=entry.get("importance", 0.5),
                    relevance=entry.get("relevance", 0.5),
                    content_type=entry.get("content_type"),
                ).clamp()
            )
        return results

    def write_brief(self, item: BriefInput, context: Context) -> Brief | None:
        payload = {
            "coverage": [
                {"source": source, "language": language, "headline": headline,
                 "excerpt": excerpt}
                for source, language, headline, excerpt in zip(
                    item.sources, item.languages, item.headlines, item.excerpts
                )
            ]
        }
        data = _parse_json(
            self._generate(
                brief_system_prompt(context),
                json.dumps(payload, ensure_ascii=False, indent=1),
                BRIEF_SCHEMA,
            )
        )
        if not isinstance(data, dict):
            return None
        headline = str(data.get("headline", "")).strip()
        summary = str(data.get("summary", "")).strip()
        if not headline or not summary:
            return None
        return Brief(
            headline=headline,
            summary=summary,
            why_it_matters=str(data.get("why_it_matters", "")).strip(),
            key_facts=[str(f).strip() for f in (data.get("key_facts") or [])][:4],
            topics=[str(t).strip().lower() for t in (data.get("topics") or [])][:4],
            importance=_optional_float(data.get("importance")),
            relevance=_optional_float(data.get("relevance")),
        )

    def same_event(self, pairs: list[PairInput], context: Context) -> dict[str, bool]:
        if not pairs:
            return {}
        payload = [
            {
                "key": pair.key,
                "a": {"title": pair.left_title, "language": pair.left_language,
                      "excerpt": pair.left_excerpt},
                "b": {"title": pair.right_title, "language": pair.right_language,
                      "excerpt": pair.right_excerpt},
            }
            for pair in pairs
        ]
        data = _parse_json(
            self._generate(
                SAME_EVENT_SYSTEM_PROMPT,
                json.dumps(payload, ensure_ascii=False, indent=1),
                SAME_EVENT_SCHEMA,
            )
        )
        if not isinstance(data, list):
            return {}
        wanted = {pair.key for pair in pairs}
        verdicts: dict[str, bool] = {}
        for entry in data:
            if isinstance(entry, dict) and str(entry.get("key")) in wanted:
                verdicts[str(entry["key"])] = bool(entry.get("same_event"))
        return verdicts


def _optional_float(value) -> float | None:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return None


def _parse_json(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(f"gemini returned invalid JSON: {exc}; got {_short(raw)}") from exc


def _short(value: str, limit: int = 300) -> str:
    value = " ".join((value or "").split())
    return value[:limit] + ("…" if len(value) > limit else "")
