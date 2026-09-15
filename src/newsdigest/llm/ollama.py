"""Local LLM via Ollama. No API, no key, no quota.

The other half of moving off a hosted free tier. Embeddings went local because
cross-lingual similarity is a small-model problem; the LLM is a harder ask, but
on a machine with real memory it is a reasonable one -- and unlike the embedding
case the argument is about hardware, not quality ceilings. A GitHub runner (2
vCPU, no GPU) cannot do this in a 45-minute job; an Apple-silicon laptop can.

Ollama is used rather than llama.cpp directly because it exposes JSON-Schema
structured output through the `format` field, and this pipeline leans on schema
enforcement the way the Gemini provider leans on `responseSchema`. Without it a
local model reliably wanders out of JSON on the array-of-objects enrich call.

Two settings are load-bearing:

* `think: false` -- Qwen3 and other hybrid models emit reasoning before the
  answer by default, which lands inside the response and breaks parsing. Ollama
  ignores the field on models that have no thinking mode, so it is safe to send
  unconditionally.
* a long timeout -- a local 8B model takes tens of seconds for a batch of 16
  articles, where a hosted API takes two. The default here is minutes, not
  seconds, because there is no rate limit to protect and nothing is billed.

There is no quota, so LLMQuotaError is never raised. A connection refused means
the server is not running, and that is reported as such rather than as a
mysterious transport failure.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

from .base import (
    SAME_EVENT_SYSTEM_PROMPT,
    Brief,
    BriefInput,
    Context,
    Enrichment,
    EnrichInput,
    LLMError,
    LLMProvider,
    PairInput,
    brief_system_prompt,
    enrich_system_prompt,
)

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
#: Multilingual, strong at instruction-following, and small enough to be quick
#: on a laptop. Override with LLM_MODEL; a 12-14B model is a clear upgrade if
#: there is memory for it.
DEFAULT_MODEL = "qwen3:8b"

_STRINGS = {"type": "array", "items": {"type": "string"}}

#: Mirrors the Gemini provider's schemas. Kept per-provider rather than shared,
#: because each one owns its wire format -- Gemini's is an OpenAPI subset while
#: this is plain JSON Schema, and they are only incidentally alike.
ENRICH_SCHEMA: dict[str, Any] = {
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

BRIEF_SCHEMA: dict[str, Any] = {
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

SAME_EVENT_SCHEMA: dict[str, Any] = {
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


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 600.0,
        max_retries: int = 2,
        num_ctx: int = 8192,
    ):
        super().__init__()
        self.model = model or DEFAULT_MODEL
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        #: Batches of 16 articles with 900-char excerpts overflow a 4k window,
        #: and an overflowed prompt is silently truncated rather than refused --
        #: which shows up as missing ids, not as an error.
        self.num_ctx = num_ctx
        self._session = requests.Session()

    # -- transport ----------------------------------------------------------

    def _generate(self, system: str, prompt: str, schema: dict) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": schema,
            "think": False,
            "options": {"temperature": 0.2, "num_ctx": self.num_ctx},
        }
        url = f"{self.base_url}/api/chat"

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                self.calls += 1
                response = self._session.post(url, json=body, timeout=self.timeout)
            except requests.ConnectionError as exc:
                # The overwhelmingly likely cause, and worth saying plainly.
                raise LLMError(
                    f"no ollama at {self.base_url} -- start it with `ollama serve` "
                    f"and pull the model with `ollama pull {self.model}`"
                ) from exc
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(1.0 + attempt)
                continue

            if response.status_code == 404:
                raise LLMError(
                    f"ollama has no model {self.model!r}; "
                    f"run `ollama pull {self.model}`"
                )
            if response.status_code != 200:
                raise LLMError(f"ollama {response.status_code}: {_short(response.text)}")
            return self._extract_text(response.json())

        raise LLMError(f"ollama unreachable after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def _extract_text(payload: dict) -> str:
        message = payload.get("message") or {}
        text = (message.get("content") or "").strip()
        if not text:
            # A model with thinking left on can answer with reasoning only.
            reason = payload.get("done_reason") or "unknown"
            raise LLMError(f"ollama returned empty content (done_reason={reason})")
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
        data = _parse_json(
            self._generate(enrich_system_prompt(context), prompt, ENRICH_SCHEMA)
        )
        if not isinstance(data, list):
            raise LLMError("ollama did not return a list of enrichments")

        wanted = {item.id for item in items}
        out: list[Enrichment] = []
        for entry in data:
            if not isinstance(entry, dict) or str(entry.get("id")) not in wanted:
                continue
            out.append(
                Enrichment(
                    id=str(entry["id"]),
                    summary=str(entry.get("summary") or ""),
                    why_it_matters=str(entry.get("why_it_matters") or ""),
                    key_facts=_strings(entry.get("key_facts")),
                    topics=_strings(entry.get("topics")),
                    entities=_strings(entry.get("entities")),
                    importance=_number(entry.get("importance"), 0.5),
                    relevance=_number(entry.get("relevance"), 0.5),
                    content_type=entry.get("content_type"),
                ).clamp()
            )
        if len(out) < len(items):
            # Not fatal: the caller retries the missing ids on the next run.
            log.debug("ollama returned %d of %d enrichments", len(out), len(items))
        return out

    def write_brief(self, item: BriefInput, context: Context) -> Brief | None:
        payload = {
            "headlines": item.headlines,
            "sources": item.sources,
            "languages": item.languages,
            "excerpts": item.excerpts,
        }
        data = _parse_json(
            self._generate(
                brief_system_prompt(context),
                json.dumps(payload, ensure_ascii=False, indent=1),
                BRIEF_SCHEMA,
            )
        )
        if not isinstance(data, dict) or not data.get("headline"):
            return None
        return Brief(
            headline=str(data["headline"]),
            summary=str(data.get("summary") or ""),
            why_it_matters=str(data.get("why_it_matters") or ""),
            key_facts=_strings(data.get("key_facts")),
            topics=_strings(data.get("topics")),
            importance=_optional_number(data.get("importance")),
            relevance=_optional_number(data.get("relevance")),
        ).clamp()

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
        return {
            str(e["key"]): bool(e.get("same_event"))
            for e in data
            if isinstance(e, dict) and str(e.get("key")) in wanted
        }


def _parse_json(raw: str):
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise LLMError(f"ollama returned unparseable JSON: {_short(raw)}") from exc


def _strings(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if str(v).strip()]


def _number(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_number(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _short(value: str, limit: int = 300) -> str:
    value = " ".join((value or "").split())
    return value if len(value) <= limit else value[: limit - 1] + "…"
