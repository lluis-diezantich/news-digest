"""Google Gemini provider (free tier).

Uses the REST endpoint directly with `requests` -- one fewer dependency than the
SDK, and `responseSchema` makes the JSON contract explicit and machine-checked.

Free-tier quotas are per-minute and per-day, so the pipeline batches many
articles into one call and caches every result by content hash; see
`enrich.py`. A 429 raises LLMQuotaError, which stops LLM work for the run
without failing it -- unenriched articles are simply picked up next time.
"""

from __future__ import annotations

import json
import logging
import random
import time

import requests

from .base import (
    BRIEF_SYSTEM_PROMPT,
    ENRICH_SYSTEM_PROMPT,
    Brief,
    BriefInput,
    Enrichment,
    EnrichInput,
    LLMError,
    LLMProvider,
    LLMQuotaError,
)

log = logging.getLogger(__name__)

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

# OpenAPI-subset schema Gemini validates its output against.
ENRICH_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "summary": {"type": "string"},
            "why_it_matters": {"type": "string"},
            "topics": {"type": "array", "items": {"type": "string"}},
            "entities": {"type": "array", "items": {"type": "string"}},
            "importance": {"type": "number"},
            "event_label": {"type": "string"},
        },
        "required": [
            "id",
            "summary",
            "why_it_matters",
            "topics",
            "entities",
            "importance",
            "event_label",
        ],
    },
}

BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "summary": {"type": "string"},
        "why_it_matters": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["headline", "summary", "why_it_matters", "topics"],
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
    ):
        super().__init__()
        if not api_key:
            raise LLMError("gemini provider requires GEMINI_API_KEY")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update(
            {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        )

    # -- transport ----------------------------------------------------------

    def _generate(self, system: str, prompt: str, schema: dict) -> str:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": schema,
                "temperature": 0.2,
                "maxOutputTokens": 8192,
            },
        }
        url = f"{API_ROOT}/{self.model}:generateContent"

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                self.calls += 1
                response = self._session.post(url, json=body, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
                self._backoff(attempt)
                continue

            if response.status_code == 429:
                raise LLMQuotaError(
                    f"gemini rate limit / quota exhausted: {_short(response.text)}"
                )
            if response.status_code in (500, 502, 503, 504):
                last_error = LLMError(f"gemini {response.status_code}")
                self._backoff(attempt)
                continue
            if response.status_code != 200:
                raise LLMError(
                    f"gemini {response.status_code}: {_short(response.text)}"
                )
            return self._extract_text(response.json())

        raise LLMError(f"gemini unreachable after {self.max_retries} attempts: {last_error}")

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

    def enrich(self, items: list[EnrichInput]) -> list[Enrichment]:
        if not items:
            return []
        prompt = json.dumps(
            [
                {
                    "id": item.id,
                    "title": item.title,
                    "source": item.source,
                    "published": item.published,
                    "excerpt": item.excerpt,
                }
                for item in items
            ],
            ensure_ascii=False,
            indent=1,
        )
        raw = self._generate(ENRICH_SYSTEM_PROMPT, prompt, ENRICH_SCHEMA)
        data = _parse_json(raw)
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
                    topics=list(entry.get("topics") or []),
                    entities=list(entry.get("entities") or []),
                    importance=entry.get("importance", 0.5),
                    event_label=str(entry.get("event_label", "")),
                ).clamp()
            )
        return results

    def write_brief(self, item: BriefInput) -> Brief | None:
        payload = {
            "coverage": [
                {"source": source, "headline": headline, "excerpt": excerpt}
                for source, headline, excerpt in zip(
                    item.sources, item.headlines, item.excerpts
                )
            ]
        }
        raw = self._generate(
            BRIEF_SYSTEM_PROMPT,
            json.dumps(payload, ensure_ascii=False, indent=1),
            BRIEF_SCHEMA,
        )
        data = _parse_json(raw)
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
            topics=[str(t).strip().lower() for t in (data.get("topics") or [])][:4],
        )


def _parse_json(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        # responseSchema makes this rare, but a truncated response is possible.
        raise LLMError(f"gemini returned invalid JSON: {exc}; got {_short(raw)}") from exc


def _short(value: str, limit: int = 300) -> str:
    value = " ".join((value or "").split())
    return value[:limit] + ("…" if len(value) > limit else "")
