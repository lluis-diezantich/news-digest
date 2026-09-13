"""Provider-layer tests. Gemini is exercised against a stubbed transport."""

import json

import pytest

from newsdigest.config import LLMSettings
from newsdigest.llm import get_provider
from newsdigest.llm.base import BriefInput, EnrichInput, LLMError, LLMQuotaError
from newsdigest.llm.gemini import GeminiProvider
from newsdigest.llm.heuristic import HeuristicProvider

ITEMS = [
    EnrichInput(id="a1", title="Central bank raises rates", source="Wire",
                published="2026-09-10T08:30:00+00:00",
                excerpt="The bank moved by half a point, citing inflation."),
    EnrichInput(id="a2", title="Rare orchid found in Peru", source="Wire",
                published=None, excerpt="Scientists described a new species."),
]


class StubResponse:
    def __init__(self, status: int, payload):
        self.status_code = status
        self._payload = payload
        self.text = payload if isinstance(payload, str) else json.dumps(payload)

    def json(self):
        return self._payload


class StubSession:
    """Stands in for requests.Session inside GeminiProvider."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.posts: list[dict] = []
        self.headers: dict = {}

    def post(self, url, json=None, timeout=None):
        self.posts.append(json)
        return self.responses.pop(0) if self.responses else self.responses[-1]


def gemini_with(*responses) -> GeminiProvider:
    provider = GeminiProvider(api_key="test-key", model="gemini-test", max_retries=1)
    provider._session = StubSession(*responses)
    return provider


def candidate(text: str, finish: str = "STOP"):
    return {"candidates": [{"finishReason": finish,
                            "content": {"parts": [{"text": text}]}}]}


class TestGemini:
    def test_parses_a_batch_response(self):
        payload = json.dumps([
            {"id": "a1", "summary": "Rates went up half a point.",
             "why_it_matters": "Borrowing gets pricier.", "topics": ["Economics"],
             "entities": ["Central Bank"], "importance": 0.72,
             "event_label": "Central Bank Rate Rise"},
            {"id": "a2", "summary": "A new orchid was described.",
             "why_it_matters": "", "topics": ["science"], "entities": ["Peru"],
             "importance": 0.2, "event_label": "peru orchid discovery"},
        ])
        provider = gemini_with(StubResponse(200, candidate(payload)))
        results = provider.enrich(ITEMS)

        assert [r.id for r in results] == ["a1", "a2"]
        # clamp() normalizes topics/labels to lowercase and bounds importance.
        assert results[0].topics == ["economics"]
        assert results[0].event_label == "central bank rate rise"
        assert 0.0 <= results[0].importance <= 1.0
        # The whole batch is one request.
        assert provider.calls == 1

    def test_request_asks_for_schema_constrained_json(self):
        provider = gemini_with(StubResponse(200, candidate("[]")))
        provider.enrich(ITEMS)
        body = provider._session.posts[0]
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        assert body["generationConfig"]["responseSchema"]["type"] == "array"
        # Only metadata and the excerpt are sent.
        sent = json.loads(body["contents"][0]["parts"][0]["text"])
        assert set(sent[0]) == {"id", "title", "source", "published", "excerpt"}

    def test_unknown_ids_are_discarded(self):
        payload = json.dumps([{"id": "hallucinated", "summary": "x", "why_it_matters": "",
                               "topics": [], "entities": [], "importance": 0.5,
                               "event_label": ""}])
        provider = gemini_with(StubResponse(200, candidate(payload)))
        assert provider.enrich(ITEMS) == []

    def test_rate_limit_raises_quota_error(self):
        provider = gemini_with(StubResponse(429, {"error": {"message": "quota"}}))
        with pytest.raises(LLMQuotaError):
            provider.enrich(ITEMS)

    def test_blocked_prompt_raises(self):
        provider = gemini_with(StubResponse(200, {"promptFeedback": {"blockReason": "SAFETY"}}))
        with pytest.raises(LLMError, match="blocked"):
            provider.enrich(ITEMS)

    def test_invalid_json_raises(self):
        provider = gemini_with(StubResponse(200, candidate("not json")))
        with pytest.raises(LLMError, match="invalid JSON"):
            provider.enrich(ITEMS)

    def test_brief_roundtrip(self):
        payload = json.dumps({"headline": "Rates rise half a point",
                              "summary": "Two outlets report the same move.",
                              "why_it_matters": "Mortgages get dearer.",
                              "topics": ["Economics"]})
        provider = gemini_with(StubResponse(200, candidate(payload)))
        brief = provider.write_brief(
            BriefInput(story_id="s1", headlines=["a", "b"], sources=["A", "B"],
                       excerpts=["x", "y"])
        )
        assert brief.headline == "Rates rise half a point"
        assert brief.topics == ["economics"]

    def test_missing_key_is_rejected(self):
        with pytest.raises(LLMError):
            GeminiProvider(api_key="")


class TestHeuristic:
    def test_produces_a_summary_and_topics_offline(self):
        results = HeuristicProvider().enrich(ITEMS)
        assert len(results) == 2
        assert results[0].summary
        assert "economics" in results[0].topics
        assert results[0].event_label

    def test_event_label_is_stable_across_wordings(self):
        provider = HeuristicProvider()
        a = provider.enrich([EnrichInput("1", "US and China agree trade deal", "A", None, "")])
        b = provider.enrich([EnrichInput("2", "China and US agree trade deal", "B", None, "")])
        assert a[0].event_label == b[0].event_label

    def test_keywords_match_whole_words_only(self):
        """Regression: "air force" and "Ukrainian" must not read as AI news."""
        provider = HeuristicProvider()
        result = provider.enrich([EnrichInput(
            "1", "Russian drone hits Ukrainian train near Polish border", "Wire", None,
            "Ukraine's air force said it had intercepted most of the drones. "
            "Rescuers said a raid on the railway chair depot caused no casualties.",
        )])[0]
        assert "ai" not in result.topics

    def test_real_ai_coverage_is_still_tagged(self):
        provider = HeuristicProvider()
        result = provider.enrich([EnrichInput(
            "1", "Lab releases a new AI model", "Wire", None,
            "The chatbot is built on a large language model.",
        )])[0]
        assert "ai" in result.topics

    def test_no_brief_offline(self):
        assert HeuristicProvider().write_brief(
            BriefInput("s", ["a"], ["A"], ["x"])
        ) is None


class TestRegistry:
    def test_missing_key_falls_back_to_offline(self):
        provider = get_provider(LLMSettings(provider="gemini", api_key=None))
        assert provider.name == "none"

    def test_unknown_provider_falls_back_to_offline(self):
        assert get_provider(LLMSettings(provider="nope")).name == "none"

    def test_gemini_is_built_when_configured(self):
        provider = get_provider(LLMSettings(provider="gemini", api_key="k", model="m"))
        assert isinstance(provider, GeminiProvider)
        assert provider.model == "m"
