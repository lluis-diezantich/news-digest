"""Ollama provider, against a stubbed transport. No server, no model download.

The local LLM half of moving off a hosted free tier. Unlike embeddings, where a
small model matched the API outright, this is a hardware argument: a CI runner
cannot do it, a machine with real memory can.
"""

import json

import pytest

from newsdigest.config import LLMSettings
from newsdigest.llm import get_provider
from newsdigest.llm.base import BriefInput, Context, EnrichInput, LLMError, PairInput
from newsdigest.llm.ollama import DEFAULT_MODEL, OllamaProvider

ITEMS = [
    EnrichInput(id="a1", title="La UE anuncia nuevas sanciones", source="El País",
                language="es", published=None, excerpt="Bruselas aprobó medidas."),
    EnrichInput(id="a2", title="Rare orchid found in Peru", source="Wire",
                language="en", published=None, excerpt="Scientists described it."),
]

ENRICHED = json.dumps([
    {"id": "a1", "summary": "The EU approved new measures.", "why_it_matters": "Trade tightens.",
     "key_facts": ["Approved in Brussels"], "topics": ["world"], "entities": ["European Union"],
     "importance": 0.72, "relevance": 0.65, "content_type": "reporting"},
    {"id": "a2", "summary": "A new orchid was described.", "why_it_matters": "",
     "key_facts": [], "topics": ["science"], "entities": [], "importance": 0.2,
     "relevance": 0.1, "content_type": "reporting"},
])


class StubResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = payload if isinstance(payload, str) else json.dumps(payload)

    def json(self):
        return self._payload


class StubSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.posts = []
        self.urls = []

    def post(self, url, json=None, timeout=None):
        self.urls.append(url)
        self.posts.append(json)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def reply(text):
    return StubResponse(200, {"message": {"role": "assistant", "content": text},
                              "done_reason": "stop"})


def provider_with(*responses, **kw):
    p = OllamaProvider(model="test-model", max_retries=1, **kw)
    p._session = StubSession(*responses)
    return p


@pytest.fixture
def context():
    return Context(output_language="en", interests=["world"], excluded_topics=["sports"])


class TestRequestShape:
    def test_posts_to_the_chat_endpoint(self, context):
        p = provider_with(reply(ENRICHED))
        p.enrich(ITEMS, context)
        assert p._session.urls == ["http://localhost:11434/api/chat"]

    def test_base_url_is_configurable(self, context):
        p = provider_with(reply(ENRICHED), base_url="http://box.local:1234/")
        p.enrich(ITEMS, context)
        assert p._session.urls[0] == "http://box.local:1234/api/chat"

    def test_a_json_schema_is_sent(self, context):
        """Schema enforcement is what keeps a local model inside JSON."""
        p = provider_with(reply(ENRICHED))
        p.enrich(ITEMS, context)
        body = p._session.posts[0]
        assert body["format"]["type"] == "array"
        assert "importance" in body["format"]["items"]["properties"]

    def test_thinking_is_disabled(self, context):
        """Hybrid models otherwise emit reasoning into the response and break it."""
        p = provider_with(reply(ENRICHED))
        p.enrich(ITEMS, context)
        assert p._session.posts[0]["think"] is False

    def test_streaming_is_off_and_context_is_set(self, context):
        p = provider_with(reply(ENRICHED), num_ctx=16384)
        p.enrich(ITEMS, context)
        body = p._session.posts[0]
        assert body["stream"] is False
        assert body["options"]["num_ctx"] == 16384


class TestEnrich:
    def test_parses_a_batch(self, context):
        out = provider_with(reply(ENRICHED)).enrich(ITEMS, context)
        assert [e.id for e in out] == ["a1", "a2"]
        assert out[0].importance == pytest.approx(0.72)
        assert out[0].content_type == "reporting"

    def test_unknown_ids_are_dropped(self, context):
        payload = json.dumps([{"id": "not-ours", "summary": "x", "why_it_matters": "",
                               "key_facts": [], "topics": [], "entities": [],
                               "importance": 0.5, "relevance": 0.5,
                               "content_type": "reporting"}])
        assert provider_with(reply(payload)).enrich(ITEMS, context) == []

    def test_out_of_range_scores_are_clamped(self, context):
        payload = json.dumps([{"id": "a1", "summary": "x", "why_it_matters": "",
                               "key_facts": [], "topics": [], "entities": [],
                               "importance": 5.0, "relevance": -2.0,
                               "content_type": "reporting"}])
        e = provider_with(reply(payload)).enrich(ITEMS, context)[0]
        assert e.importance == 1.0 and e.relevance == 0.0

    def test_empty_input_makes_no_request(self, context):
        p = provider_with(reply(ENRICHED))
        assert p.enrich([], context) == [] and p.calls == 0

    def test_a_partial_batch_is_not_an_error(self, context):
        """Missing ids are retried next run rather than failing the whole batch."""
        payload = json.dumps([json.loads(ENRICHED)[0]])
        assert len(provider_with(reply(payload)).enrich(ITEMS, context)) == 1


class TestBriefAndPairs:
    def test_writes_a_brief(self, context):
        payload = json.dumps({"headline": "EU widens sanctions", "summary": "It did.",
                              "why_it_matters": "Trade.", "key_facts": ["a"],
                              "topics": ["world"], "importance": 0.8, "relevance": 0.6})
        b = provider_with(reply(payload)).write_brief(
            BriefInput(story_id="s1", headlines=["x"], sources=["El País"],
                       languages=["es"], excerpts=["y"]), context)
        assert b.headline == "EU widens sanctions" and b.importance == pytest.approx(0.8)

    def test_a_brief_without_a_headline_is_none(self, context):
        p = provider_with(reply(json.dumps({"summary": "no headline"})))
        assert p.write_brief(BriefInput("s1", ["x"], ["s"], ["es"], ["y"]), context) is None

    def test_adjudicates_pairs_in_one_request(self, context):
        pairs = [PairInput(f"k{i}", "l", "es", "le", "r", "en", "re") for i in range(3)]
        payload = json.dumps([{"key": "k0", "same_event": True},
                              {"key": "k1", "same_event": False}])
        p = provider_with(reply(payload))
        assert p.same_event(pairs, context) == {"k0": True, "k1": False}
        assert p.calls == 1, "pairs must be batched into one request"

    def test_no_pairs_makes_no_request(self, context):
        p = provider_with(reply("[]"))
        assert p.same_event([], context) == {} and p.calls == 0


class TestErrors:
    def test_a_missing_server_says_so(self, context):
        import requests

        class Refusing:
            def post(self, *a, **k):
                raise requests.ConnectionError("refused")
        p = provider_with(reply(ENRICHED))
        p._session = Refusing()
        with pytest.raises(LLMError, match="no ollama at .*ollama serve"):
            p.enrich(ITEMS, context)

    def test_a_missing_model_says_how_to_pull_it(self, context):
        p = provider_with(StubResponse(404, {"error": "model not found"}))
        with pytest.raises(LLMError, match="ollama pull test-model"):
            p.enrich(ITEMS, context)

    def test_empty_content_is_an_error(self, context):
        p = provider_with(StubResponse(200, {"message": {"content": "  "},
                                             "done_reason": "length"}))
        with pytest.raises(LLMError, match="done_reason=length"):
            p.enrich(ITEMS, context)

    def test_unparseable_json_is_an_error(self, context):
        with pytest.raises(LLMError, match="unparseable JSON"):
            provider_with(reply("not json at all")).enrich(ITEMS, context)

    def test_http_error_surfaces(self, context):
        p = provider_with(StubResponse(500, "boom"))
        with pytest.raises(LLMError, match="ollama 500"):
            p.enrich(ITEMS, context)


class TestRegistry:
    def test_selected_by_name(self):
        assert get_provider(LLMSettings(provider="ollama")).name == "ollama"

    def test_a_gemini_model_name_is_not_carried_over(self):
        p = get_provider(LLMSettings(provider="ollama", model="gemini-3.6-flash"))
        assert p.model == DEFAULT_MODEL

    def test_cache_key_separates_it_from_other_providers(self):
        ctx = Context(output_language="en", interests=[], excluded_topics=[])
        a = get_provider(LLMSettings(provider="ollama")).cache_key(ctx)
        b = get_provider(LLMSettings(provider="gemini", api_key="k")).cache_key(ctx)
        assert a != b, "switching provider must not reuse cached enrichments"
