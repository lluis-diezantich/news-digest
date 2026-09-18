"""LLM provider tests. Gemini is exercised against a stubbed transport."""

import copy
import json

import pytest

from conftest import PER_DAY_429, PER_MINUTE_429

from newsdigest.config import LLMSettings, Preferences, Settings
from newsdigest.llm import build_context, get_provider
from newsdigest.llm.base import (
    TOPICS, Brief, BriefInput, Context, EnrichInput, Enrichment, LLMError, LLMQuotaError,
    PairInput, brief_system_prompt, enrich_system_prompt, language_name, normalize_topics,
)
from newsdigest.llm.gemini import GeminiProvider
from newsdigest.llm.heuristic import HeuristicProvider

ITEMS = [
    EnrichInput(id="a1", title="La UE anuncia nuevas sanciones contra Rusia",
                source="El País", language="es", published="2026-09-10T08:30:00+00:00",
                excerpt="Bruselas ha aprobado un nuevo paquete de medidas."),
    EnrichInput(id="a2", title="Rare orchid found in Peru", source="Wire",
                language="en", published=None, excerpt="Scientists described a new species."),
]


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
        self.headers = {}

    def post(self, url, json=None, timeout=None):
        # A snapshot, not a reference: the real transport serializes the body at
        # send time, so a later retry that edits the config must not appear to
        # have rewritten history for the call that already went out.
        self.posts.append(copy.deepcopy(json))
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def gemini_with(*responses, **kwargs) -> GeminiProvider:
    # Rate-limit waiting is off unless a test asks for it, so no test sleeps.
    kwargs.setdefault("max_rate_limit_retries", 0)
    provider = GeminiProvider(api_key="k", model="gemini-test", max_retries=1, **kwargs)
    provider._session = StubSession(*responses)
    return provider


def candidate(text, finish="STOP"):
    return {"candidates": [{"finishReason": finish, "content": {"parts": [{"text": text}]}}]}


ENRICHED = json.dumps([
    {"id": "a1", "summary": "The EU approved new measures.",
     "why_it_matters": "Trade will tighten.", "key_facts": ["Package approved in Brussels"],
     "topics": ["World", "Economics"], "entities": ["European Union"],
     "importance": 0.72, "relevance": 0.65, "content_type": "reporting"},
    {"id": "a2", "summary": "A new orchid was described.", "why_it_matters": "",
     "key_facts": [], "topics": ["science"], "entities": ["Peru"],
     "importance": 0.2, "relevance": 0.1, "content_type": "reporting"},
])


class TestGeminiEnrich:
    def test_parses_a_batch(self, context):
        provider = gemini_with(StubResponse(200, candidate(ENRICHED)))
        results = provider.enrich(ITEMS, context)
        assert [r.id for r in results] == ["a1", "a2"]
        first = results[0]
        assert first.topics == ["world", "economics"]   # clamp lowercases
        assert first.key_facts == ["Package approved in Brussels"]
        assert first.content_type == "reporting"
        assert 0.0 <= first.relevance <= 1.0
        assert provider.calls == 1                       # one request per batch

    def test_only_metadata_and_excerpt_are_sent(self, context):
        provider = gemini_with(StubResponse(200, candidate("[]")))
        provider.enrich(ITEMS, context)
        sent = json.loads(provider._session.posts[0]["contents"][0]["parts"][0]["text"])
        assert set(sent[0]) == {"id", "title", "source", "language", "published", "excerpt"}

    def test_output_language_and_interests_reach_the_prompt(self):
        provider = gemini_with(StubResponse(200, candidate("[]")))
        provider.enrich(ITEMS, Context(output_language="ca", interests=["climate"],
                                       excluded_topics=["sports"]))
        system = provider._session.posts[0]["systemInstruction"]["parts"][0]["text"]
        assert "Catalan" in system
        assert "climate" in system and "sports" in system

    def test_schema_constrains_the_response(self, context):
        provider = gemini_with(StubResponse(200, candidate("[]")))
        provider.enrich(ITEMS, context)
        config = provider._session.posts[0]["generationConfig"]
        assert config["responseMimeType"] == "application/json"
        required = config["responseSchema"]["items"]["required"]
        assert {"relevance", "key_facts", "content_type"} <= set(required)

    def test_unknown_ids_are_discarded(self, context):
        payload = json.dumps([{"id": "ghost", "summary": "x", "why_it_matters": "",
                               "key_facts": [], "topics": [], "entities": [],
                               "importance": 0.5, "relevance": 0.5,
                               "content_type": "reporting"}])
        provider = gemini_with(StubResponse(200, candidate(payload)))
        assert provider.enrich(ITEMS, context) == []

    def test_invalid_content_type_becomes_none(self, context):
        payload = json.dumps([{"id": "a1", "summary": "x", "why_it_matters": "",
                               "key_facts": [], "topics": [], "entities": [],
                               "importance": 0.5, "relevance": 0.5,
                               "content_type": "gibberish"}])
        provider = gemini_with(StubResponse(200, candidate(payload)))
        assert provider.enrich(ITEMS, context)[0].content_type is None

    def test_out_of_range_scores_are_clamped(self, context):
        payload = json.dumps([{"id": "a1", "summary": "x", "why_it_matters": "",
                               "key_facts": [], "topics": [], "entities": [],
                               "importance": 5.0, "relevance": -2.0,
                               "content_type": "reporting"}])
        provider = gemini_with(StubResponse(200, candidate(payload)))
        result = provider.enrich(ITEMS, context)[0]
        assert result.importance == 1.0 and result.relevance == 0.0

    def test_rate_limit_raises_quota_error_with_no_wait_budget(self, context):
        provider = gemini_with(StubResponse(429, {"error": {"message": "quota"}}))
        with pytest.raises(LLMQuotaError):
            provider.enrich(ITEMS, context)

    def test_per_minute_limit_is_waited_out_then_retried(self, context, no_sleep):
        """The free-tier case: a burst outruns the per-minute allowance."""
        provider = gemini_with(
            StubResponse(429, PER_MINUTE_429),
            StubResponse(200, candidate(ENRICHED)),
            max_rate_limit_retries=2,
        )
        assert len(provider.enrich(ITEMS, context)) == 2
        assert len(no_sleep) == 1

    def test_retry_info_delay_is_honoured(self, context, no_sleep):
        provider = gemini_with(
            StubResponse(429, PER_MINUTE_429),
            StubResponse(200, candidate(ENRICHED)),
            max_rate_limit_retries=2,
        )
        provider.enrich(ITEMS, context)
        # 31s from RetryInfo, plus up to 1s of jitter.
        assert 31.0 <= no_sleep[0] < 32.0

    def test_per_day_limit_is_terminal_and_never_sleeps(self, context, no_sleep):
        """Waiting cannot fix a daily quota, so it must not be attempted."""
        provider = gemini_with(StubResponse(429, PER_DAY_429), max_rate_limit_retries=5)
        with pytest.raises(LLMQuotaError, match="daily quota"):
            provider.enrich(ITEMS, context)
        assert no_sleep == []

    def test_persistent_rate_limit_gives_up_and_degrades(self, context, no_sleep):
        provider = gemini_with(StubResponse(429, PER_MINUTE_429), max_rate_limit_retries=2)
        with pytest.raises(LLMQuotaError, match="after 2 waits"):
            provider.enrich(ITEMS, context)
        assert len(no_sleep) == 2

    def test_run_wait_budget_bounds_total_sleeping(self, context, no_sleep):
        """A low allowance must not spend the workflow's whole timeout asleep."""
        provider = gemini_with(
            StubResponse(429, PER_MINUTE_429),
            max_rate_limit_retries=99,
            max_rate_limit_wait=70.0,
        )
        with pytest.raises(LLMQuotaError, match="wait budget"):
            provider.enrich(ITEMS, context)
        assert sum(no_sleep) <= 70.0

    def test_thinking_level_is_sent_at_the_cheapest_level(self, context):
        """Thought tokens are billed as output and drawn from maxOutputTokens."""
        provider = gemini_with(StubResponse(200, candidate(ENRICHED)))
        provider.enrich(ITEMS, context)
        cfg = provider._session.posts[0]["generationConfig"]
        # Nested: a top-level thinkingLevel is rejected by the live API.
        assert cfg["thinkingConfig"] == {"thinkingLevel": "low"}

    def test_thinking_level_is_omitted_when_unset(self, context):
        """Models with thinking already off want no field at all."""
        provider = gemini_with(StubResponse(200, candidate(ENRICHED)), thinking_level="")
        provider.enrich(ITEMS, context)
        assert "thinkingConfig" not in provider._session.posts[0]["generationConfig"]

    def test_a_model_rejecting_thinking_level_is_retried_without_it(self, context):
        """A pinned LLM_MODEL that predates the field must not cost the run."""
        rejection = {"error": {"code": 400, "message":
                     'Invalid JSON payload received. Unknown name "thinkingConfig".'}}
        provider = gemini_with(
            StubResponse(400, rejection), StubResponse(200, candidate(ENRICHED))
        )
        assert len(provider.enrich(ITEMS, context)) == 2
        sent = provider._session.posts
        assert "thinkingConfig" in sent[0]["generationConfig"]
        assert "thinkingConfig" not in sent[1]["generationConfig"]

    def test_thinking_level_is_dropped_for_the_rest_of_the_run(self, context):
        """One 400, not one per request."""
        rejection = {"error": {"code": 400, "message": 'Unknown name "thinkingConfig".'}}
        provider = gemini_with(
            StubResponse(400, rejection), StubResponse(200, candidate(ENRICHED))
        )
        provider.enrich(ITEMS, context)
        assert provider.thinking_level == ""
        provider._session.responses = [StubResponse(200, candidate(ENRICHED))]
        provider.enrich(ITEMS, context)
        assert "thinkingConfig" not in provider._session.posts[-1]["generationConfig"]

    def test_an_unrelated_400_still_raises(self, context):
        """Only a thinking complaint is retried; everything else is a real error."""
        provider = gemini_with(StubResponse(400, {"error": {"message": "bad api key"}}))
        with pytest.raises(LLMError, match="400"):
            provider.enrich(ITEMS, context)

    def test_blocked_prompt_raises(self, context):
        provider = gemini_with(StubResponse(200, {"promptFeedback": {"blockReason": "SAFETY"}}))
        with pytest.raises(LLMError, match="blocked"):
            provider.enrich(ITEMS, context)

    def test_invalid_json_raises(self, context):
        provider = gemini_with(StubResponse(200, candidate("not json")))
        with pytest.raises(LLMError, match="invalid JSON"):
            provider.enrich(ITEMS, context)

    def test_truncated_response_raises(self, context):
        provider = gemini_with(StubResponse(200, candidate("", finish="MAX_TOKENS")))
        with pytest.raises(LLMError, match="maxOutputTokens"):
            provider.enrich(ITEMS, context)


class TestGeminiBrief:
    def test_roundtrip(self, context):
        payload = json.dumps({"headline": "EU widens Russia sanctions",
                              "summary": "Three outlets report the same package.",
                              "why_it_matters": "Trade tightens.",
                              "key_facts": ["Adopted in Brussels"],
                              "topics": ["World"], "importance": 0.8, "relevance": 0.7})
        provider = gemini_with(StubResponse(200, candidate(payload)))
        brief = provider.write_brief(
            BriefInput(story_id="s1", headlines=["a", "b"], sources=["A", "B"],
                       languages=["en", "es"], excerpts=["x", "y"]), context)
        assert brief.headline == "EU widens Russia sanctions"
        assert brief.topics == ["world"]
        assert brief.importance == 0.8 and brief.relevance == 0.7

    def test_languages_are_sent_with_each_excerpt(self, context):
        provider = gemini_with(StubResponse(200, candidate("{}")))
        provider.write_brief(
            BriefInput(story_id="s1", headlines=["a"], sources=["A"],
                       languages=["ca"], excerpts=["x"]), context)
        sent = json.loads(provider._session.posts[0]["contents"][0]["parts"][0]["text"])
        assert sent["coverage"][0]["language"] == "ca"

    def test_missing_headline_yields_none(self, context):
        provider = gemini_with(StubResponse(200, candidate('{"summary": "only"}')))
        assert provider.write_brief(
            BriefInput("s", ["a"], ["A"], ["en"], ["x"]), context) is None


class TestGeminiSameEvent:
    PAIRS = [PairInput(key="p1", left_title="A", left_language="en", left_excerpt="x",
                       right_title="B", right_language="es", right_excerpt="y")]

    def test_returns_verdicts_by_key(self, context):
        payload = json.dumps([{"key": "p1", "same_event": True}])
        provider = gemini_with(StubResponse(200, candidate(payload)))
        assert provider.same_event(self.PAIRS, context) == {"p1": True}

    def test_unknown_keys_are_ignored(self, context):
        payload = json.dumps([{"key": "nope", "same_event": True}])
        provider = gemini_with(StubResponse(200, candidate(payload)))
        assert provider.same_event(self.PAIRS, context) == {}

    def test_no_pairs_makes_no_call(self, context):
        provider = gemini_with(StubResponse(200, candidate("[]")))
        assert provider.same_event([], context) == {}
        assert provider.calls == 0


class TestHeuristic:
    def test_classifies_all_three_languages(self, context):
        provider = HeuristicProvider()
        cases = [
            ("El Govern aprova una llei d'intel·ligència artificial",
             "El Parlament ha votat la nova llei.", "ca", "ai"),
            ("La inflación se moderó en agosto",
             "El mercado espera una bajada de tipos.", "es", "economics"),
            ("Lab releases a new AI model", "The chatbot uses a large model.", "en", "ai"),
        ]
        for title, excerpt, language, expected in cases:
            result = provider.enrich(
                [EnrichInput("1", title, "S", language, None, excerpt)], context)[0]
            assert expected in result.topics, f"{title!r} -> {result.topics}"

    def test_keywords_match_whole_words_only(self, context):
        """Regression: "air force" and "Ukrainian" must not read as AI news."""
        result = HeuristicProvider().enrich([EnrichInput(
            "1", "Russian drone hits Ukrainian train", "Wire", "en",
            None, "Ukraine's air force said it intercepted most of the drones.",
        )], context)[0]
        assert "ai" not in result.topics

    def test_excluded_topic_caps_relevance_even_in_another_language(self, context):
        """"sports" never appears in a Catalan football headline."""
        result = HeuristicProvider().enrich([EnrichInput(
            "1", "El Barça guanya la lliga", "Ara", "ca", None,
            "El partit va acabar amb un gol al minut 90.")], context)[0]
        assert "sports" in result.topics
        assert result.relevance <= 0.15

    def test_interest_hit_raises_relevance(self, context):
        low = HeuristicProvider().enrich([EnrichInput(
            "1", "Council repaves the high street", "Local", "en", None,
            "Work begins next month on the pavement.")], context)[0]
        high = HeuristicProvider().enrich([EnrichInput(
            "2", "New chip boosts AI economics", "Wire", "en", None,
            "The technology could reshape the market for inference.")], context)[0]
        assert high.relevance > low.relevance

    def test_offers_no_brief_or_verdict(self, context):
        provider = HeuristicProvider()
        assert provider.write_brief(BriefInput("s", ["a"], ["A"], ["en"], ["x"]), context) is None
        assert provider.same_event([], context) == {}


class TestTopicVocabulary:
    """The tag vocabulary is closed. These are the drifts seen in real output."""

    def test_keeps_canonical_topics(self):
        assert normalize_topics(["politics", "ai"]) == ["politics", "ai"]

    def test_folds_synonyms_onto_one_tag(self):
        """`ai` and `artificial intelligence` were two chips for one topic."""
        assert normalize_topics(["artificial intelligence"]) == ["ai"]
        assert normalize_topics(["economy"]) == ["economics"]
        assert normalize_topics(["elections"]) == ["politics"]
        assert normalize_topics(["geopolitics"]) == ["world"]
        assert normalize_topics(["football"]) == ["sports"]

    def test_folds_qualified_variants(self):
        """"spanish politics" and "us politics" split the politics filter."""
        assert normalize_topics(["spanish politics", "us politics"]) == ["politics"]

    def test_drops_place_names(self):
        """A country is an entity, not a topic -- it has its own field."""
        assert normalize_topics(["sweden", "west bank", "catalonia"]) == []

    def test_drops_the_offline_placeholder(self):
        """`general` was the second most common tag and means nothing."""
        assert normalize_topics(["general"]) == []

    def test_is_accent_and_case_insensitive(self):
        assert normalize_topics(["Economía", " AI "]) == ["economics", "ai"]

    def test_deduplicates_after_folding(self):
        assert normalize_topics(["ai", "artificial intelligence", "llm"]) == ["ai"]

    def test_caps_the_count(self):
        assert len(normalize_topics(list(TOPICS))) == 4

    def test_enrichment_clamp_enforces_the_vocabulary(self):
        e = Enrichment(id="a", summary="s", topics=["Politics", "Sweden", "elections"])
        assert e.clamp().topics == ["politics"]

    def test_brief_prompt_anchors_importance(self):
        """The brief's importance overwrites the story's, so an unanchored scale
        here flattened the whole ranking to ~0.8."""
        prompt = brief_system_prompt(Context(interests=["ai"]))
        assert "0.3 routine" in prompt and "sparing" in prompt
        assert "Wide coverage is NOT importance" in prompt

    def test_brief_clamp_enforces_the_vocabulary(self):
        b = Brief(headline="h", summary="s", topics=["technology regulation", "usa"])
        assert b.clamp().topics == ["technology"]

    def test_excluded_topics_can_still_be_matched(self):
        """A topic that can never be emitted can never be excluded.

        Closing the vocabulary silently disarmed `excluded_topics: [celebrity]`,
        leaving only the headline-substring half -- and no celebrity headline
        contains the word "celebrity".
        """
        assert normalize_topics(["celebrity"]) == ["celebrity"]
        assert normalize_topics(["sports"]) == ["sports"]
        assert normalize_topics(["gossip"]) == ["celebrity"]

    def test_celebrity_is_separate_from_culture(self):
        """Excluding gossip must not exclude the arts."""
        assert normalize_topics(["film"]) == ["culture"]
        assert normalize_topics(["paparazzi"]) != ["culture"]

    def test_prompt_lists_the_allowed_topics(self):
        """A closed vocabulary the model is never shown is only half enforced."""
        prompt = enrich_system_prompt(Context(interests=["ai"]))
        assert ", ".join(TOPICS) in prompt


class TestSportsFalseFriends:
    """The keyword list must not contain words politics also uses.

    Fixed 2026-09-18: `partido`/`partit` mean political PARTY and `madrid` names
    the city, the government and the Comunidad. Word boundaries do not help --
    these are whole words -- so on a Spanish and Catalan politics corpus they
    tagged 31 articles of one week as sport, and `excluded_topics: [sports]` then
    docked six of that week's top contenders 1.5 each.
    """

    @pytest.mark.parametrize("headline", [
        "El PP contradice al Gobierno sobre la tutela de los menores migrantes",
        "El partido de Feijóo se reúne con el comisario de Migración",
        "Un informe de la Comunidad de Madrid señala que no se justificó el gasto",
        "El Supremo arrastra los pies en la causa del procés",
        "El partit de Junts trenca amb el Govern espanyol",
    ])
    def test_political_headlines_are_not_sport(self, heuristic, headline):
        enrichment = heuristic.enrich(
            [EnrichInput(id="a1", title=headline, source="S", language="es",
                         published=None, excerpt=headline)],
            Context(),
        )[0]
        assert "sports" not in enrichment.topics, enrichment.topics

    @pytest.mark.parametrize("headline", [
        "El Barça guanya la lliga amb un gol al minut 90",
        "Real Madrid wins the cup final after extra time",
        "El entrenador deja el equipo tras el torneo olímpico",
    ])
    def test_actual_sport_is_still_caught(self, heuristic, headline):
        enrichment = heuristic.enrich(
            [EnrichInput(id="a1", title=headline, source="S", language="es",
                         published=None, excerpt=headline)],
            Context(),
        )[0]
        assert "sports" in enrichment.topics, enrichment.topics


class TestRegistryAndContext:
    def test_missing_key_falls_back_to_offline(self):
        assert get_provider(LLMSettings(provider="gemini", api_key=None)).name == "none"

    def test_unknown_provider_falls_back_to_offline(self):
        assert get_provider(LLMSettings(provider="nope")).name == "none"

    def test_gemini_is_built_when_configured(self):
        provider = get_provider(LLMSettings(provider="gemini", api_key="k", model="m"))
        assert isinstance(provider, GeminiProvider) and provider.model == "m"

    def test_context_is_built_from_config_highest_weight_first(self):
        ctx = build_context(
            Settings(output_language="es"),
            Preferences(topics={"sports": 0.2, "ai": 1.9, "climate": 1.4},
                        excluded_topics=["celebrity"]),
        )
        assert ctx.output_language == "es"
        assert ctx.interests[:2] == ["ai", "climate"]
        assert ctx.excluded_topics == ["celebrity"]

    def test_cache_key_covers_language_and_interests(self):
        provider = HeuristicProvider()
        a = provider.cache_key(Context(output_language="en", interests=["ai"]))
        b = provider.cache_key(Context(output_language="ca", interests=["ai"]))
        c = provider.cache_key(Context(output_language="en", interests=["climate"]))
        assert a != b and a != c

    def test_language_name_falls_back_to_the_code(self):
        assert language_name("ca") == "Catalan"
        assert language_name("zz") == "zz"
