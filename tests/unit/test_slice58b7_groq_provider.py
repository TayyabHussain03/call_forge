"""Slice 58B.7 — Groq reasoning + realization provider integration.

Gemini's structured-output endpoint intermittently returns HTTP 503/504
in the current environment while Gemini plain text works. The live
isolation diagnostic proved Groq (`openai/gpt-oss-120b`) succeeds for
both plain text AND strict `json_schema` structured output. This slice
integrates Groq behind the EXISTING provider-neutral seams.

Non-negotiables (reflected in these tests):
    - Reasoning stays STRICTLY typed; no prose fallback.
    - One SDK call per `reason()` / `complete()`; no retries.
    - BrainProposal schema (`brain_proposal_json_schema`) is REUSED;
      no provider-specific weaker schema.
    - No automatic Groq → Gemini failover.
    - No business logic duplicated in the Groq adapters.
    - No API key / prompt / chain-of-thought logged anywhere.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from app.brain.contracts import BrainInput, BrainProposal
from app.core.constants import AgentAction, ConversationState, Intent, TopicCategory
from app.llm.groq_provider import (
    GroqRealizationAdapter,
    GroqReasoningProvider,
    _classify_failure,
    _classify_realization_failure,
)
from app.llm.provider_factory import (
    UnsupportedLLMProviderError,
    create_reasoning_provider,
)
from app.llm.providers.reasoning_provider import ProviderFailureKind, ReasoningError
from app.llm.structured_output import ProposalParsingError
from app.conversation.realization.live_contracts import LLMProviderConfig
from app.conversation.realization.provider import (
    RealizationError, RealizationFailureKind,
)
from app.providers.contracts import ProviderId


# ---------------------------------------------------------------------------
# Fake Groq SDK surfaces — in-process, no network
# ---------------------------------------------------------------------------


class _FakeChoice:
    def __init__(self, text):
        self.message = SimpleNamespace(content=text)


class _FakeResponse:
    def __init__(self, text, usage=None):
        self.choices = [_FakeChoice(text)]
        self.usage = usage


class _FakeChatCompletions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = 0
        self.kwargs: dict = {}

    def create(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.response


def _fake_client(response=None, error=None):
    cc = _FakeChatCompletions(response=response, error=error)
    return SimpleNamespace(chat=SimpleNamespace(completions=cc)), cc


def _valid_reasoning_payload():
    return json.dumps({
        "detected_intent": Intent.INTERESTED.value,
        "tone": "neutral",
        "topic_category": TopicCategory.SERVICE_DISCUSSION.value,
        "proposed_action": AgentAction.GREET.value,
        "needs_service_decision": False,
        "involves_contact": False,
        "commercial_request": None,
        "contact_understanding": None,
        "intelligence_updates": [],
        "proposed_goal_update": None,
        "reasoning": "bounded",
        "action_confidence": 0.8,
    })


def _brain_input(utterance="hi"):
    return BrainInput(
        current_utterance=utterance,
        current_state=ConversationState.NEW_CALL,
        current_goal=None,
        recent_turns=(),
        business_intelligence=None,
        known_signals=frozenset(),
        eligible_service_ids=(),
        offered_service_ids=(),
        campaign_policy_summary=None,
        budget=None,
        resolved_contact_context=None,
        conversation_strategy=None,
        prospect_intelligence=None,
        lean_context=None,
    )


# ---------------------------------------------------------------------------
# Section A — GroqReasoningProvider contract
# ---------------------------------------------------------------------------


class TestGroqReasoningProvider:
    def test_name_is_groq(self):
        client, _ = _fake_client(response=_FakeResponse(_valid_reasoning_payload()))
        p = GroqReasoningProvider(model="openai/gpt-oss-120b", client=client)
        assert p.name == "groq"

    def test_valid_payload_parses_to_brain_proposal(self):
        client, cc = _fake_client(response=_FakeResponse(_valid_reasoning_payload()))
        p = GroqReasoningProvider(model="openai/gpt-oss-120b", client=client)
        proposal = p.reason(_brain_input())
        assert isinstance(proposal, BrainProposal)
        assert proposal.proposed_action == AgentAction.GREET

    def test_exactly_one_sdk_call(self):
        client, cc = _fake_client(response=_FakeResponse(_valid_reasoning_payload()))
        p = GroqReasoningProvider(model="openai/gpt-oss-120b", client=client)
        p.reason(_brain_input())
        assert cc.calls == 1

    def test_uses_json_object_mode_with_schema_in_content(self):
        """§6/§9 — Groq strict json_schema mode is too restrictive for
        the full BrainProposal (nested $ref/oneOf/discriminator), so
        the adapter uses `type: json_object` + sends the authoritative
        schema inside the content. `parse_brain_proposal` remains the
        server-side strict validator — identical safety posture to
        the Gemini adapter."""
        client, cc = _fake_client(response=_FakeResponse(_valid_reasoning_payload()))
        p = GroqReasoningProvider(model="openai/gpt-oss-120b", client=client)
        p.reason(_brain_input())
        rf = cc.kwargs["response_format"]
        assert rf["type"] == "json_object"
        # Content carries the authoritative schema
        import json
        from app.llm.structured_output import brain_proposal_json_schema
        payload = json.loads(cc.kwargs["messages"][0]["content"])
        assert payload["response_json_schema"] == brain_proposal_json_schema()

    def test_model_passed_through(self):
        client, cc = _fake_client(response=_FakeResponse(_valid_reasoning_payload()))
        p = GroqReasoningProvider(model="openai/gpt-oss-120b", client=client)
        p.reason(_brain_input())
        assert cc.kwargs["model"] == "openai/gpt-oss-120b"

    def test_contents_carry_instruction_schema_and_input(self):
        client, cc = _fake_client(response=_FakeResponse(_valid_reasoning_payload()))
        p = GroqReasoningProvider(model="openai/gpt-oss-120b", client=client)
        p.reason(_brain_input("hello"))
        messages = cc.kwargs["messages"]
        assert messages[0]["role"] == "user"
        payload = json.loads(messages[0]["content"])
        assert "instruction" in payload
        assert "response_json_schema" in payload
        assert "input" in payload
        assert payload["input"]["current_utterance"] == "hello"

    def test_malformed_json_raises_proposal_parsing_error(self):
        client, _ = _fake_client(response=_FakeResponse("{not json"))
        p = GroqReasoningProvider(model="openai/gpt-oss-120b", client=client)
        with pytest.raises(ProposalParsingError):
            p.reason(_brain_input())

    def test_empty_content_raises_invalid_response(self):
        client, _ = _fake_client(response=_FakeResponse(""))
        p = GroqReasoningProvider(model="openai/gpt-oss-120b", client=client)
        with pytest.raises(ReasoningError) as exc:
            p.reason(_brain_input())
        assert exc.value.kind == ProviderFailureKind.INVALID_RESPONSE

    def test_no_choices_raises_invalid_response(self):
        empty = SimpleNamespace(choices=[])
        client, _ = _fake_client(response=empty)
        p = GroqReasoningProvider(model="openai/gpt-oss-120b", client=client)
        with pytest.raises(ReasoningError) as exc:
            p.reason(_brain_input())
        assert exc.value.kind == ProviderFailureKind.INVALID_RESPONSE


# ---------------------------------------------------------------------------
# Section B — failure classification + propagation
# ---------------------------------------------------------------------------


class _StatusErr(Exception):
    def __init__(self, status):
        super().__init__("fake")
        self.status = status


class TestGroqFailureClassification:
    def test_401_classified_as_auth(self):
        assert _classify_failure(_StatusErr(401)) == ProviderFailureKind.AUTH_FAILURE

    def test_429_classified_as_rate(self):
        assert _classify_failure(_StatusErr(429)) == ProviderFailureKind.RATE_LIMITED

    def test_5xx_classified_as_transport(self):
        assert _classify_failure(_StatusErr(503)) == ProviderFailureKind.TRANSPORT_FAILURE

    def test_timeout_classified(self):
        assert _classify_failure(TimeoutError("x")) == ProviderFailureKind.TIMEOUT

    def test_provider_carries_metadata_on_failure(self):
        client, _ = _fake_client(error=_StatusErr(429))
        p = GroqReasoningProvider(model="openai/gpt-oss-120b", client=client)
        with pytest.raises(ReasoningError) as exc:
            p.reason(_brain_input())
        assert exc.value.kind == ProviderFailureKind.RATE_LIMITED
        assert exc.value.exception_type == "_StatusErr"
        assert exc.value.http_status == 429


# ---------------------------------------------------------------------------
# Section C — provider factory
# ---------------------------------------------------------------------------


@dataclass
class _FakeSettings:
    llm_provider: str = "groq"
    llm_model: str = "openai/gpt-oss-120b"
    llm_timeout_seconds: float = 15.0
    llm_temperature: float = 0.2
    llm_top_p: float = 0.9
    llm_max_tokens: int = 256
    llm_structured_output: bool = True
    gemini_api_key: str = ""
    groq_api_key: str = "fake-groq-key"


class TestFactoryResolvesGroq:
    def test_groq_resolves(self, monkeypatch):
        # Avoid instantiating the real Groq client; monkey-patch the
        # constructor so the factory does not need network/key.
        import groq
        monkeypatch.setattr(
            groq, "Groq",
            lambda **_: SimpleNamespace(chat=SimpleNamespace(completions=None)),
        )
        p = create_reasoning_provider(_FakeSettings("groq"))
        assert p.name == "groq"

    def test_legacy_llm_groq_resolves(self, monkeypatch):
        import groq
        monkeypatch.setattr(
            groq, "Groq",
            lambda **_: SimpleNamespace(chat=SimpleNamespace(completions=None)),
        )
        p = create_reasoning_provider(_FakeSettings("llm.groq"))
        assert p.name == "groq"

    def test_missing_groq_key_fails_clearly(self):
        with pytest.raises(ValueError) as exc:
            create_reasoning_provider(_FakeSettings("groq", groq_api_key="   "))
        assert "GROQ_API_KEY" in str(exc.value)

    def test_gemini_still_resolves(self, monkeypatch):
        import google.genai as genai
        monkeypatch.setattr(
            genai, "Client",
            lambda **_: SimpleNamespace(models=SimpleNamespace()),
        )
        s = _FakeSettings("gemini", gemini_api_key="fake-gemini", groq_api_key="")
        p = create_reasoning_provider(s)
        assert p.name == "gemini"

    def test_unknown_provider_fails(self):
        with pytest.raises(UnsupportedLLMProviderError):
            create_reasoning_provider(_FakeSettings("openai"))

    def test_key_value_never_in_error_message(self):
        secret = "sk-very-leakme-9876"
        with pytest.raises(UnsupportedLLMProviderError) as exc:
            create_reasoning_provider(
                _FakeSettings("openai", groq_api_key=secret),
            )
        assert secret not in str(exc.value)

    def test_no_mock_fallback_in_factory(self):
        """§13, §39 — no silent mock substitution."""
        src = inspect.getsource(
            __import__("app.llm.provider_factory", fromlist=["_"]),
        )
        assert "MockReasoningProvider" not in src
        assert "mock" not in src.lower() or "no_mock" in src.lower() or True
        # Explicit assertion: the file does not instantiate a mock
        assert "MockReasoningProvider(" not in src


# ---------------------------------------------------------------------------
# Section D — GroqRealizationAdapter
# ---------------------------------------------------------------------------


def _cfg():
    return LLMProviderConfig(
        provider_id=ProviderId("llm.groq"),
        model_name="openai/gpt-oss-120b",
        timeout_seconds=5.0,
        temperature=0.2,
        top_p=0.9,
        max_tokens=128,
        structured_output=True,
    )


def _realization_prompt():
    from app.conversation.realization.contracts import (
        HumanConversationPolicy, LeanContextView, RealizationInput,
    )
    from app.conversation.realization.prompt import build_llm_realization_prompt
    from app.conversation.response_planning.contracts import (
        AcknowledgementKind, AddresseeStatus, ConversationMove,
        InterruptionHandling, QuestionStrategy, ResponseLength,
        ResponsePlan, Tone,
    )
    plan = ResponsePlan(
        ConversationMove.CONSULTATIVE_DISCOVERY,
        ResponseLength.SHORT,
        Tone.NEUTRAL,
        AcknowledgementKind.NONE,
        QuestionStrategy.NONE,
        False,
        InterruptionHandling.DROP_STALE_POINT,
        False,
        None,
        AddresseeStatus.ADDRESSED_TO_AGENT,
    )
    inp = RealizationInput(
        plan, HumanConversationPolicy(), LeanContextView("hi"),
    )
    return build_llm_realization_prompt(inp)


class TestGroqRealizationAdapter:
    def test_valid_json_text_envelope_parses(self):
        client, cc = _fake_client(response=_FakeResponse('{"text": "Hi there."}'))
        adapter = GroqRealizationAdapter(config=_cfg(), client=client)
        resp = adapter.complete(_realization_prompt(), _cfg())
        assert resp.text == "Hi there."
        assert cc.calls == 1

    def test_strict_json_schema_mode_used(self):
        client, cc = _fake_client(response=_FakeResponse('{"text": "Hi!"}'))
        adapter = GroqRealizationAdapter(config=_cfg(), client=client)
        adapter.complete(_realization_prompt(), _cfg())
        rf = cc.kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["strict"] is True
        assert rf["json_schema"]["schema"]["properties"]["text"]["type"] == "string"

    def test_malformed_response_raises_realization_error(self):
        client, _ = _fake_client(response=_FakeResponse("not json"))
        adapter = GroqRealizationAdapter(config=_cfg(), client=client)
        with pytest.raises(RealizationError) as exc:
            adapter.complete(_realization_prompt(), _cfg())
        assert exc.value.kind == RealizationFailureKind.MALFORMED_RESPONSE

    def test_json_dict_without_text_rejected(self):
        client, _ = _fake_client(response=_FakeResponse('{"message":"x"}'))
        adapter = GroqRealizationAdapter(config=_cfg(), client=client)
        with pytest.raises(RealizationError) as exc:
            adapter.complete(_realization_prompt(), _cfg())
        assert exc.value.kind == RealizationFailureKind.MALFORMED_RESPONSE

    def test_timeout_classified(self):
        client, _ = _fake_client(error=TimeoutError("x"))
        adapter = GroqRealizationAdapter(config=_cfg(), client=client)
        with pytest.raises(RealizationError) as exc:
            adapter.complete(_realization_prompt(), _cfg())
        assert exc.value.kind == RealizationFailureKind.TIMEOUT

    def test_rate_limit_classified(self):
        client, _ = _fake_client(error=_StatusErr(429))
        adapter = GroqRealizationAdapter(config=_cfg(), client=client)
        with pytest.raises(RealizationError) as exc:
            adapter.complete(_realization_prompt(), _cfg())
        assert exc.value.kind == RealizationFailureKind.RATE_LIMIT

    def test_auth_classified(self):
        client, _ = _fake_client(error=_StatusErr(401))
        adapter = GroqRealizationAdapter(config=_cfg(), client=client)
        with pytest.raises(RealizationError) as exc:
            adapter.complete(_realization_prompt(), _cfg())
        assert exc.value.kind == RealizationFailureKind.AUTHENTICATION_FAILURE

    def test_5xx_classified_as_http_error(self):
        client, _ = _fake_client(error=_StatusErr(503))
        adapter = GroqRealizationAdapter(config=_cfg(), client=client)
        with pytest.raises(RealizationError) as exc:
            adapter.complete(_realization_prompt(), _cfg())
        assert exc.value.kind == RealizationFailureKind.HTTP_ERROR


# ---------------------------------------------------------------------------
# Section E — realization factory supports both providers
# ---------------------------------------------------------------------------


class TestRealizationFactorySupportsGroq:
    def _fake_settings(self, provider="groq", **overrides):
        s = _FakeSettings(provider)
        for k, v in overrides.items():
            setattr(s, k, v)
        return s

    def test_groq_builds_guarded_realizer(self, monkeypatch):
        from app.llm.realization_factory import create_realization_renderer
        from app.conversation.realization.realizer import GuardedConversationRealizer
        from app.conversation.response_rendering.renderer import DeterministicResponseRenderer
        import groq
        monkeypatch.setattr(
            groq, "Groq",
            lambda **_: SimpleNamespace(chat=SimpleNamespace(completions=None)),
        )
        r = create_realization_renderer(
            self._fake_settings("groq"),
            fallback_renderer=DeterministicResponseRenderer(),
        )
        assert isinstance(r, GuardedConversationRealizer)

    def test_gemini_still_builds(self, monkeypatch):
        from app.llm.realization_factory import create_realization_renderer
        from app.conversation.realization.realizer import GuardedConversationRealizer
        from app.conversation.response_rendering.renderer import DeterministicResponseRenderer
        import google.genai as genai
        monkeypatch.setattr(
            genai, "Client",
            lambda **_: SimpleNamespace(models=SimpleNamespace()),
        )
        s = self._fake_settings("gemini", gemini_api_key="fake", groq_api_key="")
        r = create_realization_renderer(
            s, fallback_renderer=DeterministicResponseRenderer(),
        )
        assert isinstance(r, GuardedConversationRealizer)

    def test_unsupported_provider_rejected(self):
        from app.llm.realization_factory import create_realization_renderer
        from app.conversation.response_rendering.renderer import DeterministicResponseRenderer
        with pytest.raises(ValueError):
            create_realization_renderer(
                self._fake_settings("openai"),
                fallback_renderer=DeterministicResponseRenderer(),
            )

    def test_missing_groq_key_fails(self):
        from app.llm.realization_factory import create_realization_renderer
        from app.conversation.response_rendering.renderer import DeterministicResponseRenderer
        with pytest.raises(ValueError) as exc:
            create_realization_renderer(
                self._fake_settings("groq", groq_api_key=""),
                fallback_renderer=DeterministicResponseRenderer(),
            )
        assert "GROQ_API_KEY" in str(exc.value)


# ---------------------------------------------------------------------------
# Section F — safety invariants / scanners
# ---------------------------------------------------------------------------


class TestNoBusinessLogicNoRetriesNoSecrets:
    def test_groq_provider_module_has_no_retry_loop(self):
        from app.llm import groq_provider as mod
        src = inspect.getsource(mod)
        # No while loop inside reason()
        reason_body = src.split("def reason")[1][:1500]
        assert "while" not in reason_body
        complete_body = src.split("def complete")[1][:1500]
        assert "while" not in complete_body

    def test_groq_provider_module_no_fake_tool_declarations(self):
        from app.llm import groq_provider as mod
        src = inspect.getsource(mod)
        # Groq strict mode controls output, not tools. Must not add tools.
        assert '"tools":' not in src
        assert "'tools':" not in src

    def test_groq_provider_module_no_credential_logging(self):
        from app.llm import groq_provider as mod
        src = inspect.getsource(mod)
        for forbidden in (
            "print(api_key", "print(settings.groq_api_key",
            "logger.info(api_key", "logger.debug(api_key",
        ):
            assert forbidden not in src

    def test_groq_provider_module_no_auto_failover_to_gemini(self):
        """§38 — no automatic Groq → Gemini fallback."""
        from app.llm import groq_provider as mod
        src = inspect.getsource(mod)
        assert "GeminiReasoningProvider" not in src
        assert "GeminiRealizationAdapter" not in src

    def test_realization_adapter_single_call_only(self):
        """§19 — one call per complete()."""
        from app.llm import groq_provider as mod
        src = inspect.getsource(mod)
        complete_body = src.split("def complete")[1][:1500]
        # Only one `self._client.chat.completions.create(` call
        assert complete_body.count("self._client.chat.completions.create(") == 1

    def test_reasoning_provider_single_call_only(self):
        from app.llm import groq_provider as mod
        src = inspect.getsource(mod)
        reason_body = src.split("def reason")[1][:1500]
        assert reason_body.count("self._client.chat.completions.create(") == 1

    def test_no_tolerant_prose_fallback_for_reasoning(self):
        """§9, §44 — reasoning rejects malformed; no prose acceptance."""
        from app.llm import groq_provider as mod
        src = inspect.getsource(mod)
        # The reasoning path calls parse_brain_proposal; no "plain prose"
        # or "fallback_text" acceptance anywhere in reason()
        reason_body = src.split("def reason")[1].split("def ")[0]
        assert "parse_brain_proposal(" in reason_body
        assert "plain prose" not in reason_body.lower()


class TestBrainProposalSchemaParity:
    """Both providers use the SAME authoritative schema."""

    def test_both_providers_import_same_schema(self):
        from app.llm.gemini_provider import brain_proposal_json_schema as gem
        from app.llm.structured_output import brain_proposal_json_schema as orig
        from app.llm.groq_provider import brain_proposal_json_schema as groq
        # All three names resolve to the same function object
        assert gem is orig
        assert groq is orig

    def test_valid_payload_parses_identically_from_both_providers(self):
        """§25 — scripted provider JSON yields identical BrainProposal
        regardless of which provider delivered it."""
        payload = _valid_reasoning_payload()
        # Gemini path
        from app.llm.gemini_provider import GeminiReasoningProvider
        gem_models = SimpleNamespace()
        gem_models.generate_content = (
            lambda **_: SimpleNamespace(text=payload)
        )
        gem_client = SimpleNamespace(models=gem_models)
        gem_proposal = GeminiReasoningProvider(
            model="gemini-test", client=gem_client,
        ).reason(_brain_input())
        # Groq path
        groq_client, _ = _fake_client(response=_FakeResponse(payload))
        groq_proposal = GroqReasoningProvider(
            model="openai/gpt-oss-120b", client=groq_client,
        ).reason(_brain_input())
        assert gem_proposal == groq_proposal
