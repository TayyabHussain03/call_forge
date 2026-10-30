"""Slice 58A — Gemini adapter hardening against AFC warning.

Proves the Gemini reasoning provider explicitly disables automatic
function calling (AFC), preserves strict structured output, keeps
one-call semantics, and never introduces fake tools to silence the
SDK warning. Also exercises the failure-classification taxonomy.

Fixture pattern mirrors `tests/unit/test_reasoning_structured_output.py`
— an in-process fake Gemini `models` object that captures the kwargs
passed to `generate_content`, so no network / API key is required.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from app.brain.contracts import BrainInput
from app.core.constants import AgentAction, ConversationState, Intent, TopicCategory
from app.llm.gemini_provider import GeminiReasoningProvider
from app.llm.provider_factory import (
    UnsupportedLLMProviderError,
    create_reasoning_provider,
)
from app.llm.providers.reasoning_provider import ProviderFailureKind, ReasoningError
from app.llm.structured_output import ProposalParsingError


# ---------------------------------------------------------------------------
# Minimal fake SDK client
# ---------------------------------------------------------------------------


class _Models:
    def __init__(self, *, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = 0
        self.kwargs: dict = {}

    def generate_content(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.response


def _make_provider(models: _Models) -> GeminiReasoningProvider:
    return GeminiReasoningProvider(
        model="gemini-test",
        client=SimpleNamespace(models=models),
    )


def _valid_payload() -> dict:
    return {
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
    }


def _brain_input() -> BrainInput:
    return BrainInput(
        current_utterance="hello",
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
# AFC configuration invariants
# ---------------------------------------------------------------------------


class TestAfcDisabled:
    def _call_once(self):
        models = _Models(
            response=SimpleNamespace(text=json.dumps(_valid_payload())),
        )
        provider = _make_provider(models)
        proposal = provider.reason(_brain_input())
        return models, proposal

    def test_automatic_function_calling_explicitly_disabled(self):
        """§5, §6 — the config carries an explicit AFC-disable marker."""
        models, _ = self._call_once()
        cfg = models.kwargs["config"]
        afc = cfg.get("automatic_function_calling")
        assert afc is not None, "AFC config is missing — SDK will emit AFC warning"
        # Accept either a dict ({'disable': True}) or a typed SDK object
        # exposing a .disable attribute, since the SDK accepts both.
        disable = (
            afc.get("disable") if isinstance(afc, dict)
            else getattr(afc, "disable", None)
        )
        assert disable is True

    def test_structured_output_preserved(self):
        """§9 — strict JSON schema and MIME remain configured."""
        models, _ = self._call_once()
        cfg = models.kwargs["config"]
        assert cfg["response_mime_type"] == "application/json"
        assert cfg.get("response_json_schema") is not None

    def test_no_tools_configured(self):
        """§8 — no fake tool/function declaration smuggled in."""
        models, _ = self._call_once()
        cfg = models.kwargs["config"]
        assert "tools" not in cfg
        assert "tool_config" not in cfg

    def test_exactly_one_sdk_call_per_reason(self):
        """§10 — one invocation per `reason()`; no retries, no chat loop."""
        models, _ = self._call_once()
        assert models.calls == 1

    def test_model_passed_correctly(self):
        models, _ = self._call_once()
        assert models.kwargs["model"] == "gemini-test"

    def test_contents_contains_serialized_brain_input(self):
        """BrainInput is JSON-serialized into `contents` deterministically.

        Slice 58B.1 nested the input under `.input` and added an
        `.instruction` role preamble. Fields remain reachable.
        """
        models, _ = self._call_once()
        raw = models.kwargs["contents"]
        parsed = json.loads(raw)
        payload = parsed["input"]
        assert payload["current_utterance"] == "hello"
        assert payload["current_state"] == ConversationState.NEW_CALL.value

    def test_successful_response_parses_to_brain_proposal(self):
        _, proposal = self._call_once()
        assert proposal.detected_intent == Intent.INTERESTED
        assert proposal.proposed_action == AgentAction.GREET


class TestNoGlobalWarningSuppression:
    def test_provider_does_not_touch_global_logging_or_warnings(self):
        """§7 — explicit disable, never suppression."""
        src = inspect.getsource(
            __import__("app.llm.gemini_provider", fromlist=["_"]),
        )
        for forbidden in (
            "warnings.filterwarnings",
            "warnings.simplefilter",
            "logging.disable",
            "logging.getLogger(\"google_genai",
            "logging.getLogger('google_genai",
        ):
            assert forbidden not in src, (
                f"forbidden symptom patch {forbidden} in Gemini adapter"
            )


class TestNoChatMigration:
    def test_provider_does_not_use_chat_session(self):
        """§3 — AFC fix must not migrate to stateful Chat.send_message."""
        src = inspect.getsource(
            __import__("app.llm.gemini_provider", fromlist=["_"]),
        )
        for forbidden in (
            "chats.create",
            "chat.send_message",
            ".send_message(",
            ".send_message_stream(",
        ):
            assert forbidden not in src


# ---------------------------------------------------------------------------
# Failure-classification regression
# ---------------------------------------------------------------------------


class TestFailureClassification:
    def test_timeout_classified_as_timeout(self):
        models = _Models(error=TimeoutError("deadline"))
        provider = _make_provider(models)
        with pytest.raises(ReasoningError) as exc:
            provider.reason(_brain_input())
        assert exc.value.kind == ProviderFailureKind.TIMEOUT

    def test_auth_401_classified_as_auth_failure(self):
        class _Auth(Exception):
            status_code = 401
        models = _Models(error=_Auth("unauthorized"))
        provider = _make_provider(models)
        with pytest.raises(ReasoningError) as exc:
            provider.reason(_brain_input())
        assert exc.value.kind == ProviderFailureKind.AUTH_FAILURE

    def test_auth_403_classified_as_auth_failure(self):
        class _Forbidden(Exception):
            status_code = 403
        models = _Models(error=_Forbidden("forbidden"))
        provider = _make_provider(models)
        with pytest.raises(ReasoningError) as exc:
            provider.reason(_brain_input())
        assert exc.value.kind == ProviderFailureKind.AUTH_FAILURE

    def test_rate_limit_429_classified_as_rate_limited(self):
        class _RL(Exception):
            status_code = 429
        models = _Models(error=_RL("too many"))
        provider = _make_provider(models)
        with pytest.raises(ReasoningError) as exc:
            provider.reason(_brain_input())
        assert exc.value.kind == ProviderFailureKind.RATE_LIMITED

    def test_other_exception_falls_through_to_transport_failure(self):
        models = _Models(error=RuntimeError("some generic SDK boom"))
        provider = _make_provider(models)
        with pytest.raises(ReasoningError) as exc:
            provider.reason(_brain_input())
        assert exc.value.kind == ProviderFailureKind.TRANSPORT_FAILURE

    def test_missing_text_classified_as_invalid_response(self):
        models = _Models(response=SimpleNamespace())  # no .text attribute
        provider = _make_provider(models)
        with pytest.raises(ReasoningError) as exc:
            provider.reason(_brain_input())
        assert exc.value.kind == ProviderFailureKind.INVALID_RESPONSE

    def test_malformed_json_raises_proposal_parsing_error(self):
        """§33 — strict parse failure surfaces as ProposalParsingError."""
        models = _Models(response=SimpleNamespace(text="{not json"))
        provider = _make_provider(models)
        with pytest.raises(ProposalParsingError):
            provider.reason(_brain_input())


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


class _FakeSettings:
    def __init__(self, provider="gemini", api_key="test-key",
                 model="gemini-1.5-flash", timeout=30.0):
        self.llm_provider = provider
        self.gemini_api_key = api_key
        self.llm_model = model
        self.llm_timeout_seconds = timeout


class TestProviderFactory:
    def test_gemini_resolves(self, monkeypatch):
        """Avoid instantiating the real SDK — monkeypatch the Client
        constructor to a stub."""
        import google.genai as genai
        monkeypatch.setattr(
            genai, "Client",
            lambda **kwargs: SimpleNamespace(models=SimpleNamespace()),
        )
        p = create_reasoning_provider(_FakeSettings("gemini"))
        assert p.name == "gemini"

    def test_legacy_llm_gemini_resolves(self, monkeypatch):
        """Historical spelling still accepted (§12)."""
        import google.genai as genai
        monkeypatch.setattr(
            genai, "Client",
            lambda **kwargs: SimpleNamespace(models=SimpleNamespace()),
        )
        p = create_reasoning_provider(_FakeSettings("llm.gemini"))
        assert p.name == "gemini"

    def test_unsupported_provider_fails_clearly(self):
        with pytest.raises(UnsupportedLLMProviderError):
            create_reasoning_provider(_FakeSettings("openai"))

    def test_missing_api_key_fails_clearly(self):
        with pytest.raises(ValueError) as exc:
            create_reasoning_provider(_FakeSettings("gemini", api_key="   "))
        assert "GEMINI_API_KEY" in str(exc.value)

    def test_api_key_not_in_error_message(self):
        """§13 — never leak key value, even on failure."""
        key = "secret-key-12345"
        try:
            # Force a different failure path with a valid-looking key.
            with pytest.raises(UnsupportedLLMProviderError) as exc:
                create_reasoning_provider(_FakeSettings("openai", api_key=key))
            assert key not in str(exc.value)
        except AssertionError:
            raise
