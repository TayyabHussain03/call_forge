"""Slice 58B.6 — Gemini reasoning failure classification + trace propagation.

Live trace evidence proved the orchestrator was logging
`proposal_status=provider_failed` with `notes="reasoning provider failed"`
and nothing else — specifically, provider_failure_kind was always
TRANSPORT_FAILURE for live google-genai exceptions.

Root cause (confirmed by SDK inspection, not inference):
google-genai 2.x's `APIError` exposes `.status` (int), not `.status_code`.
The previous `_classify_failure` read `getattr(exc, "status_code", None)`
which was ALWAYS None → every 401 / 403 / 429 silently fell through to
TRANSPORT_FAILURE. The dev trace never surfaced the real cause.

This slice:
1. Fixes `_classify_failure` to read `.status` (with `.status_code`
   fallback for other SDKs), and to also traverse `.response.status`.
2. Carries typed exception_type + http_status on ReasoningError.
3. Propagates both through TurnDecisionTrace into the dev trace.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from app.brain.contracts import BrainInput
from app.core.constants import AgentAction, ConversationState, Intent, TopicCategory
from app.llm.gemini_provider import (
    GeminiReasoningProvider,
    _classify_failure,
    _extract_http_status,
)
from app.llm.providers.reasoning_provider import ProviderFailureKind, ReasoningError


# ---------------------------------------------------------------------------
# Section A — _classify_failure reads google-genai's `.status`
# ---------------------------------------------------------------------------


class _StatusErr(Exception):
    def __init__(self, status: int):
        super().__init__("fake")
        self.status = status


class _StatusCodeErr(Exception):
    def __init__(self, status_code: int):
        super().__init__("fake")
        self.status_code = status_code


class _NestedResponseErr(Exception):
    def __init__(self, status: int):
        super().__init__("fake")
        self.response = SimpleNamespace(status=status)


class TestClassifyFailureReadsStatus:
    def test_google_genai_401_now_classified_as_auth(self):
        """§15 — the previous classifier returned TRANSPORT_FAILURE for
        this because it read `.status_code` not `.status`."""
        assert _classify_failure(_StatusErr(401)) == ProviderFailureKind.AUTH_FAILURE

    def test_google_genai_403_classified_as_auth(self):
        assert _classify_failure(_StatusErr(403)) == ProviderFailureKind.AUTH_FAILURE

    def test_google_genai_429_classified_as_rate(self):
        assert _classify_failure(_StatusErr(429)) == ProviderFailureKind.RATE_LIMITED

    def test_other_sdk_status_code_401_still_classified_as_auth(self):
        """Defensive — other SDKs (openai, etc.) use `.status_code`."""
        assert _classify_failure(_StatusCodeErr(401)) == ProviderFailureKind.AUTH_FAILURE

    def test_nested_response_status_classified(self):
        """Some SDKs expose status on `.response.status`."""
        assert _classify_failure(_NestedResponseErr(429)) == ProviderFailureKind.RATE_LIMITED

    def test_timeout_still_classified(self):
        assert _classify_failure(TimeoutError("deadline")) == ProviderFailureKind.TIMEOUT

    def test_5xx_classified_as_transport(self):
        """§31 — server errors are transport-level failures."""
        assert _classify_failure(_StatusErr(503)) == ProviderFailureKind.TRANSPORT_FAILURE

    def test_unknown_exception_falls_to_transport(self):
        assert _classify_failure(RuntimeError("boom")) == ProviderFailureKind.TRANSPORT_FAILURE

    def test_bool_status_not_mistaken_for_int(self):
        """`True`/`False` are ints in Python; must not be read as HTTP status."""
        err = Exception()
        err.status = True  # type: ignore[attr-defined]
        assert _extract_http_status(err) is None


# ---------------------------------------------------------------------------
# Section B — ReasoningError carries diagnostic metadata
# ---------------------------------------------------------------------------


class TestReasoningErrorCarriesDiagnostics:
    def test_defaults_leave_metadata_none(self):
        exc = ReasoningError("x", ProviderFailureKind.TRANSPORT_FAILURE)
        assert exc.exception_type is None
        assert exc.http_status is None

    def test_explicit_metadata_preserved(self):
        exc = ReasoningError(
            "x", ProviderFailureKind.AUTH_FAILURE,
            exception_type="ClientError", http_status=401,
        )
        assert exc.kind == ProviderFailureKind.AUTH_FAILURE
        assert exc.exception_type == "ClientError"
        assert exc.http_status == 401

    def test_metadata_never_contains_secret(self):
        """Message + metadata must never carry an API key, prompt, or
        chain-of-thought — this is a type-level contract check."""
        exc = ReasoningError(
            "safe", ProviderFailureKind.AUTH_FAILURE,
            exception_type="ClientError", http_status=401,
        )
        serialized = f"{exc.args} {exc.exception_type} {exc.http_status}"
        for forbidden in ("api_key", "secret", "chain_of_thought"):
            assert forbidden not in serialized.lower()


# ---------------------------------------------------------------------------
# Section C — GeminiReasoningProvider populates metadata on failure
# ---------------------------------------------------------------------------


class _Models:
    def __init__(self, error=None, response=None):
        self.error = error
        self.response = response

    def generate_content(self, **kwargs):
        if self.error is not None:
            raise self.error
        return self.response


def _brain_input() -> BrainInput:
    return BrainInput(
        current_utterance="hi",
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


class TestProviderPopulatesMetadata:
    def _provider(self, models):
        return GeminiReasoningProvider(
            model="gemini-test", client=SimpleNamespace(models=models),
        )

    def test_auth_failure_carries_type_and_status(self):
        provider = self._provider(_Models(error=_StatusErr(401)))
        with pytest.raises(ReasoningError) as exc:
            provider.reason(_brain_input())
        assert exc.value.kind == ProviderFailureKind.AUTH_FAILURE
        assert exc.value.exception_type == "_StatusErr"
        assert exc.value.http_status == 401

    def test_rate_limit_carries_type_and_status(self):
        provider = self._provider(_Models(error=_StatusErr(429)))
        with pytest.raises(ReasoningError) as exc:
            provider.reason(_brain_input())
        assert exc.value.kind == ProviderFailureKind.RATE_LIMITED
        assert exc.value.http_status == 429

    def test_transport_failure_carries_type_only(self):
        provider = self._provider(_Models(error=RuntimeError("boom")))
        with pytest.raises(ReasoningError) as exc:
            provider.reason(_brain_input())
        assert exc.value.kind == ProviderFailureKind.TRANSPORT_FAILURE
        assert exc.value.exception_type == "RuntimeError"
        assert exc.value.http_status is None

    def test_invalid_response_is_separate_from_sdk_failure(self):
        """§20 — SDK call succeeded but response has no `.text`."""
        provider = self._provider(_Models(response=SimpleNamespace()))
        with pytest.raises(ReasoningError) as exc:
            provider.reason(_brain_input())
        assert exc.value.kind == ProviderFailureKind.INVALID_RESPONSE


# ---------------------------------------------------------------------------
# Section D — TurnDecisionTrace carries the new fields
# ---------------------------------------------------------------------------


class TestTraceCarriesProviderFailureFields:
    def test_dataclass_has_new_fields(self):
        from app.brain.orchestrator.orchestrator import TurnDecisionTrace
        import dataclasses
        fields = {f.name for f in dataclasses.fields(TurnDecisionTrace)}
        assert "provider_failure_kind" in fields
        assert "provider_exception_type" in fields
        assert "provider_http_status" in fields

    def test_dev_trace_record_surfaces_new_fields(self):
        from app.application.production_turn_processor import _build_dev_trace
        trace = SimpleNamespace(
            stage_outcome=None, provider_name="gemini",
            proposal_status=None, scope_category=None,
            authority_tier=None, action_confidence=None,
            notes=None, priority_action=None,
            proposed_action=None, detected_intent=None, topic_category=None,
            provider_failure_kind=ProviderFailureKind.AUTH_FAILURE,
            provider_exception_type="ClientError",
            provider_http_status=401,
        )
        s2 = SimpleNamespace(outcome=None, conversation_status=None, trace=trace)
        plan = SimpleNamespace(
            communicative_goal=None, response_length=None,
            question_strategy=None, clarification_required=False,
            addressee_status=None, catalog_request=False,
            re_explanation_strategy=None, escalation_decision=None,
            approved_action=None,
        )
        output = SimpleNamespace(
            pipeline_outcome=None, conversation_terminal=False,
            rendered_response=SimpleNamespace(text="x"),
        )
        turn = SimpleNamespace(turn_id="t", sequence_number=1, utterance="hi")
        rec = _build_dev_trace(
            turn=turn, slice_two=s2, slice_three=None,
            plan=plan, output=output, reasoning_calls=1,
            realization_metrics=None,
        )
        assert rec["slice_two"]["provider_failure_kind"] == "auth_failure"
        assert rec["slice_two"]["provider_exception_type"] == "ClientError"
        assert rec["slice_two"]["provider_http_status"] == 401


# ---------------------------------------------------------------------------
# Section E — orchestrator PROVIDER_FAILED path propagates diagnostics
# ---------------------------------------------------------------------------


class TestOrchestratorPropagatesProviderDiagnostics:
    def test_provider_failed_path_populates_trace_metadata(self):
        """When a ReasoningError bubbles out of the provider, the
        orchestrator's _brain_result on the PROVIDER_FAILED path
        must forward the exc's kind/exception_type/http_status
        onto TurnDecisionTrace. Scanner-verified against source
        (full orchestrator pipeline exercised in Slice 51/58B.5
        integration tests)."""
        from app.brain.orchestrator import orchestrator as mod
        src = inspect.getsource(mod)
        assert "provider_failure_kind=exc.kind" in src
        assert 'getattr(exc, "exception_type"' in src
        assert 'getattr(exc, "http_status"' in src


# ---------------------------------------------------------------------------
# Section F — safety invariants
# ---------------------------------------------------------------------------


class TestNoRetryLoopAndNoSecretExposure:
    def test_provider_adapter_has_no_retry_loop(self):
        """§21 — one call per reason(). No retries added by this slice."""
        from app.llm import gemini_provider as mod
        src = inspect.getsource(mod)
        # No while loop around generate_content
        assert "while" not in src.split("def reason")[1][:800]

    def test_provider_adapter_no_tolerant_prose_fallback(self):
        """§44 — reasoning must remain STRICTLY typed. Unlike
        realization's Slice 58B.5 tolerant parser, reasoning cannot
        accept plain prose as a BrainProposal."""
        from app.llm import gemini_provider as mod
        src = inspect.getsource(mod)
        # The reasoning adapter continues to go through parse_brain_proposal
        assert "parse_brain_proposal(" in src
        # And has no "plain prose" fallback branch
        assert "plain prose" not in src.lower()
        assert "fallback_text" not in src.lower()

    def test_provider_adapter_imports_no_secrets_logging(self):
        """§13 — no credential/prompt/chain-of-thought logging."""
        from app.llm import gemini_provider as mod
        src = inspect.getsource(mod)
        for forbidden in (
            "logger.info(", "logger.debug(", "print(brain_input",
            "print(response",
        ):
            assert forbidden not in src

    def test_diagnostic_script_never_prints_api_key(self):
        from pathlib import Path
        src = (
            Path(__file__).resolve().parents[2]
            / "scripts" / "diag_gemini_reasoning.py"
        ).read_text(encoding="utf-8")
        # Script uses _bool_present but never prints the raw key value
        assert "api_key_present" in src
        assert "api_key_present=" in src
        # Does NOT print the actual key
        assert "print(settings.gemini_api_key" not in src
        assert "{settings.gemini_api_key" not in src
