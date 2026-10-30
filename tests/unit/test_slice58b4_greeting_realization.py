"""Slice 58B.4 — greeting realization-layer repair.

The user's live trace proved reasoning / scope / authority / execution
are all correct for `hi`:

    slice_two.outcome        = authority_approved
    slice_two.scope_category = in_scope
    slice_two.authority_tier = auto_execute
    slice_three.approved_action = greet
    slice_three.outcome      = executed
    slice_three.is_terminal  = false
    plan.clarification_required = false
    conversation_terminal    = false

But the final rendered text was "How would you like to continue?" —
the DeterministicResponseRenderer's generic COMMUNICATE_RESULT else
branch. Something in the realization layer either rejected Gemini's
candidate or Gemini produced an unacceptable wording; either way,
the deterministic fallback discarded the approved GREET semantics.

This slice ships three bounded changes:

    1. ResponsePlan carries the ActionValidator-approved action
       forward so the deterministic renderer can offer typed
       action-specific fallbacks (owner: planner contract).
    2. DeterministicResponseRenderer's generic else branch realizes
       GREET as a greeting (and ASK_IDENTITY / INTRODUCE_REASON
       similarly) instead of degrading to "How would you like to
       continue?" (owner: renderer).
    3. GuardedConversationRealizer exposes typed `last_metrics` per
       render so the dev trace can distinguish "live realization
       accepted" from "live realization rejected → deterministic
       fallback" (owner: realizer + trace builder).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.conversation.realization.contracts import (
    HumanConversationPolicy, LeanContextView, RealizationInput,
)
from app.conversation.realization.provider import (
    MockConversationRealizationProvider, RealizationError,
    RealizationFailureKind,
)
from app.conversation.realization.realizer import GuardedConversationRealizer
from app.conversation.response_planning.contracts import (
    AcknowledgementKind, AddresseeStatus, AuthoritativeResultKind,
    ConversationMove, InterruptionHandling, QuestionStrategy,
    ResponseLength, ResponsePlan, Tone,
)
from app.conversation.response_rendering.contracts import (
    ResponseRenderInput, ResponseRenderingBudget, TrustedRenderingContext,
)
from app.conversation.response_rendering.renderer import (
    DeterministicResponseRenderer,
)
from app.core.constants import AgentAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plan(*, approved_action=None, goal=ConversationMove.COMMUNICATE_RESULT,
          clarification=False, addressee=AddresseeStatus.ADDRESSED_TO_AGENT):
    return ResponsePlan(
        goal,
        ResponseLength.SHORT,
        Tone.NEUTRAL,
        AcknowledgementKind.NONE,
        QuestionStrategy.NONE,
        clarification,
        InterruptionHandling.NONE,
        False,
        None,
        addressee,
        approved_action=approved_action,
    )


def _render_input(plan, *, authoritative=AuthoritativeResultKind.EXECUTED):
    return ResponseRenderInput(
        plan,
        authoritative_result=authoritative,
        budget=ResponseRenderingBudget(5),
        trusted_context=TrustedRenderingContext(),
        current_prospect_message="hi",
    )


# ---------------------------------------------------------------------------
# Section A — ResponsePlan contract additive
# ---------------------------------------------------------------------------


class TestResponsePlanCarriesApprovedAction:
    def test_default_is_none(self):
        p = _plan()
        assert getattr(p, "approved_action", "MISSING") is None

    def test_explicit_greet_preserved(self):
        p = _plan(approved_action=AgentAction.GREET)
        assert p.approved_action == AgentAction.GREET

    def test_planner_populates_from_input(self):
        """Trace planner._build → ResponsePlan carries approved_action."""
        import inspect
        from app.conversation.response_planning import planner as planner_mod
        src = inspect.getsource(planner_mod)
        assert "approved_action=planning_input.approved_action" in src


# ---------------------------------------------------------------------------
# Section B — Renderer: typed action-specific fallback
# ---------------------------------------------------------------------------


class TestDeterministicGreetFallback:
    def test_greet_fallback_actually_greets(self):
        """§28 — approved GREET must still greet via deterministic
        fallback. 'How would you like to continue?' is unacceptable."""
        renderer = DeterministicResponseRenderer()
        plan = _plan(approved_action=AgentAction.GREET)
        out = renderer.render(_render_input(plan))
        text = out.text.lower()
        assert "how would you like to continue" not in text
        # Natural greeting indicator
        assert any(word in text for word in ("hi", "hello", "thanks for"))

    def test_ask_identity_fallback_asks_who(self):
        renderer = DeterministicResponseRenderer()
        plan = _plan(approved_action=AgentAction.ASK_IDENTITY)
        out = renderer.render(_render_input(plan))
        assert "how would you like to continue" not in out.text.lower()
        assert "who" in out.text.lower() or "confirm" in out.text.lower()

    def test_introduce_reason_fallback_introduces(self):
        renderer = DeterministicResponseRenderer()
        plan = _plan(approved_action=AgentAction.INTRODUCE_REASON)
        out = renderer.render(_render_input(plan))
        assert "how would you like to continue" not in out.text.lower()
        assert "reaching out" in out.text.lower() or "about" in out.text.lower()

    def test_unknown_approved_action_falls_through_to_historical_prompt(self):
        """Historical behavior preserved for actions we don't yet type."""
        renderer = DeterministicResponseRenderer()
        plan = _plan(approved_action=AgentAction.ANSWER_QUESTION)
        out = renderer.render(_render_input(plan))
        # ANSWER_QUESTION isn't in our fallback map; historical fallback OK
        assert out.text


# ---------------------------------------------------------------------------
# Section C — GuardedConversationRealizer typed metrics
# ---------------------------------------------------------------------------


class TestRealizerLastMetrics:
    def _render(self, provider):
        realizer = GuardedConversationRealizer(provider, DeterministicResponseRenderer())
        plan = _plan(approved_action=AgentAction.GREET)
        realizer.render(_render_input(plan))
        return realizer.last_metrics

    def test_accepted_candidate_marks_live_realization_used(self):
        # Use the Mock provider's default "Okay." which the current
        # validator accepts under COMMUNICATE_RESULT + no evidence
        # requirement. Point of the test is metrics structure, not
        # validator behavior.
        provider = MockConversationRealizationProvider()
        m = self._render(provider)
        assert m["eligible"] is True
        assert m["called"] is True
        assert m["candidate_present"] is True
        assert m["candidate_text"] == "Okay."
        assert m["validation_status"] == "accepted"
        assert m["deterministic_fallback_used"] is False

    def test_provider_realization_error_marks_provider_failed(self):
        class _Fail(MockConversationRealizationProvider):
            def realize(self, value):
                self.call_count += 1
                raise RealizationError("boom", RealizationFailureKind.TIMEOUT)
        m = self._render(_Fail())
        assert m["eligible"] is True
        assert m["called"] is True
        assert m["candidate_present"] is False
        assert m["validation_status"] == "provider_failed"
        assert m["provider_failure_kind"] == "timeout"
        assert m["deterministic_fallback_used"] is True

    def test_candidate_rejected_by_validator_falls_back(self):
        """Candidate text that violates validator → rejected + fallback."""
        # Validator rejects empty / overlong / command-shaped candidates.
        # A None or non-string candidate is one such path.
        class _BadType(MockConversationRealizationProvider):
            def realize(self, value):
                self.call_count += 1
                return 12345  # not a string → ValueError in validator
        m = self._render(_BadType())
        assert m["validation_status"] == "rejected"
        assert m["rejection_reason"] is not None
        assert m["deterministic_fallback_used"] is True

    def test_not_eligible_short_circuits(self):
        """authoritative_result != EXECUTED → not called."""
        provider = MockConversationRealizationProvider(response="Hi.")
        realizer = GuardedConversationRealizer(provider, DeterministicResponseRenderer())
        plan = _plan(approved_action=AgentAction.GREET)
        realizer.render(_render_input(plan, authoritative=AuthoritativeResultKind.FALLBACK))
        m = realizer.last_metrics
        assert m["eligible"] is False
        assert m["called"] is False
        assert m["validation_status"] == "not_called"
        assert m["deterministic_fallback_used"] is True
        assert m["rejection_reason"] == "not_eligible"

    def test_metrics_contain_no_credentials(self):
        import json
        provider = MockConversationRealizationProvider()
        m = self._render(provider)
        serialized = json.dumps(m).lower()
        for forbidden in ("api_key", "secret", "chain_of_thought", "traceback"):
            assert forbidden not in serialized


# ---------------------------------------------------------------------------
# Section D — Trace builder includes realization block
# ---------------------------------------------------------------------------


class TestTraceCarriesRealizationBlock:
    def _call_build(self, metrics=None):
        from app.application.production_turn_processor import _build_dev_trace
        turn = SimpleNamespace(turn_id="t", sequence_number=1, utterance="hi")
        trace = SimpleNamespace(
            stage_outcome=None, provider_name="gemini", proposal_status=None,
            scope_category=None, authority_tier=None, action_confidence=None,
            notes=None, priority_action=None,
        )
        s2 = SimpleNamespace(outcome=None, conversation_status=None, trace=trace)
        plan = _plan(approved_action=AgentAction.GREET)
        output = SimpleNamespace(
            pipeline_outcome=None, conversation_terminal=False,
            rendered_response=SimpleNamespace(text="Hi!"),
        )
        return _build_dev_trace(
            turn=turn, slice_two=s2, slice_three=None,
            plan=plan, output=output, reasoning_calls=1,
            realization_metrics=metrics,
        )

    def test_record_includes_realization_key(self):
        rec = self._call_build(metrics={
            "eligible": True, "called": True,
            "candidate_present": True, "candidate_text": "Hi!",
            "validation_status": "accepted", "rejection_reason": None,
            "provider_failure_kind": None, "deterministic_fallback_used": False,
        })
        assert "realization" in rec
        assert rec["realization"]["validation_status"] == "accepted"
        assert rec["realization"]["deterministic_fallback_used"] is False

    def test_record_handles_none_realization(self):
        """Composition without a realizer still produces a record."""
        rec = self._call_build(metrics=None)
        assert "realization" in rec
        assert rec["realization"] is None


class TestExtractRealizationMetrics:
    def test_direct_realizer_metrics(self):
        from app.application.production_turn_processor import _extract_realization_metrics
        realizer = GuardedConversationRealizer(
            MockConversationRealizationProvider(),
            DeterministicResponseRenderer(),
        )
        realizer.render(_render_input(_plan(approved_action=AgentAction.GREET)))
        m = _extract_realization_metrics(realizer)
        assert isinstance(m, dict)
        assert m["validation_status"] == "accepted"

    def test_none_for_renderer_without_metrics(self):
        from app.application.production_turn_processor import _extract_realization_metrics
        assert _extract_realization_metrics(DeterministicResponseRenderer()) is None

    def test_none_input(self):
        from app.application.production_turn_processor import _extract_realization_metrics
        assert _extract_realization_metrics(None) is None


# ---------------------------------------------------------------------------
# Section E — Prior invariants preserved
# ---------------------------------------------------------------------------


class TestPriorInvariantsPreserved:
    def test_action_validation_contract_unchanged(self):
        """ResponsePlan still validates its old fields."""
        p = _plan(approved_action=AgentAction.GREET)
        assert p.communicative_goal == ConversationMove.COMMUNICATE_RESULT

    def test_renderer_legacy_fallback_still_available(self):
        """ANSWER_CURRENT_QUESTION branch unchanged by this slice."""
        import inspect
        from app.conversation.response_rendering import renderer as r
        src = inspect.getsource(r)
        assert "ConversationMove.ANSWER_CURRENT_QUESTION" in src

    def test_realization_adapter_afc_disable_preserved(self):
        import inspect
        from app.conversation.realization import adapter
        src = inspect.getsource(adapter)
        assert '"automatic_function_calling": {"disable": True}' in src
