"""Slice 58B.5 — realization parser resilience + non-terminal fallback.

Two independent defects proven by live traces:

(A) Realization MALFORMED_RESPONSE
    Live trace: `provider_failure_kind: "malformed_response"`,
    `candidate_present: false`. Root cause: `_parse_response` required
    `set(payload) == {"text"}`, rejecting responses with extra envelope
    keys and leaving no graceful path for plain-prose Gemini output.

(B) Unsafe terminal END_CALL fallback
    Live trace for `hola`:
        slice_three.approved_action = end_call
        slice_three.fallback_used   = true
        is_terminal                 = true
        plan.clarification_required = true
    Root cause: the second greeting hit `greeting` state where GREET
    is not allowed; ActionValidator rejected; FallbackEngine fell to
    `default: { action: end_call }` because `greeting` had no
    state-specific fallback entry.

Both fixed in this slice.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Section A — resilient realization parser
# ---------------------------------------------------------------------------


class TestRealizationParserResilience:
    def _parse(self, raw):
        from app.conversation.realization.adapter import _parse_response
        return _parse_response(SimpleNamespace(text=raw))

    def test_strict_schema_response_parses(self):
        r = self._parse(json.dumps({"text": "Hi!"}))
        assert r.text == "Hi!"

    def test_response_with_extra_envelope_keys_parses(self):
        """Tolerant: extra fields are ignored, text is extracted."""
        r = self._parse(json.dumps({
            "text": "Hello there.",
            "metadata": {"candidate_count": 1},
            "model_version": "2.5",
        }))
        assert r.text == "Hello there."

    def test_plain_prose_response_accepted_when_not_json(self):
        """§5 — realization is wording. If the model partially drops
        JSON mode and returns raw text, use it as the realized text."""
        r = self._parse("Hi, thanks for taking the call.")
        assert r.text == "Hi, thanks for taking the call."

    def test_empty_text_rejected_as_malformed(self):
        from app.conversation.realization.provider import (
            RealizationError, RealizationFailureKind,
        )
        with pytest.raises(RealizationError) as exc:
            self._parse("")
        assert exc.value.kind == RealizationFailureKind.MALFORMED_RESPONSE

    def test_none_text_rejected_as_malformed(self):
        from app.conversation.realization.provider import (
            RealizationError, RealizationFailureKind,
        )
        with pytest.raises(RealizationError) as exc:
            self._parse(None)
        assert exc.value.kind == RealizationFailureKind.MALFORMED_RESPONSE

    def test_json_dict_without_text_key_rejected(self):
        from app.conversation.realization.provider import (
            RealizationError, RealizationFailureKind,
        )
        with pytest.raises(RealizationError) as exc:
            self._parse(json.dumps({"message": "Hi"}))
        assert exc.value.kind == RealizationFailureKind.MALFORMED_RESPONSE

    def test_json_list_rejected(self):
        from app.conversation.realization.provider import (
            RealizationError, RealizationFailureKind,
        )
        with pytest.raises(RealizationError) as exc:
            self._parse("[1, 2, 3]")
        assert exc.value.kind == RealizationFailureKind.MALFORMED_RESPONSE

    def test_oversized_text_bounded(self):
        """Candidate text is bounded; downstream validator enforces more."""
        r = self._parse(json.dumps({"text": "x" * 5000}))
        assert len(r.text) <= 2000

    def test_text_is_none_in_json_rejected(self):
        from app.conversation.realization.provider import (
            RealizationError, RealizationFailureKind,
        )
        with pytest.raises(RealizationError) as exc:
            self._parse(json.dumps({"text": None}))
        assert exc.value.kind == RealizationFailureKind.MALFORMED_RESPONSE

    def test_bytes_response_parses(self):
        r = self._parse(b'{"text": "Hi!"}')
        assert r.text == "Hi!"


# ---------------------------------------------------------------------------
# Section B — non-terminal default fallback for conversational states
# ---------------------------------------------------------------------------


class TestNonTerminalFallbackForGreetingStates:
    def _load_config(self):
        from app.conversation.state_machine.states import load_config
        from app.config.settings import get_settings
        return load_config(get_settings().conversation_config_path)

    def test_greeting_state_has_non_terminal_fallback(self):
        """§15 — second greeting from `greeting` state must NOT be
        silently terminated by the fallback engine."""
        config = self._load_config()
        from app.core.constants import ConversationState, AgentAction
        mapping = config.fallbacks.by_state.get(ConversationState.GREETING)
        assert mapping is not None, "greeting must have an explicit fallback"
        # The chosen action must not be END_CALL or MARK_DNC
        assert mapping.action not in (AgentAction.END_CALL, AgentAction.MARK_DNC)

    def test_identify_person_has_non_terminal_fallback(self):
        config = self._load_config()
        from app.core.constants import ConversationState, AgentAction
        mapping = config.fallbacks.by_state.get(ConversationState.IDENTIFY_PERSON)
        assert mapping is not None
        assert mapping.action not in (AgentAction.END_CALL, AgentAction.MARK_DNC)

    def test_reason_for_call_has_non_terminal_fallback(self):
        config = self._load_config()
        from app.core.constants import ConversationState, AgentAction
        mapping = config.fallbacks.by_state.get(ConversationState.REASON_FOR_CALL)
        assert mapping is not None
        assert mapping.action not in (AgentAction.END_CALL, AgentAction.MARK_DNC)

    def test_listen_has_non_terminal_fallback(self):
        config = self._load_config()
        from app.core.constants import ConversationState, AgentAction
        mapping = config.fallbacks.by_state.get(ConversationState.LISTEN)
        assert mapping is not None
        assert mapping.action not in (AgentAction.END_CALL, AgentAction.MARK_DNC)

    def test_fallback_select_returns_non_terminal_for_greeting(self):
        """End-to-end: FallbackEngine.select from `greeting` state does
        NOT return a terminal decision for an ordinary rejection."""
        from app.conversation.guardrails.fallbacks import FallbackEngine
        from app.contracts.validation import ValidationCategory, ValidationResult
        from app.core.constants import ConversationState, AgentAction
        config = self._load_config()
        engine = FallbackEngine(config)
        # An ordinary rejection (not dnc_conflict, not unknown_requirement)
        rejection = ValidationResult(
            allowed=False,
            category=ValidationCategory.NOT_ELIGIBLE,
            reason="greet not allowed from greeting",
        )
        decision = engine.select(ConversationState.GREETING, rejection)
        assert decision.is_terminal is False
        assert decision.action != AgentAction.END_CALL


# ---------------------------------------------------------------------------
# Section C — DNC + explicit terminal preserved
# ---------------------------------------------------------------------------


class TestTerminalRegression:
    def _load_config(self):
        from app.conversation.state_machine.states import load_config
        from app.config.settings import get_settings
        return load_config(get_settings().conversation_config_path)

    def test_dnc_conflict_category_still_terminal(self):
        """§22 — DNC conflict must remain terminal with mark_dnc."""
        config = self._load_config()
        from app.core.constants import AgentAction, ConversationState
        from app.conversation.guardrails.fallbacks import FallbackEngine
        from app.contracts.validation import ValidationCategory, ValidationResult
        engine = FallbackEngine(config)
        result = ValidationResult(
            allowed=False,
            category=ValidationCategory.DNC_CONFLICT,
            reason="dnc flag",
        )
        decision = engine.select(ConversationState.LISTEN, result)
        assert decision.is_terminal is True
        assert decision.action == AgentAction.MARK_DNC

    def test_unknown_requirement_still_terminal(self):
        """§21 — unknown_requirement must still safe-close terminally."""
        config = self._load_config()
        from app.core.constants import AgentAction, ConversationState
        from app.conversation.guardrails.fallbacks import FallbackEngine
        from app.contracts.validation import ValidationCategory, ValidationResult
        engine = FallbackEngine(config)
        result = ValidationResult(
            allowed=False,
            category=ValidationCategory.UNKNOWN_REQUIREMENT,
            reason="unknown",
        )
        decision = engine.select(ConversationState.LISTEN, result)
        assert decision.is_terminal is True

    def test_terminal_safe_path_still_terminal(self):
        """Last-resort terminal_safe is unchanged."""
        config = self._load_config()
        from app.conversation.guardrails.fallbacks import FallbackEngine
        from app.core.constants import AgentAction
        engine = FallbackEngine(config)
        decision = engine.terminal_safe(reason="test")
        assert decision.is_terminal is True
        assert decision.action == AgentAction.END_CALL


# ---------------------------------------------------------------------------
# Section D — terminal/clarification consistency invariant (§25)
# ---------------------------------------------------------------------------


class TestTerminalClarificationInvariant:
    def test_processor_clears_clarification_on_terminal(self):
        """Scanner enforces: the processor downgrades
        plan.clarification_required=True to False when
        terminal=True so renderer cannot ask the user to clarify
        something after the session has ended."""
        from app.application import production_turn_processor as mod
        src = inspect.getsource(mod)
        # The invariant block is present
        assert "terminal and (" in src
        assert "plan.clarification_required" in src
        assert "QuestionStrategy.NONE" in src
        assert "clarification_required=False" in src


# ---------------------------------------------------------------------------
# Section E — typed BrainProposal fields on the trace
# ---------------------------------------------------------------------------


class TestTraceCarriesProposalFields:
    def test_trace_dataclass_has_proposed_action(self):
        from app.brain.orchestrator.orchestrator import TurnDecisionTrace
        import dataclasses
        fields = {f.name for f in dataclasses.fields(TurnDecisionTrace)}
        assert "proposed_action" in fields
        assert "detected_intent" in fields
        assert "topic_category" in fields

    def test_dev_trace_record_surfaces_proposal_fields(self):
        from app.application.production_turn_processor import _build_dev_trace
        from types import SimpleNamespace
        from app.core.constants import AgentAction, Intent, TopicCategory
        trace = SimpleNamespace(
            stage_outcome=None, provider_name="gemini", proposal_status=None,
            scope_category=None, authority_tier=None, action_confidence=0.9,
            notes=None, priority_action=None,
            proposed_action=AgentAction.GREET,
            detected_intent=Intent.INTERESTED,
            topic_category=TopicCategory.QUALIFICATION,
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
            rendered_response=SimpleNamespace(text="Hi"),
        )
        turn = SimpleNamespace(turn_id="t", sequence_number=1, utterance="hi")
        rec = _build_dev_trace(
            turn=turn, slice_two=s2, slice_three=None,
            plan=plan, output=output, reasoning_calls=1,
            realization_metrics=None,
        )
        assert rec["slice_two"]["proposed_action"] == "greet"
        assert rec["slice_two"]["detected_intent"] == "interested"
        assert rec["slice_two"]["topic_category"] == "qualification"
