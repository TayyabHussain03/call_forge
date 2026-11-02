"""Slice 58B.8 — fallback semantic propagation + realization fidelity.

Live Groq browser evidence proved two independent defects:

(A) PLANNER_MAPPING — FallbackEngine selected ASK_IDENTITY for `salam`
    (hola → INTRODUCE_REASON), but ResponsePlanner's `_safety_plan`
    collapsed every FALLBACK into SAFE_RECOVERY +
    CLARIFY_CURRENT_INPUT, discarding the authoritative recovery
    action. Final text became "Could you clarify that for me?"
    even though the deterministic system had already chosen a valid
    non-terminal typed action.

(B) REALIZATION_PROMPT_FIDELITY — Groq's `hi` candidate
    "Hello! How can I assist you today?" asked a question even
    though `plan.question_strategy=none`. Validator correctly
    rejected; deterministic fallback produced greeting prose. The
    rejection is correct behavior; the slice tightens the realization
    prompt to minimise unnecessary rejection + fallback cycles.
"""

from __future__ import annotations

import inspect
import json

import pytest

from app.core.constants import AgentAction, ConversationState, Tone
from app.conversation.response_planning.contracts import (
    AcknowledgementKind, AddresseeStatus, AuthoritativeResultKind,
    ConversationMove, InterruptionCategory, InterruptionContext,
    QuestionStrategy, ResponseLength, ResponsePlan, ResponsePlanningInput,
)
from app.conversation.response_planning.planner import (
    ResponsePlanner, _is_non_terminal_recovery_action,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _planning_input(*, authoritative, approved_action=None):
    return ResponsePlanningInput(
        authoritative_state=ConversationState.GREETING,
        authoritative_result=authoritative,
        current_prospect_message="x",
        approved_action=approved_action,
        tone=Tone.NEUTRAL,
        conversation_category=InterruptionCategory.OTHER,
        addressee_status=AddresseeStatus.ADDRESSED_TO_AGENT,
        interruption=InterruptionContext(),
    )


# ---------------------------------------------------------------------------
# Section A — _is_non_terminal_recovery_action classifier
# ---------------------------------------------------------------------------


class TestNonTerminalRecoveryClassifier:
    def test_none_action_is_not_recovery(self):
        assert _is_non_terminal_recovery_action(None) is False

    def test_end_call_is_not_non_terminal_recovery(self):
        """§6 — END_CALL is explicitly terminal, not non-terminal recovery."""
        assert _is_non_terminal_recovery_action(AgentAction.END_CALL) is False

    def test_mark_dnc_is_not_non_terminal_recovery(self):
        """§6 — MARK_DNC remains deterministic terminal behavior."""
        assert _is_non_terminal_recovery_action(AgentAction.MARK_DNC) is False

    @pytest.mark.parametrize("action", [
        AgentAction.ASK_IDENTITY,
        AgentAction.INTRODUCE_REASON,
        AgentAction.ANSWER_QUESTION,
        AgentAction.ASK_EMAIL,
        AgentAction.CONFIRM_CONTACT,
        AgentAction.CLARIFY_CONTACT,
        AgentAction.SCHEDULE_CALLBACK,
        AgentAction.OFFER_SERVICE,
        AgentAction.GREET,
    ])
    def test_conversational_actions_are_non_terminal_recovery(self, action):
        assert _is_non_terminal_recovery_action(action) is True


# ---------------------------------------------------------------------------
# Section B — planner preserves typed fallback recovery
# ---------------------------------------------------------------------------


class TestPlannerPreservesFallbackAction:
    def test_fallback_with_ask_identity_does_not_produce_clarify(self):
        """§1, §4 — the salam case. FALLBACK + approved_action=ASK_IDENTITY
        must NOT yield SAFE_RECOVERY + CLARIFY_CURRENT_INPUT."""
        inp = _planning_input(
            authoritative=AuthoritativeResultKind.FALLBACK,
            approved_action=AgentAction.ASK_IDENTITY,
        )
        plan = ResponsePlanner().plan(inp)
        assert plan.communicative_goal != ConversationMove.SAFE_RECOVERY
        assert plan.question_strategy != QuestionStrategy.CLARIFY_CURRENT_INPUT
        assert plan.clarification_required is False
        # And the plan carries the approved action forward (Slice 58B.4
        # renderer dispatch key)
        assert plan.approved_action == AgentAction.ASK_IDENTITY

    def test_fallback_with_introduce_reason_does_not_produce_clarify(self):
        """§1, §11 — the hola case."""
        inp = _planning_input(
            authoritative=AuthoritativeResultKind.FALLBACK,
            approved_action=AgentAction.INTRODUCE_REASON,
        )
        plan = ResponsePlanner().plan(inp)
        assert plan.communicative_goal != ConversationMove.SAFE_RECOVERY
        assert plan.question_strategy == QuestionStrategy.NONE
        assert plan.clarification_required is False
        assert plan.approved_action == AgentAction.INTRODUCE_REASON

    def test_fallback_with_no_approved_action_still_falls_to_safe_recovery(self):
        """§7 — generic fallback stays generic when no action was selected."""
        inp = _planning_input(
            authoritative=AuthoritativeResultKind.FALLBACK,
            approved_action=None,
        )
        plan = ResponsePlanner().plan(inp)
        assert plan.communicative_goal == ConversationMove.SAFE_RECOVERY
        assert plan.question_strategy == QuestionStrategy.CLARIFY_CURRENT_INPUT
        assert plan.clarification_required is True

    def test_fallback_with_end_call_falls_through_to_generic(self):
        """§6 — terminal END_CALL doesn't take the non-terminal branch.
        (The processor's terminal-consistency invariant from Slice 58B.5
        then clears clarification on the actually-terminal output.)"""
        inp = _planning_input(
            authoritative=AuthoritativeResultKind.FALLBACK,
            approved_action=AgentAction.END_CALL,
        )
        plan = ResponsePlanner().plan(inp)
        # END_CALL takes the generic branch (safe_recovery) because
        # it's terminal and must not be realized as a conversation move
        assert plan.communicative_goal == ConversationMove.SAFE_RECOVERY

    def test_fallback_with_mark_dnc_falls_through_to_generic(self):
        inp = _planning_input(
            authoritative=AuthoritativeResultKind.FALLBACK,
            approved_action=AgentAction.MARK_DNC,
        )
        plan = ResponsePlanner().plan(inp)
        assert plan.communicative_goal == ConversationMove.SAFE_RECOVERY

    def test_non_fallback_results_unchanged(self):
        """REDIRECT and ESCALATE paths unchanged by this slice."""
        inp = _planning_input(
            authoritative=AuthoritativeResultKind.REDIRECT,
            approved_action=AgentAction.ASK_IDENTITY,
        )
        plan = ResponsePlanner().plan(inp)
        assert plan.communicative_goal == ConversationMove.REDIRECT_SAFELY


# ---------------------------------------------------------------------------
# Section C — fallback provenance remains observable
# ---------------------------------------------------------------------------


class TestFallbackProvenancePreserved:
    def test_authoritative_result_fallback_not_rewritten(self):
        """§3 — do not falsely pretend the primary action executed.
        The plan is action-specific but authoritative_result stays FALLBACK."""
        inp = _planning_input(
            authoritative=AuthoritativeResultKind.FALLBACK,
            approved_action=AgentAction.ASK_IDENTITY,
        )
        plan = ResponsePlanner().plan(inp)
        # Trace carries the original authoritative result; planner input is
        # the source of truth downstream. Scanner enforces the planner
        # doesn't mask it.
        import inspect as _inspect
        from app.conversation.response_planning import planner as pmod
        src = _inspect.getsource(pmod._safety_plan if hasattr(pmod, '_safety_plan') else ResponsePlanner._safety_plan)
        # The _safety_plan function does NOT mutate planning_input.authoritative_result
        assert "authoritative_result = " not in src.replace("authoritative_result = planning_input.authoritative_result", "")


# ---------------------------------------------------------------------------
# Section D — realization prompt forbids questions under NONE strategy
# ---------------------------------------------------------------------------


class TestRealizationPromptQuestionFidelity:
    def _instruction(self) -> str:
        from app.conversation.realization.contracts import (
            HumanConversationPolicy, LeanContextView, RealizationInput,
        )
        from app.conversation.realization.prompt import (
            build_llm_realization_prompt, serialize_llm_realization_prompt,
        )
        plan = ResponsePlan(
            ConversationMove.CONSULTATIVE_DISCOVERY,
            ResponseLength.SHORT,
            Tone.NEUTRAL,
            AcknowledgementKind.NONE,
            QuestionStrategy.NONE,
            False,
            __import__("app.conversation.response_planning.contracts", fromlist=["InterruptionHandling"]).InterruptionHandling.DROP_STALE_POINT,
            False,
            None,
            AddresseeStatus.ADDRESSED_TO_AGENT,
        )
        inp = RealizationInput(plan, HumanConversationPolicy(), LeanContextView("hi"))
        raw = serialize_llm_realization_prompt(build_llm_realization_prompt(inp))
        return json.loads(raw)["instruction"]

    def test_instruction_explicitly_forbids_questions_under_none_strategy(self):
        """§15 — hi + question_strategy=NONE must not get Groq to
        append 'How can I assist you today?'"""
        text = self._instruction().lower()
        assert "question_strategy" in text
        assert "'none'" in text or '"none"' in text or "is 'none'" in text or 'is "none"' in text
        # The prohibition is explicit
        assert "must not" in text or "do not" in text
        assert "no question" in text or "no clarifying question" in text or "not ask a question" in text

    def test_instruction_still_binds_to_plan(self):
        """Slice 58B.1 invariant preserved."""
        text = self._instruction().lower()
        assert "communicative_goal" in text
        assert "do not reconsider" in text


class TestValidatorStillRejectsQuestionWhenNoneStrategy:
    """§14 — the validator must continue to reject the symptom even if
    the prompt is strengthened. Defense in depth."""

    def test_validator_rejects_question_candidate_under_none_strategy(self):
        from app.conversation.realization.contracts import (
            HumanConversationPolicy, LeanContextView, RealizationInput,
        )
        from app.conversation.realization.validator import (
            RealizationValidationError, validate_realization,
        )
        plan = ResponsePlan(
            ConversationMove.CONSULTATIVE_DISCOVERY,
            ResponseLength.SHORT,
            Tone.NEUTRAL,
            AcknowledgementKind.NONE,
            QuestionStrategy.NONE,
            False,
            __import__("app.conversation.response_planning.contracts", fromlist=["InterruptionHandling"]).InterruptionHandling.DROP_STALE_POINT,
            False,
            None,
            AddresseeStatus.ADDRESSED_TO_AGENT,
        )
        value = RealizationInput(plan, HumanConversationPolicy(), LeanContextView("hi"))
        # Candidate contains a question. Validator must reject.
        with pytest.raises(RealizationValidationError):
            validate_realization("Hello! How can I assist you today?", value)


# ---------------------------------------------------------------------------
# Section E — renderer realizes typed fallback action
# ---------------------------------------------------------------------------


class TestRendererRealizesFallbackAction:
    """§4 — the renderer's Slice 58B.4 `_typed_action_fallback` already
    dispatches on `plan.approved_action` to emit action-specific prose.
    Combined with the planner fix, FALLBACK + ASK_IDENTITY now produces
    an ASK_IDENTITY-shaped response, not 'Could you clarify that for me?'."""

    def _render(self, approved_action):
        from app.conversation.response_rendering.contracts import (
            ResponseRenderInput, ResponseRenderingBudget,
            TrustedRenderingContext,
        )
        from app.conversation.response_rendering.renderer import (
            DeterministicResponseRenderer,
        )
        from app.conversation.response_planning.contracts import (
            InterruptionHandling,
        )
        plan = ResponsePlan(
            ConversationMove.COMMUNICATE_RESULT,
            ResponseLength.SHORT,
            Tone.NEUTRAL,
            AcknowledgementKind.NONE,
            QuestionStrategy.NONE,
            False,
            InterruptionHandling.HOLD_PRIOR_CONTEXT,
            False,
            None,
            AddresseeStatus.ADDRESSED_TO_AGENT,
            approved_action=approved_action,
        )
        inp = ResponseRenderInput(
            plan,
            authoritative_result=AuthoritativeResultKind.FALLBACK,
            budget=ResponseRenderingBudget(5),
            trusted_context=TrustedRenderingContext(),
            current_prospect_message="salam",
        )
        return DeterministicResponseRenderer().render(inp)

    def test_ask_identity_fallback_asks_who(self):
        out = self._render(AgentAction.ASK_IDENTITY)
        lower = out.text.lower()
        assert "could you clarify that for me" not in lower
        assert "who" in lower or "confirm" in lower

    def test_introduce_reason_fallback_introduces(self):
        out = self._render(AgentAction.INTRODUCE_REASON)
        lower = out.text.lower()
        assert "could you clarify that for me" not in lower
        assert "reaching out" in lower or "about" in lower
