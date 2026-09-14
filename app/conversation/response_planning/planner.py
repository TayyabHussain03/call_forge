"""Deterministic post-decision response planner."""

from __future__ import annotations

from app.conversation.response_planning.contracts import (
    AddresseeStatus,
    AcknowledgementKind,
    AuthoritativeResultKind,
    ConversationMove,
    ExplanationNeed,
    InterruptionCategory,
    InterruptionHandling,
    PendingConversationIntent,
    QuestionStrategy,
    ResponseLength,
    ResponsePlan,
    ResponsePlanningInput,
)


class ResponsePlanner:
    """Choose conversational strategy without changing authoritative state."""

    _ACKNOWLEDGEMENTS = (
        AcknowledgementKind.GOT_IT,
        AcknowledgementKind.MAKES_SENSE,
        AcknowledgementKind.OKAY,
        AcknowledgementKind.RIGHT,
        AcknowledgementKind.UNDERSTOOD,
    )

    def plan(self, planning_input: ResponsePlanningInput) -> ResponsePlan:
        """Return a pure deterministic communication plan."""
        pending = _active_pending(planning_input.interruption.previous_intent)

        if planning_input.addressee_status == AddresseeStatus.ADDRESSEE_UNCERTAIN:
            return self._build(
                planning_input,
                ConversationMove.CLARIFY_ADDRESSEE,
                ResponseLength.SHORT,
                QuestionStrategy.CONFIRM_ADDRESSEE,
                clarification=True,
                handling=InterruptionHandling.HOLD_PRIOR_CONTEXT,
                pending=pending,
            )

        if planning_input.addressee_status == AddresseeStatus.NOT_ADDRESSED_TO_AGENT:
            return self._build(
                planning_input,
                ConversationMove.CONTINUE_PRIOR_CONTEXT,
                ResponseLength.SHORT,
                QuestionStrategy.NONE,
                handling=InterruptionHandling.HOLD_PRIOR_CONTEXT,
                resume=pending is not None,
                pending=pending,
            )

        escalation_plan = self._escalation_plan(planning_input, pending)
        if escalation_plan is not None:
            return escalation_plan

        safety_plan = self._safety_plan(planning_input, pending)
        if safety_plan is not None:
            return safety_plan

        category = planning_input.conversation_category
        interrupted = planning_input.interruption.was_interrupted
        if category == InterruptionCategory.QUESTION:
            return self._build(
                planning_input,
                ConversationMove.ANSWER_CURRENT_QUESTION,
                _explanatory_length(planning_input.explanation_need),
                QuestionStrategy.ANSWER_THEN_FOLLOW_UP,
                handling=(
                    InterruptionHandling.INTEGRATE_IF_USEFUL
                    if interrupted and pending is not None
                    else InterruptionHandling.NONE
                ),
                resume=interrupted and pending is not None,
                pending=pending,
            )
        if category == InterruptionCategory.OBJECTION:
            return self._build(
                planning_input,
                ConversationMove.EXPLORE_OBJECTION,
                _explanatory_length(planning_input.explanation_need),
                QuestionStrategy.EXPLORE_WITH_ONE_QUESTION,
                acknowledgement=self._next_acknowledgement(planning_input),
                handling=InterruptionHandling.DROP_STALE_POINT,
            )
        if category == InterruptionCategory.CORRECTION:
            return self._build(
                planning_input,
                ConversationMove.INCORPORATE_CORRECTION,
                ResponseLength.MODERATE,
                QuestionStrategy.NONE,
                acknowledgement=self._next_acknowledgement(planning_input),
                handling=InterruptionHandling.DROP_STALE_POINT,
            )
        if category == InterruptionCategory.TOPIC_SHIFT:
            return self._build(
                planning_input,
                ConversationMove.FOLLOW_NEW_DIRECTION,
                _ordinary_length(planning_input.explanation_need),
                QuestionStrategy.NONE,
                handling=InterruptionHandling.DROP_STALE_POINT,
            )
        if category == InterruptionCategory.CLARIFICATION:
            return self._build(
                planning_input,
                ConversationMove.CLARIFY_MEANING,
                _ordinary_length(planning_input.explanation_need),
                QuestionStrategy.CLARIFY_CURRENT_INPUT,
                clarification=True,
                handling=InterruptionHandling.DROP_STALE_POINT,
            )

        return self._build(
            planning_input,
            ConversationMove.COMMUNICATE_RESULT,
            _ordinary_length(planning_input.explanation_need),
            QuestionStrategy.NONE,
            handling=(
                InterruptionHandling.INTEGRATE_IF_USEFUL
                if interrupted and pending is not None
                else InterruptionHandling.NONE
            ),
            resume=interrupted and pending is not None,
            pending=pending,
        )

    def _safety_plan(
        self,
        planning_input: ResponsePlanningInput,
        pending: PendingConversationIntent | None,
    ) -> ResponsePlan | None:
        result = planning_input.authoritative_result
        if result == AuthoritativeResultKind.FALLBACK:
            return self._build(
                planning_input,
                ConversationMove.SAFE_RECOVERY,
                ResponseLength.SHORT,
                QuestionStrategy.CLARIFY_CURRENT_INPUT,
                clarification=True,
                handling=InterruptionHandling.HOLD_PRIOR_CONTEXT,
                pending=pending,
            )
        if result == AuthoritativeResultKind.REDIRECT:
            return self._build(
                planning_input,
                ConversationMove.REDIRECT_SAFELY,
                ResponseLength.SHORT,
                QuestionStrategy.NONE,
                handling=InterruptionHandling.DROP_STALE_POINT,
            )
        if result == AuthoritativeResultKind.ESCALATE:
            return self._build(
                planning_input,
                ConversationMove.ACKNOWLEDGE_ESCALATION,
                ResponseLength.SHORT,
                QuestionStrategy.NONE,
                acknowledgement=self._next_acknowledgement(planning_input),
                handling=InterruptionHandling.HOLD_PRIOR_CONTEXT,
                pending=pending,
            )
        return None

    def _escalation_plan(
        self,
        planning_input: ResponsePlanningInput,
        pending: PendingConversationIntent | None,
    ) -> ResponsePlan | None:
        from app.conversation.escalation.contracts import (
            EscalationCapability,
            RecoveryMode,
        )

        decision = planning_input.escalation_decision
        if decision is None or decision.recovery_mode == RecoveryMode.PROCEED_NORMALLY:
            return None
        common = {
            "handling": (
                InterruptionHandling.INTEGRATE_IF_USEFUL
                if decision.resume_previous_goal and pending is not None
                else InterruptionHandling.HOLD_PRIOR_CONTEXT
            ),
            "resume": decision.resume_previous_goal and pending is not None,
            "pending": pending,
        }
        if decision.recovery_mode == RecoveryMode.ANSWER_WITH_EVIDENCE:
            return self._build(
                planning_input,
                ConversationMove.ANSWER_APPROVED_EVIDENCE,
                ResponseLength.MODERATE,
                QuestionStrategy.NONE,
                **common,
            )
        if decision.recovery_mode == RecoveryMode.ASK_CLARIFYING_QUESTION:
            needs_contact_explanation = decision.capability_required in {
                EscalationCapability.EMAIL,
                EscalationCapability.MESSAGE,
            }
            return self._build(
                planning_input,
                ConversationMove.CLARIFY_MEANING,
                (
                    ResponseLength.MODERATE
                    if needs_contact_explanation
                    else ResponseLength.SHORT
                ),
                QuestionStrategy.CLARIFY_CURRENT_INPUT,
                clarification=True,
                **common,
            )
        if decision.recovery_mode == RecoveryMode.SAFE_REDIRECT:
            return self._build(
                planning_input,
                ConversationMove.REDIRECT_SAFELY,
                ResponseLength.SHORT,
                QuestionStrategy.NONE,
                handling=InterruptionHandling.DROP_STALE_POINT,
            )
        if decision.recovery_mode == RecoveryMode.POLITE_WRAP_UP:
            return self._build(
                planning_input,
                ConversationMove.POLITE_WRAP_UP,
                ResponseLength.SHORT,
                QuestionStrategy.NONE,
                handling=InterruptionHandling.DROP_STALE_POINT,
            )
        goal = (
            ConversationMove.ACKNOWLEDGE_UNCERTAINTY
            if decision.recovery_mode == RecoveryMode.ACKNOWLEDGE_UNKNOWN
            else ConversationMove.OFFER_SUPPORTED_NEXT_STEP
        )
        return self._build(
            planning_input,
            goal,
            (
                ResponseLength.MODERATE
                if decision.resume_previous_goal
                or decision.recovery_mode
                in {
                    RecoveryMode.OFFER_HUMAN_FOLLOW_UP,
                    RecoveryMode.OFFER_CALLBACK,
                    RecoveryMode.OFFER_INFORMATION_FOLLOW_UP,
                }
                else ResponseLength.SHORT
            ),
            QuestionStrategy.NONE,
            **common,
        )

    def _next_acknowledgement(
        self, planning_input: ResponsePlanningInput
    ) -> AcknowledgementKind:
        previous = planning_input.previous_acknowledgement
        if previous not in self._ACKNOWLEDGEMENTS:
            return self._ACKNOWLEDGEMENTS[0]
        index = self._ACKNOWLEDGEMENTS.index(previous)
        return self._ACKNOWLEDGEMENTS[(index + 1) % len(self._ACKNOWLEDGEMENTS)]

    @staticmethod
    def _build(
        planning_input: ResponsePlanningInput,
        goal: ConversationMove,
        length: ResponseLength,
        question: QuestionStrategy,
        *,
        acknowledgement: AcknowledgementKind = AcknowledgementKind.NONE,
        clarification: bool = False,
        handling: InterruptionHandling = InterruptionHandling.NONE,
        resume: bool = False,
        pending: PendingConversationIntent | None = None,
    ) -> ResponsePlan:
        return ResponsePlan(
            communicative_goal=goal,
            response_length=length,
            tone=planning_input.tone,
            acknowledgement=acknowledgement,
            question_strategy=question,
            clarification_required=clarification,
            interruption_handling=handling,
            resume_previous_point=resume,
            pending_intent=pending,
            addressee_status=planning_input.addressee_status,
            trusted_context_summary=planning_input.trusted_context_summary,
            escalation_decision=planning_input.escalation_decision,
        )


def _active_pending(
    pending: PendingConversationIntent | None,
) -> PendingConversationIntent | None:
    return pending if pending is not None and pending.active else None


def _ordinary_length(need: ExplanationNeed) -> ResponseLength:
    if need == ExplanationNeed.SIMPLE:
        return ResponseLength.SHORT
    if need == ExplanationNeed.COMPLEX:
        return ResponseLength.DETAILED
    return ResponseLength.MODERATE


def _explanatory_length(need: ExplanationNeed) -> ResponseLength:
    return (
        ResponseLength.DETAILED
        if need == ExplanationNeed.COMPLEX
        else ResponseLength.MODERATE
    )
