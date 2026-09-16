"""Pure consultative decision engine selecting one useful conversational move."""

from __future__ import annotations

from app.conversation.consultative.contracts import (
    AcknowledgementIntent,
    ConsultativeConversationDecision,
    ConsultativeDecisionInput,
    ConsultativeMove,
    ConsultativeObjective,
    ProblemField,
    QuestionPolicy,
    ServiceFitStatus,
)
from app.conversation.prospect_intelligence.contracts import (
    ObjectionType,
    ProspectRole,
    SolutionSatisfaction,
)
from app.conversation.response_planning.contracts import (
    AddresseeStatus,
    InterruptionCategory,
    ResponseLength,
)
from app.conversation.strategy.contracts import SalesStage


class ConsultativeDecisionEngine:
    """Diagnose before prescribing, without action or state authority."""

    def decide(
        self, value: ConsultativeDecisionInput
    ) -> ConsultativeConversationDecision:
        if value.addressee_status != AddresseeStatus.ADDRESSED_TO_AGENT:
            return _decision(
                ConsultativeObjective.RECOVER,
                ConsultativeMove.CLARIFY,
                ProblemField.UNDERLYING_PROBLEM,
                value,
            )
        if value.signals.commercial_purpose_question:
            return ConsultativeConversationDecision(
                ConsultativeObjective.ANSWER,
                ConsultativeMove.ANSWER_DIRECT_QUESTION,
                problem_summary=_summary(value),
                explanation_depth=ResponseLength.SHORT,
                commercial_transparency_required=True,
                resume_previous_goal=False,
            )
        if value.signals.misunderstanding:
            return ConsultativeConversationDecision(
                ConsultativeObjective.RECOVER,
                ConsultativeMove.SIMPLIFY_EXPLANATION,
                problem_summary=_summary(value),
                explanation_depth=ResponseLength.SHORT,
                simplification_required=True,
                resume_previous_goal=_resume(value),
            )
        if value.signals.direct_question:
            return ConsultativeConversationDecision(
                ConsultativeObjective.ANSWER,
                ConsultativeMove.ANSWER_DIRECT_QUESTION,
                problem_summary=_summary(value),
                service_fit=value.service_fit,
                explanation_depth=(
                    ResponseLength.DETAILED
                    if value.signals.explanation_requested
                    else ResponseLength.SHORT
                ),
                resume_previous_goal=_resume(value),
            )
        if _busy_and_interested(value):
            return ConsultativeConversationDecision(
                ConsultativeObjective.PROGRESS,
                ConsultativeMove.LOW_PRESSURE_CALLBACK,
                problem_summary=_summary(value),
                explanation_depth=ResponseLength.SHORT,
            )
        if value.signals.explicit_no_problem:
            return ConsultativeConversationDecision(
                ConsultativeObjective.CLOSE,
                ConsultativeMove.GRACEFUL_CLOSE,
                explanation_depth=ResponseLength.SHORT,
            )
        if _routing_role(value):
            return _decision(
                ConsultativeObjective.UNDERSTAND,
                ConsultativeMove.ROUTE_TO_DECISION_MAKER,
                ProblemField.ROLE_ROUTING,
                value,
            )
        if _current_provider_needs_context(value):
            return _decision(
                ConsultativeObjective.UNDERSTAND,
                ConsultativeMove.UNDERSTAND_EXISTING_SOLUTION,
                ProblemField.PROVIDER_SATISFACTION,
                value,
            )
        if _unknown_objection_reason(value):
            return _decision(
                ConsultativeObjective.UNDERSTAND,
                ConsultativeMove.HANDLE_OBJECTION,
                ProblemField.OBJECTION_REASON,
                value,
            )
        if value.signals.correction or (
            value.interruption_category == InterruptionCategory.CORRECTION
        ):
            if not value.problem.meaningful:
                return _decision(
                    ConsultativeObjective.UNDERSTAND,
                    ConsultativeMove.CLARIFY,
                    ProblemField.UNDERLYING_PROBLEM,
                    value,
                )
        if value.signals.service_information_request:
            if value.service_fit.status == ServiceFitStatus.SUPPORTED_FIT:
                return _explain(value)
            return _decision(
                ConsultativeObjective.UNDERSTAND,
                ConsultativeMove.DISCOVER_PROBLEM,
                ProblemField.UNDERLYING_PROBLEM,
                value,
            )
        if value.service_fit.status == ServiceFitStatus.SUPPORTED_FIT:
            if value.signals.buying_signal:
                return ConsultativeConversationDecision(
                    ConsultativeObjective.PROGRESS,
                    ConsultativeMove.ASK_MICRO_COMMITMENT,
                    problem_summary=_summary(value),
                    service_fit=value.service_fit,
                    explanation_depth=ResponseLength.SHORT,
                )
            return _explain(value)
        if value.problem.requested_solution and not value.problem.meaningful:
            return _decision(
                ConsultativeObjective.UNDERSTAND,
                ConsultativeMove.DISCOVER_PROBLEM,
                ProblemField.UNDERLYING_PROBLEM,
                value,
            )
        if (
            not value.problem.meaningful
            and value.strategy is not None
            and value.strategy.sales_stage == SalesStage.OPENING
        ):
            return _decision(
                ConsultativeObjective.OPEN_TRUTHFULLY,
                ConsultativeMove.OPEN_CONVERSATION,
                ProblemField.UNDERLYING_PROBLEM,
                value,
            )
        if not value.problem.meaningful:
            return _decision(
                ConsultativeObjective.UNDERSTAND,
                ConsultativeMove.DISCOVER_PROBLEM,
                ProblemField.UNDERLYING_PROBLEM,
                value,
            )
        if value.problem.current_process is None:
            return _decision(
                ConsultativeObjective.UNDERSTAND,
                ConsultativeMove.UNDERSTAND_CURRENT_PROCESS,
                ProblemField.CURRENT_PROCESS,
                value,
            )
        if value.problem.impact is None:
            return _decision(
                ConsultativeObjective.UNDERSTAND,
                ConsultativeMove.UNDERSTAND_IMPACT,
                ProblemField.IMPACT,
                value,
            )
        if value.service_fit.status == ServiceFitStatus.NO_AUTHORIZED_FIT:
            return ConsultativeConversationDecision(
                ConsultativeObjective.CLOSE,
                ConsultativeMove.GRACEFUL_CLOSE,
                problem_summary=_summary(value),
                explanation_depth=ResponseLength.SHORT,
            )
        gap = (
            value.service_fit.missing_information[0]
            if value.service_fit.missing_information
            else ProblemField.DESIRED_OUTCOME
        )
        return _decision(
            ConsultativeObjective.UNDERSTAND,
            ConsultativeMove.CLARIFY,
            gap,
            value,
        )


def _decision(
    objective: ConsultativeObjective,
    move: ConsultativeMove,
    gap: ProblemField,
    value: ConsultativeDecisionInput,
) -> ConsultativeConversationDecision:
    return ConsultativeConversationDecision(
        objective,
        move,
        gap,
        _summary(value),
        value.service_fit,
        ResponseLength.SHORT,
        (
            QuestionPolicy.SHORT_CONTRAST
            if value.signals.two_way_ambiguity
            else QuestionPolicy.ONE_PRIMARY
        ),
        AcknowledgementIntent.REFLECT if value.problem.meaningful else AcknowledgementIntent.NONE,
        False,
        resume_previous_goal=False,
    )


def _explain(value: ConsultativeDecisionInput) -> ConsultativeConversationDecision:
    return ConsultativeConversationDecision(
        ConsultativeObjective.EXPLAIN,
        ConsultativeMove.EXPLAIN_RELEVANT_FIT,
        problem_summary=_summary(value),
        service_fit=value.service_fit,
        explanation_depth=(
            ResponseLength.DETAILED
            if value.signals.explanation_requested
            else ResponseLength.MODERATE
        ),
        acknowledgement_intent=AcknowledgementIntent.REFLECT,
    )


def _summary(value: ConsultativeDecisionInput) -> str | None:
    return value.problem.explicit_description or value.problem.friction


def _resume(value: ConsultativeDecisionInput) -> bool:
    pending = value.pending_intent
    return bool(pending is not None and pending.active)


def _busy_and_interested(value: ConsultativeDecisionInput) -> bool:
    if value.prospect is None:
        return False
    busy = value.prospect.busy
    interested = value.prospect.interested
    return bool(busy is not None and busy.value and interested is not None and interested.value)


def _routing_role(value: ConsultativeDecisionInput) -> bool:
    if value.prospect is None or value.prospect.explicit_role is None:
        return False
    return value.prospect.explicit_role.value in {
        ProspectRole.RECEPTIONIST,
        ProspectRole.GATEKEEPER,
    }


def _unknown_objection_reason(value: ConsultativeDecisionInput) -> bool:
    if value.prospect is None or value.prospect.explicit_objection is None:
        return False
    return value.prospect.explicit_objection.value not in {
        ObjectionType.NONE,
        ObjectionType.UNKNOWN,
    } and value.problem.friction is None


def _current_provider_needs_context(value: ConsultativeDecisionInput) -> bool:
    if value.prospect is None or value.prospect.current_solution is None:
        return False
    return (
        value.prospect.current_solution.satisfaction
        == SolutionSatisfaction.UNKNOWN
    )
