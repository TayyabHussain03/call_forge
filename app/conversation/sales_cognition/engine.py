"""Pure deterministic human-sales cognition derived from structured evidence."""

from __future__ import annotations

from app.conversation.consultative.contracts import (
    ProblemField,
    ServiceFitStatus,
)
from app.conversation.prospect_intelligence.contracts import (
    ObjectionType,
    PreferredNextStep,
    ProspectRole,
    SolutionSatisfaction,
)
from app.conversation.response_planning.contracts import ResponseLength
from app.conversation.sales_cognition.contracts import (
    BuyingReadinessGuidance,
    ConversationEnergy,
    ConversationMomentum,
    CuriosityFocus,
    DiscoveryReadiness,
    EmotionalPosture,
    InterestStrength,
    ObjectionUnderstanding,
    PressureState,
    QuestionPriority,
    RelationshipState,
    RoleConversationStyle,
    SalesCognitionInput,
    SalesConversationGuidance,
    TrustState,
)
from app.conversation.sales_cognition.validator import validate_sales_guidance


class HumanSalesCognitionEngine:
    """Produce advisory conversation guidance without changing domain decisions."""

    def evaluate(self, value: SalesCognitionInput) -> SalesConversationGuidance:
        interest = _interest(value)
        objection = _objection(value)
        readiness = _readiness(value, interest)
        question_focus = _question_focus(value)
        guidance = SalesConversationGuidance(
            momentum=_momentum(value, interest),
            trust=_trust(value),
            interest=interest,
            discovery_readiness=readiness,
            objection=objection,
            energy=_energy(value),
            question_priority=_question_priority(question_focus),
            question_focus=question_focus,
            relationship=_relationship(value),
            pressure=_pressure(value, interest),
            curiosity_focus=_curiosity_focus(question_focus),
            role_style=_role_style(value),
            emotional_posture=_emotion(value),
            buying_guidance=_buying_guidance(value, interest, objection),
            recommended_response_depth=_depth(value),
        )
        return validate_sales_guidance(guidance, value)


def _observed(value):  # type: ignore[no-untyped-def]
    return value.value if value is not None else None


def _interest(value: SalesCognitionInput) -> InterestStrength:
    next_step = _observed(value.prospect.explicit_next_step)
    if next_step in {
        PreferredNextStep.DEMO,
        PreferredNextStep.HUMAN_FOLLOW_UP,
        PreferredNextStep.CALLBACK,
    }:
        return InterestStrength.BUYING_SIGNAL
    if value.signals.research_only or next_step == PreferredNextStep.SEND_INFORMATION:
        return InterestStrength.RESEARCH_MODE
    interested = _observed(value.prospect.interested)
    if interested is False:
        return InterestStrength.NONE
    if interested is True:
        if (
            value.prior_guidance is not None
            and value.prior_guidance.interest
            in {InterestStrength.MODERATE, InterestStrength.STRONG}
        ):
            return InterestStrength.STRONG
        return InterestStrength.MODERATE
    if value.problem.meaningful:
        return InterestStrength.WEAK
    return InterestStrength.UNKNOWN


def _momentum(
    value: SalesCognitionInput, interest: InterestStrength
) -> ConversationMomentum:
    if value.signals.correction or value.signals.confused:
        return ConversationMomentum.RECOVERING
    if value.signals.explicit_no_need or interest == InterestStrength.NONE:
        return ConversationMomentum.DECLINING
    if interest in {InterestStrength.STRONG, InterestStrength.BUYING_SIGNAL}:
        return ConversationMomentum.IMPROVING
    if value.prior_guidance is not None or value.problem.meaningful:
        return ConversationMomentum.STABLE
    return ConversationMomentum.UNKNOWN


def _trust(value: SalesCognitionInput) -> TrustState:
    if value.signals.confused or value.signals.skeptical or value.signals.correction:
        return TrustState.FRAGILE
    if value.returning_discussion:
        return TrustState.ESTABLISHED
    if (
        value.prior_guidance is not None
        and value.prior_guidance.trust == TrustState.BUILDING
        and _observed(value.prospect.interested) is True
    ):
        return TrustState.ESTABLISHED
    if value.problem.meaningful or _observed(value.prospect.interested) is not None:
        return TrustState.BUILDING
    return TrustState.UNKNOWN


def _objection(value: SalesCognitionInput) -> ObjectionUnderstanding:
    objection = _observed(value.prospect.explicit_objection)
    return {
        ObjectionType.NONE: ObjectionUnderstanding.NONE,
        ObjectionType.NO_TIME: ObjectionUnderstanding.NO_TIME,
        ObjectionType.BAD_TIMING: ObjectionUnderstanding.NO_TIME,
        ObjectionType.PRICE: ObjectionUnderstanding.BUDGET,
        ObjectionType.ALREADY_HAVE_PROVIDER: ObjectionUnderstanding.EXISTING_PROVIDER,
        ObjectionType.NO_NEED: ObjectionUnderstanding.NO_NEED,
        ObjectionType.TRUST: ObjectionUnderstanding.NO_TRUST,
        ObjectionType.INTERNAL_TEAM: ObjectionUnderstanding.INTERNAL_TEAM,
        ObjectionType.CONTRACT_LOCK: ObjectionUnderstanding.CONTRACT_LOCK,
        ObjectionType.SEND_INFORMATION: ObjectionUnderstanding.RESEARCH_ONLY,
    }.get(objection, ObjectionUnderstanding.UNKNOWN)


def _readiness(
    value: SalesCognitionInput, interest: InterestStrength
) -> DiscoveryReadiness:
    if interest == InterestStrength.BUYING_SIGNAL and (
        value.service_fit.status == ServiceFitStatus.SUPPORTED_FIT
    ):
        return DiscoveryReadiness.READY_FOR_NEXT_STEP
    if value.service_fit.status == ServiceFitStatus.SUPPORTED_FIT:
        return DiscoveryReadiness.READY_FOR_FIT
    if not value.problem.meaningful:
        return DiscoveryReadiness.NEED_PROBLEM
    if value.problem.current_process is None:
        return DiscoveryReadiness.NEED_WORKFLOW
    if value.problem.impact is None:
        return DiscoveryReadiness.NEED_IMPACT
    return DiscoveryReadiness.NEED_DECISION_CONTEXT


def _question_focus(value: SalesCognitionInput) -> ProblemField | None:
    candidates: tuple[ProblemField | None, ...] = (
        (
            ProblemField.PROVIDER_SATISFACTION
            if value.prospect.current_solution is not None
            and value.prospect.current_solution.satisfaction
            == SolutionSatisfaction.UNKNOWN
            else None
        ),
        value.consultative_decision.primary_information_gap,
        ProblemField.UNDERLYING_PROBLEM if not value.problem.meaningful else None,
        ProblemField.CURRENT_PROCESS if value.problem.current_process is None else None,
        ProblemField.IMPACT if value.problem.impact is None else None,
        ProblemField.DESIRED_OUTCOME if value.problem.desired_outcome is None else None,
    )
    recent = set(value.recent_question_concepts)
    return next((item for item in candidates if item is not None and item not in recent), None)


def _question_priority(focus: ProblemField | None) -> QuestionPriority:
    return {
        None: QuestionPriority.NONE,
        ProblemField.UNDERLYING_PROBLEM: QuestionPriority.PROBLEM,
        ProblemField.CURRENT_PROCESS: QuestionPriority.WORKFLOW,
        ProblemField.IMPACT: QuestionPriority.IMPACT,
        ProblemField.PROVIDER_SATISFACTION: QuestionPriority.PROVIDER_GAP,
        ProblemField.ROLE_ROUTING: QuestionPriority.DECISION_CONTEXT,
    }.get(focus, QuestionPriority.FIT_CLARIFICATION)


def _curiosity_focus(focus: ProblemField | None) -> CuriosityFocus:
    return {
        None: CuriosityFocus.NONE,
        ProblemField.UNDERLYING_PROBLEM: CuriosityFocus.UNDERLYING_PROBLEM,
        ProblemField.CURRENT_PROCESS: CuriosityFocus.CURRENT_WORKFLOW,
        ProblemField.IMPACT: CuriosityFocus.BUSINESS_IMPACT,
        ProblemField.PROVIDER_SATISFACTION: CuriosityFocus.PROVIDER_EXPERIENCE,
        ProblemField.ROLE_ROUTING: CuriosityFocus.ROLE_ROUTING,
    }.get(focus, CuriosityFocus.FIT_UNCERTAINTY)


def _relationship(value: SalesCognitionInput) -> RelationshipState:
    if value.returning_discussion:
        return RelationshipState.RETURNING_DISCUSSION
    if value.prior_guidance is not None and value.problem.meaningful:
        return RelationshipState.KNOWN_CONTEXT
    if value.problem.meaningful or value.prior_guidance is not None:
        return RelationshipState.DEVELOPING
    return RelationshipState.FIRST_CONTACT


def _energy(value: SalesCognitionInput) -> ConversationEnergy:
    if value.signals.fatigued:
        return ConversationEnergy.FATIGUED
    if _observed(value.prospect.busy) is True:
        return ConversationEnergy.FAST
    if value.signals.confused or value.signals.frustrated:
        return ConversationEnergy.SLOW
    return ConversationEnergy.NORMAL


def _pressure(
    value: SalesCognitionInput, interest: InterestStrength
) -> PressureState:
    if value.signals.confused or value.signals.correction:
        return PressureState.NEEDS_CLARIFICATION
    if value.signals.fatigued or _observed(value.prospect.busy) is True:
        return PressureState.NEEDS_SLOWING
    if interest in {InterestStrength.STRONG, InterestStrength.BUYING_SIGNAL}:
        return PressureState.READY_TO_CONTINUE
    return PressureState.COMFORTABLE


def _role_style(value: SalesCognitionInput) -> RoleConversationStyle:
    role = _observed(value.prospect.explicit_role)
    return {
        ProspectRole.RECEPTIONIST: RoleConversationStyle.BRIEF_ROUTING,
        ProspectRole.GATEKEEPER: RoleConversationStyle.BRIEF_ROUTING,
        ProspectRole.OWNER: RoleConversationStyle.OWNER_VALUE,
        ProspectRole.FOUNDER: RoleConversationStyle.OWNER_VALUE,
        ProspectRole.MANAGER: RoleConversationStyle.OPERATIONS_WORKFLOW,
        ProspectRole.OPERATIONS_MANAGER: RoleConversationStyle.OPERATIONS_WORKFLOW,
        ProspectRole.TECHNICAL_MANAGER: RoleConversationStyle.TECHNICAL_PRECISION,
        ProspectRole.SALES_MANAGER: RoleConversationStyle.SALES_PROCESS,
        ProspectRole.FINANCE_CONTACT: RoleConversationStyle.FINANCIAL_RESTRAINT,
        ProspectRole.ASSISTANT_MANAGER: RoleConversationStyle.SUPPORTIVE,
        ProspectRole.ASSISTANT: RoleConversationStyle.SUPPORTIVE,
    }.get(role, RoleConversationStyle.GENERAL_CONSULTATIVE)


def _emotion(value: SalesCognitionInput) -> EmotionalPosture:
    if value.signals.confused:
        return EmotionalPosture.CONFUSED
    if value.signals.skeptical:
        return EmotionalPosture.SKEPTICAL
    if value.signals.frustrated:
        return EmotionalPosture.FRUSTRATED
    if _observed(value.prospect.busy) is True:
        return EmotionalPosture.BUSY
    if value.signals.curious:
        return EmotionalPosture.CURIOUS
    if _observed(value.prospect.interested) is True:
        return EmotionalPosture.INTERESTED
    return EmotionalPosture.NEUTRAL


def _buying_guidance(
    value: SalesCognitionInput,
    interest: InterestStrength,
    objection: ObjectionUnderstanding,
) -> BuyingReadinessGuidance:
    if value.signals.explicit_no_need or interest == InterestStrength.NONE:
        return BuyingReadinessGuidance.GRACEFUL_CLOSE
    if objection not in {ObjectionUnderstanding.NONE, ObjectionUnderstanding.UNKNOWN}:
        return BuyingReadinessGuidance.HANDLE_OBJECTION
    if (
        interest == InterestStrength.BUYING_SIGNAL
        and value.service_fit.status == ServiceFitStatus.SUPPORTED_FIT
    ):
        return BuyingReadinessGuidance.SUGGEST_MICRO_COMMITMENT
    if value.service_fit.status == ServiceFitStatus.SUPPORTED_FIT:
        return BuyingReadinessGuidance.EXPLAIN_FIT
    return BuyingReadinessGuidance.CONTINUE_DISCOVERY


def _depth(value: SalesCognitionInput) -> ResponseLength:
    if (
        value.signals.confused
        or value.signals.fatigued
        or _observed(value.prospect.busy) is True
    ):
        return ResponseLength.SHORT
    if value.signals.detailed_explanation_requested:
        return ResponseLength.DETAILED
    if value.signals.curious or value.signals.research_only:
        return ResponseLength.MODERATE
    return value.consultative_decision.explanation_depth
