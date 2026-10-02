"""Deterministic Sales Knowledge Intelligence Engine.

Transforms approved evidence into a structured conversational teaching strategy.
All decisions are deterministic — no LLM calls, no retrieval, no state mutation.
"""

from __future__ import annotations

from app.conversation.business_conversation.contracts import (
    BusinessFactKind,
    BusinessProblemKind,
    ConversationTopic,
)
from app.conversation.business_diagnostic.contracts import (
    BusinessMaturity,
    CapabilityArea,
)
from app.conversation.consultative.contracts import (
    ProblemCategory,
    ServiceFitStatus,
)
from app.conversation.conversation_steering.contracts import (
    ConversationReadiness,
    CuriosityBudgetState,
)
from app.conversation.prospect_intelligence.contracts import (
    BuyingStage,
    InformationLevel,
    ProspectRole,
)
from app.conversation.response_planning.contracts import ResponseLength
from app.conversation.sales_cognition.contracts import (
    ConversationMomentum,
    DiscoveryReadiness,
    InterestStrength,
    ObjectionUnderstanding,
    TrustState,
)
from app.conversation.sales_playbook.contracts import (
    BenefitCategory,
    ConsultantStep,
    PitchReadiness,
)
from app.conversation.understanding.contracts import ConversationalRegister
from app.knowledge.evidence_validation.contracts import (
    ApprovedEvidence,
    EvidenceGroup,
    EvidenceQuality,
)
from app.knowledge.sales_intelligence.contracts import (
    AnalogyStrategy,
    BusinessValueFocus,
    CognitiveLoad,
    ConversationObjective,
    DisclosureLevel,
    ExplanationDepth,
    ExplanationPlan,
    FollowUpStyle,
    KnowledgeConfidence,
    KnowledgeConversationContext,
    KnowledgeIntent,
    KnowledgePriority,
    MomentumSignal,
    ProgressiveDisclosurePlan,
    ResponseComplexity,
    SalesKnowledgeInput,
    SelectedEvidence,
    SuppressedEvidence,
    SuppressionReason,
)
from app.knowledge.sales_intelligence.validator import validate_knowledge_context


class SalesKnowledgeIntelligenceEngine:
    """Produce a conversational teaching strategy from approved evidence."""

    def design(self, value: SalesKnowledgeInput) -> KnowledgeConversationContext:
        depth = _depth(value)
        intent = _knowledge_intent(value)
        disclosure_level = _disclosure_level(value)
        max_items = _max_evidence(disclosure_level, value)
        topic_focus = _topic_focus(value)

        ranked = _rank_evidence(value, topic_focus)
        selected, suppressed = _select_and_suppress(
            ranked, max_items, value, topic_focus
        )

        business_focus = _primary_business_value(value)
        selected_items = tuple(
            SelectedEvidence(
                evidence=evidence,
                priority=KnowledgePriority.PRIMARY if i == 0 else KnowledgePriority.SUPPORTING,
                business_value_focus=business_focus,
                sequence_position=i,
            )
            for i, evidence in enumerate(selected)
        )

        deferred_hint = _deferred_group_hint(suppressed)
        follow_up = _follow_up_style(value, suppressed)
        confidence = _knowledge_confidence(selected)
        objective = _conversation_objective(value, intent)
        cognitive_load = _cognitive_load(value, depth)
        momentum = _momentum_signal(value)
        saturated = _knowledge_saturated(value)

        context = KnowledgeConversationContext(
            selected=selected_items,
            suppressed=tuple(suppressed),
            explanation=ExplanationPlan(
                depth=depth,
                knowledge_intent=intent,
                analogy=_analogy(value, depth),
                business_value_focus=business_focus,
                response_complexity=_response_complexity(value, depth),
                confidence=confidence,
            ),
            disclosure=ProgressiveDisclosurePlan(
                disclosure_level=disclosure_level,
                max_evidence_this_turn=max_items,
                follow_up=follow_up,
                deferred_group_hint=deferred_hint,
            ),
            conversation_objective=objective,
            cognitive_load=cognitive_load,
            momentum=momentum,
            knowledge_saturated=saturated,
        )
        validate_knowledge_context(context, value)
        return context


# ---------------------------------------------------------------------------
# Explanation depth
# ---------------------------------------------------------------------------


def _depth(value: SalesKnowledgeInput) -> ExplanationDepth:
    guidance = value.sales_guidance
    if guidance.recommended_response_depth == ResponseLength.SHORT:
        return ExplanationDepth.SIMPLE

    has_technical_facts = any(
        fact.kind in {BusinessFactKind.SOFTWARE, BusinessFactKind.INTEGRATION}
        for fact in value.conversation.facts
    )
    if has_technical_facts and guidance.recommended_response_depth == ResponseLength.DETAILED:
        return ExplanationDepth.DETAILED

    register = value.language_profile.conversational_register
    if register == ConversationalRegister.FORMAL:
        return ExplanationDepth.NORMAL

    role = _effective_role(value)
    if role in {ProspectRole.TECHNICAL_MANAGER, ProspectRole.OPERATIONS_MANAGER}:
        return ExplanationDepth.DETAILED

    if role in {
        ProspectRole.RECEPTIONIST,
        ProspectRole.GATEKEEPER,
        ProspectRole.STAFF,
        ProspectRole.ASSISTANT,
    }:
        return ExplanationDepth.SIMPLE

    if guidance.recommended_response_depth == ResponseLength.DETAILED:
        return ExplanationDepth.DETAILED

    return ExplanationDepth.NORMAL


# ---------------------------------------------------------------------------
# Knowledge intent
# ---------------------------------------------------------------------------


def _knowledge_intent(value: SalesKnowledgeInput) -> KnowledgeIntent:
    problem = value.problem
    if problem.requested_solution is not None:
        return KnowledgeIntent.ANSWER_QUESTION

    playbook = value.playbook_guidance
    if playbook is not None and playbook.consultant_step == ConsultantStep.EDUCATE:
        return KnowledgeIntent.EDUCATE

    if problem.category == ProblemCategory.UNKNOWN and not problem.meaningful:
        return KnowledgeIntent.BUILD_AWARENESS

    has_comparison_facts = any(
        fact.kind == BusinessFactKind.SOFTWARE for fact in value.conversation.facts
    )
    if has_comparison_facts and problem.requested_solution is not None:
        return KnowledgeIntent.COMPARE

    interest = value.sales_guidance.interest
    if interest in {InterestStrength.STRONG, InterestStrength.BUYING_SIGNAL}:
        return KnowledgeIntent.ILLUSTRATE_BENEFIT

    if interest == InterestStrength.NONE:
        return KnowledgeIntent.BUILD_AWARENESS

    return KnowledgeIntent.EDUCATE


# ---------------------------------------------------------------------------
# Topic focus from conversation context
# ---------------------------------------------------------------------------

_TOPIC_TO_GROUPS: dict[ConversationTopic | None, frozenset[EvidenceGroup]] = {
    ConversationTopic.WEBSITE: frozenset(
        {EvidenceGroup.FEATURE, EvidenceGroup.BENEFIT, EvidenceGroup.FAQ, EvidenceGroup.PROCESS}
    ),
    ConversationTopic.CRM: frozenset(
        {EvidenceGroup.FEATURE, EvidenceGroup.BENEFIT, EvidenceGroup.IMPLEMENTATION}
    ),
    ConversationTopic.OPERATIONS: frozenset(
        {EvidenceGroup.PROCESS, EvidenceGroup.IMPLEMENTATION, EvidenceGroup.CASE_STUDY}
    ),
    ConversationTopic.MARKETING: frozenset(
        {EvidenceGroup.BENEFIT, EvidenceGroup.CASE_STUDY, EvidenceGroup.FEATURE}
    ),
    ConversationTopic.SALES: frozenset(
        {EvidenceGroup.BENEFIT, EvidenceGroup.CASE_STUDY, EvidenceGroup.FEATURE}
    ),
    ConversationTopic.REPORTING: frozenset(
        {EvidenceGroup.FEATURE, EvidenceGroup.PROCESS, EvidenceGroup.IMPLEMENTATION}
    ),
    None: frozenset(EvidenceGroup),
}


def _topic_focus(value: SalesKnowledgeInput) -> frozenset[EvidenceGroup]:
    topic = value.conversation.current_focus
    return _TOPIC_TO_GROUPS.get(topic, _TOPIC_TO_GROUPS[None])


# ---------------------------------------------------------------------------
# Evidence ranking
# ---------------------------------------------------------------------------

_GROUP_BASE_PRIORITY: dict[EvidenceGroup, int] = {
    EvidenceGroup.BENEFIT: 0,
    EvidenceGroup.FEATURE: 1,
    EvidenceGroup.FAQ: 2,
    EvidenceGroup.PROCESS: 3,
    EvidenceGroup.CASE_STUDY: 4,
    EvidenceGroup.IMPLEMENTATION: 5,
    EvidenceGroup.LIMITATION: 6,
    EvidenceGroup.COMMERCIAL_POLICY: 7,
    EvidenceGroup.COMPLIANCE: 8,
}

_QUALITY_BONUS: dict[EvidenceQuality, int] = {
    EvidenceQuality.DIRECT: 0,
    EvidenceQuality.SUPPORTED: 1,
    EvidenceQuality.LIMITED: 2,
}


def _rank_evidence(
    value: SalesKnowledgeInput,
    topic_groups: frozenset[EvidenceGroup],
) -> list[ApprovedEvidence]:
    items = list(value.approved_evidence.items)
    if not items:
        return []

    def sort_key(evidence: ApprovedEvidence) -> tuple[int, int, str]:
        topic_penalty = 0 if evidence.group in topic_groups else 10
        group_rank = _GROUP_BASE_PRIORITY.get(evidence.group, 9)
        quality_rank = _QUALITY_BONUS.get(evidence.quality, 2)
        return (topic_penalty + group_rank, quality_rank, evidence.approval_id)

    return sorted(items, key=sort_key)


# ---------------------------------------------------------------------------
# Selection and suppression
# ---------------------------------------------------------------------------


def _select_and_suppress(
    ranked: list[ApprovedEvidence],
    max_items: int,
    value: SalesKnowledgeInput,
    topic_groups: frozenset[EvidenceGroup],
) -> tuple[list[ApprovedEvidence], list[SuppressedEvidence]]:
    selected: list[ApprovedEvidence] = []
    suppressed: list[SuppressedEvidence] = []

    for evidence in ranked:
        if evidence.approval_id in value.already_discussed_evidence_ids:
            suppressed.append(SuppressedEvidence(evidence, SuppressionReason.ALREADY_DISCUSSED))
            continue

        if evidence.group not in topic_groups:
            suppressed.append(SuppressedEvidence(evidence, SuppressionReason.OFF_TOPIC))
            continue

        if _is_premature(evidence, value):
            suppressed.append(SuppressedEvidence(evidence, SuppressionReason.PREMATURE))
            continue

        if _is_wrong_depth(evidence, value):
            suppressed.append(SuppressedEvidence(evidence, SuppressionReason.WRONG_DEPTH))
            continue

        if len(selected) >= max_items:
            suppressed.append(SuppressedEvidence(evidence, SuppressionReason.OVERLOAD))
            continue

        selected.append(evidence)

    return selected, suppressed


def _is_premature(evidence: ApprovedEvidence, value: SalesKnowledgeInput) -> bool:
    guidance = value.sales_guidance
    if evidence.group == EvidenceGroup.COMMERCIAL_POLICY:
        if guidance.trust == TrustState.BUILDING or guidance.trust == TrustState.UNKNOWN:
            return True

    if evidence.group == EvidenceGroup.CASE_STUDY:
        if guidance.interest in {InterestStrength.NONE, InterestStrength.UNKNOWN}:
            return True

    playbook = value.playbook_guidance
    if playbook is not None and playbook.pitch_readiness == PitchReadiness.DISCOVERY_REQUIRED:
        if evidence.group in {EvidenceGroup.IMPLEMENTATION, EvidenceGroup.COMMERCIAL_POLICY}:
            return True

    return False


def _is_wrong_depth(evidence: ApprovedEvidence, value: SalesKnowledgeInput) -> bool:
    guidance = value.sales_guidance
    if guidance.recommended_response_depth == ResponseLength.SHORT:
        if evidence.group in {
            EvidenceGroup.IMPLEMENTATION,
            EvidenceGroup.COMPLIANCE,
            EvidenceGroup.CASE_STUDY,
        }:
            return True
    return False


# ---------------------------------------------------------------------------
# Disclosure level
# ---------------------------------------------------------------------------


def _disclosure_level(value: SalesKnowledgeInput) -> DisclosureLevel:
    guidance = value.sales_guidance
    interest = guidance.interest

    if guidance.recommended_response_depth == ResponseLength.SHORT:
        return DisclosureLevel.MINIMAL

    if interest in {InterestStrength.STRONG, InterestStrength.BUYING_SIGNAL}:
        return DisclosureLevel.FULL

    if interest in {InterestStrength.NONE, InterestStrength.UNKNOWN, InterestStrength.WEAK}:
        return DisclosureLevel.MINIMAL

    return DisclosureLevel.MODERATE


def _max_evidence(level: DisclosureLevel, value: SalesKnowledgeInput) -> int:
    if level == DisclosureLevel.MINIMAL:
        return 1
    if level == DisclosureLevel.FULL:
        return 4
    return 2


# ---------------------------------------------------------------------------
# Business value focus
# ---------------------------------------------------------------------------

_PROBLEM_TO_VALUE: dict[ProblemCategory, BusinessValueFocus] = {
    ProblemCategory.LEAD_FLOW: BusinessValueFocus.REVENUE,
    ProblemCategory.FOLLOW_UP: BusinessValueFocus.OPERATIONAL_EFFICIENCY,
    ProblemCategory.CUSTOMER_QUESTIONS: BusinessValueFocus.CUSTOMER_EXPERIENCE,
    ProblemCategory.WORKFLOW: BusinessValueFocus.OPERATIONAL_EFFICIENCY,
    ProblemCategory.WEBSITE: BusinessValueFocus.VISIBILITY,
    ProblemCategory.VISIBILITY: BusinessValueFocus.VISIBILITY,
    ProblemCategory.BRANDING: BusinessValueFocus.TRUST_BUILDING,
    ProblemCategory.OTHER: BusinessValueFocus.NONE,
    ProblemCategory.UNKNOWN: BusinessValueFocus.NONE,
}

_GOAL_TO_VALUE: dict[str, BusinessValueFocus] = {
    "increase_sales": BusinessValueFocus.REVENUE,
    "reduce_time": BusinessValueFocus.TIME_SAVINGS,
    "reduce_cost": BusinessValueFocus.COST_SAVINGS,
    "grow_business": BusinessValueFocus.GROWTH,
    "scale_operations": BusinessValueFocus.OPERATIONAL_EFFICIENCY,
    "improve_customer_experience": BusinessValueFocus.CUSTOMER_EXPERIENCE,
    "generate_leads": BusinessValueFocus.REVENUE,
    "visibility": BusinessValueFocus.VISIBILITY,
    "efficiency": BusinessValueFocus.OPERATIONAL_EFFICIENCY,
}


def _primary_business_value(value: SalesKnowledgeInput) -> BusinessValueFocus:
    goals = value.conversation.goals
    if goals:
        first_goal = next(iter(goals))
        mapped = _GOAL_TO_VALUE.get(first_goal.value)
        if mapped is not None:
            return mapped

    return _PROBLEM_TO_VALUE.get(value.problem.category, BusinessValueFocus.NONE)


# ---------------------------------------------------------------------------
# Analogy strategy
# ---------------------------------------------------------------------------


def _analogy(value: SalesKnowledgeInput, depth: ExplanationDepth) -> AnalogyStrategy:
    if depth == ExplanationDepth.DETAILED:
        return AnalogyStrategy.NONE

    has_industry = any(
        fact.kind == BusinessFactKind.INDUSTRY for fact in value.conversation.facts
    )

    if has_industry and depth == ExplanationDepth.SIMPLE:
        return AnalogyStrategy.INDUSTRY_EXAMPLE

    if value.problem.current_process is not None:
        return AnalogyStrategy.WORKFLOW_PARALLEL

    if depth == ExplanationDepth.SIMPLE:
        return AnalogyStrategy.EVERYDAY_COMPARISON

    return AnalogyStrategy.NONE


# ---------------------------------------------------------------------------
# Response complexity
# ---------------------------------------------------------------------------


def _response_complexity(
    value: SalesKnowledgeInput, depth: ExplanationDepth
) -> ResponseComplexity:
    if depth == ExplanationDepth.SIMPLE:
        return ResponseComplexity.SIMPLE

    has_technical = any(
        fact.kind in {BusinessFactKind.SOFTWARE, BusinessFactKind.INTEGRATION}
        for fact in value.conversation.facts
    )
    if has_technical and depth == ExplanationDepth.DETAILED:
        return ResponseComplexity.TECHNICAL

    return ResponseComplexity.MODERATE


# ---------------------------------------------------------------------------
# Follow-up style
# ---------------------------------------------------------------------------


def _follow_up_style(
    value: SalesKnowledgeInput,
    suppressed: list[SuppressedEvidence],
) -> FollowUpStyle:
    deferred = [
        item
        for item in suppressed
        if item.reason in {SuppressionReason.OVERLOAD, SuppressionReason.PREMATURE}
    ]
    if not deferred:
        return FollowUpStyle.NONE

    guidance = value.sales_guidance
    if guidance.interest in {InterestStrength.STRONG, InterestStrength.BUYING_SIGNAL}:
        return FollowUpStyle.OFFER_NEXT_TOPIC

    if guidance.trust in {TrustState.BUILDING, TrustState.UNKNOWN}:
        return FollowUpStyle.ASK_PERMISSION

    return FollowUpStyle.CURIOSITY_HOOK


# ---------------------------------------------------------------------------
# Deferred group hint
# ---------------------------------------------------------------------------


def _deferred_group_hint(
    suppressed: list[SuppressedEvidence],
) -> EvidenceGroup | None:
    deferred = [
        item
        for item in suppressed
        if item.reason in {SuppressionReason.OVERLOAD, SuppressionReason.PREMATURE}
    ]
    if not deferred:
        return None
    groups = [item.evidence.group for item in deferred]
    return max(set(groups), key=groups.count)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _effective_role(value: SalesKnowledgeInput) -> ProspectRole:
    prospect = value.prospect
    if prospect.explicit_role is not None:
        return prospect.explicit_role.value
    if prospect.likely_role is not None:
        return prospect.likely_role.value
    return ProspectRole.UNKNOWN


# ---------------------------------------------------------------------------
# Knowledge confidence
# ---------------------------------------------------------------------------


def _knowledge_confidence(
    selected: list[ApprovedEvidence],
) -> KnowledgeConfidence:
    if not selected:
        return KnowledgeConfidence.MINIMAL

    qualities = [item.quality for item in selected]
    if all(q == EvidenceQuality.DIRECT for q in qualities):
        return KnowledgeConfidence.HIGH
    if any(q == EvidenceQuality.LIMITED for q in qualities):
        return KnowledgeConfidence.MINIMAL
    return KnowledgeConfidence.MEDIUM


# ---------------------------------------------------------------------------
# Conversation objective
# ---------------------------------------------------------------------------


def _conversation_objective(
    value: SalesKnowledgeInput,
    intent: KnowledgeIntent,
) -> ConversationObjective:
    guidance = value.sales_guidance

    if guidance.objection not in {ObjectionUnderstanding.NONE, ObjectionUnderstanding.UNKNOWN}:
        return ConversationObjective.HANDLE_OBJECTION

    if intent == KnowledgeIntent.COMPARE:
        return ConversationObjective.COMPARE

    if intent == KnowledgeIntent.ANSWER_QUESTION:
        return ConversationObjective.CLARIFY

    if guidance.trust in {TrustState.BUILDING, TrustState.UNKNOWN}:
        return ConversationObjective.BUILD_TRUST

    if intent == KnowledgeIntent.ILLUSTRATE_BENEFIT:
        if guidance.interest in {InterestStrength.STRONG, InterestStrength.BUYING_SIGNAL}:
            return ConversationObjective.CLOSE_GRACEFULLY
        return ConversationObjective.CONFIRM

    if intent == KnowledgeIntent.EDUCATE:
        return ConversationObjective.EDUCATE

    if intent == KnowledgeIntent.BUILD_AWARENESS:
        return ConversationObjective.DISCOVER

    if guidance.discovery_readiness == DiscoveryReadiness.READY_FOR_FIT:
        return ConversationObjective.DISCOVER

    return ConversationObjective.CONTINUE_DISCUSSION


# ---------------------------------------------------------------------------
# Cognitive load
# ---------------------------------------------------------------------------


def _cognitive_load(
    value: SalesKnowledgeInput,
    depth: ExplanationDepth,
) -> CognitiveLoad:
    if depth == ExplanationDepth.SIMPLE:
        return CognitiveLoad.LOW

    role = _effective_role(value)
    if role in {ProspectRole.TECHNICAL_MANAGER, ProspectRole.OPERATIONS_MANAGER}:
        return CognitiveLoad.HIGH

    if depth == ExplanationDepth.DETAILED:
        has_technical = any(
            fact.kind in {BusinessFactKind.SOFTWARE, BusinessFactKind.INTEGRATION}
            for fact in value.conversation.facts
        )
        if has_technical:
            return CognitiveLoad.HIGH

    return CognitiveLoad.NORMAL


# ---------------------------------------------------------------------------
# Momentum signal
# ---------------------------------------------------------------------------


def _momentum_signal(value: SalesKnowledgeInput) -> MomentumSignal:
    guidance = value.sales_guidance
    conversation = value.conversation

    if conversation.pending_topic is not None and conversation.pending_topic != conversation.current_focus:
        return MomentumSignal.PIVOT

    if guidance.momentum in {ConversationMomentum.DECLINING, ConversationMomentum.RECOVERING}:
        return MomentumSignal.STAY_ON_TOPIC

    if guidance.objection not in {ObjectionUnderstanding.NONE, ObjectionUnderstanding.UNKNOWN}:
        return MomentumSignal.STAY_ON_TOPIC

    return MomentumSignal.CONTINUE


# ---------------------------------------------------------------------------
# Knowledge saturation
# ---------------------------------------------------------------------------


def _knowledge_saturated(value: SalesKnowledgeInput) -> bool:
    if not value.already_discussed_evidence_ids:
        return False
    total = len(value.approved_evidence.items)
    if total == 0:
        return True
    discussed = sum(
        1
        for item in value.approved_evidence.items
        if item.approval_id in value.already_discussed_evidence_ids
    )
    return discussed >= total
