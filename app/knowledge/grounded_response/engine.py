"""Deterministic Grounded Response Composer.

Transforms SKIE's knowledge conversation context into a grounded conversation
plan. All decisions are deterministic — no LLM calls, no retrieval, no state
mutation.
"""

from __future__ import annotations

from app.conversation.business_conversation.contracts import BusinessFactKind
from app.conversation.prospect_intelligence.contracts import ProspectRole
from app.conversation.sales_cognition.contracts import (
    InterestStrength,
    TrustState,
)
from app.conversation.understanding.contracts import ConversationalRegister
from app.knowledge.sales_intelligence.contracts import (
    AnalogyStrategy,
    CognitiveLoad,
    ConversationObjective,
    ExplanationDepth,
    KnowledgeIntent,
    KnowledgePriority,
    MomentumSignal,
)

from app.knowledge.grounded_response.contracts import (
    EvidenceFraming,
    ExplanationTone,
    GroundedConversationPlan,
    GroundedEvidenceItem,
    GroundedResponseInput,
    LanguageAlignment,
    ResponseGuardrail,
    ResponseNaturalness,
    TransitionStyle,
    ValueIntegrationStyle,
)
from app.knowledge.grounded_response.validator import validate_grounded_plan


class GroundedResponseComposer:
    """Produce a grounded conversation plan from SKIE output."""

    def compose(self, value: GroundedResponseInput) -> GroundedConversationPlan:
        kc = value.knowledge_context

        tone = _explanation_tone(value)
        alignment = _language_alignment(value)
        guardrail = _response_guardrail(value, kc.cognitive_load)

        grounded_items = tuple(
            GroundedEvidenceItem(
                evidence=sel.evidence,
                framing=_evidence_framing(value, sel.priority, kc.explanation.knowledge_intent),
                transition=_transition_style(i, len(kc.selected)),
                value_integration=_value_integration(value, sel.priority),
                sequence_position=i,
            )
            for i, sel in enumerate(kc.selected)
        )

        plan = GroundedConversationPlan(
            grounded_items=grounded_items,
            explanation_tone=tone,
            language_alignment=alignment,
            guardrail=guardrail,
            knowledge_context=kc,
        )
        validate_grounded_plan(plan, value)
        return plan


# ---------------------------------------------------------------------------
# Explanation tone
# ---------------------------------------------------------------------------


def _explanation_tone(value: GroundedResponseInput) -> ExplanationTone:
    kc = value.knowledge_context
    objective = kc.conversation_objective

    if objective == ConversationObjective.HANDLE_OBJECTION:
        return ExplanationTone.REASSURING

    if objective == ConversationObjective.BUILD_TRUST:
        return ExplanationTone.CONVERSATIONAL

    if objective == ConversationObjective.CLARIFY:
        return ExplanationTone.DIRECT

    if objective == ConversationObjective.CLOSE_GRACEFULLY:
        return ExplanationTone.CONSULTATIVE

    guidance = value.sales_guidance
    if guidance.trust in {TrustState.BUILDING, TrustState.UNKNOWN}:
        return ExplanationTone.CONVERSATIONAL

    if kc.explanation.knowledge_intent == KnowledgeIntent.EDUCATE:
        return ExplanationTone.EDUCATIONAL

    register = value.language_profile.conversational_register
    if register == ConversationalRegister.FORMAL:
        return ExplanationTone.CONSULTATIVE

    return ExplanationTone.CONVERSATIONAL


# ---------------------------------------------------------------------------
# Language alignment
# ---------------------------------------------------------------------------


def _language_alignment(value: GroundedResponseInput) -> LanguageAlignment:
    kc = value.knowledge_context

    if kc.cognitive_load == CognitiveLoad.LOW:
        return LanguageAlignment.SIMPLIFY

    register = value.language_profile.conversational_register
    if register == ConversationalRegister.FORMAL:
        return LanguageAlignment.ELEVATE

    has_technical = any(
        fact.kind in {BusinessFactKind.SOFTWARE, BusinessFactKind.INTEGRATION}
        for fact in value.conversation.facts
    )
    if has_technical and kc.cognitive_load == CognitiveLoad.HIGH:
        return LanguageAlignment.MIRROR

    if register == ConversationalRegister.CASUAL:
        return LanguageAlignment.MIRROR

    return LanguageAlignment.NEUTRAL


# ---------------------------------------------------------------------------
# Response guardrail
# ---------------------------------------------------------------------------


def _response_guardrail(
    value: GroundedResponseInput,
    cognitive_load: CognitiveLoad,
) -> ResponseGuardrail:
    kc = value.knowledge_context

    if cognitive_load == CognitiveLoad.LOW:
        max_concepts = 1
    elif cognitive_load == CognitiveLoad.HIGH:
        max_concepts = 3
    else:
        max_concepts = 2

    avoid_jargon = kc.explanation.depth == ExplanationDepth.SIMPLE
    avoid_feature_listing = kc.explanation.depth != ExplanationDepth.DETAILED
    require_benefit = kc.explanation.knowledge_intent in {
        KnowledgeIntent.EDUCATE,
        KnowledgeIntent.ILLUSTRATE_BENEFIT,
        KnowledgeIntent.BUILD_AWARENESS,
    }

    if kc.knowledge_saturated:
        naturalness = ResponseNaturalness.MINIMAL
    elif value.sales_guidance.trust in {TrustState.BUILDING, TrustState.UNKNOWN}:
        naturalness = ResponseNaturalness.CAUTIOUS
    else:
        naturalness = ResponseNaturalness.NATURAL

    return ResponseGuardrail(
        max_concepts_per_sentence=max_concepts,
        avoid_jargon=avoid_jargon,
        avoid_feature_listing=avoid_feature_listing,
        require_benefit_framing=require_benefit,
        naturalness=naturalness,
    )


# ---------------------------------------------------------------------------
# Evidence framing
# ---------------------------------------------------------------------------


def _evidence_framing(
    value: GroundedResponseInput,
    priority: KnowledgePriority,
    intent: KnowledgeIntent,
) -> EvidenceFraming:
    kc = value.knowledge_context

    if intent == KnowledgeIntent.ANSWER_QUESTION:
        return EvidenceFraming.QUESTION_ANSWER

    if intent == KnowledgeIntent.COMPARE:
        return EvidenceFraming.CONTRAST

    if kc.explanation.analogy in {
        AnalogyStrategy.INDUSTRY_EXAMPLE,
        AnalogyStrategy.EVERYDAY_COMPARISON,
    }:
        return EvidenceFraming.EXAMPLE

    if kc.explanation.analogy == AnalogyStrategy.WORKFLOW_PARALLEL:
        return EvidenceFraming.STORY

    if priority == KnowledgePriority.PRIMARY:
        return EvidenceFraming.DIRECT_STATEMENT

    return EvidenceFraming.DIRECT_STATEMENT


# ---------------------------------------------------------------------------
# Transition style
# ---------------------------------------------------------------------------


def _transition_style(position: int, total: int) -> TransitionStyle:
    if total <= 1 or position == 0:
        return TransitionStyle.NONE

    if position == 1:
        return TransitionStyle.BUILDING_ON

    if position == total - 1:
        return TransitionStyle.RELATED_POINT

    return TransitionStyle.NATURAL_BRIDGE


# ---------------------------------------------------------------------------
# Value integration
# ---------------------------------------------------------------------------


def _value_integration(
    value: GroundedResponseInput,
    priority: KnowledgePriority,
) -> ValueIntegrationStyle:
    kc = value.knowledge_context

    if kc.conversation_objective == ConversationObjective.CLOSE_GRACEFULLY:
        return ValueIntegrationStyle.LEADING

    if kc.conversation_objective == ConversationObjective.HANDLE_OBJECTION:
        return ValueIntegrationStyle.CLOSING_HOOK

    if priority == KnowledgePriority.PRIMARY:
        return ValueIntegrationStyle.EMBEDDED

    if kc.explanation.knowledge_intent == KnowledgeIntent.BUILD_AWARENESS:
        return ValueIntegrationStyle.CLOSING_HOOK

    if kc.momentum == MomentumSignal.PIVOT:
        return ValueIntegrationStyle.LEADING

    return ValueIntegrationStyle.NONE
