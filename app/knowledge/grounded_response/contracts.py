"""Immutable contracts for the Grounded Response Composer.

GRC transforms SKIE's KnowledgeConversationContext into a grounded conversation
plan that ensures: explanations are natural, only approved evidence is used,
business value is woven in, customer language is maintained, and responses never
read like a brochure.

TRUST BOUNDARY: all inputs are trusted (SKIE output, deterministic snapshots).
Output is advisory context consumed by Response Planning and Human Conversation
Realization.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.conversation.business_conversation.contracts import (
    BusinessConversationSnapshot,
)
from app.conversation.consultative.contracts import ProspectProblem
from app.conversation.prospect_intelligence.contracts import (
    ProspectIntelligenceSummary,
)
from app.conversation.sales_cognition.contracts import SalesConversationGuidance
from app.conversation.sales_playbook.contracts import OpportunityGuidance
from app.conversation.understanding.contracts import LanguageProfile
from app.knowledge.evidence_validation.contracts import ApprovedEvidence
from app.knowledge.sales_intelligence.contracts import (
    KnowledgeConversationContext,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ExplanationTone(str, Enum):
    """Natural tone for the knowledge explanation."""

    CONVERSATIONAL = "conversational"
    CONSULTATIVE = "consultative"
    EDUCATIONAL = "educational"
    REASSURING = "reassuring"
    DIRECT = "direct"


class ValueIntegrationStyle(str, Enum):
    """How business value is woven into the explanation."""

    EMBEDDED = "embedded"
    CLOSING_HOOK = "closing_hook"
    LEADING = "leading"
    NONE = "none"


class LanguageAlignment(str, Enum):
    """How closely the response should mirror the customer's language."""

    MIRROR = "mirror"
    SIMPLIFY = "simplify"
    ELEVATE = "elevate"
    NEUTRAL = "neutral"


class ResponseNaturalness(str, Enum):
    """Guard against brochure-like or robotic responses."""

    NATURAL = "natural"
    CAUTIOUS = "cautious"
    MINIMAL = "minimal"


class EvidenceFraming(str, Enum):
    """How an individual evidence item should be presented."""

    DIRECT_STATEMENT = "direct_statement"
    EXAMPLE = "example"
    QUESTION_ANSWER = "question_answer"
    STORY = "story"
    CONTRAST = "contrast"


class TransitionStyle(str, Enum):
    """How to transition between evidence items when multiple are selected."""

    NONE = "none"
    NATURAL_BRIDGE = "natural_bridge"
    BUILDING_ON = "building_on"
    RELATED_POINT = "related_point"


# ---------------------------------------------------------------------------
# Evidence grounding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundedEvidenceItem:
    """One evidence item with its grounding instructions."""

    evidence: ApprovedEvidence
    framing: EvidenceFraming
    transition: TransitionStyle
    value_integration: ValueIntegrationStyle
    sequence_position: int

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, ApprovedEvidence):
            raise TypeError("evidence must be approved evidence")
        if not isinstance(self.framing, EvidenceFraming):
            raise TypeError("framing must be typed")
        if not isinstance(self.transition, TransitionStyle):
            raise TypeError("transition must be typed")
        if not isinstance(self.value_integration, ValueIntegrationStyle):
            raise TypeError("value integration must be typed")
        if not isinstance(self.sequence_position, int) or self.sequence_position < 0:
            raise ValueError("sequence position must be a non-negative integer")


@dataclass(frozen=True)
class ResponseGuardrail:
    """Deterministic guardrails for grounded response generation."""

    max_concepts_per_sentence: int
    avoid_jargon: bool
    avoid_feature_listing: bool
    require_benefit_framing: bool
    naturalness: ResponseNaturalness

    def __post_init__(self) -> None:
        if not 1 <= self.max_concepts_per_sentence <= 3:
            raise ValueError("max concepts per sentence must be 1-3")
        if not isinstance(self.naturalness, ResponseNaturalness):
            raise TypeError("naturalness must be typed")


# ---------------------------------------------------------------------------
# Input / Output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundedResponseInput:
    """All trusted inputs GRC consumes."""

    knowledge_context: KnowledgeConversationContext
    prospect: ProspectIntelligenceSummary
    problem: ProspectProblem
    sales_guidance: SalesConversationGuidance
    playbook_guidance: OpportunityGuidance | None
    language_profile: LanguageProfile
    conversation: BusinessConversationSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.knowledge_context, KnowledgeConversationContext):
            raise TypeError("knowledge context must be a knowledge conversation context")
        if not isinstance(self.prospect, ProspectIntelligenceSummary):
            raise TypeError("prospect must be a prospect intelligence summary")
        if not isinstance(self.problem, ProspectProblem):
            raise TypeError("problem must be a prospect problem")
        if not isinstance(self.sales_guidance, SalesConversationGuidance):
            raise TypeError("sales guidance must be sales conversation guidance")
        if not isinstance(self.language_profile, LanguageProfile):
            raise TypeError("language profile must be a language profile")
        if not isinstance(self.conversation, BusinessConversationSnapshot):
            raise TypeError("conversation must be a business conversation snapshot")


@dataclass(frozen=True)
class GroundedConversationPlan:
    """GRC output: a grounded conversation plan for Response Planning.

    This is advisory context. It carries no transition, persistence, service
    authorization, pricing, wording, or execution authority.
    """

    grounded_items: tuple[GroundedEvidenceItem, ...]
    explanation_tone: ExplanationTone
    language_alignment: LanguageAlignment
    guardrail: ResponseGuardrail
    knowledge_context: KnowledgeConversationContext

    def __post_init__(self) -> None:
        items = tuple(self.grounded_items)
        if len(items) > 4:
            raise ValueError("grounded items must not exceed four")
        if any(not isinstance(item, GroundedEvidenceItem) for item in items):
            raise TypeError("grounded items must be grounded evidence items")
        if not isinstance(self.explanation_tone, ExplanationTone):
            raise TypeError("explanation tone must be typed")
        if not isinstance(self.language_alignment, LanguageAlignment):
            raise TypeError("language alignment must be typed")
        if not isinstance(self.guardrail, ResponseGuardrail):
            raise TypeError("guardrail must be typed")
        if not isinstance(self.knowledge_context, KnowledgeConversationContext):
            raise TypeError("knowledge context must be typed")

        positions = [item.sequence_position for item in items]
        if positions != sorted(positions) or len(set(positions)) != len(positions):
            raise ValueError("grounded items must have unique ascending positions")

        object.__setattr__(self, "grounded_items", items)
