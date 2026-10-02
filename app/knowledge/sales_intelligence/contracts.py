"""Immutable contracts for the Sales Knowledge Intelligence Engine.

SKIE transforms approved evidence into a structured conversational teaching
strategy. It decides WHAT knowledge to present, HOW MUCH, in what ORDER, and at
what DEPTH — but never generates wording, authorizes services, retrieves
knowledge, or mutates any upstream state.

TRUST BOUNDARY: all inputs are trusted (approved evidence, deterministic
intelligence snapshots). Output is advisory context consumed by Response
Planning and Human Conversation Realization.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.conversation.business_conversation.contracts import (
    BusinessConversationSnapshot,
    BusinessFactKind,
)
from app.conversation.business_diagnostic.contracts import BusinessDiagnosticSnapshot
from app.conversation.business_memory.contracts import BusinessMentalModelSnapshot
from app.conversation.consultative.contracts import (
    ProspectProblem,
    ServiceFitDecision,
)
from app.conversation.conversation_steering.contracts import (
    ConversationPrioritySnapshot,
)
from app.conversation.prospect_intelligence.contracts import (
    ProspectIntelligenceSummary,
)
from app.conversation.qualification.contracts import QualificationSnapshot
from app.conversation.sales_cognition.contracts import SalesConversationGuidance
from app.conversation.sales_playbook.contracts import OpportunityGuidance
from app.conversation.understanding.contracts import LanguageProfile
from app.knowledge.evidence_validation.contracts import (
    ApprovedEvidence,
    ApprovedEvidenceSet,
    EvidenceGroup,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ExplanationDepth(str, Enum):
    """How deep the explanation should go for this turn."""

    SIMPLE = "simple"
    NORMAL = "normal"
    DETAILED = "detailed"


class DisclosureLevel(str, Enum):
    """How many pieces of knowledge to expose in this turn."""

    MINIMAL = "minimal"
    MODERATE = "moderate"
    FULL = "full"


class KnowledgeIntent(str, Enum):
    """The primary educational objective for this turn's knowledge."""

    EDUCATE = "educate"
    COMPARE = "compare"
    ANSWER_QUESTION = "answer_question"
    ILLUSTRATE_BENEFIT = "illustrate_benefit"
    ADDRESS_MISCONCEPTION = "address_misconception"
    BUILD_AWARENESS = "build_awareness"


class SuppressionReason(str, Enum):
    """Why a piece of approved evidence was suppressed for this turn."""

    OFF_TOPIC = "off_topic"
    PREMATURE = "premature"
    ALREADY_DISCUSSED = "already_discussed"
    OVERLOAD = "overload"
    CUSTOMER_NOT_READY = "customer_not_ready"
    WRONG_DEPTH = "wrong_depth"


class AnalogyStrategy(str, Enum):
    """Whether and how to use an analogy in explanation."""

    NONE = "none"
    INDUSTRY_EXAMPLE = "industry_example"
    EVERYDAY_COMPARISON = "everyday_comparison"
    WORKFLOW_PARALLEL = "workflow_parallel"


class FollowUpStyle(str, Enum):
    """How to offer the next knowledge step without dumping."""

    NONE = "none"
    OFFER_NEXT_TOPIC = "offer_next_topic"
    ASK_PERMISSION = "ask_permission"
    CURIOSITY_HOOK = "curiosity_hook"


class KnowledgePriority(str, Enum):
    """Relative importance of this evidence item in the teaching plan."""

    PRIMARY = "primary"
    SUPPORTING = "supporting"
    DEFERRED = "deferred"


class BusinessValueFocus(str, Enum):
    """The business outcome lens for presenting knowledge."""

    REVENUE = "revenue"
    COST_SAVINGS = "cost_savings"
    TIME_SAVINGS = "time_savings"
    CUSTOMER_EXPERIENCE = "customer_experience"
    OPERATIONAL_EFFICIENCY = "operational_efficiency"
    VISIBILITY = "visibility"
    TRUST_BUILDING = "trust_building"
    GROWTH = "growth"
    NONE = "none"


class ResponseComplexity(str, Enum):
    """Overall complexity of the knowledge response."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    TECHNICAL = "technical"


# ---------------------------------------------------------------------------
# Evidence selection contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectedEvidence:
    """One approved evidence item selected for this turn's teaching plan."""

    evidence: ApprovedEvidence
    priority: KnowledgePriority
    business_value_focus: BusinessValueFocus
    sequence_position: int

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, ApprovedEvidence):
            raise TypeError("selected evidence must be approved evidence")
        if not isinstance(self.priority, KnowledgePriority):
            raise TypeError("evidence priority must be typed")
        if not isinstance(self.business_value_focus, BusinessValueFocus):
            raise TypeError("business value focus must be typed")
        if not isinstance(self.sequence_position, int) or self.sequence_position < 0:
            raise ValueError("sequence position must be a non-negative integer")


@dataclass(frozen=True)
class SuppressedEvidence:
    """One approved evidence item suppressed for this turn."""

    evidence: ApprovedEvidence
    reason: SuppressionReason

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, ApprovedEvidence):
            raise TypeError("suppressed evidence must be approved evidence")
        if not isinstance(self.reason, SuppressionReason):
            raise TypeError("suppression reason must be typed")


# ---------------------------------------------------------------------------
# Teaching plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExplanationPlan:
    """The structured teaching strategy for selected evidence."""

    depth: ExplanationDepth
    knowledge_intent: KnowledgeIntent
    analogy: AnalogyStrategy
    business_value_focus: BusinessValueFocus
    response_complexity: ResponseComplexity

    def __post_init__(self) -> None:
        if not isinstance(self.depth, ExplanationDepth):
            raise TypeError("explanation depth must be typed")
        if not isinstance(self.knowledge_intent, KnowledgeIntent):
            raise TypeError("knowledge intent must be typed")
        if not isinstance(self.analogy, AnalogyStrategy):
            raise TypeError("analogy strategy must be typed")
        if not isinstance(self.business_value_focus, BusinessValueFocus):
            raise TypeError("business value focus must be typed")
        if not isinstance(self.response_complexity, ResponseComplexity):
            raise TypeError("response complexity must be typed")


@dataclass(frozen=True)
class ProgressiveDisclosurePlan:
    """How much to reveal now and how to offer more later."""

    disclosure_level: DisclosureLevel
    max_evidence_this_turn: int
    follow_up: FollowUpStyle
    deferred_group_hint: EvidenceGroup | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.disclosure_level, DisclosureLevel):
            raise TypeError("disclosure level must be typed")
        if not 1 <= self.max_evidence_this_turn <= 4:
            raise ValueError("max evidence per turn must be between one and four")
        if not isinstance(self.follow_up, FollowUpStyle):
            raise TypeError("follow up style must be typed")
        if self.deferred_group_hint is not None and not isinstance(
            self.deferred_group_hint, EvidenceGroup
        ):
            raise TypeError("deferred group hint must be an evidence group")


# ---------------------------------------------------------------------------
# Input / Output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SalesKnowledgeInput:
    """All trusted inputs SKIE consumes. No transcript, no retrieval, no HRE."""

    approved_evidence: ApprovedEvidenceSet
    prospect: ProspectIntelligenceSummary
    problem: ProspectProblem
    service_fit: ServiceFitDecision
    sales_guidance: SalesConversationGuidance
    playbook_guidance: OpportunityGuidance | None
    language_profile: LanguageProfile
    conversation: BusinessConversationSnapshot
    diagnostic: BusinessDiagnosticSnapshot | None = None
    steering: ConversationPrioritySnapshot | None = None
    qualification: QualificationSnapshot | None = None
    business_memory: BusinessMentalModelSnapshot | None = None
    already_discussed_evidence_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.approved_evidence, ApprovedEvidenceSet):
            raise TypeError("approved evidence must be an approved evidence set")
        if not isinstance(self.prospect, ProspectIntelligenceSummary):
            raise TypeError("prospect must be a prospect intelligence summary")
        if not isinstance(self.problem, ProspectProblem):
            raise TypeError("problem must be a prospect problem")
        if not isinstance(self.service_fit, ServiceFitDecision):
            raise TypeError("service fit must be a service fit decision")
        if not isinstance(self.sales_guidance, SalesConversationGuidance):
            raise TypeError("sales guidance must be sales conversation guidance")
        if not isinstance(self.language_profile, LanguageProfile):
            raise TypeError("language profile must be a language profile")
        if not isinstance(self.conversation, BusinessConversationSnapshot):
            raise TypeError("conversation must be a business conversation snapshot")
        ids = frozenset(self.already_discussed_evidence_ids)
        if any(not item.strip() or len(item) > 100 for item in ids):
            raise ValueError("already discussed evidence ids must be bounded")
        object.__setattr__(self, "already_discussed_evidence_ids", ids)


@dataclass(frozen=True)
class KnowledgeConversationContext:
    """SKIE output: a complete conversational teaching strategy.

    This is advisory context for Response Planning and Human Conversation
    Realization. It carries no transition, persistence, service authorization,
    pricing, wording, or execution authority.
    """

    selected: tuple[SelectedEvidence, ...]
    suppressed: tuple[SuppressedEvidence, ...]
    explanation: ExplanationPlan
    disclosure: ProgressiveDisclosurePlan

    def __post_init__(self) -> None:
        selected = tuple(self.selected)
        suppressed = tuple(self.suppressed)
        if len(selected) > 4:
            raise ValueError("selected evidence must not exceed four items")
        if any(not isinstance(item, SelectedEvidence) for item in selected):
            raise TypeError("selected items must be selected evidence")
        if any(not isinstance(item, SuppressedEvidence) for item in suppressed):
            raise TypeError("suppressed items must be suppressed evidence")
        if not isinstance(self.explanation, ExplanationPlan):
            raise TypeError("explanation must be an explanation plan")
        if not isinstance(self.disclosure, ProgressiveDisclosurePlan):
            raise TypeError("disclosure must be a progressive disclosure plan")

        positions = [item.sequence_position for item in selected]
        if positions != sorted(positions) or len(set(positions)) != len(positions):
            raise ValueError("selected evidence must have unique ascending positions")

        selected_ids = {item.evidence.approval_id for item in selected}
        suppressed_ids = {item.evidence.approval_id for item in suppressed}
        if selected_ids & suppressed_ids:
            raise ValueError("evidence cannot be both selected and suppressed")

        if len(selected) > self.disclosure.max_evidence_this_turn:
            raise ValueError(
                "selected evidence exceeds disclosure plan limit"
            )

        object.__setattr__(self, "selected", selected)
        object.__setattr__(self, "suppressed", suppressed)
