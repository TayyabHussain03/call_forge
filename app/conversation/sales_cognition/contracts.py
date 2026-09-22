"""Immutable bounded contracts for advisory sales-conversation cognition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.conversation.consultative.contracts import (
    ConsultativeConversationDecision,
    ProblemField,
    ProspectProblem,
    ServiceFitDecision,
)
from app.conversation.prospect_intelligence.contracts import ProspectIntelligenceSummary
from app.conversation.response_planning.contracts import ResponseLength
from app.conversation.strategy.contracts import ConversationStrategy


class ConversationMomentum(str, Enum):
    UNKNOWN = "unknown"
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    RECOVERING = "recovering"


class TrustState(str, Enum):
    UNKNOWN = "unknown"
    BUILDING = "building"
    ESTABLISHED = "established"
    FRAGILE = "fragile"


class InterestStrength(str, Enum):
    UNKNOWN = "unknown"
    NONE = "none"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    BUYING_SIGNAL = "buying_signal"
    RESEARCH_MODE = "research_mode"


class DiscoveryReadiness(str, Enum):
    NEED_PROBLEM = "need_problem"
    NEED_WORKFLOW = "need_workflow"
    NEED_IMPACT = "need_impact"
    NEED_DECISION_CONTEXT = "need_decision_context"
    READY_FOR_FIT = "ready_for_fit"
    READY_FOR_NEXT_STEP = "ready_for_next_step"


class ObjectionUnderstanding(str, Enum):
    NONE = "none"
    NO_TIME = "no_time"
    BUDGET = "budget"
    EXISTING_PROVIDER = "existing_provider"
    NO_NEED = "no_need"
    NO_TRUST = "no_trust"
    INTERNAL_TEAM = "internal_team"
    CONTRACT_LOCK = "contract_lock"
    RESEARCH_ONLY = "research_only"
    UNKNOWN = "unknown"


class ConversationEnergy(str, Enum):
    FAST = "fast"
    NORMAL = "normal"
    SLOW = "slow"
    FATIGUED = "fatigued"


class QuestionPriority(str, Enum):
    NONE = "none"
    PROBLEM = "problem"
    WORKFLOW = "workflow"
    IMPACT = "impact"
    PROVIDER_GAP = "provider_gap"
    DECISION_CONTEXT = "decision_context"
    FIT_CLARIFICATION = "fit_clarification"


class RelationshipState(str, Enum):
    FIRST_CONTACT = "first_contact"
    DEVELOPING = "developing"
    RETURNING_DISCUSSION = "returning_discussion"
    KNOWN_CONTEXT = "known_context"


class PressureState(str, Enum):
    COMFORTABLE = "comfortable"
    NEEDS_SLOWING = "needs_slowing"
    NEEDS_CLARIFICATION = "needs_clarification"
    READY_TO_CONTINUE = "ready_to_continue"


class CuriosityFocus(str, Enum):
    UNDERLYING_PROBLEM = "underlying_problem"
    CURRENT_WORKFLOW = "current_workflow"
    BUSINESS_IMPACT = "business_impact"
    PROVIDER_EXPERIENCE = "provider_experience"
    ROLE_ROUTING = "role_routing"
    FIT_UNCERTAINTY = "fit_uncertainty"
    NONE = "none"


class RoleConversationStyle(str, Enum):
    BRIEF_ROUTING = "brief_routing"
    OWNER_VALUE = "owner_value"
    OPERATIONS_WORKFLOW = "operations_workflow"
    TECHNICAL_PRECISION = "technical_precision"
    SALES_PROCESS = "sales_process"
    FINANCIAL_RESTRAINT = "financial_restraint"
    SUPPORTIVE = "supportive"
    GENERAL_CONSULTATIVE = "general_consultative"


class EmotionalPosture(str, Enum):
    NEUTRAL = "neutral"
    CONFUSED = "confused"
    SKEPTICAL = "skeptical"
    FRUSTRATED = "frustrated"
    CURIOUS = "curious"
    BUSY = "busy"
    INTERESTED = "interested"


class BuyingReadinessGuidance(str, Enum):
    CONTINUE_DISCOVERY = "continue_discovery"
    EXPLAIN_FIT = "explain_fit"
    HANDLE_OBJECTION = "handle_objection"
    SUGGEST_MICRO_COMMITMENT = "suggest_micro_commitment"
    GRACEFUL_CLOSE = "graceful_close"


@dataclass(frozen=True)
class CognitionSignals:
    """Trusted typed current-turn evidence; raw-text classification is excluded."""

    confused: bool = False
    skeptical: bool = False
    frustrated: bool = False
    curious: bool = False
    fatigued: bool = False
    correction: bool = False
    research_only: bool = False
    explicit_no_need: bool = False
    detailed_explanation_requested: bool = False

    def __post_init__(self) -> None:
        if any(not isinstance(value, bool) for value in self.__dict__.values()):
            raise TypeError("cognition signals must be boolean")


@dataclass(frozen=True)
class SalesCognitionInput:
    prospect: ProspectIntelligenceSummary
    problem: ProspectProblem
    service_fit: ServiceFitDecision
    strategy: ConversationStrategy | None
    consultative_decision: ConsultativeConversationDecision
    signals: CognitionSignals = CognitionSignals()
    recent_question_concepts: tuple[ProblemField, ...] = ()
    prior_guidance: SalesConversationGuidance | None = None
    returning_discussion: bool = False

    def __post_init__(self) -> None:
        concepts = tuple(self.recent_question_concepts)
        if len(concepts) > 4 or any(
            not isinstance(item, ProblemField) for item in concepts
        ):
            raise ValueError("recent question concepts must be typed and bounded")
        if not isinstance(self.returning_discussion, bool):
            raise TypeError("returning discussion flag must be boolean")
        object.__setattr__(self, "recent_question_concepts", concepts)


@dataclass(frozen=True)
class SalesConversationGuidance:
    """Advisory communication intelligence with no execution or authority fields."""

    momentum: ConversationMomentum
    trust: TrustState
    interest: InterestStrength
    discovery_readiness: DiscoveryReadiness
    objection: ObjectionUnderstanding
    energy: ConversationEnergy
    question_priority: QuestionPriority
    question_focus: ProblemField | None
    relationship: RelationshipState
    pressure: PressureState
    curiosity_focus: CuriosityFocus
    role_style: RoleConversationStyle
    emotional_posture: EmotionalPosture
    buying_guidance: BuyingReadinessGuidance
    recommended_response_depth: ResponseLength
    max_primary_questions: int = 1

    def __post_init__(self) -> None:
        if self.max_primary_questions != 1:
            raise ValueError("sales cognition permits exactly one primary question")
        if (self.question_priority == QuestionPriority.NONE) != (
            self.question_focus is None
        ):
            raise ValueError("question priority and focus must agree")
