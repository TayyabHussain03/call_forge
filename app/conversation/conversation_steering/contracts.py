"""Provider-neutral, advisory-only contracts for discovery steering."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.conversation.business_conversation.contracts import BusinessConversationSnapshot
from app.conversation.business_diagnostic.contracts import BusinessDiagnosticSnapshot
from app.conversation.prospect_intelligence.contracts import ProspectIntelligenceSummary
from app.conversation.strategy.contracts import ConversationStrategy
from app.core.constants import ConversationState


class DiagnosticPriority(str, Enum):
    WEBSITE_PERFORMANCE = "website_performance"
    SALES_PROCESS = "sales_process"
    LEAD_HANDLING = "lead_handling"
    WORKFLOW = "workflow"
    REPORTING = "reporting"
    AUTOMATION_READINESS = "automation_readiness"
    DECISION_PROCESS = "decision_process"
    MARKETING_VISIBILITY = "marketing_visibility"
    BUSINESS_CONTEXT = "business_context"


class PriorityEvolution(str, Enum):
    UNCHANGED = "unchanged"
    RAISED = "raised"
    LOWERED = "lowered"
    REPLACED = "replaced"
    DEFERRED = "deferred"
    BLOCKED = "blocked"


class DiscoveryStatus(str, Enum):
    DISCOVERY_REQUIRED = "discovery_required"
    DISCOVERY_IN_PROGRESS = "discovery_in_progress"
    DISCOVERY_COMPLETE = "discovery_complete"
    DISCOVERY_DEFERRED = "discovery_deferred"
    DISCOVERY_BLOCKED = "discovery_blocked"


class ConversationReadiness(str, Enum):
    UNDERSTAND_MORE = "understand_more"
    READY_TO_EDUCATE = "ready_to_educate"
    READY_FOR_VALUE_DISCUSSION = "ready_for_value_discussion"
    READY_FOR_PLAYBOOK = "ready_for_playbook"
    NO_ACTION = "no_action"


class CuriosityBudgetState(str, Enum):
    OPEN = "open"
    LIMITED = "limited"
    EXHAUSTED = "exhausted"


class QuestionDecision(str, Enum):
    ASK_ONE = "ask_one"
    ASK_NONE = "ask_none"
    DEFER = "defer"
    WAIT_FOR_CUSTOMER = "wait_for_customer"
    CHANGE_TOPIC = "change_topic"


class QuestionReason(str, Enum):
    MISSING_WORKFLOW = "missing_workflow"
    NEED_CLARIFICATION = "need_clarification"
    CUSTOMER_ALREADY_ANSWERED = "customer_already_answered"
    REPEATED_TOPIC = "repeated_topic"
    NATURAL_INTERRUPTION = "natural_interruption"
    DIRECT_QUESTION = "direct_question"
    NEED_CONFIRMATION = "need_confirmation"
    CONVERSATION_FATIGUE = "conversation_fatigue"


class DiscussionTiming(str, Enum):
    PREMATURE = "premature"
    NATURAL = "natural"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class CustomerFatigue(str, Enum):
    ENGAGED = "engaged"
    NEUTRAL = "neutral"
    QUESTION_FATIGUE = "question_fatigue"
    TOPIC_FATIGUE = "topic_fatigue"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CuriosityMemory:
    explored: frozenset[DiagnosticPriority] = frozenset()
    still_unknown: frozenset[DiagnosticPriority] = frozenset()
    refused: frozenset[DiagnosticPriority] = frozenset()
    deferred: frozenset[DiagnosticPriority] = frozenset()
    corrected: frozenset[DiagnosticPriority] = frozenset()


@dataclass(frozen=True)
class ConversationSteeringInput:
    conversation: BusinessConversationSnapshot
    diagnostic: BusinessDiagnosticSnapshot
    prospect: ProspectIntelligenceSummary | None
    strategy: ConversationStrategy | None
    current_state: ConversationState
    prior: ConversationPrioritySnapshot | None = None


@dataclass(frozen=True)
class ConversationPrioritySnapshot:
    """One non-commercial next-understanding objective, never an action or service."""

    current_priority: DiagnosticPriority
    evolution: PriorityEvolution
    previous_priority: DiagnosticPriority | None
    pending_priority: DiagnosticPriority | None
    discovery_status: DiscoveryStatus
    readiness: ConversationReadiness
    curiosity_budget: CuriosityBudgetState
    question_decision: QuestionDecision
    question_reason: QuestionReason
    discussion_timing: DiscussionTiming
    fatigue: CustomerFatigue
    memory: CuriosityMemory

    def __post_init__(self) -> None:
        forbidden = {"service", "product", "action", "authority", "state", "price", "execution", "recommend"}
        if any(any(word in field for word in forbidden) for field in self.__dataclass_fields__):
            raise ValueError("steering contract contains forbidden fields")
