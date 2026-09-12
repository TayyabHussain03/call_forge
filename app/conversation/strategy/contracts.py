"""Bounded strategy contracts layered above the authoritative FSM."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.conversation.response_planning.contracts import InterruptionContext
from app.core.constants import ConversationState


class SalesStage(str, Enum):
    """Broad advisory sales progression, independent of the domain FSM."""

    OPENING = "opening"
    DISCOVERY = "discovery"
    QUALIFICATION = "qualification"
    VALUE_EXPLORATION = "value_exploration"
    NEXT_STEP = "next_step"


class ConversationMode(str, Enum):
    """Temporary conversational posture that does not replace sales stage."""

    NORMAL = "normal"
    QUESTION_DETOUR = "question_detour"
    OBJECTION = "objection"
    BUSY = "busy"
    CLARIFICATION = "clarification"


class StrategyType(str, Enum):
    """Generic category for the single recommended conversational objective."""

    OPEN_CONVERSATION = "open_conversation"
    DISCOVER_NEED = "discover_need"
    UNDERSTAND_CURRENT_SOLUTION = "understand_current_solution"
    QUALIFY_NEED = "qualify_need"
    HANDLE_QUESTION = "handle_question"
    HANDLE_OBJECTION = "handle_objection"
    EXPLAIN_VALUE = "explain_value"
    ASK_MICRO_COMMITMENT = "ask_micro_commitment"
    PROPOSE_NEXT_STEP = "propose_next_step"
    CLARIFY = "clarify"


class InformationGap(str, Enum):
    """Typed fact the strategy has not yet received as structured input."""

    CURRENT_WORKFLOW = "current_workflow"
    PAIN_POINT = "pain_point"
    CURRENT_SOLUTION = "current_solution"
    SATISFACTION = "satisfaction"
    DECISION_AUTHORITY = "decision_authority"
    URGENCY = "urgency"
    NEXT_STEP_PREFERENCE = "next_step_preference"


class MicroCommitment(str, Enum):
    """Small next step to recommend, never an accepted confirmation."""

    NONE = "none"
    PERMISSION_TO_CONTINUE = "permission_to_continue"
    ANSWER_ONE_QUESTION = "answer_one_question"
    SEND_INFORMATION = "send_information"
    SHARE_CONTACT = "share_contact"
    CALLBACK = "callback"
    DEMO = "demo"
    HUMAN_FOLLOW_UP = "human_follow_up"


@dataclass(frozen=True)
class ConversationStrategyInput:
    """Structured facts/advisory signals; raw utterance is deliberately absent."""

    current_stage: SalesStage
    current_state: ConversationState
    campaign_goal: str | None = None
    satisfied_information: frozenset[InformationGap] = frozenset()
    pain_present: bool = False
    meaningful_interest: bool = False
    busy: bool = False
    current_solution_present: bool = False
    objection_present: bool = False
    question_present: bool = False
    clarification_needed: bool = False
    explicit_next_step: MicroCommitment | None = None
    interruption: InterruptionContext = InterruptionContext()
    campaign_context_label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.current_stage, SalesStage):
            raise TypeError("current_stage must be a SalesStage")
        if not isinstance(self.current_state, ConversationState):
            raise TypeError("current_state must be a ConversationState")
        information = frozenset(self.satisfied_information)
        if any(not isinstance(gap, InformationGap) for gap in information):
            raise TypeError("satisfied information must contain InformationGap values")
        object.__setattr__(self, "satisfied_information", information)
        if self.explicit_next_step is not None and not isinstance(
            self.explicit_next_step, MicroCommitment
        ):
            raise TypeError("explicit_next_step must be a MicroCommitment")
        if not isinstance(self.interruption, InterruptionContext):
            raise TypeError("interruption must be an InterruptionContext")
        if self.campaign_goal is not None and len(self.campaign_goal) > 200:
            raise ValueError("campaign goal exceeds bounded length")
        if self.campaign_context_label is not None and len(
            self.campaign_context_label
        ) > 100:
            raise ValueError("campaign context label exceeds bounded length")


@dataclass(frozen=True)
class ConversationStrategy:
    """One advisory conversational direction with no execution authority."""

    sales_stage: SalesStage
    conversation_mode: ConversationMode
    communication_goal: str
    strategy_type: StrategyType
    information_gaps: tuple[InformationGap, ...] = ()
    micro_commitment: MicroCommitment = MicroCommitment.NONE
    resume_previous_goal: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.sales_stage, SalesStage):
            raise TypeError("sales_stage must be a SalesStage")
        if not isinstance(self.conversation_mode, ConversationMode):
            raise TypeError("conversation_mode must be a ConversationMode")
        if not isinstance(self.strategy_type, StrategyType):
            raise TypeError("strategy_type must be a StrategyType")
        if not isinstance(self.micro_commitment, MicroCommitment):
            raise TypeError("micro_commitment must be a MicroCommitment")
        gaps = tuple(self.information_gaps)
        if any(not isinstance(gap, InformationGap) for gap in gaps):
            raise TypeError("information gaps must contain InformationGap values")
        object.__setattr__(self, "information_gaps", gaps)
        if not self.communication_goal.strip() or len(self.communication_goal) > 240:
            raise ValueError("strategy communication goal must be bounded")
        if len(gaps) > 3 or len(set(gaps)) != len(gaps):
            raise ValueError("information gaps must be unique and bounded")
