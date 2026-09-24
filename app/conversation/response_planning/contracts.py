"""Typed, bounded contracts for post-decision conversation strategy.

These contracts describe how to communicate an authoritative result. They carry
no transition, persistence, contact-confirmation, service, pricing, discount, or
human-approval authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from app.core.constants import AgentAction, ConversationState, Tone

if TYPE_CHECKING:
    from app.conversation.consultative.contracts import (
        ConsultativeConversationDecision,
        ServiceAnswerContext,
    )
    from app.conversation.escalation.contracts import EscalationDecision
    from app.conversation.sales_cognition.contracts import SalesConversationGuidance
    from app.conversation.sales_playbook.contracts import OpportunityGuidance
    from app.conversation.understanding.contracts import LanguageProfile


class ResponseLength(str, Enum):
    """Adaptive amount of explanation; no fixed sentence count is implied."""

    SHORT = "short"
    MODERATE = "moderate"
    DETAILED = "detailed"


class ExplanationNeed(str, Enum):
    """Trusted/advisory indication of how much useful explanation is needed."""

    SIMPLE = "simple"
    STANDARD = "standard"
    COMPLEX = "complex"


class InterruptionCategory(str, Enum):
    """Advisory semantic classification; it grants no execution authority."""

    ANSWER = "answer"
    QUESTION = "question"
    OBJECTION = "objection"
    CORRECTION = "correction"
    TOPIC_SHIFT = "topic_shift"
    CLARIFICATION = "clarification"
    ADDRESSEE_UNCERTAIN = "addressee_uncertain"
    OTHER = "other"


class AddresseeStatus(str, Enum):
    """Who a prospect utterance is understood to address."""

    ADDRESSED_TO_AGENT = "addressed_to_agent"
    NOT_ADDRESSED_TO_AGENT = "not_addressed_to_agent"
    ADDRESSEE_UNCERTAIN = "addressee_uncertain"


class AcknowledgementKind(str, Enum):
    """Optional acknowledgement style, kept semantic rather than final prose."""

    NONE = "none"
    GOT_IT = "got_it"
    MAKES_SENSE = "makes_sense"
    OKAY = "okay"
    RIGHT = "right"
    UNDERSTOOD = "understood"


class ConversationMove(str, Enum):
    """The single primary communication objective for the next response."""

    COMMUNICATE_RESULT = "communicate_result"
    ANSWER_CURRENT_QUESTION = "answer_current_question"
    EXPLORE_OBJECTION = "explore_objection"
    INCORPORATE_CORRECTION = "incorporate_correction"
    FOLLOW_NEW_DIRECTION = "follow_new_direction"
    CLARIFY_MEANING = "clarify_meaning"
    CLARIFY_ADDRESSEE = "clarify_addressee"
    CONTINUE_PRIOR_CONTEXT = "continue_prior_context"
    SAFE_RECOVERY = "safe_recovery"
    REDIRECT_SAFELY = "redirect_safely"
    ACKNOWLEDGE_ESCALATION = "acknowledge_escalation"
    ANSWER_APPROVED_EVIDENCE = "answer_approved_evidence"
    ACKNOWLEDGE_UNCERTAINTY = "acknowledge_uncertainty"
    OFFER_SUPPORTED_NEXT_STEP = "offer_supported_next_step"
    POLITE_WRAP_UP = "polite_wrap_up"
    CONSULTATIVE_DISCOVERY = "consultative_discovery"
    EXPLAIN_RELEVANT_FIT = "explain_relevant_fit"
    LOW_PRESSURE_CONTINUATION = "low_pressure_continuation"
    LANGUAGE_RECOVERY = "language_recovery"
    TRUTHFUL_COMMERCIAL_ANSWER = "truthful_commercial_answer"


class QuestionStrategy(str, Enum):
    """Whether and why a follow-up question is useful."""

    NONE = "none"
    ANSWER_THEN_FOLLOW_UP = "answer_then_follow_up"
    EXPLORE_WITH_ONE_QUESTION = "explore_with_one_question"
    CLARIFY_CURRENT_INPUT = "clarify_current_input"
    CONFIRM_ADDRESSEE = "confirm_addressee"


class InterruptionHandling(str, Enum):
    """How an unfinished prior point relates to the new response."""

    NONE = "none"
    INTEGRATE_IF_USEFUL = "integrate_if_useful"
    DROP_STALE_POINT = "drop_stale_point"
    HOLD_PRIOR_CONTEXT = "hold_prior_context"


class AuthoritativeResultKind(str, Enum):
    """Normalized read-only result supplied by the deterministic pipeline."""

    EXECUTED = "executed"
    FALLBACK = "fallback"
    REDIRECT = "redirect"
    ESCALATE = "escalate"
    PIPELINE_STOPPED = "pipeline_stopped"
    AUTHORITY_APPROVED = "authority_approved"


@dataclass(frozen=True)
class VoiceActivityMetadata:
    """Future-runtime metadata only; no audio processing occurs here."""

    barge_in_detected: bool = False
    overlapping_speech: bool = False
    speech_started_during_agent_output: bool = False


@dataclass(frozen=True)
class PendingConversationIntent:
    """At most one bounded unfinished conversational intent, never reasoning."""

    goal: str
    summary: str
    completed: bool = False
    still_relevant: bool = True

    def __post_init__(self) -> None:
        if not self.goal.strip() or len(self.goal) > 120:
            raise ValueError("pending goal must contain 1-120 characters")
        if not self.summary.strip() or len(self.summary) > 240:
            raise ValueError("pending summary must contain 1-240 characters")

    @property
    def active(self) -> bool:
        """Whether this point may be integrated into a later response."""
        return not self.completed and self.still_relevant


@dataclass(frozen=True)
class InterruptionContext:
    """Bounded continuity metadata about an interrupted prior response."""

    was_interrupted: bool = False
    category: InterruptionCategory = InterruptionCategory.OTHER
    previous_intent: PendingConversationIntent | None = None
    voice_activity: VoiceActivityMetadata = VoiceActivityMetadata()


@dataclass(frozen=True)
class ResponsePlanningInput:
    """Bounded post-decision input assembled by trusted application code."""

    authoritative_state: ConversationState
    authoritative_result: AuthoritativeResultKind
    current_prospect_message: str
    approved_action: AgentAction | None = None
    tone: Tone = Tone.NEUTRAL
    conversation_category: InterruptionCategory = InterruptionCategory.OTHER
    addressee_status: AddresseeStatus = AddresseeStatus.ADDRESSED_TO_AGENT
    interruption: InterruptionContext = InterruptionContext()
    explanation_need: ExplanationNeed = ExplanationNeed.STANDARD
    trusted_context_summary: tuple[str, ...] = ()
    previous_acknowledgement: AcknowledgementKind = AcknowledgementKind.NONE
    escalation_decision: EscalationDecision | None = None
    language_profile: LanguageProfile | None = None
    consultative_decision: ConsultativeConversationDecision | None = None
    service_answer_context: ServiceAnswerContext | None = None
    sales_guidance: SalesConversationGuidance | None = None
    playbook_guidance: OpportunityGuidance | None = None
    business_conversation: object | None = None
    business_diagnostic: object | None = None
    conversation_priority: object | None = None
    qualification: object | None = None
    business_memory: object | None = None

    def __post_init__(self) -> None:
        if len(self.current_prospect_message) > 2000:
            raise ValueError("current prospect message exceeds bounded length")
        if len(self.trusted_context_summary) > 6:
            raise ValueError("trusted context summary exceeds six items")
        if any(len(item) > 240 for item in self.trusted_context_summary):
            raise ValueError("trusted context summary item exceeds bounded length")
        _validate_language_profile(self.language_profile)
        _validate_consultative(
            self.consultative_decision, self.service_answer_context
        )
        _validate_sales_guidance(self.sales_guidance)
        _validate_playbook_guidance(self.playbook_guidance)
        _validate_business_conversation(self.business_conversation)
        _validate_business_diagnostic(self.business_diagnostic)
        _validate_conversation_priority(self.conversation_priority)
        _validate_qualification(self.qualification)
        _validate_business_memory(self.business_memory)


@dataclass(frozen=True)
class ResponsePlan:
    """Communication strategy only; final prose and execution are both absent."""

    communicative_goal: ConversationMove
    response_length: ResponseLength
    tone: Tone
    acknowledgement: AcknowledgementKind
    question_strategy: QuestionStrategy
    clarification_required: bool
    interruption_handling: InterruptionHandling
    resume_previous_point: bool
    pending_intent: PendingConversationIntent | None
    addressee_status: AddresseeStatus
    trusted_context_summary: tuple[str, ...] = ()
    escalation_decision: EscalationDecision | None = None
    language_profile: LanguageProfile | None = None
    consultative_decision: ConsultativeConversationDecision | None = None
    service_answer_context: ServiceAnswerContext | None = None
    sales_guidance: SalesConversationGuidance | None = None
    playbook_guidance: OpportunityGuidance | None = None
    business_conversation: object | None = None
    business_diagnostic: object | None = None
    conversation_priority: object | None = None
    qualification: object | None = None
    business_memory: object | None = None

    def __post_init__(self) -> None:
        _validate_language_profile(self.language_profile)
        _validate_consultative(
            self.consultative_decision, self.service_answer_context
        )
        _validate_sales_guidance(self.sales_guidance)
        _validate_playbook_guidance(self.playbook_guidance)
        _validate_business_conversation(self.business_conversation)
        _validate_business_diagnostic(self.business_diagnostic)
        _validate_conversation_priority(self.conversation_priority)
        _validate_qualification(self.qualification)
        _validate_business_memory(self.business_memory)


def _validate_language_profile(value: object) -> None:
    if value is None:
        return
    from app.conversation.understanding.contracts import LanguageProfile

    if not isinstance(value, LanguageProfile):
        raise TypeError("language profile has an invalid type")


def _validate_consultative(decision: object, answer: object) -> None:
    if decision is None and answer is None:
        return
    from app.conversation.consultative.contracts import (
        ConsultativeConversationDecision,
        ServiceAnswerContext,
    )

    if decision is not None and not isinstance(
        decision, ConsultativeConversationDecision
    ):
        raise TypeError("consultative decision has an invalid type")
    if answer is not None and not isinstance(answer, ServiceAnswerContext):
        raise TypeError("service answer context has an invalid type")


def _validate_sales_guidance(value: object) -> None:
    if value is None:
        return
    from app.conversation.sales_cognition.contracts import SalesConversationGuidance

    if not isinstance(value, SalesConversationGuidance):
        raise TypeError("sales guidance has an invalid type")


def _validate_playbook_guidance(value: object) -> None:
    if value is None:
        return
    from app.conversation.sales_playbook.contracts import OpportunityGuidance

    if not isinstance(value, OpportunityGuidance):
        raise TypeError("playbook guidance has an invalid type")


def _validate_business_conversation(value: object) -> None:
    if value is None:
        return
    from app.conversation.business_conversation.contracts import BusinessConversationSnapshot
    if not isinstance(value, BusinessConversationSnapshot):
        raise TypeError("business conversation has an invalid type")


def _validate_business_diagnostic(value: object) -> None:
    if value is None:
        return
    from app.conversation.business_diagnostic.contracts import BusinessDiagnosticSnapshot
    if not isinstance(value, BusinessDiagnosticSnapshot):
        raise TypeError("business diagnostic has an invalid type")


def _validate_conversation_priority(value: object) -> None:
    if value is None:
        return
    from app.conversation.conversation_steering.contracts import ConversationPrioritySnapshot
    if not isinstance(value, ConversationPrioritySnapshot):
        raise TypeError("conversation priority has an invalid type")

def _validate_qualification(value: object) -> None:
    if value is None: return
    from app.conversation.qualification.contracts import QualificationSnapshot
    if not isinstance(value, QualificationSnapshot): raise TypeError("qualification has an invalid type")

def _validate_business_memory(value: object) -> None:
    if value is None: return
    from app.conversation.business_memory.contracts import BusinessMentalModelSnapshot
    if not isinstance(value, BusinessMentalModelSnapshot): raise TypeError("business memory has an invalid type")
