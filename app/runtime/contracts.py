"""Transport-neutral runtime event, turn, delivery, and session contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from app.conversation.response_planning.contracts import (
    AddresseeStatus,
    AuthoritativeResultKind,
    InterruptionCategory,
    InterruptionContext,
    ResponsePlan,
)
from app.conversation.response_rendering.contracts import RenderedResponse


class RuntimeEventType(str, Enum):
    CALL_STARTED = "call_started"
    USER_SPEECH_STARTED = "user_speech_started"
    USER_UTTERANCE_FINAL = "user_utterance_final"
    AGENT_DELIVERY_STARTED = "agent_delivery_started"
    AGENT_DELIVERY_COMPLETED = "agent_delivery_completed"
    AGENT_DELIVERY_INTERRUPTED = "agent_delivery_interrupted"
    CALL_DISCONNECTED = "call_disconnected"
    RUNTIME_ERROR = "runtime_error"


class DeliveryAction(str, Enum):
    SPEAK = "speak"
    CANCEL_CURRENT = "cancel_current"
    NO_OUTPUT = "no_output"


class DeliveryStatus(str, Enum):
    PENDING = "pending"
    DELIVERING = "delivering"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


class DeliveryProgress(str, Enum):
    NOT_STARTED = "not_started"
    PARTIAL = "partial"
    COMPLETED = "completed"


class CoordinationOutcome(str, Enum):
    HANDLED = "handled"
    TURN_PROCESSED = "turn_processed"
    DUPLICATE_IGNORED = "duplicate_ignored"
    STALE_IGNORED = "stale_ignored"
    DISCONNECTED_IGNORED = "disconnected_ignored"
    FAILED = "failed"


class RuntimeFailureKind(str, Enum):
    WRONG_SESSION = "wrong_session"
    TURN_PROCESSING_FAILED = "turn_processing_failed"
    INVALID_EVENT = "invalid_event"
    RUNTIME_ERROR = "runtime_error"


@dataclass(frozen=True)
class RuntimeEvent:
    """One ordered runtime event containing no provider SDK object."""

    event_id: str
    call_id: str
    sequence_number: int
    event_type: RuntimeEventType
    turn_id: str | None = None
    utterance: str | None = None
    barge_in: bool = False
    overlapping_speech: bool = False
    possible_background_speech: bool = False
    addressee_status: AddresseeStatus = AddresseeStatus.ADDRESSED_TO_AGENT
    conversation_category: InterruptionCategory = InterruptionCategory.OTHER

    def __post_init__(self) -> None:
        if not self.event_id or not self.call_id or self.sequence_number < 0:
            raise ValueError("event identity and non-negative sequence are required")
        if self.event_type == RuntimeEventType.USER_UTTERANCE_FINAL:
            if not self.turn_id or not self.utterance or len(self.utterance) > 2000:
                raise ValueError("final utterance requires a bounded utterance and turn_id")
        if self.event_type in {
            RuntimeEventType.AGENT_DELIVERY_STARTED,
            RuntimeEventType.AGENT_DELIVERY_COMPLETED,
            RuntimeEventType.AGENT_DELIVERY_INTERRUPTED,
        } and not self.turn_id:
            raise ValueError("delivery event requires turn_id")


@dataclass(frozen=True)
class CoordinatedUserTurn:
    """Finalized logical user turn passed once to the authoritative pipeline."""

    turn_id: str
    utterance: str
    interruption: InterruptionContext
    addressee_status: AddresseeStatus
    conversation_category: InterruptionCategory


@dataclass(frozen=True)
class CoordinatedTurnOutput:
    """Communication output returned by the composed authoritative pipeline."""

    turn_id: str
    response_plan: ResponsePlan
    rendered_response: RenderedResponse
    unfinished_point_summary: str | None = None
    pipeline_outcome: AuthoritativeResultKind | None = None
    conversation_terminal: bool = False

    def __post_init__(self) -> None:
        if self.unfinished_point_summary is not None and len(
            self.unfinished_point_summary
        ) > 240:
            raise ValueError("unfinished point summary exceeds bounded length")


class TurnProcessor(Protocol):
    """Composition seam implemented by the existing pipeline plus response layers."""

    def process_turn(self, turn: CoordinatedUserTurn) -> CoordinatedTurnOutput:
        """Process one finalized logical turn exactly once."""


@dataclass(frozen=True)
class ActiveDelivery:
    """Runtime delivery state, deliberately separate from conversation state."""

    turn_id: str
    response_plan: ResponsePlan
    rendered_response: RenderedResponse
    unfinished_point_summary: str | None
    status: DeliveryStatus = DeliveryStatus.PENDING
    progress: DeliveryProgress = DeliveryProgress.NOT_STARTED


@dataclass(frozen=True)
class DeliveryInstruction:
    """Small transport-neutral output with no domain authority."""

    action: DeliveryAction
    turn_id: str | None = None
    text: str | None = None
    interruptible: bool = True

    def __post_init__(self) -> None:
        if self.action == DeliveryAction.SPEAK and (not self.turn_id or not self.text):
            raise ValueError("SPEAK requires turn_id and text")
        if self.action == DeliveryAction.CANCEL_CURRENT and not self.turn_id:
            raise ValueError("CANCEL_CURRENT requires turn_id")
        if self.action != DeliveryAction.SPEAK and self.text is not None:
            raise ValueError("only SPEAK may carry text")


@dataclass(frozen=True)
class RuntimeSessionState:
    """Immutable call-scoped mechanics; not conversation-domain context."""

    call_id: str
    current_turn_id: str | None = None
    last_sequence: int = -1
    active_delivery: ActiveDelivery | None = None
    disconnected: bool = False
    pending_interruption: InterruptionContext | None = None
    processed_event_ids: frozenset[str] = field(default_factory=frozenset)
    processed_turn_ids: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class CoordinationResult:
    """Result of one event without any business decision fields."""

    outcome: CoordinationOutcome
    state: RuntimeSessionState
    instructions: tuple[DeliveryInstruction, ...] = ()
    processed_turn: CoordinatedUserTurn | None = None
    failure: RuntimeFailureKind | None = None
    turn_output: CoordinatedTurnOutput | None = None
