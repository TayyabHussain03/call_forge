"""Focused Slice 8 tests for runtime event and delivery coordination."""

from __future__ import annotations

from dataclasses import fields

from app.conversation.response_planning.contracts import (
    AddresseeStatus,
    AcknowledgementKind,
    ConversationMove,
    InterruptionCategory,
    InterruptionHandling,
    QuestionStrategy,
    ResponseLength,
    ResponsePlan,
)
from app.conversation.response_rendering.contracts import RenderedResponse
from app.core.constants import Tone
from app.runtime.contracts import (
    CoordinatedTurnOutput,
    CoordinationOutcome,
    DeliveryAction,
    DeliveryInstruction,
    DeliveryProgress,
    DeliveryStatus,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeFailureKind,
)
from app.runtime.turn_coordinator import TurnCoordinator


class CountingProcessor:
    def __init__(self) -> None:
        self.calls = 0
        self.turns = []
        self.domain_transition_count = 0

    def process_turn(self, turn):  # type: ignore[no-untyped-def]
        self.calls += 1
        self.turns.append(turn)
        self.domain_transition_count += 1
        plan = _plan()
        return CoordinatedTurnOutput(
            turn.turn_id,
            plan,
            RenderedResponse(
                "Here is the approved response.",
                plan.communicative_goal,
                plan.response_length,
                False,
            ),
            "explain the approved service value",
        )


def _plan() -> ResponsePlan:
    return ResponsePlan(
        ConversationMove.COMMUNICATE_RESULT,
        ResponseLength.MODERATE,
        Tone.NEUTRAL,
        AcknowledgementKind.NONE,
        QuestionStrategy.NONE,
        False,
        InterruptionHandling.NONE,
        False,
        None,
        AddresseeStatus.ADDRESSED_TO_AGENT,
    )


def _event(
    sequence: int,
    event_type: RuntimeEventType,
    *,
    event_id: str | None = None,
    turn_id: str | None = None,
    utterance: str | None = None,
    call_id: str = "call",
    **metadata: object,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_id or f"event-{sequence}",
        call_id,
        sequence,
        event_type,
        turn_id,
        utterance,
        **metadata,  # type: ignore[arg-type]
    )


def _final(sequence: int, turn_id: str = "turn-1", **metadata: object) -> RuntimeEvent:
    return _event(
        sequence,
        RuntimeEventType.USER_UTTERANCE_FINAL,
        turn_id=turn_id,
        utterance="final user input",
        **metadata,
    )


def test_normal_turn_delivery_lifecycle() -> None:
    processor = CountingProcessor()
    coordinator = TurnCoordinator("call", processor)

    result = coordinator.handle(_final(1))
    assert result.outcome == CoordinationOutcome.TURN_PROCESSED
    assert result.instructions[0].action == DeliveryAction.SPEAK
    assert coordinator.state.active_delivery is not None
    assert coordinator.state.active_delivery.status == DeliveryStatus.PENDING

    coordinator.handle(_event(2, RuntimeEventType.AGENT_DELIVERY_STARTED, turn_id="turn-1"))
    completed = coordinator.handle(
        _event(3, RuntimeEventType.AGENT_DELIVERY_COMPLETED, turn_id="turn-1")
    )
    assert completed.state.active_delivery is not None
    assert completed.state.active_delivery.status == DeliveryStatus.COMPLETED
    assert completed.state.active_delivery.progress == DeliveryProgress.COMPLETED


def test_partial_speech_cancels_active_delivery_without_processing_turn() -> None:
    processor = CountingProcessor()
    coordinator = TurnCoordinator("call", processor)
    coordinator.handle(_final(1))
    coordinator.handle(_event(2, RuntimeEventType.AGENT_DELIVERY_STARTED, turn_id="turn-1"))

    interrupted = coordinator.handle(
        _event(
            3,
            RuntimeEventType.USER_SPEECH_STARTED,
            barge_in=True,
            overlapping_speech=True,
        )
    )
    assert interrupted.instructions == (
        DeliveryInstruction(DeliveryAction.CANCEL_CURRENT, "turn-1"),
    )
    assert interrupted.state.active_delivery is not None
    assert interrupted.state.active_delivery.status == DeliveryStatus.INTERRUPTED
    assert processor.calls == 1


def test_partial_speech_without_final_input_never_invokes_processor() -> None:
    processor = CountingProcessor()
    coordinator = TurnCoordinator("call", processor)
    coordinator.handle(_event(1, RuntimeEventType.USER_SPEECH_STARTED, barge_in=True))
    assert processor.calls == 0


def test_interruption_metadata_reaches_next_finalized_turn() -> None:
    processor = CountingProcessor()
    coordinator = TurnCoordinator("call", processor)
    coordinator.handle(_final(1))
    coordinator.handle(_event(2, RuntimeEventType.AGENT_DELIVERY_STARTED, turn_id="turn-1"))
    coordinator.handle(
        _event(
            3,
            RuntimeEventType.USER_SPEECH_STARTED,
            barge_in=True,
            conversation_category=InterruptionCategory.QUESTION,
        )
    )
    coordinator.handle(_final(4, "turn-2"))

    next_turn = processor.turns[-1]
    assert next_turn.interruption.was_interrupted
    assert next_turn.interruption.voice_activity.barge_in_detected
    assert next_turn.interruption.previous_intent is not None
    assert next_turn.interruption.previous_intent.summary == "explain the approved service value"


def test_old_response_is_not_restarted_after_barge_in() -> None:
    processor = CountingProcessor()
    coordinator = TurnCoordinator("call", processor)
    coordinator.handle(_final(1))
    coordinator.handle(_event(2, RuntimeEventType.USER_SPEECH_STARTED, barge_in=True))
    result = coordinator.handle(_final(3, "turn-2"))

    assert [item.action for item in result.instructions] == [DeliveryAction.SPEAK]
    assert result.instructions[0].turn_id == "turn-2"


def test_stale_old_delivery_event_is_ignored_after_new_turn() -> None:
    processor = CountingProcessor()
    coordinator = TurnCoordinator("call", processor)
    coordinator.handle(_final(1))
    coordinator.handle(_event(2, RuntimeEventType.USER_SPEECH_STARTED, barge_in=True))
    coordinator.handle(_final(3, "turn-2"))

    stale = coordinator.handle(
        _event(4, RuntimeEventType.AGENT_DELIVERY_COMPLETED, turn_id="turn-1")
    )
    assert stale.outcome == CoordinationOutcome.STALE_IGNORED
    assert coordinator.state.current_turn_id == "turn-2"


def test_duplicate_final_event_and_turn_are_processed_once() -> None:
    processor = CountingProcessor()
    coordinator = TurnCoordinator("call", processor)
    event = _final(1)
    coordinator.handle(event)
    duplicate_event = coordinator.handle(event)
    duplicate_turn = coordinator.handle(
        _final(2, "turn-1", event_id="another-event")
    )

    assert duplicate_event.outcome == CoordinationOutcome.DUPLICATE_IGNORED
    assert duplicate_turn.outcome == CoordinationOutcome.DUPLICATE_IGNORED
    assert processor.calls == 1
    assert processor.domain_transition_count == 1


def test_out_of_order_event_fails_closed() -> None:
    coordinator = TurnCoordinator("call", CountingProcessor())
    coordinator.handle(_event(5, RuntimeEventType.CALL_STARTED))
    result = coordinator.handle(_event(4, RuntimeEventType.USER_SPEECH_STARTED))
    assert result.outcome == CoordinationOutcome.STALE_IGNORED
    assert coordinator.state.last_sequence == 5


def test_disconnect_cancels_delivery_and_blocks_delayed_events() -> None:
    processor = CountingProcessor()
    coordinator = TurnCoordinator("call", processor)
    coordinator.handle(_final(1))
    disconnected = coordinator.handle(_event(2, RuntimeEventType.CALL_DISCONNECTED))

    assert disconnected.state.disconnected
    assert disconnected.instructions[0].action == DeliveryAction.CANCEL_CURRENT
    assert coordinator.handle(_final(3, "turn-2")).outcome == (
        CoordinationOutcome.DISCONNECTED_IGNORED
    )
    assert coordinator.handle(
        _event(4, RuntimeEventType.AGENT_DELIVERY_STARTED, turn_id="turn-1")
    ).outcome == CoordinationOutcome.DISCONNECTED_IGNORED
    assert processor.calls == 1


def test_delivery_interruption_does_not_rollback_domain_transition() -> None:
    processor = CountingProcessor()
    coordinator = TurnCoordinator("call", processor)
    coordinator.handle(_final(1))
    coordinator.handle(_event(2, RuntimeEventType.AGENT_DELIVERY_STARTED, turn_id="turn-1"))
    coordinator.handle(_event(3, RuntimeEventType.AGENT_DELIVERY_INTERRUPTED, turn_id="turn-1"))

    assert processor.domain_transition_count == 1
    assert coordinator.state.active_delivery is not None
    assert coordinator.state.active_delivery.status == DeliveryStatus.INTERRUPTED


def test_addressee_metadata_is_advisory_and_carried_without_authority() -> None:
    processor = CountingProcessor()
    coordinator = TurnCoordinator("call", processor)
    coordinator.handle(
        _final(
            1,
            possible_background_speech=True,
        )
    )
    assert processor.turns[0].addressee_status == AddresseeStatus.ADDRESSEE_UNCERTAIN
    forbidden = {"next_state", "approved_discount", "persist_contact"}
    assert forbidden.isdisjoint({item.name for item in fields(DeliveryInstruction)})


def test_two_call_scoped_coordinators_do_not_leak_state() -> None:
    first_processor = CountingProcessor()
    second_processor = CountingProcessor()
    first = TurnCoordinator("first", first_processor)
    second = TurnCoordinator("second", second_processor)

    first.handle(_final(1, call_id="first"))
    assert first.state.current_turn_id == "turn-1"
    assert second.state.current_turn_id is None
    assert second_processor.calls == 0


def test_same_event_sequence_replays_deterministically() -> None:
    events = (
        _event(0, RuntimeEventType.CALL_STARTED),
        _final(1),
        _event(2, RuntimeEventType.AGENT_DELIVERY_STARTED, turn_id="turn-1"),
        _event(3, RuntimeEventType.USER_SPEECH_STARTED, barge_in=True),
        _final(4, "turn-2"),
    )

    def replay():  # type: ignore[no-untyped-def]
        coordinator = TurnCoordinator("call", CountingProcessor())
        return tuple(coordinator.handle(event) for event in events)

    assert replay() == replay()


def test_runtime_error_is_typed_and_does_not_invoke_turn_processor() -> None:
    processor = CountingProcessor()
    result = TurnCoordinator("call", processor).handle(
        _event(1, RuntimeEventType.RUNTIME_ERROR)
    )
    assert result.outcome == CoordinationOutcome.FAILED
    assert result.failure == RuntimeFailureKind.RUNTIME_ERROR
    assert processor.calls == 0


def test_failed_logical_turn_is_not_retried_by_duplicate_runtime_event() -> None:
    class FailingProcessor:
        calls = 0

        def process_turn(self, turn):  # type: ignore[no-untyped-def]
            self.calls += 1
            raise RuntimeError("offline failure")

    processor = FailingProcessor()
    coordinator = TurnCoordinator("call", processor)
    failed = coordinator.handle(_final(1, "failed-turn"))
    duplicate = coordinator.handle(
        _final(2, "failed-turn", event_id="retry-event")
    )

    assert failed.failure == RuntimeFailureKind.TURN_PROCESSING_FAILED
    assert duplicate.outcome == CoordinationOutcome.DUPLICATE_IGNORED
    assert processor.calls == 1
