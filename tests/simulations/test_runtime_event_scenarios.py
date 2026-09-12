"""Event-driven runtime scenarios U-Z without audio or network providers."""

from __future__ import annotations

from app.brain.contracts import BrainInput, BrainProposal
from app.conversation.response_planning.contracts import (
    AddresseeStatus,
    AcknowledgementKind,
    ConversationMove,
    InterruptionHandling,
    QuestionStrategy,
    ResponseLength,
    ResponsePlan,
)
from app.conversation.response_rendering.contracts import RenderedResponse
from app.core.constants import AgentAction, ConversationState, Intent, TopicCategory, Tone
from app.llm.providers.reasoning_provider import MockReasoningProvider
from app.runtime.contracts import (
    CoordinatedTurnOutput,
    CoordinationOutcome,
    DeliveryAction,
    RuntimeEvent,
    RuntimeEventType,
)
from app.runtime.turn_coordinator import TurnCoordinator


class OfflineRuntimeProcessor:
    """Deterministic stand-in for the already-tested composed pipeline."""

    def __init__(self) -> None:
        self.provider = MockReasoningProvider(
            default=BrainProposal(
                detected_intent=Intent.INTERESTED,
                topic_category=TopicCategory.QUALIFICATION,
                proposed_action=AgentAction.GREET,
            )
        )
        self.turns = []

    def process_turn(self, turn):  # type: ignore[no-untyped-def]
        self.turns.append(turn)
        self.provider.reason(BrainInput(turn.utterance, ConversationState.NEW_CALL))
        plan = ResponsePlan(
            ConversationMove.COMMUNICATE_RESULT,
            ResponseLength.SHORT,
            Tone.NEUTRAL,
            AcknowledgementKind.NONE,
            QuestionStrategy.NONE,
            False,
            InterruptionHandling.NONE,
            False,
            None,
            AddresseeStatus.ADDRESSED_TO_AGENT,
        )
        return CoordinatedTurnOutput(
            turn.turn_id,
            plan,
            RenderedResponse(
                "How can I help?",
                plan.communicative_goal,
                plan.response_length,
                False,
            ),
            "ask how the prospect would like to continue",
        )


def _event(
    sequence: int,
    kind: RuntimeEventType,
    *,
    turn_id: str | None = None,
    utterance: str | None = None,
    event_id: str | None = None,
    call_id: str = "call",
    barge_in: bool = False,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_id or f"e-{sequence}",
        call_id,
        sequence,
        kind,
        turn_id,
        utterance,
        barge_in,
    )


def _final(sequence: int, turn_id: str, *, call_id: str = "call") -> RuntimeEvent:
    return _event(
        sequence,
        RuntimeEventType.USER_UTTERANCE_FINAL,
        turn_id=turn_id,
        utterance="final input",
        call_id=call_id,
    )


def test_scenario_u_normal_delivery() -> None:
    processor = OfflineRuntimeProcessor()
    coordinator = TurnCoordinator("call", processor)
    speak = coordinator.handle(_final(1, "t1"))
    coordinator.handle(_event(2, RuntimeEventType.AGENT_DELIVERY_STARTED, turn_id="t1"))
    completed = coordinator.handle(
        _event(3, RuntimeEventType.AGENT_DELIVERY_COMPLETED, turn_id="t1")
    )
    assert speak.instructions[0].action == DeliveryAction.SPEAK
    assert completed.state.active_delivery is not None
    assert completed.state.active_delivery.status.value == "completed"


def test_scenario_v_user_interrupts_agent() -> None:
    processor = OfflineRuntimeProcessor()
    coordinator = TurnCoordinator("call", processor)
    coordinator.handle(_final(1, "t1"))
    coordinator.handle(_event(2, RuntimeEventType.AGENT_DELIVERY_STARTED, turn_id="t1"))
    cancel = coordinator.handle(
        _event(3, RuntimeEventType.USER_SPEECH_STARTED, barge_in=True)
    )
    coordinator.handle(_final(4, "t2"))
    assert cancel.instructions[0].action == DeliveryAction.CANCEL_CURRENT
    assert processor.turns[-1].interruption.was_interrupted


def test_scenario_w_stale_old_response() -> None:
    processor = OfflineRuntimeProcessor()
    coordinator = TurnCoordinator("call", processor)
    coordinator.handle(_final(1, "t1"))
    coordinator.handle(_event(2, RuntimeEventType.USER_SPEECH_STARTED, barge_in=True))
    coordinator.handle(_final(3, "t2"))
    late = coordinator.handle(
        _event(4, RuntimeEventType.AGENT_DELIVERY_COMPLETED, turn_id="t1")
    )
    assert late.outcome == CoordinationOutcome.STALE_IGNORED
    assert coordinator.state.current_turn_id == "t2"


def test_scenario_x_duplicate_final_invokes_brain_once() -> None:
    processor = OfflineRuntimeProcessor()
    coordinator = TurnCoordinator("call", processor)
    event = _final(1, "t1")
    coordinator.handle(event)
    coordinator.handle(event)
    coordinator.handle(
        _event(
            2,
            RuntimeEventType.USER_UTTERANCE_FINAL,
            turn_id="t1",
            utterance="duplicate final",
            event_id="different-event",
        )
    )
    assert processor.provider.call_count == 1


def test_scenario_y_disconnect_during_response() -> None:
    processor = OfflineRuntimeProcessor()
    coordinator = TurnCoordinator("call", processor)
    coordinator.handle(_final(1, "t1"))
    disconnected = coordinator.handle(_event(2, RuntimeEventType.CALL_DISCONNECTED))
    delayed = coordinator.handle(_final(3, "t2"))
    assert disconnected.instructions[0].action == DeliveryAction.CANCEL_CURRENT
    assert delayed.outcome == CoordinationOutcome.DISCONNECTED_IGNORED
    assert processor.provider.call_count == 1


def test_scenario_z_two_independent_calls() -> None:
    first_processor = OfflineRuntimeProcessor()
    second_processor = OfflineRuntimeProcessor()
    first = TurnCoordinator("first", first_processor)
    second = TurnCoordinator("second", second_processor)
    first.handle(_final(1, "first-turn", call_id="first"))
    second.handle(_final(1, "second-turn", call_id="second"))
    first.handle(_event(2, RuntimeEventType.CALL_DISCONNECTED, call_id="first"))
    assert first.state.disconnected
    assert not second.state.disconnected
    assert first_processor.provider.call_count == second_processor.provider.call_count == 1
