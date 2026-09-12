"""Call-scoped deterministic runtime event coordinator."""

from __future__ import annotations

from dataclasses import replace

from app.conversation.response_planning.contracts import (
    AddresseeStatus,
    InterruptionCategory,
    InterruptionContext,
    PendingConversationIntent,
    VoiceActivityMetadata,
)
from app.runtime.contracts import (
    ActiveDelivery,
    CoordinatedTurnOutput,
    CoordinatedUserTurn,
    CoordinationOutcome,
    CoordinationResult,
    DeliveryAction,
    DeliveryInstruction,
    DeliveryProgress,
    DeliveryStatus,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeFailureKind,
    RuntimeSessionState,
    TurnProcessor,
)


class TurnCoordinator:
    """Coordinate one call's event ordering, turns, and response delivery."""

    def __init__(self, call_id: str, turn_processor: TurnProcessor) -> None:
        self._processor = turn_processor
        self._state = RuntimeSessionState(call_id=call_id)

    @property
    def state(self) -> RuntimeSessionState:
        """Return the current immutable runtime state snapshot."""
        return self._state

    def handle(self, event: RuntimeEvent) -> CoordinationResult:
        """Handle one event idempotently and in monotonic sequence order."""
        if event.call_id != self._state.call_id:
            return self._result(
                CoordinationOutcome.FAILED, RuntimeFailureKind.WRONG_SESSION
            )
        if event.event_id in self._state.processed_event_ids:
            return self._result(CoordinationOutcome.DUPLICATE_IGNORED)
        if self._state.disconnected:
            return self._result(CoordinationOutcome.DISCONNECTED_IGNORED)
        if event.sequence_number <= self._state.last_sequence:
            self._record_event(event, advance_sequence=False)
            return self._result(CoordinationOutcome.STALE_IGNORED)

        self._record_event(event)
        if event.event_type == RuntimeEventType.CALL_DISCONNECTED:
            return self._disconnect()
        if event.event_type == RuntimeEventType.RUNTIME_ERROR:
            return self._result(
                CoordinationOutcome.FAILED, RuntimeFailureKind.RUNTIME_ERROR
            )
        if event.event_type == RuntimeEventType.USER_SPEECH_STARTED:
            return self._speech_started(event)
        if event.event_type == RuntimeEventType.USER_UTTERANCE_FINAL:
            return self._final_utterance(event)
        if event.event_type == RuntimeEventType.AGENT_DELIVERY_STARTED:
            return self._delivery_update(event, DeliveryStatus.DELIVERING)
        if event.event_type == RuntimeEventType.AGENT_DELIVERY_COMPLETED:
            return self._delivery_update(event, DeliveryStatus.COMPLETED)
        if event.event_type == RuntimeEventType.AGENT_DELIVERY_INTERRUPTED:
            return self._delivery_update(event, DeliveryStatus.INTERRUPTED)
        return self._result(CoordinationOutcome.HANDLED)

    def _speech_started(self, event: RuntimeEvent) -> CoordinationResult:
        active = self._state.active_delivery
        if active is None or active.status in {
            DeliveryStatus.COMPLETED,
            DeliveryStatus.CANCELLED,
            DeliveryStatus.INTERRUPTED,
        }:
            return self._result(CoordinationOutcome.HANDLED)
        interrupted = replace(
            active,
            status=DeliveryStatus.INTERRUPTED,
            progress=(
                DeliveryProgress.PARTIAL
                if active.status == DeliveryStatus.DELIVERING
                else DeliveryProgress.NOT_STARTED
            ),
        )
        self._state = replace(
            self._state,
            active_delivery=interrupted,
            pending_interruption=self._interruption(event, active),
        )
        return self._result(
            CoordinationOutcome.HANDLED,
            instructions=(
                DeliveryInstruction(DeliveryAction.CANCEL_CURRENT, active.turn_id),
            ),
        )

    def _final_utterance(self, event: RuntimeEvent) -> CoordinationResult:
        assert event.turn_id is not None and event.utterance is not None
        if event.turn_id in self._state.processed_turn_ids:
            return self._result(CoordinationOutcome.DUPLICATE_IGNORED)

        instructions: list[DeliveryInstruction] = []
        active = self._state.active_delivery
        if active is not None and active.status in {
            DeliveryStatus.PENDING,
            DeliveryStatus.DELIVERING,
        }:
            instructions.append(
                DeliveryInstruction(DeliveryAction.CANCEL_CURRENT, active.turn_id)
            )
            active = replace(active, status=DeliveryStatus.CANCELLED)
            self._state = replace(
                self._state,
                active_delivery=active,
                pending_interruption=self._interruption(event, active),
            )

        interruption = self._state.pending_interruption or InterruptionContext(
            category=(
                InterruptionCategory.ADDRESSEE_UNCERTAIN
                if event.possible_background_speech
                else event.conversation_category
            ),
            voice_activity=VoiceActivityMetadata(
                overlapping_speech=event.overlapping_speech
            ),
        )
        addressee_status = (
            AddresseeStatus.ADDRESSEE_UNCERTAIN
            if event.possible_background_speech
            else event.addressee_status
        )
        turn = CoordinatedUserTurn(
            event.turn_id,
            event.utterance,
            interruption,
            addressee_status,
            event.conversation_category,
        )
        self._state = replace(
            self._state,
            processed_turn_ids=self._state.processed_turn_ids | {turn.turn_id},
        )
        try:
            output = self._processor.process_turn(turn)
        except Exception:
            return self._result(
                CoordinationOutcome.FAILED,
                RuntimeFailureKind.TURN_PROCESSING_FAILED,
                instructions=tuple(instructions),
                processed_turn=turn,
            )
        if output.turn_id != turn.turn_id:
            return self._result(
                CoordinationOutcome.FAILED,
                RuntimeFailureKind.INVALID_EVENT,
                instructions=tuple(instructions),
                processed_turn=turn,
            )

        delivery = ActiveDelivery(
            turn_id=turn.turn_id,
            response_plan=output.response_plan,
            rendered_response=output.rendered_response,
            unfinished_point_summary=output.unfinished_point_summary,
        )
        self._state = replace(
            self._state,
            current_turn_id=turn.turn_id,
            active_delivery=delivery,
            pending_interruption=None,
        )
        instructions.append(
            DeliveryInstruction(
                DeliveryAction.SPEAK,
                turn.turn_id,
                output.rendered_response.text,
            )
        )
        return self._result(
            CoordinationOutcome.TURN_PROCESSED,
            instructions=tuple(instructions),
            processed_turn=turn,
            turn_output=output,
        )

    def _delivery_update(
        self, event: RuntimeEvent, status: DeliveryStatus
    ) -> CoordinationResult:
        active = self._state.active_delivery
        if active is None or event.turn_id != self._state.current_turn_id:
            return self._result(CoordinationOutcome.STALE_IGNORED)
        if active.status in {
            DeliveryStatus.CANCELLED,
            DeliveryStatus.INTERRUPTED,
            DeliveryStatus.COMPLETED,
        }:
            return self._result(CoordinationOutcome.STALE_IGNORED)
        progress = {
            DeliveryStatus.DELIVERING: DeliveryProgress.PARTIAL,
            DeliveryStatus.COMPLETED: DeliveryProgress.COMPLETED,
            DeliveryStatus.INTERRUPTED: DeliveryProgress.PARTIAL,
        }[status]
        self._state = replace(
            self._state,
            active_delivery=replace(active, status=status, progress=progress),
            pending_interruption=(
                self._interruption(event, active)
                if status == DeliveryStatus.INTERRUPTED
                else self._state.pending_interruption
            ),
        )
        return self._result(CoordinationOutcome.HANDLED)

    def _disconnect(self) -> CoordinationResult:
        active = self._state.active_delivery
        instructions: tuple[DeliveryInstruction, ...] = ()
        if active is not None and active.status not in {
            DeliveryStatus.COMPLETED,
            DeliveryStatus.CANCELLED,
        }:
            instructions = (
                DeliveryInstruction(DeliveryAction.CANCEL_CURRENT, active.turn_id),
            )
            active = replace(active, status=DeliveryStatus.CANCELLED)
        self._state = replace(
            self._state,
            disconnected=True,
            active_delivery=active,
            pending_interruption=None,
        )
        return self._result(CoordinationOutcome.HANDLED, instructions=instructions)

    @staticmethod
    def _interruption(
        event: RuntimeEvent, active: ActiveDelivery
    ) -> InterruptionContext:
        summary = active.unfinished_point_summary
        if summary is None and active.response_plan.pending_intent is not None:
            summary = active.response_plan.pending_intent.summary
        pending = (
            PendingConversationIntent(
                goal=active.response_plan.communicative_goal.value,
                summary=summary,
            )
            if summary
            else None
        )
        category = event.conversation_category
        if event.possible_background_speech:
            category = InterruptionCategory.ADDRESSEE_UNCERTAIN
        return InterruptionContext(
            was_interrupted=True,
            category=category,
            previous_intent=pending,
            voice_activity=VoiceActivityMetadata(
                barge_in_detected=event.barge_in,
                overlapping_speech=event.overlapping_speech,
                speech_started_during_agent_output=True,
            ),
        )

    def _record_event(self, event: RuntimeEvent, *, advance_sequence: bool = True) -> None:
        self._state = replace(
            self._state,
            last_sequence=(
                event.sequence_number if advance_sequence else self._state.last_sequence
            ),
            processed_event_ids=self._state.processed_event_ids | {event.event_id},
        )

    def _result(
        self,
        outcome: CoordinationOutcome,
        failure: RuntimeFailureKind | None = None,
        *,
        instructions: tuple[DeliveryInstruction, ...] = (),
        processed_turn: CoordinatedUserTurn | None = None,
        turn_output: CoordinatedTurnOutput | None = None,
    ) -> CoordinationResult:
        return CoordinationResult(
            outcome,
            self._state,
            instructions,
            processed_turn,
            failure,
            turn_output,
        )
