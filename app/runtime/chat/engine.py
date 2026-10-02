"""Chat Sales Runtime engine.

Wraps the existing ProductionTurnProcessor as orchestration — no
duplicated business logic. Each chat turn is translated into a
CoordinatedUserTurn, processed through the production pipeline, and
the result is mapped back to a ChatTurnResult.

The runtime owns session lifecycle and transcript accumulation.
The processor owns every business decision.
"""

from __future__ import annotations

from dataclasses import replace

from app.conversation.response_planning.contracts import (
    AddresseeStatus,
    InterruptionCategory,
    InterruptionContext,
)
from app.runtime.contracts import CoordinatedTurnOutput, CoordinatedUserTurn, TurnProcessor
from app.runtime.chat.contracts import (
    ChatSession,
    ChatTurnInput,
    ChatTurnResult,
    RuntimeMode,
    SessionStatus,
    TranscriptEntry,
    TranscriptRole,
    TurnDiagnosticTrace,
)
from app.runtime.chat.repository import InMemoryChatSessionRepository


class DuplicateTurnError(Exception):
    """Raised when a turn sequence has already been processed."""


class SessionTerminatedError(Exception):
    """Raised when a turn is submitted to a non-active session."""


class ChatRuntime:
    """Text-based runtime exercising the production conversation pipeline.

    This is orchestration around existing processors — it does NOT
    contain conversation logic, state machine transitions, or
    business decisions. Those live in ProductionTurnProcessor and
    the layers beneath it.
    """

    def __init__(
        self,
        repository: InMemoryChatSessionRepository,
        processors: dict[str, TurnProcessor] | None = None,
    ) -> None:
        self._repository = repository
        self._processors: dict[str, TurnProcessor] = dict(processors or {})

    def register_processor(self, session_id: str, processor: TurnProcessor) -> None:
        self._processors[session_id] = processor

    def process_turn(self, turn_input: ChatTurnInput) -> ChatTurnResult:
        session = self._repository.get(turn_input.session_id)

        if session.status != SessionStatus.ACTIVE:
            raise SessionTerminatedError(
                f"session {session.session_id} is {session.status.value}"
            )

        if turn_input.turn_sequence <= session.turn_count:
            raise DuplicateTurnError(
                f"turn {turn_input.turn_sequence} already processed"
            )

        processor = self._processors.get(session.session_id)
        if processor is None:
            raise ValueError(f"no processor registered for session {session.session_id}")

        coordinated_turn = _to_coordinated_turn(turn_input)
        output = processor.process_turn(coordinated_turn)

        new_status = (
            SessionStatus.COMPLETED if output.conversation_terminal
            else SessionStatus.ACTIVE
        )

        prospect_entry = TranscriptEntry(
            turn_sequence=turn_input.turn_sequence,
            role=TranscriptRole.PROSPECT,
            text=turn_input.message,
        )
        agent_entry = TranscriptEntry(
            turn_sequence=turn_input.turn_sequence,
            role=TranscriptRole.AGENT,
            text=output.rendered_response.text,
        )

        updated_session = replace(
            session,
            turn_count=turn_input.turn_sequence,
            status=new_status,
            transcript=(*session.transcript, prospect_entry, agent_entry),
        )
        self._repository.update(updated_session)

        diagnostics = _build_diagnostics(turn_input.turn_sequence, output)

        return ChatTurnResult(
            session_id=session.session_id,
            turn_sequence=turn_input.turn_sequence,
            agent_response=output.rendered_response.text,
            conversation_terminal=output.conversation_terminal,
            pipeline_outcome=(
                output.pipeline_outcome.value if output.pipeline_outcome else None
            ),
            diagnostics=diagnostics,
        )

    def get_session(self, session_id: str) -> ChatSession:
        return self._repository.get(session_id)

    def get_transcript(self, session_id: str) -> tuple[TranscriptEntry, ...]:
        return self._repository.get(session_id).transcript

    def terminate_session(self, session_id: str) -> ChatSession:
        session = self._repository.get(session_id)
        if session.status != SessionStatus.ACTIVE:
            return session
        updated = replace(session, status=SessionStatus.TERMINATED)
        return self._repository.update(updated)

    def run_scenario(
        self,
        session_id: str,
        prospect_turns: tuple[str, ...],
    ) -> tuple[ChatTurnResult, ...]:
        results: list[ChatTurnResult] = []
        for i, message in enumerate(prospect_turns):
            turn_input = ChatTurnInput(
                session_id=session_id,
                message=message,
                turn_sequence=i + 1,
            )
            result = self.process_turn(turn_input)
            results.append(result)
            if result.conversation_terminal:
                break
        return tuple(results)


def _to_coordinated_turn(turn_input: ChatTurnInput) -> CoordinatedUserTurn:
    return CoordinatedUserTurn(
        turn_id=f"chat_{turn_input.session_id}_t{turn_input.turn_sequence}",
        sequence_number=turn_input.turn_sequence,
        utterance=turn_input.message,
        interruption=InterruptionContext(),
        addressee_status=AddresseeStatus.ADDRESSED_TO_AGENT,
        conversation_category=InterruptionCategory.OTHER,
    )


def _build_diagnostics(
    turn_sequence: int,
    output: CoordinatedTurnOutput,
) -> TurnDiagnosticTrace:
    return TurnDiagnosticTrace(
        turn_sequence=turn_sequence,
        conversation_state=(
            output.pipeline_outcome.value if output.pipeline_outcome else "unknown"
        ),
        pipeline_outcome=(
            output.pipeline_outcome.value if output.pipeline_outcome else None
        ),
        is_terminal=output.conversation_terminal,
    )
