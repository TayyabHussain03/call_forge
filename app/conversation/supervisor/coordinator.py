"""Explicit non-background seam between a supervisor provider and buffer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.conversation.supervisor.buffer import (
    BufferWriteOutcome,
    StrategyBuffer,
    StrategyBufferSnapshot,
)
from app.conversation.supervisor.contracts import SupervisorInput, SupervisorInsight
from app.conversation.supervisor.provider import SupervisorError, SupervisorProvider


class SupervisorRunOutcome(str, Enum):
    """Typed result of one optional supervisor analysis."""

    STORED = "stored"
    STALE_REJECTED = "stale_rejected"
    DUPLICATE_IGNORED = "duplicate_ignored"
    FAILED = "failed"
    INVALID_OUTPUT = "invalid_output"


@dataclass(frozen=True)
class SupervisorRunResult:
    """Sanitized result containing no provider reasoning or exception detail."""

    outcome: SupervisorRunOutcome
    buffer: StrategyBufferSnapshot


class SupervisorCoordinator:
    """Run optional analysis explicitly; production scheduling remains future work."""

    def __init__(
        self,
        provider: SupervisorProvider,
        buffer: StrategyBuffer,
    ) -> None:
        self._provider = provider
        self._buffer = buffer

    def analyze(self, supervisor_input: SupervisorInput) -> SupervisorRunResult:
        """Call once and store only a well-formed identity-matched result."""
        before = self._buffer.snapshot()
        try:
            insight = self._provider.analyze(supervisor_input)
        except SupervisorError:
            return SupervisorRunResult(SupervisorRunOutcome.FAILED, before)
        if not _matches_input(insight, supervisor_input):
            return SupervisorRunResult(SupervisorRunOutcome.INVALID_OUTPUT, before)
        write = self._buffer.record(insight)
        outcome = {
            BufferWriteOutcome.ACCEPTED: SupervisorRunOutcome.STORED,
            BufferWriteOutcome.STALE_REJECTED: SupervisorRunOutcome.STALE_REJECTED,
            BufferWriteOutcome.DUPLICATE_IGNORED: SupervisorRunOutcome.DUPLICATE_IGNORED,
            BufferWriteOutcome.INVALID_REJECTED: SupervisorRunOutcome.INVALID_OUTPUT,
        }[write]
        return SupervisorRunResult(outcome, self._buffer.snapshot())


def _matches_input(value: object, expected: SupervisorInput) -> bool:
    return bool(
        isinstance(value, SupervisorInsight)
        and value.call_id == expected.call_id
        and value.source_turn_id == expected.turn_id
        and value.source_turn_sequence == expected.source_turn_sequence
    )
