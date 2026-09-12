"""Call-scoped, versioned, in-memory advisory insight buffer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.conversation.supervisor.contracts import SupervisorInsight


class BufferWriteOutcome(str, Enum):
    """Deterministic result of attempting to record one insight."""

    ACCEPTED = "accepted"
    STALE_REJECTED = "stale_rejected"
    DUPLICATE_IGNORED = "duplicate_ignored"
    INVALID_REJECTED = "invalid_rejected"


@dataclass(frozen=True)
class StrategyBufferSnapshot:
    """Immutable view of a call's latest and consumed advisory versions."""

    call_id: str
    latest: SupervisorInsight | None = None
    consumed_source_sequence: int | None = None


class StrategyBuffer:
    """Keep only the newest insight; first result wins for a duplicate version."""

    def __init__(self, call_id: str) -> None:
        if not call_id or len(call_id) > 100:
            raise ValueError("buffer call id must contain 1-100 characters")
        self._call_id = call_id
        self._latest: SupervisorInsight | None = None
        self._consumed_source_sequence: int | None = None

    @property
    def call_id(self) -> str:
        """Return the isolated call identity for this buffer."""
        return self._call_id

    def record(self, insight: SupervisorInsight) -> BufferWriteOutcome:
        """Accept only a valid, newer insight for this call."""
        if not isinstance(insight, SupervisorInsight) or insight.call_id != self._call_id:
            return BufferWriteOutcome.INVALID_REJECTED
        if self._latest is not None:
            if insight.source_turn_sequence < self._latest.source_turn_sequence:
                return BufferWriteOutcome.STALE_REJECTED
            if insight.source_turn_sequence == self._latest.source_turn_sequence:
                return BufferWriteOutcome.DUPLICATE_IGNORED
        self._latest = insight
        self._consumed_source_sequence = None
        return BufferWriteOutcome.ACCEPTED

    def latest(self) -> SupervisorInsight | None:
        """Return the immutable latest insight, if any."""
        return self._latest

    def latest_applicable_for(self, turn_sequence: int) -> SupervisorInsight | None:
        """Return an insight only for a strictly later turn."""
        if turn_sequence < 0:
            raise ValueError("turn sequence must be non-negative")
        insight = self._latest
        if insight is None or insight.source_turn_sequence >= turn_sequence:
            return None
        if insight.source_turn_sequence == self._consumed_source_sequence:
            return None
        return insight

    def consume_for(self, turn_sequence: int) -> SupervisorInsight | None:
        """Consume the latest applicable version at most once."""
        insight = self.latest_applicable_for(turn_sequence)
        if insight is not None:
            self._consumed_source_sequence = insight.source_turn_sequence
        return insight

    def snapshot(self) -> StrategyBufferSnapshot:
        """Return an immutable copy of current buffer state."""
        return StrategyBufferSnapshot(
            self._call_id,
            self._latest,
            self._consumed_source_sequence,
        )
