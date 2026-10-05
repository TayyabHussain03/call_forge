"""Immutable contracts for the sales session lifecycle.

Orchestrate, do not re-reason. Customer outcome and runtime status
are separate axes. Conversation truth supersedes pre-call preparation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.runtime.chat.contracts import ChatSession, ChatTurnResult
from app.runtime.chat.outcome.contracts import ExecutionStatus, LeadOutcome


class LifecycleStatus(str, Enum):
    """High-level lifecycle state."""

    BOOTSTRAP_REJECTED = "bootstrap_rejected"
    ACTIVE = "active"
    TURN_REJECTED = "turn_rejected"
    PROVIDER_FAILURE = "provider_failure"
    RUNTIME_FAILURE = "runtime_failure"
    TERMINATED = "terminated"
    COMPLETED = "completed"


@dataclass(frozen=True)
class LifecycleDiagnostics:
    """Bounded audit metadata for the lifecycle."""

    session_id: str | None = None
    turns_processed: int = 0
    terminal_turn: int | None = None
    bootstrap_status: str | None = None
    failure_detail: str | None = None


@dataclass(frozen=True)
class LifecycleResult:
    """Immutable result of a full session lifecycle."""

    status: LifecycleStatus
    session: ChatSession | None = None
    latest_turn_result: ChatTurnResult | None = None
    lead_outcome: LeadOutcome | None = None
    execution_status: ExecutionStatus | None = None
    diagnostics: LifecycleDiagnostics | None = None

    def __post_init__(self) -> None:
        if self.status == LifecycleStatus.COMPLETED:
            if self.lead_outcome is None:
                raise ValueError("COMPLETED lifecycle must include lead_outcome")
