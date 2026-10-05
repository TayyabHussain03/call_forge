"""Immutable contracts for the Chat Sales Runtime.

The chat runtime wraps the production ProductionTurnProcessor — it is
orchestration, not a second brain. Every turn flows through the same
authoritative pipeline intended for future voice.

TRUST BOUNDARY: ChatTurnInput carries untrusted user text. Session state
is authoritative. DNC, state transitions, and all business decisions
remain inside the existing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.precall.contracts import PreCallConversationPlan


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RuntimeMode(str, Enum):
    """Whether the runtime is running a live session or a scripted scenario."""

    INTERACTIVE = "interactive"
    SIMULATION = "simulation"


class SessionStatus(str, Enum):
    """Lifecycle of a chat session."""

    ACTIVE = "active"
    COMPLETED = "completed"
    TERMINATED = "terminated"
    ERROR = "error"


class TranscriptRole(str, Enum):
    """Speaker in a transcript entry."""

    PROSPECT = "prospect"
    AGENT = "agent"
    SYSTEM = "system"


# ---------------------------------------------------------------------------
# Turn input / output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChatTurnInput:
    """One prospect message entering the chat runtime."""

    session_id: str
    message: str
    turn_sequence: int

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise ValueError("session_id must not be empty")
        if not self.message or not self.message.strip():
            raise ValueError("message must not be empty")
        if len(self.message) > 2000:
            raise ValueError("message exceeds maximum length")
        if self.turn_sequence < 0:
            raise ValueError("turn_sequence must be non-negative")


@dataclass(frozen=True)
class ChatTurnResult:
    """Result of processing one chat turn through the production pipeline."""

    session_id: str
    turn_sequence: int
    agent_response: str
    conversation_terminal: bool = False
    pipeline_outcome: str | None = None
    diagnostics: TurnDiagnosticTrace | None = None

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if not self.agent_response:
            raise ValueError("agent_response must not be empty")


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TranscriptEntry:
    """One entry in the conversation transcript."""

    turn_sequence: int
    role: TranscriptRole
    text: str

    def __post_init__(self) -> None:
        if self.turn_sequence < 0:
            raise ValueError("turn_sequence must be non-negative")
        if not isinstance(self.role, TranscriptRole):
            raise TypeError("role must be TranscriptRole")
        if not self.text:
            raise ValueError("text must not be empty")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnDiagnosticTrace:
    """Non-authoritative diagnostic snapshot for observability."""

    turn_sequence: int
    conversation_state: str
    pipeline_outcome: str | None = None
    sales_stage: str | None = None
    is_terminal: bool = False
    reasoning_provider: str | None = None
    budget_turns_remaining: int | None = None


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChatSession:
    """Immutable snapshot of a chat session's state."""

    session_id: str
    tenant_id: str
    campaign_id: str
    lead_id: str

    status: SessionStatus = SessionStatus.ACTIVE
    mode: RuntimeMode = RuntimeMode.INTERACTIVE
    revision: int = 0

    transcript: tuple[TranscriptEntry, ...] = ()
    turn_count: int = 0

    pre_call_plan: PreCallConversationPlan | None = None

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise ValueError("session_id must not be empty")
        if not self.tenant_id or not self.tenant_id.strip():
            raise ValueError("tenant_id must not be empty")
        if not self.campaign_id or not self.campaign_id.strip():
            raise ValueError("campaign_id must not be empty")
        if not self.lead_id or not self.lead_id.strip():
            raise ValueError("lead_id must not be empty")
        if self.revision < 0:
            raise ValueError("revision must be non-negative")
        if self.turn_count < 0:
            raise ValueError("turn_count must be non-negative")


# ---------------------------------------------------------------------------
# Simulation scenario
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SalesSimulationScenario:
    """Scripted scenario for deterministic simulation testing."""

    scenario_id: str
    name: str
    prospect_turns: tuple[str, ...]
    expected_terminal: bool = False

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.scenario_id.strip():
            raise ValueError("scenario_id must not be empty")
        if not self.name or not self.name.strip():
            raise ValueError("name must not be empty")
        if not self.prospect_turns:
            raise ValueError("scenario must have at least one prospect turn")
