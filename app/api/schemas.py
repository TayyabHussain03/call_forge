"""Pydantic request/response schemas for the HTTP + WS transport.

Transport-only models. These do NOT mirror runtime internals — they are
the stable public API shape. Mappers (`app.api.mappers`) adapt runtime
results into these before returning to clients.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# Length bounds — must align with runtime limits in
# app.runtime.chat.contracts.ChatTurnInput (max 2000 chars).
MAX_MESSAGE_LENGTH = 2000
MAX_ID_LENGTH = 120


class StartSessionRequest(BaseModel):
    """Minimum external inputs to bootstrap a session.

    Internal state (participant state, coverage, Prospect Intelligence,
    FSM state, reasoning_calls) is NEVER accepted here.
    """

    model_config = ConfigDict(extra="forbid")

    campaign_id: str = Field(..., min_length=1, max_length=MAX_ID_LENGTH)
    lead_id: str = Field(..., min_length=1, max_length=MAX_ID_LENGTH)
    session_id: str = Field(..., min_length=1, max_length=MAX_ID_LENGTH)
    bootstrap_request_id: str = Field(..., min_length=1, max_length=MAX_ID_LENGTH)
    trusted_time_iso: str = Field(..., min_length=1, max_length=40)


class StartSessionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    status: str
    campaign_id: str
    lead_id: str
    tenant_id: str


class ProcessTurnRequest(BaseModel):
    """One prospect message. `turn_request_id` is mandatory for retries."""

    model_config = ConfigDict(extra="forbid")

    turn_request_id: str = Field(..., min_length=1, max_length=MAX_ID_LENGTH)
    text: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)


class ProcessTurnResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    turn_request_id: str
    turn_sequence: int
    text: str
    session_status: str
    terminal: bool
    revision: int
    replayed: bool = False


class SessionView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    tenant_id: str
    campaign_id: str
    lead_id: str
    status: str
    turn_count: int
    revision: int


class FinishSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_dnc: bool = False
    is_not_interested: bool = False
    session_error: bool = False


class FinishSessionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    outcome_status: str
    interest_level: str
    dnc_status: str
    execution_status: str
    termination_reason: str


# ---------------------------------------------------------------------------
# WebSocket message envelopes — stable across releases
# ---------------------------------------------------------------------------


class WsUserTurnMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = Field(..., pattern="^user_turn$")
    turn_request_id: str = Field(..., min_length=1, max_length=MAX_ID_LENGTH)
    text: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)


class WsAgentTurnMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = "agent_turn"
    turn_request_id: str
    session_id: str
    turn_sequence: int
    text: str
    session_status: str
    terminal: bool
    revision: int
    replayed: bool = False


class WsErrorMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = "error"
    code: str
    message: str
