"""Runtime → public schema mappers.

One mapper per runtime type. Both HTTP and WebSocket use these so a turn
processed through either transport produces identical public shape.
Internal fields (snapshot payloads, diagnostics with reasoning traces,
provider secrets) are never emitted.
"""

from __future__ import annotations

from app.api.schemas import (
    ProcessTurnResponse,
    SessionView,
    StartSessionResponse,
    FinishSessionResponse,
    WsAgentTurnMessage,
)
from app.application.sales_runtime.persistent_runtime import (
    PersistentLifecycleResult,
    PersistentTurnResult,
)
from app.runtime.chat.bootstrap.contracts import SessionBootstrapResult
from app.runtime.chat.contracts import ChatSession


def map_bootstrap_to_response(
    result: SessionBootstrapResult,
) -> StartSessionResponse:
    """Only called for CREATED results; others are mapped to errors."""
    session = result.session
    if session is None:
        raise ValueError("map_bootstrap_to_response requires a session")
    return StartSessionResponse(
        session_id=session.session_id,
        status=session.status.value,
        campaign_id=session.campaign_id,
        lead_id=session.lead_id,
        tenant_id=session.tenant_id,
    )


def map_session_to_view(session: ChatSession) -> SessionView:
    return SessionView(
        session_id=session.session_id,
        tenant_id=session.tenant_id,
        campaign_id=session.campaign_id,
        lead_id=session.lead_id,
        status=session.status.value,
        turn_count=session.turn_count,
        revision=session.revision,
    )


def map_turn_to_http_response(
    result: PersistentTurnResult,
    *,
    turn_request_id: str,
    session_status: str,
) -> ProcessTurnResponse:
    return ProcessTurnResponse(
        session_id=result.session_id,
        turn_request_id=turn_request_id,
        turn_sequence=result.turn_sequence,
        text=result.agent_response,
        session_status=session_status,
        terminal=result.conversation_terminal,
        revision=result.committed_revision,
        replayed=result.replayed_from_idempotency,
    )


def map_turn_to_ws_message(
    result: PersistentTurnResult,
    *,
    turn_request_id: str,
    session_status: str,
) -> WsAgentTurnMessage:
    return WsAgentTurnMessage(
        turn_request_id=turn_request_id,
        session_id=result.session_id,
        turn_sequence=result.turn_sequence,
        text=result.agent_response,
        session_status=session_status,
        terminal=result.conversation_terminal,
        revision=result.committed_revision,
        replayed=result.replayed_from_idempotency,
    )


def map_finish_to_response(
    result: PersistentLifecycleResult,
) -> FinishSessionResponse:
    outcome = result.lead_outcome
    return FinishSessionResponse(
        session_id=outcome.session_id,
        outcome_status=outcome.outcome_status.value,
        interest_level=outcome.interest_level.value,
        dnc_status=outcome.dnc_status.value,
        execution_status=result.execution_status.value,
        termination_reason=outcome.termination_reason.value,
    )
