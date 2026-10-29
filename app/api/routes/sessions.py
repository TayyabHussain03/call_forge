"""HTTP routes for session lifecycle and turn processing.

Thin transport: validate, call runtime, map result. No sales logic lives
here. HTTP and WebSocket both call the same runtime methods and share the
same mappers (`app.api.mappers`).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.api.deps import (
    get_campaign_lookup,
    get_runtime,
    get_session_scoped,
    get_tenant_id,
)
from app.api.errors import (
    ApiErrorCode,
    map_bootstrap_rejected,
    map_internal,
    map_session_not_found,
    map_stale_revision,
)
from app.api.mappers import (
    map_bootstrap_to_response,
    map_finish_to_response,
    map_session_to_view,
    map_turn_to_http_response,
)
from app.api.schemas import (
    FinishSessionRequest,
    FinishSessionResponse,
    ProcessTurnRequest,
    ProcessTurnResponse,
    SessionView,
    StartSessionRequest,
    StartSessionResponse,
)
from app.application.sales_runtime.persistent_runtime import PersistentSalesRuntime
from app.persistence.errors import NotFoundError, StaleRevisionError as PersistStaleError
from app.runtime.chat.bootstrap.contracts import (
    BootstrapStatus,
    SessionBootstrapInput,
)
from app.runtime.chat.contracts import ChatSession, SessionStatus
from app.runtime.chat.outcome.contracts import TerminationReason
from app.runtime.chat.repository import (
    SessionNotFoundError,
    StaleRevisionError as RuntimeStaleError,
)

router = APIRouter(prefix="/api/v1")


# ---------------------------------------------------------------------------
# POST /sessions — bootstrap
# ---------------------------------------------------------------------------


@router.post("/sessions", response_model=StartSessionResponse, status_code=201)
async def start_session(
    body: StartSessionRequest,
    runtime: Annotated[PersistentSalesRuntime, Depends(get_runtime)],
    tenant_id: Annotated[str, Depends(get_tenant_id)],
    campaign_lookup=Depends(get_campaign_lookup),
) -> StartSessionResponse:
    campaign = campaign_lookup(body.campaign_id)
    if campaign is None:
        err = map_bootstrap_rejected(f"campaign '{body.campaign_id}' not configured")
        raise HTTPException(status_code=err.http_status, detail=err.to_payload())
    if campaign.tenant_id != tenant_id:
        err = map_bootstrap_rejected("campaign does not belong to current tenant")
        raise HTTPException(status_code=err.http_status, detail=err.to_payload())
    bootstrap_input = SessionBootstrapInput(
        tenant_id=tenant_id,
        campaign_id=body.campaign_id,
        lead_id=body.lead_id,
        session_id=body.session_id,
        bootstrap_request_id=body.bootstrap_request_id,
        trusted_time_iso=body.trusted_time_iso,
    )
    try:
        result = await run_in_threadpool(
            runtime.start_session, bootstrap_input, campaign,
        )
    except Exception:
        err = map_internal()
        raise HTTPException(status_code=err.http_status, detail=err.to_payload())
    if result.status != BootstrapStatus.CREATED:
        err = map_bootstrap_rejected(
            result.rejection_reason or f"bootstrap status: {result.status.value}",
        )
        raise HTTPException(status_code=err.http_status, detail=err.to_payload())
    return map_bootstrap_to_response(result)


# ---------------------------------------------------------------------------
# POST /sessions/{session_id}/turns — process one prospect message
# ---------------------------------------------------------------------------


@router.post(
    "/sessions/{session_id}/turns",
    response_model=ProcessTurnResponse,
)
async def process_turn(
    session_id: str,
    body: ProcessTurnRequest,
    session: Annotated[ChatSession, Depends(get_session_scoped)],
    runtime: Annotated[PersistentSalesRuntime, Depends(get_runtime)],
) -> ProcessTurnResponse:
    # Compute next sequence from the authoritative session state —
    # clients never dictate turn order (§11).
    next_sequence = session.turn_count + 1
    try:
        result = await run_in_threadpool(
            runtime.process_turn,
            session.session_id,
            body.text,
            next_sequence,
            body.turn_request_id,
        )
    except (RuntimeStaleError, PersistStaleError):
        err = map_stale_revision()
        raise HTTPException(status_code=err.http_status, detail=err.to_payload())
    except (SessionNotFoundError, NotFoundError):
        err = map_session_not_found(session_id)
        raise HTTPException(status_code=err.http_status, detail=err.to_payload())
    except HTTPException:
        raise
    except Exception:
        err = map_internal()
        raise HTTPException(status_code=err.http_status, detail=err.to_payload())
    # Re-read status for the response (persisted at commit)
    refreshed = await run_in_threadpool(runtime.sessions.get, session.session_id)
    return map_turn_to_http_response(
        result,
        turn_request_id=body.turn_request_id,
        session_status=refreshed.status.value,
    )


# ---------------------------------------------------------------------------
# GET /sessions/{session_id} — public view
# ---------------------------------------------------------------------------


@router.get("/sessions/{session_id}", response_model=SessionView)
async def get_session(
    session: Annotated[ChatSession, Depends(get_session_scoped)],
) -> SessionView:
    return map_session_to_view(session)


# ---------------------------------------------------------------------------
# POST /sessions/{session_id}/finish
# ---------------------------------------------------------------------------


@router.post(
    "/sessions/{session_id}/finish",
    response_model=FinishSessionResponse,
)
async def finish_session(
    body: FinishSessionRequest,
    session: Annotated[ChatSession, Depends(get_session_scoped)],
    runtime: Annotated[PersistentSalesRuntime, Depends(get_runtime)],
) -> FinishSessionResponse:
    reason = TerminationReason.NATURAL_COMPLETION
    if body.is_dnc:
        reason = TerminationReason.DNC_REQUEST
    elif body.is_not_interested:
        reason = TerminationReason.NOT_INTERESTED_EXIT
    elif body.session_error:
        reason = TerminationReason.RUNTIME_ERROR
    try:
        result = await run_in_threadpool(
            runtime.finish_session,
            session.session_id,
            is_dnc=body.is_dnc,
            is_not_interested=body.is_not_interested,
            session_error=body.session_error,
            termination_reason=reason,
        )
    except Exception:
        err = map_internal()
        raise HTTPException(status_code=err.http_status, detail=err.to_payload())
    return map_finish_to_response(result)
