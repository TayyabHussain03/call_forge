"""WebSocket chat transport.

One message in, one message out, over the same `runtime.process_turn`
call the HTTP route uses. The transport does not reorder, batch, or
transform semantic content — it is literally a different envelope over
the same runtime method.

Message envelopes (strict JSON):
    client → server: {"type": "user_turn",   "turn_request_id": "...", "text": "..."}
    server → client: {"type": "agent_turn",  "turn_request_id": "...", ...}
    server → client: {"type": "error",       "code": "...", "message": "..."}

Flow control: one in-flight turn per connection. The receive loop reads
one message, awaits runtime.process_turn, sends the result, then loops.
This is the bounded flow-control model per §69/§70.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, WebSocket
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect, WebSocketState

from app.api.errors import ApiErrorCode
from app.api.mappers import map_turn_to_ws_message
from app.api.schemas import WsErrorMessage, WsUserTurnMessage
from app.persistence.errors import (
    NotFoundError,
    StaleRevisionError as PersistStaleError,
)
from app.runtime.chat.repository import (
    SessionNotFoundError,
    StaleRevisionError as RuntimeStaleError,
)

router = APIRouter()


# WebSocket close codes (RFC 6455 + application range 4000-4999)
_WS_POLICY_VIOLATION = 1008
_WS_FORBIDDEN = 4403
_WS_SESSION_NOT_FOUND = 4404


async def _send_error(ws: WebSocket, code: ApiErrorCode, message: str) -> None:
    err = WsErrorMessage(code=code.value, message=message)
    await ws.send_text(err.model_dump_json())


@router.websocket("/ws/sessions/{session_id}")
async def chat_ws(websocket: WebSocket, session_id: str) -> None:
    # Tenant from the first subprotocol token or an "X-Tenant-Id" header.
    tenant_id = websocket.headers.get("x-tenant-id") or websocket.query_params.get("tenant")
    if not tenant_id:
        await websocket.close(code=_WS_POLICY_VIOLATION)
        return

    runtime = getattr(websocket.app.state, "runtime", None)
    if runtime is None:
        await websocket.close(code=_WS_POLICY_VIOLATION)
        return

    # Resolve + authorize session BEFORE accepting (close with 4xxx)
    try:
        session = runtime.sessions.get(session_id)
    except (SessionNotFoundError, NotFoundError):
        await websocket.close(code=_WS_SESSION_NOT_FOUND)
        return
    if session.tenant_id != tenant_id:
        await websocket.close(code=_WS_FORBIDDEN)
        return

    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                parsed = json.loads(raw)
                message = WsUserTurnMessage.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError) as e:
                await _send_error(
                    websocket, ApiErrorCode.VALIDATION_ERROR,
                    f"malformed message: {type(e).__name__}",
                )
                continue

            # Always re-read latest session for the authoritative turn
            # sequence — never trust the client to dictate order (§11, §28).
            try:
                refreshed = await run_in_threadpool(runtime.sessions.get, session_id)
            except (SessionNotFoundError, NotFoundError):
                await _send_error(
                    websocket, ApiErrorCode.SESSION_NOT_FOUND,
                    f"session '{session_id}' not found",
                )
                break
            next_sequence = refreshed.turn_count + 1
            try:
                result = await run_in_threadpool(
                    runtime.process_turn,
                    session_id,
                    message.text,
                    next_sequence,
                    message.turn_request_id,
                )
            except (RuntimeStaleError, PersistStaleError):
                await _send_error(
                    websocket, ApiErrorCode.STALE_REVISION,
                    "session changed before this turn could commit; retry with current revision",
                )
                continue
            except (SessionNotFoundError, NotFoundError):
                await _send_error(
                    websocket, ApiErrorCode.SESSION_NOT_FOUND,
                    f"session '{session_id}' not found",
                )
                break
            except Exception:
                await _send_error(
                    websocket, ApiErrorCode.INTERNAL_ERROR,
                    "unexpected server error",
                )
                continue

            latest = await run_in_threadpool(runtime.sessions.get, session_id)
            agent_msg = map_turn_to_ws_message(
                result,
                turn_request_id=message.turn_request_id,
                session_status=latest.status.value,
            )
            await websocket.send_text(agent_msg.model_dump_json())
    except WebSocketDisconnect:
        return
    finally:
        if websocket.application_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close()
            except Exception:
                pass
