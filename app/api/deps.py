"""FastAPI dependencies for runtime, principal, and campaign resolution.

Trust boundary (Slice 56): the only path from untrusted request state
to the runtime goes through the configured `PrincipalResolver`.
Routes depend on `Principal`, never on raw `X-Tenant-Id`.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request

from app.api.auth import (
    AuthenticationError,
    Principal,
    PrincipalResolver,
    raise_http_unauthenticated,
)
from app.api.errors import map_session_forbidden, map_session_not_found
from app.application.sales_runtime.persistent_runtime import PersistentSalesRuntime
from app.leads.campaign.contracts import CampaignConfig
from app.persistence.errors import NotFoundError
from app.runtime.chat.contracts import ChatSession
from app.runtime.chat.repository import SessionNotFoundError


def get_runtime(request: Request) -> PersistentSalesRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(
            status_code=500,
            detail={"code": "INTERNAL_ERROR", "message": "runtime not configured"},
        )
    return runtime


def get_campaign_lookup(request: Request):
    lookup = getattr(request.app.state, "campaign_lookup", None)
    if lookup is None:
        def _not_configured(_cid: str) -> CampaignConfig | None:
            return None
        return _not_configured
    return lookup


def get_principal_resolver(request: Request) -> PrincipalResolver:
    resolver = getattr(request.app.state, "principal_resolver", None)
    if resolver is None:
        # The factory refuses to construct `app` without a resolver.
        # If this ever triggers at runtime, treat it as an internal
        # configuration failure, NOT an open gate.
        raise HTTPException(
            status_code=500,
            detail={"code": "INTERNAL_ERROR", "message": "no principal resolver configured"},
        )
    return resolver


def get_principal(
    request: Request,
    resolver: Annotated[PrincipalResolver, Depends(get_principal_resolver)],
) -> Principal:
    try:
        return resolver.resolve_http(request)
    except AuthenticationError:
        raise_http_unauthenticated()


def get_session_scoped(
    session_id: str,
    runtime: Annotated[PersistentSalesRuntime, Depends(get_runtime)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> ChatSession:
    """Resolve a session AND verify it belongs to the current tenant.

    Tenant isolation at the dependency boundary. A principal from a
    different tenant looking up a known session id receives 403.
    """
    try:
        session = runtime.sessions.get(session_id)
    except (SessionNotFoundError, NotFoundError):
        err = map_session_not_found(session_id)
        raise HTTPException(status_code=err.http_status, detail=err.to_payload())
    if session.tenant_id != principal.tenant_id:
        err = map_session_forbidden()
        raise HTTPException(status_code=err.http_status, detail=err.to_payload())
    return session
