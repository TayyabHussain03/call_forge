"""FastAPI dependencies for runtime, tenant, and campaign resolution.

Runtime is application-scoped — the composition root builds one
`PersistentSalesRuntime` and attaches it to `app.state.runtime`. Request
handlers depend on `get_runtime` to retrieve it.

Tenant identity is a dependency seam. Production will plug an auth
provider; today a trusted `X-Tenant-Id` header serves development.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request

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
    """Return a callable (campaign_id) -> CampaignConfig | None.

    Keeps campaign resolution a seam the composition root can plug.
    """
    lookup = getattr(request.app.state, "campaign_lookup", None)
    if lookup is None:
        def _not_configured(_cid: str) -> CampaignConfig | None:
            return None
        return _not_configured
    return lookup


def get_tenant_id(
    x_tenant_id: Annotated[str | None, Header(alias="X-Tenant-Id")] = None,
) -> str:
    """Dev-time trusted tenant identity header.

    §40–§41: this is explicitly a development seam. Production replaces
    this with a real principal-resolver dependency. Never hardcoded.
    """
    if not x_tenant_id or not x_tenant_id.strip():
        raise HTTPException(
            status_code=401,
            detail={"code": "VALIDATION_ERROR", "message": "X-Tenant-Id header is required"},
        )
    return x_tenant_id.strip()


def get_session_scoped(
    session_id: str,
    runtime: Annotated[PersistentSalesRuntime, Depends(get_runtime)],
    tenant_id: Annotated[str, Depends(get_tenant_id)],
) -> ChatSession:
    """Resolve a session AND verify it belongs to the current tenant.

    §14 — tenant isolation at the dependency boundary. Session IDs are
    opaque to one tenant; another tenant looking up the same ID must
    receive 404-equivalent (never leak existence across tenants). We
    emit SESSION_FORBIDDEN rather than 404 purely to keep the server's
    audit trail distinguishing "wrong tenant" from "no such session";
    both resolve to the same client-visible denial.
    """
    try:
        session = runtime.sessions.get(session_id)
    except (SessionNotFoundError, NotFoundError):
        err = map_session_not_found(session_id)
        raise HTTPException(status_code=err.http_status, detail=err.to_payload())
    if session.tenant_id != tenant_id:
        err = map_session_forbidden()
        raise HTTPException(status_code=err.http_status, detail=err.to_payload())
    return session
