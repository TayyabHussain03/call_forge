"""Principal model + resolver protocol + development resolver.

The transport depends on a validated `Principal`, never on a raw header.
`PrincipalResolver` is the one trust boundary between untrusted request
state (headers, cookies, query strings) and the runtime's authorization
checks. Routes and WebSocket handlers call `resolve_http` / `resolve_ws`
and never read raw credentials themselves.

Only `DevelopmentHeaderPrincipalResolver` is shipped. Real IdP resolvers
(JWT/OIDC/session) will be added in later slices tied to a chosen
provider — this slice deliberately does not guess issuer/audience/claim
mapping (§60).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from fastapi import HTTPException, Request
from starlette.websockets import WebSocket

from app.api.errors import ApiError, ApiErrorCode


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Principal:
    """Immutable authenticated identity. Only built by a resolver.

    `subject_id` and `tenant_id` are deliberately distinct (§4, §5):
    one principal represents one user; a tenant may hold many users.
    """

    subject_id: str
    tenant_id: str

    def __post_init__(self) -> None:
        for name in ("subject_id", "tenant_id"):
            val = getattr(self, name)
            if not isinstance(val, str):
                raise ValueError(f"{name} must be a string")
            if not val or not val.strip() or val != val.strip():
                raise ValueError(f"{name} must be non-empty and not padded")
            if len(val) > 120:
                raise ValueError(f"{name} exceeds 120 chars")


# ---------------------------------------------------------------------------
# Resolver protocol
# ---------------------------------------------------------------------------


class AuthenticationError(Exception):
    """Resolver-raised when credentials are missing/malformed/untrusted."""


@runtime_checkable
class PrincipalResolver(Protocol):
    """Trust boundary. Every implementation produces a `Principal` or
    raises `AuthenticationError`. Future JWT/OIDC/session resolvers
    satisfy this same contract."""

    mode: str  # "development" or "production" — a declared label, not auth proof

    def resolve_http(self, request: Request) -> Principal: ...

    def resolve_ws(self, websocket: WebSocket) -> Principal: ...


# ---------------------------------------------------------------------------
# Development header resolver
# ---------------------------------------------------------------------------


class DevelopmentHeaderPrincipalResolver:
    """Reads `X-Tenant-Id` + optional `X-Subject-Id` headers (or `tenant`
    + `subject` query params for WS). Explicitly development-only.

    Production deployments MUST use a real resolver — the app factory
    refuses to start with this resolver when `APP_ENV=production`.
    """

    mode = "development"

    _DEFAULT_DEV_SUBJECT = "dev-subject"

    def resolve_http(self, request: Request) -> Principal:
        tenant = request.headers.get("x-tenant-id")
        subject = request.headers.get("x-subject-id") or self._DEFAULT_DEV_SUBJECT
        return self._build(tenant, subject)

    def resolve_ws(self, websocket: WebSocket) -> Principal:
        tenant = websocket.headers.get("x-tenant-id") or websocket.query_params.get("tenant")
        subject = (
            websocket.headers.get("x-subject-id")
            or websocket.query_params.get("subject")
            or self._DEFAULT_DEV_SUBJECT
        )
        return self._build(tenant, subject)

    def _build(self, tenant, subject) -> Principal:
        if not tenant or not isinstance(tenant, str) or not tenant.strip():
            raise AuthenticationError("tenant identity missing")
        tenant_stripped = tenant.strip()
        if len(tenant_stripped) > 120:
            raise AuthenticationError("tenant identity too long")
        try:
            return Principal(subject_id=subject.strip(), tenant_id=tenant_stripped)
        except ValueError as e:
            raise AuthenticationError(str(e))


# ---------------------------------------------------------------------------
# HTTP mapping helpers
# ---------------------------------------------------------------------------


def http_unauthenticated_error() -> ApiError:
    return ApiError(
        code=ApiErrorCode.VALIDATION_ERROR,
        message="Authentication required.",
        http_status=401,
    )


def raise_http_unauthenticated() -> None:
    err = http_unauthenticated_error()
    raise HTTPException(status_code=err.http_status, detail=err.to_payload())
