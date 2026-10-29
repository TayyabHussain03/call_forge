"""Transport error taxonomy and typed exception mapping.

The transport does not interpret business meaning; it maps known runtime
exceptions to stable public codes + HTTP status. Unknown exceptions are
mapped to INTERNAL_ERROR with no leaked internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ApiErrorCode(str, Enum):
    """Stable external error taxonomy. Frontend branches on these."""

    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_FORBIDDEN = "SESSION_FORBIDDEN"
    INVALID_SESSION_STATE = "INVALID_SESSION_STATE"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    STALE_REVISION = "STALE_REVISION"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    BOOTSTRAP_REJECTED = "BOOTSTRAP_REJECTED"
    TERMINAL_SESSION = "TERMINAL_SESSION"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True)
class ApiError:
    """Typed error envelope. Serialized the same way on HTTP + WS."""

    code: ApiErrorCode
    message: str
    http_status: int

    def to_payload(self) -> dict:
        return {"code": self.code.value, "message": self.message}


def map_stale_revision() -> ApiError:
    return ApiError(
        code=ApiErrorCode.STALE_REVISION,
        message="The session changed before this turn could be committed.",
        http_status=409,
    )


def map_session_not_found(session_id: str) -> ApiError:
    return ApiError(
        code=ApiErrorCode.SESSION_NOT_FOUND,
        message=f"Session '{session_id}' was not found.",
        http_status=404,
    )


def map_session_forbidden() -> ApiError:
    return ApiError(
        code=ApiErrorCode.SESSION_FORBIDDEN,
        message="This session is not accessible under the current tenant.",
        http_status=403,
    )


def map_bootstrap_rejected(reason: str) -> ApiError:
    return ApiError(
        code=ApiErrorCode.BOOTSTRAP_REJECTED,
        message=reason or "Session bootstrap rejected.",
        http_status=409,
    )


def map_validation(detail: str) -> ApiError:
    return ApiError(
        code=ApiErrorCode.VALIDATION_ERROR,
        message=detail or "Request failed validation.",
        http_status=422,
    )


def map_internal() -> ApiError:
    return ApiError(
        code=ApiErrorCode.INTERNAL_ERROR,
        message="An unexpected server error occurred.",
        http_status=500,
    )
