"""Security-headers middleware.

Adds a strict Content-Security-Policy plus a small set of high-value
defense-in-depth headers to every HTTP response. Configuration is
environment-driven: callers pass explicit `connect_src_origins` from
deployment config so the policy correctly covers API/WS endpoints that
may live on a different origin.

Scope:
    - This slice adds headers to the HTTP response surface. WebSocket
      upgrades are governed by `connect-src` on the HTML page itself,
      not by response headers on the WS endpoint.
    - CSP ownership lives at the HTTP response/middleware boundary —
      runtime, Brain, and route handlers remain unaware.
"""

from __future__ import annotations

from typing import Iterable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach a strict CSP + nosniff + referrer-policy + bounded
    permissions-policy to every HTTP response.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        connect_src_origins: tuple[str, ...] = (),
        enable_hsts: bool = False,
    ) -> None:
        super().__init__(app)
        self._csp = _build_csp(connect_src_origins)
        self._enable_hsts = enable_hsts

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers.setdefault("Content-Security-Policy", self._csp)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=(), payment=()",
        )
        if self._enable_hsts:
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=31536000; includeSubDomains",
            )
        return response


def _build_csp(connect_src_origins: Iterable[str]) -> str:
    """Compose the CSP policy string.

    Base policy derived from the current single-page chat client:
        - loads chat.css + chat.js from /static (same origin)
        - fetches /api/v1/* and connects to /ws/sessions/* (same origin
          by default; configurable origins supported for split deployments)
        - no inline scripts, no inline event handlers, no eval
        - no framing of this page
    """
    cs = ["'self'"]
    for origin in connect_src_origins:
        if origin and origin not in cs:
            cs.append(origin)
    # For WebSocket from same-origin, modern browsers accept `'self'`.
    # Including `ws:` and `wss:` explicitly covers older behavior and
    # is harmless; we only add them when deployment hasn't passed an
    # explicit WS origin. Keep it minimal: let callers configure.
    directives = [
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self'",
        "img-src 'self' data:",
        f"connect-src {' '.join(cs)}",
        "object-src 'none'",
        "base-uri 'none'",
        "frame-ancestors 'none'",
        "form-action 'self'",
    ]
    return "; ".join(directives)
