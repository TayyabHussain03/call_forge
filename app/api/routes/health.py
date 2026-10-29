"""Health and readiness probes.

`/health` is a liveness check — process is alive. No DB, no LLM.
`/ready` is a readiness check — minimum runtime dependencies accessible.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/ready")
def ready(request: Request) -> JSONResponse:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        return JSONResponse(
            status_code=503, content={"status": "not_ready", "reason": "runtime missing"},
        )
    try:
        runtime._c.raw.execute("SELECT 1").fetchone()
    except Exception:
        return JSONResponse(
            status_code=503, content={"status": "not_ready", "reason": "db unavailable"},
        )
    return JSONResponse(status_code=200, content={"status": "ready"})
