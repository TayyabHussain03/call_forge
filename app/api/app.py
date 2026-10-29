"""FastAPI application factory.

Transport is thin: validate → runtime call → map. The factory accepts
the composed runtime at creation time so the composition root owns
runtime lifetime and lifecycle — nothing is instantiated at module
import.

Dependencies injected via `app.state`:
    - app.state.runtime: PersistentSalesRuntime (application-scoped)
    - app.state.campaign_lookup: (campaign_id: str) -> CampaignConfig | None
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes.health import router as health_router
from app.api.routes.sessions import router as sessions_router
from app.api.websocket.chat import router as ws_router
from app.application.sales_runtime.persistent_runtime import PersistentSalesRuntime
from app.leads.campaign.contracts import CampaignConfig


_STATIC_DIR = Path(__file__).resolve().parent / "static"


CampaignLookup = Callable[[str], "CampaignConfig | None"]


def create_app(
    *,
    runtime: PersistentSalesRuntime,
    campaign_lookup: CampaignLookup,
) -> FastAPI:
    """Create the production FastAPI application.

    Caller composes `runtime` and `campaign_lookup`. The transport does
    not build runtime or know campaign resolution details. Test and
    production share the same factory with different injections.
    """
    app = FastAPI(title="Outbound Agent API", version="1.0.0")
    app.state.runtime = runtime
    app.state.campaign_lookup = campaign_lookup

    app.include_router(health_router)
    app.include_router(sessions_router)
    app.include_router(ws_router)

    # Static chat client. Slice 54 ships `chat.html` as a vanilla-JS
    # single-page client. Served at `/static/*` — the page lives at
    # `/static/chat.html`. No build step, no framework dependencies —
    # the transport-only client code is intentionally self-contained
    # per AGENTS.md bounded-change discipline.
    if _STATIC_DIR.is_dir():
        app.mount(
            "/static", StaticFiles(directory=str(_STATIC_DIR)), name="static",
        )

    return app
