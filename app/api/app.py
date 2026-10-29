"""FastAPI application factory.

Trust boundary ownership (Slice 56):
    - `principal_resolver` is REQUIRED. There is no implicit default.
    - In production mode (`APP_ENV=production` env var, or `app_env`
      kwarg), a `DevelopmentHeaderPrincipalResolver` is REFUSED at
      startup — fail-closed, never silently insecure.

Dependencies injected via `app.state`:
    - app.state.runtime            : PersistentSalesRuntime (app-scoped)
    - app.state.campaign_lookup    : (campaign_id) -> CampaignConfig | None
    - app.state.principal_resolver : PrincipalResolver
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.auth import (
    DevelopmentHeaderPrincipalResolver,
    PrincipalResolver,
)
from app.api.routes.health import router as health_router
from app.api.routes.sessions import router as sessions_router
from app.api.websocket.chat import router as ws_router
from app.application.sales_runtime.persistent_runtime import PersistentSalesRuntime
from app.leads.campaign.contracts import CampaignConfig


_STATIC_DIR = Path(__file__).resolve().parent / "static"


CampaignLookup = Callable[[str], "CampaignConfig | None"]


class InsecureProductionConfigurationError(RuntimeError):
    """Raised when the app factory would start a production deployment
    with a development-only principal resolver."""


def _resolve_app_env(app_env: str | None) -> str:
    if app_env is not None:
        return app_env.strip().lower()
    return (os.environ.get("APP_ENV") or "development").strip().lower()


def _validate_cors(
    allowed_origins: Iterable[str], allow_credentials: bool,
) -> list[str]:
    origins = [o for o in allowed_origins if o and o != "*"] if allowed_origins else []
    saw_wildcard = any(o == "*" for o in (allowed_origins or []))
    if saw_wildcard and allow_credentials:
        raise InsecureProductionConfigurationError(
            "CORS wildcard origin + allow_credentials=True is not permitted"
        )
    return (["*"] if saw_wildcard and not allow_credentials else origins)


def create_app(
    *,
    runtime: PersistentSalesRuntime,
    campaign_lookup: CampaignLookup,
    principal_resolver: PrincipalResolver,
    app_env: str | None = None,
    cors_allowed_origins: Iterable[str] = (),
    cors_allow_credentials: bool = False,
) -> FastAPI:
    """Create the production FastAPI application.

    `principal_resolver` is required. Fail-closed on production +
    development resolver combination (§10, §11, §48, §49, §65).
    """
    env = _resolve_app_env(app_env)
    if env == "production" and isinstance(
        principal_resolver, DevelopmentHeaderPrincipalResolver,
    ):
        raise InsecureProductionConfigurationError(
            "DevelopmentHeaderPrincipalResolver cannot run in production. "
            "Configure a real PrincipalResolver before deploying."
        )
    cors_origins = _validate_cors(cors_allowed_origins, cors_allow_credentials)

    app = FastAPI(title="Outbound Agent API", version="1.0.0")
    app.state.runtime = runtime
    app.state.campaign_lookup = campaign_lookup
    app.state.principal_resolver = principal_resolver
    app.state.app_env = env

    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=cors_allow_credentials,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-Tenant-Id", "X-Subject-Id"],
        )

    app.include_router(health_router)
    app.include_router(sessions_router)
    app.include_router(ws_router)

    if _STATIC_DIR.is_dir():
        app.mount(
            "/static", StaticFiles(directory=str(_STATIC_DIR)), name="static",
        )

    return app
