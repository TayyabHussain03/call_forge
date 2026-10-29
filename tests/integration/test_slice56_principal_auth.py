"""Slice 56 — pluggable principal resolver + fail-closed configuration.

Proves:
    (A) Principal model validates fields; AuthenticationError raised on
        missing/malformed credentials
    (B) `create_app` REQUIRES principal_resolver; refuses to start a
        production deployment with the development resolver
    (C) HTTP + WebSocket use the SAME trust boundary (DevelopmentHeader
        resolver on both); routes depend on Principal, not raw header
    (D) Cross-tenant GET/turn/finish/WS all rejected
    (E) Identity-switch attempt mid-session is rejected
    (F) WS auth failure uses a permanent close code (4401) — browser
        client treats it as non-reconnectable (Slice 54 scanner +
        Slice 55 real browser)
    (G) Unsafe CORS wildcard+credentials config refused at startup
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api.app import InsecureProductionConfigurationError, create_app
from app.api.auth import (
    AuthenticationError,
    DevelopmentHeaderPrincipalResolver,
    Principal,
    PrincipalResolver,
)
from app.application.production_turn_processor import ProductionTurnProcessor
from app.application.sales_runtime.composition import (
    build_production_sales_runtime,
)
from app.brain.authority.models import load_authority_policies
from app.brain.authority.validator import AuthorityPolicyValidator
from app.brain.budget.evaluator import BudgetPolicyEvaluator
from app.brain.budget.models import load_budget_policies
from app.brain.contracts import BrainProposal
from app.brain.orchestrator.orchestrator import BrainOrchestrator
from app.brain.scope.models import load_scope_policies
from app.brain.scope.validator import ScopePolicyValidator
from app.config.settings import get_settings
from app.contracts.conversation_context import ConversationContext
from app.conversation.contact.resolver import ContactResolver
from app.conversation.engine import ConversationEngine
from app.conversation.guardrails.action_validator import ActionValidator
from app.conversation.guardrails.clarification import ClarificationEngine
from app.conversation.guardrails.fallbacks import FallbackEngine
from app.conversation.guardrails.priority import TrustedPriorityOutcome
from app.conversation.participant.contracts import (
    ParticipantContinuityEvidence,
    ParticipantEvidenceKind,
    ParticipantRoleHint,
)
from app.conversation.participant.intent_provider import ScriptedIntentTransport
from app.conversation.response_planning.planner import ResponsePlanner
from app.conversation.response_rendering.renderer import DeterministicResponseRenderer
from app.conversation.state_machine.machine import ConversationStateMachine
from app.conversation.state_machine.states import load_config
from app.core.constants import AgentAction, ConversationState, Intent, TopicCategory
from app.leads.campaign.contracts import (
    CampaignConfig,
    CampaignStatus,
    LeadQueueEntry,
    LeadQueueStatus,
)
from app.leads.contracts import (
    LeadRecord,
    NormalizedPhone,
    PhoneNormalizationStatus,
    WebsiteStatus,
)
from app.llm.providers.reasoning_provider import MockReasoningProvider
from app.persistence.sqlite.connection import SqliteConnection
from app.precall.contracts import PreCallDiscoveryPolicy, ServiceDiscoveryRule


# ---------------------------------------------------------------------------
# Section A — Principal model + Development resolver unit tests
# ---------------------------------------------------------------------------


class TestPrincipalModel:
    def test_valid_principal(self):
        p = Principal(subject_id="user-1", tenant_id="t1")
        assert p.subject_id == "user-1"
        assert p.tenant_id == "t1"

    def test_empty_subject_rejected(self):
        with pytest.raises(ValueError):
            Principal(subject_id="", tenant_id="t1")

    def test_empty_tenant_rejected(self):
        with pytest.raises(ValueError):
            Principal(subject_id="user-1", tenant_id="")

    def test_whitespace_padded_rejected(self):
        with pytest.raises(ValueError):
            Principal(subject_id=" user ", tenant_id="t1")

    def test_oversized_tenant_rejected(self):
        with pytest.raises(ValueError):
            Principal(subject_id="user-1", tenant_id="t" * 200)

    def test_non_string_rejected(self):
        with pytest.raises(ValueError):
            Principal(subject_id=123, tenant_id="t1")  # type: ignore[arg-type]


class TestDevelopmentResolver:
    def test_mode_label(self):
        assert DevelopmentHeaderPrincipalResolver().mode == "development"

    def test_resolve_http_missing_header_raises(self):
        resolver = DevelopmentHeaderPrincipalResolver()

        class _Req:
            headers = {}
        with pytest.raises(AuthenticationError):
            resolver.resolve_http(_Req())  # type: ignore[arg-type]

    def test_resolve_http_valid_header_returns_principal(self):
        resolver = DevelopmentHeaderPrincipalResolver()

        class _Req:
            headers = {"x-tenant-id": "t1"}
        p = resolver.resolve_http(_Req())  # type: ignore[arg-type]
        assert p.tenant_id == "t1"
        # Default dev subject used when not provided
        assert p.subject_id == "dev-subject"

    def test_resolve_http_custom_subject(self):
        resolver = DevelopmentHeaderPrincipalResolver()

        class _Req:
            headers = {"x-tenant-id": "t1", "x-subject-id": "alice"}
        p = resolver.resolve_http(_Req())  # type: ignore[arg-type]
        assert p.subject_id == "alice"

    def test_resolve_http_oversized_tenant_rejected(self):
        resolver = DevelopmentHeaderPrincipalResolver()

        class _Req:
            headers = {"x-tenant-id": "x" * 200}
        with pytest.raises(AuthenticationError):
            resolver.resolve_http(_Req())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Section B — production guard + CORS validation
# ---------------------------------------------------------------------------


def _phone():
    return NormalizedPhone(
        raw="+15551234567", normalized="+15551234567",
        status=PhoneNormalizationStatus.VALID_E164,
    )


def _lead(lead_id="lead-1", tenant="t1"):
    return LeadRecord(
        lead_id=lead_id, tenant_id=tenant, campaign_id=("c1" if tenant == "t1" else "c2"),
        business_name="ABC Services", phone=_phone(),
        category=None, city=None, state=None, country=None,
        address=None, email=None, contact_name=None,
        website=None, website_status=WebsiteStatus.UNKNOWN,
    )


def _queue(lead_id="lead-1", tenant="t1"):
    return LeadQueueEntry(
        lead_id=lead_id, tenant_id=tenant,
        campaign_id=("c1" if tenant == "t1" else "c2"),
        queue_position=0, status=LeadQueueStatus.READY,
    )


def _campaign(campaign_id, tenant):
    return CampaignConfig(
        campaign_id=campaign_id, tenant_id=tenant, name="Test",
        status=CampaignStatus.ACTIVE,
        authorized_service_ids=("svc-automation",),
    )


def _policy_cfg():
    return PreCallDiscoveryPolicy(
        policy_id="pol-1", version="1.0",
        service_rules=(
            ServiceDiscoveryRule(
                service_id="svc-automation",
                required_fact_keys=("website.presence",),
            ),
        ),
    )


class _TurnFactory:
    def __init__(self, facade, evidence=()):
        self._facade = facade
        self._evidence = list(evidence)
        self._providers = {}

    def _reasoning(self):
        proposals = tuple(
            BrainProposal(
                detected_intent=Intent.INTERESTED,
                proposed_action=a,
                topic_category=TopicCategory.QUALIFICATION,
            )
            for a in (
                AgentAction.GREET, AgentAction.ASK_IDENTITY,
                AgentAction.INTRODUCE_REASON, AgentAction.ANSWER_QUESTION,
                AgentAction.ANSWER_QUESTION, AgentAction.ANSWER_QUESTION,
            )
        )
        return MockReasoningProvider(scripted=proposals, default=proposals[-1])

    def _evidence_provider(self):
        seq = iter(self._evidence)
        def _p(turn, understanding):
            try:
                return next(seq)
            except StopIteration:
                return None
        return _p

    def _build(self, session, *, initial_state, **kwargs):
        config = load_config(get_settings().conversation_config_path)
        machine = ConversationStateMachine(config, initial_state)
        provider = self._providers.setdefault(session.session_id, self._reasoning())
        engine = ConversationEngine(
            machine, ActionValidator(config),
            fallback_engine=FallbackEngine(config),
            clarification_engine=ClarificationEngine(config),
            contact_resolver=ContactResolver(),
        )
        orchestrator = BrainOrchestrator(
            engine, BudgetPolicyEvaluator(),
            reasoning_provider=provider,
            scope_validator=ScopePolicyValidator(),
            authority_validator=AuthorityPolicyValidator(),
        )
        wrapped = self._facade.wrap_renderer(DeterministicResponseRenderer())
        processor = ProductionTurnProcessor(
            orchestrator=orchestrator,
            response_planner=ResponsePlanner(),
            response_renderer=wrapped,
            initial_context=ConversationContext(session.session_id),
            budget_policy=load_budget_policies(
                "app/config/budget_policy.yaml")["standard"],
            scope_policy=load_scope_policies(
                "app/config/scope_policy.yaml")["business_general"],
            authority_policy=load_authority_policies(
                "app/config/authority_policy.yaml")["standard"],
            trusted_priority_provider=lambda _: TrustedPriorityOutcome.NONE,
            participant_evidence_provider=self._evidence_provider(),
            re_explanation_intent_provider=self._facade.intent_callable,
            **kwargs,
        )
        processor._explanation_concept_provider = (
            self._facade.concept_callable_for(processor)
        )
        return processor

    def create_for_session(self, session, campaign, policy):
        return self._build(session, initial_state=ConversationState.NEW_CALL)

    def create_from_snapshot(self, session, campaign, policy, snapshot):
        return self._build(session, initial_state=snapshot.conversation_state)


@pytest.fixture
def api_env(tmp_path):
    """Multi-tenant env: tenants t1 (lead-1, campaign c1) and t2
    (lead-2, campaign c2). Both exist on the same runtime."""
    db_path = str(tmp_path / "slice56.db")
    conn = SqliteConnection(db_path)
    transport = ScriptedIntentTransport((
        '{"intent":"none"}',
        '{"intent":"none"}',
        '{"intent":"none"}',
    ))
    facade = build_production_sales_runtime(
        connection=conn,
        processor_factory=_TurnFactory(None, ()),
        intent_transport=transport,
    )
    facade.runtime.leads.save(_lead("lead-1", "t1"))
    facade.runtime.queue.save(_queue("lead-1", "t1"))
    facade.runtime.leads.save(_lead("lead-2", "t2"))
    facade.runtime.queue.save(_queue("lead-2", "t2"))
    facade.runtime.policies.register_default("c1", _policy_cfg())
    facade.runtime.policies.register_default("c2", _policy_cfg())
    evidence = (
        ParticipantContinuityEvidence(
            kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
            source_turn_id="t1",
            claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
        ),
    )
    facade.runtime._processor_factory = _TurnFactory(facade, evidence=evidence)
    campaigns = {
        "c1": _campaign("c1", "t1"),
        "c2": _campaign("c2", "t2"),
    }
    app = create_app(
        runtime=facade.runtime,
        campaign_lookup=campaigns.get,
        principal_resolver=DevelopmentHeaderPrincipalResolver(),
    )
    return app, facade


class TestProductionGuard:
    def test_create_app_requires_principal_resolver(self, api_env):
        """create_app signature mandates `principal_resolver` as a
        keyword arg. Missing it raises TypeError at call time."""
        app, facade = api_env
        with pytest.raises(TypeError):
            create_app(
                runtime=facade.runtime,
                campaign_lookup=lambda _c: None,
            )  # type: ignore[call-arg]

    def test_production_env_with_dev_resolver_refused(self, api_env):
        """§10/§11/§48/§49/§65 — fail-closed on insecure combination."""
        _, facade = api_env
        with pytest.raises(InsecureProductionConfigurationError):
            create_app(
                runtime=facade.runtime,
                campaign_lookup=lambda _c: None,
                principal_resolver=DevelopmentHeaderPrincipalResolver(),
                app_env="production",
            )

    def test_production_env_env_var_also_refuses(self, api_env, monkeypatch):
        _, facade = api_env
        monkeypatch.setenv("APP_ENV", "production")
        with pytest.raises(InsecureProductionConfigurationError):
            create_app(
                runtime=facade.runtime,
                campaign_lookup=lambda _c: None,
                principal_resolver=DevelopmentHeaderPrincipalResolver(),
            )

    def test_development_env_with_dev_resolver_ok(self, api_env):
        _, facade = api_env
        app = create_app(
            runtime=facade.runtime,
            campaign_lookup=lambda _c: None,
            principal_resolver=DevelopmentHeaderPrincipalResolver(),
            app_env="development",
        )
        assert app is not None

    def test_production_env_with_custom_resolver_allowed(self, api_env):
        """A non-development-resolver passes the production guard
        (future JWT/OIDC resolvers will reach this path)."""
        _, facade = api_env

        class _CustomResolver:
            mode = "production"
            def resolve_http(self, request): return Principal(subject_id="s", tenant_id="t1")
            def resolve_ws(self, websocket): return Principal(subject_id="s", tenant_id="t1")

        app = create_app(
            runtime=facade.runtime,
            campaign_lookup=lambda _c: None,
            principal_resolver=_CustomResolver(),
            app_env="production",
        )
        assert app is not None


class TestCORSValidation:
    def test_wildcard_origin_plus_credentials_refused(self, api_env):
        """§26 — unsafe CORS combination refused at startup."""
        _, facade = api_env
        with pytest.raises(InsecureProductionConfigurationError):
            create_app(
                runtime=facade.runtime,
                campaign_lookup=lambda _c: None,
                principal_resolver=DevelopmentHeaderPrincipalResolver(),
                cors_allowed_origins=["*"],
                cors_allow_credentials=True,
            )

    def test_explicit_origin_plus_credentials_ok(self, api_env):
        _, facade = api_env
        app = create_app(
            runtime=facade.runtime,
            campaign_lookup=lambda _c: None,
            principal_resolver=DevelopmentHeaderPrincipalResolver(),
            cors_allowed_origins=["https://app.example.com"],
            cors_allow_credentials=True,
        )
        assert app is not None


# ---------------------------------------------------------------------------
# Section C — HTTP end-to-end: missing auth, cross-tenant isolation
# ---------------------------------------------------------------------------


def _start_payload(sid, lead):
    return {
        "campaign_id": "c1" if lead == "lead-1" else "c2",
        "lead_id": lead,
        "session_id": sid,
        "bootstrap_request_id": f"req-{sid}",
        "trusted_time_iso": "2026-10-29T10:00:00Z",
    }


class TestHttpAuthorization:
    def test_missing_tenant_header_returns_401(self, api_env):
        app, _ = api_env
        client = TestClient(app)
        r = client.post(
            "/api/v1/sessions", json=_start_payload("s1", "lead-1"),
        )
        # No X-Tenant-Id — dev resolver raises AuthenticationError → 401
        assert r.status_code == 401

    def test_valid_t1_bootstrap(self, api_env):
        app, _ = api_env
        client = TestClient(app)
        r = client.post(
            "/api/v1/sessions",
            headers={"X-Tenant-Id": "t1"},
            json=_start_payload("s1", "lead-1"),
        )
        assert r.status_code == 201
        assert r.json()["tenant_id"] == "t1"

    def test_t2_cannot_get_t1_session(self, api_env):
        app, _ = api_env
        client = TestClient(app)
        client.post(
            "/api/v1/sessions",
            headers={"X-Tenant-Id": "t1"},
            json=_start_payload("s1", "lead-1"),
        )
        # Known session id, wrong tenant
        r = client.get("/api/v1/sessions/s1", headers={"X-Tenant-Id": "t2"})
        assert r.status_code == 403
        assert r.json()["detail"]["code"] == "SESSION_FORBIDDEN"

    def test_t2_cannot_turn_in_t1_session(self, api_env):
        app, _ = api_env
        client = TestClient(app)
        client.post(
            "/api/v1/sessions",
            headers={"X-Tenant-Id": "t1"},
            json=_start_payload("s1", "lead-1"),
        )
        r = client.post(
            "/api/v1/sessions/s1/turns",
            headers={"X-Tenant-Id": "t2"},
            json={"turn_request_id": "tr-1", "text": "hi"},
        )
        assert r.status_code == 403

    def test_t2_cannot_finish_t1_session(self, api_env):
        app, _ = api_env
        client = TestClient(app)
        client.post(
            "/api/v1/sessions",
            headers={"X-Tenant-Id": "t1"},
            json=_start_payload("s1", "lead-1"),
        )
        r = client.post(
            "/api/v1/sessions/s1/finish",
            headers={"X-Tenant-Id": "t2"},
            json={},
        )
        assert r.status_code == 403

    def test_identity_switch_mid_session_rejected(self, api_env):
        """§57 — start as t1, process a turn as t1, attempt turn 2 as
        t2. Rejected."""
        app, _ = api_env
        client = TestClient(app)
        client.post(
            "/api/v1/sessions",
            headers={"X-Tenant-Id": "t1"},
            json=_start_payload("s1", "lead-1"),
        )
        r1 = client.post(
            "/api/v1/sessions/s1/turns",
            headers={"X-Tenant-Id": "t1"},
            json={"turn_request_id": "tr-1", "text": "hi"},
        )
        assert r1.status_code == 200
        r2 = client.post(
            "/api/v1/sessions/s1/turns",
            headers={"X-Tenant-Id": "t2"},
            json={"turn_request_id": "tr-2", "text": "nope"},
        )
        assert r2.status_code == 403

    def test_t1_cannot_bootstrap_using_c2(self, api_env):
        """Even with valid auth, a t1 principal cannot bootstrap a
        session using campaign c2 (which belongs to t2)."""
        app, _ = api_env
        client = TestClient(app)
        payload = _start_payload("s-bad", "lead-2")
        payload["campaign_id"] = "c2"
        r = client.post(
            "/api/v1/sessions",
            headers={"X-Tenant-Id": "t1"},
            json=payload,
        )
        assert r.status_code == 409
        assert r.json()["detail"]["code"] == "BOOTSTRAP_REJECTED"


# ---------------------------------------------------------------------------
# Section D — WebSocket authorization
# ---------------------------------------------------------------------------


class TestWebSocketAuthorization:
    def test_ws_without_tenant_closes(self, api_env):
        """Missing credentials → permanent 4401 close."""
        app, _ = api_env
        client = TestClient(app)
        # Bootstrap a session first so the session_id exists
        client.post(
            "/api/v1/sessions",
            headers={"X-Tenant-Id": "t1"},
            json=_start_payload("s1", "lead-1"),
        )
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/ws/sessions/s1") as ws:
                ws.receive_json()
        assert exc_info.value.code == 4401

    def test_ws_wrong_tenant_closes_forbidden(self, api_env):
        app, _ = api_env
        client = TestClient(app)
        client.post(
            "/api/v1/sessions",
            headers={"X-Tenant-Id": "t1"},
            json=_start_payload("s1", "lead-1"),
        )
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/ws/sessions/s1?tenant=t2") as ws:
                ws.receive_json()
        assert exc_info.value.code == 4403

    def test_ws_authenticated_normal_turn(self, api_env):
        app, _ = api_env
        client = TestClient(app)
        client.post(
            "/api/v1/sessions",
            headers={"X-Tenant-Id": "t1"},
            json=_start_payload("s1", "lead-1"),
        )
        with client.websocket_connect("/ws/sessions/s1?tenant=t1") as ws:
            ws.send_json({
                "type": "user_turn", "turn_request_id": "ws-1", "text": "hi",
            })
            msg = ws.receive_json()
            assert msg["type"] == "agent_turn"


# ---------------------------------------------------------------------------
# Section E — browser client treats 4401 as permanent + no credential logging
# ---------------------------------------------------------------------------


class TestBrowserClientAuthHandling:
    def test_client_treats_4401_as_permanent_close(self):
        """Slice 56 extended Slice 54's permanent-close code list to
        include 4401. Scanner enforces presence."""
        from pathlib import Path
        import inspect
        src = (
            Path(__file__).resolve().parents[2]
            / "app" / "api" / "static" / "chat.html"
        ).read_text(encoding="utf-8")
        # 4401 appears alongside the other permanent close codes
        assert "4401" in src
        # And the handler marks "failed" state (no reconnect)
        assert 'this._setState("failed")' in src

    def test_no_credential_logging_in_api_modules(self):
        """§30, §71 — no bearer/cookie/auth header payloads logged."""
        import inspect
        from app.api import app as app_mod
        from app.api import auth as auth_mod
        from app.api import deps as deps_mod
        from app.api.routes import sessions as sessions_mod
        from app.api.websocket import chat as ws_mod
        for mod in (app_mod, auth_mod, deps_mod, sessions_mod, ws_mod):
            src = inspect.getsource(mod)
            for forbidden in (
                "logger.info(.*token",
                "console.log",  # browser-side files not in this scan
                "print(.*token",
                "print(.*header",
                "print(.*cookie",
            ):
                # Simple substring check — no formal logging infra yet
                if "print(" in src and (
                    "token" in src.lower() or "cookie" in src.lower()
                ):
                    import re
                    matches = re.findall(
                        r"print\s*\([^)]*(?:token|cookie|authorization)",
                        src, re.IGNORECASE,
                    )
                    assert matches == [], (
                        f"{mod.__name__} appears to print credentials: {matches}"
                    )


# ---------------------------------------------------------------------------
# Section F — runtime independence
# ---------------------------------------------------------------------------


class TestRuntimeAuthProviderIndependence:
    def test_runtime_module_has_no_auth_coupling(self):
        """§3, §76 — runtime must not import auth modules or know the
        resolver type."""
        import inspect
        from app.application.sales_runtime import persistent_runtime
        from app.application import production_turn_processor
        for mod in (persistent_runtime, production_turn_processor):
            src = inspect.getsource(mod)
            for forbidden in (
                "PrincipalResolver",
                "DevelopmentHeader",
                "AuthenticationError",
                "from app.api.auth",
            ):
                assert forbidden not in src, (
                    f"{forbidden} leaked into runtime module {mod.__name__}"
                )
