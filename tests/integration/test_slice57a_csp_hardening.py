"""Slice 57A — CSP + static-asset separation + security headers.

Proves:
    (A) static assets (chat.html, chat.js, chat.css) served with
        correct MIME types
    (B) every HTTP response carries CSP + nosniff + referrer-policy
        + permissions-policy
    (C) CSP contains the strict directives — no 'unsafe-inline', no
        'unsafe-eval', explicit connect-src with configurable origins
    (D) chat.html contains NO inline <script>…</script> block and NO
        inline event handlers; JS lives in chat.js
    (E) chat.js is free of eval() / new Function() / document.write()
    (F) auth regression — Slice 56 development-resolver browser flow
        still works under security headers
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.api.auth import DevelopmentHeaderPrincipalResolver
from app.api.middleware.security_headers import _build_csp
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


_STATIC_DIR = Path(__file__).resolve().parents[2] / "app" / "api" / "static"


# ---------------------------------------------------------------------------
# Section A — static-asset separation (file layout)
# ---------------------------------------------------------------------------


class TestStaticAssetSeparation:
    def test_chat_html_exists(self):
        assert (_STATIC_DIR / "chat.html").is_file()

    def test_chat_js_exists(self):
        assert (_STATIC_DIR / "chat.js").is_file()

    def test_chat_css_exists(self):
        assert (_STATIC_DIR / "chat.css").is_file()

    def test_chat_html_has_no_inline_script_block(self):
        """§4, §5 — the <script>…</script> block was externalized so
        CSP can be `script-src 'self'`."""
        html = (_STATIC_DIR / "chat.html").read_text(encoding="utf-8")
        # <script src="…"></script> is allowed; inline <script> body is not
        inline_script_block = re.search(
            r"<script\s*>[\s\S]*?</script>", html, re.IGNORECASE,
        )
        assert inline_script_block is None, (
            "chat.html must not carry an inline <script> body"
        )
        # External script reference IS present
        assert '<script src="/static/chat.js"' in html

    def test_chat_html_has_no_inline_style_block(self):
        """§6 Option B — CSS externalized so `style-src 'self'` works."""
        html = (_STATIC_DIR / "chat.html").read_text(encoding="utf-8")
        inline_style_block = re.search(
            r"<style\s*>[\s\S]*?</style>", html, re.IGNORECASE,
        )
        assert inline_style_block is None
        assert '<link rel="stylesheet" href="/static/chat.css"' in html

    def test_chat_html_has_no_inline_event_handlers(self):
        """§18 — no `onclick=`, `onerror=`, `onload=`, etc. attributes."""
        html = (_STATIC_DIR / "chat.html").read_text(encoding="utf-8")
        for attr in ("onclick=", "onerror=", "onload=", "onchange=",
                     "onsubmit=", "onmouseover=", "onfocus=", "onblur="):
            assert attr not in html, f"inline handler {attr} in chat.html"


# ---------------------------------------------------------------------------
# Section B — chat.js source invariants
# ---------------------------------------------------------------------------


class TestChatJsInvariants:
    def test_no_eval_or_dynamic_script(self):
        """§17, §19 — no eval(), no new Function(), no document.write()."""
        js = (_STATIC_DIR / "chat.js").read_text(encoding="utf-8")
        for forbidden in ("eval(", "new Function(", "document.write("):
            assert forbidden not in js, f"forbidden construct: {forbidden}"

    def test_no_innerhtml_regression(self):
        """§20 — Slice 54 invariant preserved after externalization."""
        js = (_STATIC_DIR / "chat.js").read_text(encoding="utf-8")
        assert ".innerHTML" not in js

    def test_use_strict_present(self):
        js = (_STATIC_DIR / "chat.js").read_text(encoding="utf-8")
        assert '"use strict";' in js


# ---------------------------------------------------------------------------
# Section C — CSP policy builder unit tests
# ---------------------------------------------------------------------------


class TestCspBuilder:
    def test_default_csp_contains_strict_directives(self):
        csp = _build_csp(())
        assert "default-src 'self'" in csp
        assert "script-src 'self'" in csp
        assert "style-src 'self'" in csp
        assert "img-src 'self' data:" in csp
        assert "connect-src 'self'" in csp
        assert "object-src 'none'" in csp
        assert "base-uri 'none'" in csp
        assert "frame-ancestors 'none'" in csp
        assert "form-action 'self'" in csp

    def test_csp_never_contains_unsafe_values(self):
        csp = _build_csp(("ws://api.example.com",))
        assert "'unsafe-inline'" not in csp
        assert "'unsafe-eval'" not in csp
        # No wildcard origin, even in connect-src
        assert "connect-src *" not in csp

    def test_connect_src_includes_configured_origins(self):
        csp = _build_csp(("https://api.example.com", "wss://ws.example.com"))
        assert "https://api.example.com" in csp
        assert "wss://ws.example.com" in csp

    def test_connect_src_deduplicates(self):
        csp = _build_csp(("'self'", "'self'", "https://api.example.com"))
        # 'self' only appears once in connect-src
        connect_segment = [
            d for d in csp.split(";") if "connect-src" in d
        ][0]
        assert connect_segment.count("'self'") == 1


# ---------------------------------------------------------------------------
# Section D — security headers wired end-to-end
# ---------------------------------------------------------------------------


def _phone():
    return NormalizedPhone(
        raw="+15551234567", normalized="+15551234567",
        status=PhoneNormalizationStatus.VALID_E164,
    )


def _lead():
    return LeadRecord(
        lead_id="lead-1", tenant_id="t1", campaign_id="c1",
        business_name="ABC Services", phone=_phone(),
        category=None, city=None, state=None, country=None,
        address=None, email=None, contact_name=None,
        website=None, website_status=WebsiteStatus.UNKNOWN,
    )


def _queue():
    return LeadQueueEntry(
        lead_id="lead-1", tenant_id="t1", campaign_id="c1",
        queue_position=0, status=LeadQueueStatus.READY,
    )


def _campaign():
    return CampaignConfig(
        campaign_id="c1", tenant_id="t1", name="Test",
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
    db_path = str(tmp_path / "slice57a.db")
    conn = SqliteConnection(db_path)
    transport = ScriptedIntentTransport(('{"intent":"none"}',))
    facade = build_production_sales_runtime(
        connection=conn,
        processor_factory=_TurnFactory(None, ()),
        intent_transport=transport,
    )
    facade.runtime.leads.save(_lead())
    facade.runtime.queue.save(_queue())
    facade.runtime.policies.register_default("c1", _policy_cfg())
    evidence = (
        ParticipantContinuityEvidence(
            kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
            source_turn_id="t1",
            claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
        ),
    )
    facade.runtime._processor_factory = _TurnFactory(facade, evidence=evidence)
    campaigns = {"c1": _campaign()}
    app = create_app(
        runtime=facade.runtime,
        campaign_lookup=campaigns.get,
        principal_resolver=DevelopmentHeaderPrincipalResolver(),
    )
    return app


class TestStaticServingAndHeaders:
    def test_chat_html_served_with_html_mime(self, api_env):
        client = TestClient(api_env)
        r = client.get("/static/chat.html")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")

    def test_chat_js_served_with_js_mime(self, api_env):
        client = TestClient(api_env)
        r = client.get("/static/chat.js")
        assert r.status_code == 200
        ct = r.headers.get("content-type", "")
        assert "javascript" in ct, f"got {ct}"

    def test_chat_css_served_with_css_mime(self, api_env):
        client = TestClient(api_env)
        r = client.get("/static/chat.css")
        assert r.status_code == 200
        assert "text/css" in r.headers.get("content-type", "")

    def test_security_headers_on_html_response(self, api_env):
        client = TestClient(api_env)
        r = client.get("/static/chat.html")
        assert r.status_code == 200
        # §11 — all four high-value headers
        csp = r.headers.get("content-security-policy", "")
        assert "default-src 'self'" in csp
        assert "script-src 'self'" in csp
        assert "'unsafe-inline'" not in csp
        assert "'unsafe-eval'" not in csp
        assert r.headers.get("x-content-type-options") == "nosniff"
        assert r.headers.get("referrer-policy") == "no-referrer"
        permissions = r.headers.get("permissions-policy", "")
        assert "camera=()" in permissions

    def test_security_headers_on_api_response(self, api_env):
        client = TestClient(api_env)
        r = client.get("/health")
        csp = r.headers.get("content-security-policy", "")
        assert csp  # present on every response, not just HTML

    def test_hsts_not_enabled_in_development(self, api_env):
        """§13 — HSTS must NOT be enabled in dev HTTP."""
        client = TestClient(api_env)
        r = client.get("/health")
        assert "strict-transport-security" not in {
            k.lower() for k in r.headers.keys()
        }

    def test_hsts_only_enabled_in_production_with_opt_in(self, tmp_path):
        """HSTS requires explicit opt-in AND production env — a mis-flag
        in dev must not emit the header."""
        from app.application.sales_runtime.composition import (
            build_production_sales_runtime,
        )

        class _CustomResolver:
            mode = "production"
            def resolve_http(self, request):
                from app.api.auth import Principal
                return Principal(subject_id="s", tenant_id="t1")
            def resolve_ws(self, websocket):
                from app.api.auth import Principal
                return Principal(subject_id="s", tenant_id="t1")

        from app.persistence.sqlite.connection import SqliteConnection
        conn = SqliteConnection(str(tmp_path / "hsts.db"))
        facade = build_production_sales_runtime(
            connection=conn,
            processor_factory=_TurnFactory(None, ()),
            intent_transport=None,
        )
        prod_app = create_app(
            runtime=facade.runtime,
            campaign_lookup=lambda _c: None,
            principal_resolver=_CustomResolver(),
            app_env="production",
            enable_hsts=True,
        )
        client = TestClient(prod_app)
        r = client.get("/health")
        assert "strict-transport-security" in {
            k.lower() for k in r.headers.keys()
        }


class TestConfigurableConnectSrc:
    def test_configured_origins_appear_in_csp(self, tmp_path):
        from app.application.sales_runtime.composition import (
            build_production_sales_runtime,
        )
        from app.persistence.sqlite.connection import SqliteConnection
        conn = SqliteConnection(str(tmp_path / "ccs.db"))
        facade = build_production_sales_runtime(
            connection=conn,
            processor_factory=_TurnFactory(None, ()),
            intent_transport=None,
        )
        app = create_app(
            runtime=facade.runtime,
            campaign_lookup=lambda _c: None,
            principal_resolver=DevelopmentHeaderPrincipalResolver(),
            csp_connect_src_origins=(
                "https://api.example.com",
                "wss://ws.example.com",
            ),
        )
        client = TestClient(app)
        r = client.get("/health")
        csp = r.headers.get("content-security-policy", "")
        assert "https://api.example.com" in csp
        assert "wss://ws.example.com" in csp


# ---------------------------------------------------------------------------
# Section E — auth/CORS/CSP interaction preserved
# ---------------------------------------------------------------------------


class TestAuthRegression:
    def test_dev_resolver_still_authenticates_with_security_headers(self, api_env):
        """§34 — security headers must not break the Slice 56 flow."""
        client = TestClient(api_env)
        r = client.post(
            "/api/v1/sessions",
            headers={"X-Tenant-Id": "t1"},
            json={
                "campaign_id": "c1",
                "lead_id": "lead-1",
                "session_id": "sess-57a",
                "bootstrap_request_id": "req-1",
                "trusted_time_iso": "2026-10-29T10:00:00Z",
            },
        )
        assert r.status_code == 201
        # And the response carries security headers
        assert r.headers.get("content-security-policy")
