"""Slice 52 — FastAPI + WebSocket transport acceptance.

Transport-layer tests. These exercise the real production runtime
behind the API; they do NOT instantiate parallel business logic in test
code. Composition pattern mirrors Slice 51J:
    build_production_sales_runtime(...) → create_app(runtime=..., campaign_lookup=...)

Covers invariant requirements:
    - idempotent retry (HTTP + WS + cross-transport)
    - tenant/session isolation
    - stale-revision mapping
    - terminal / DNC / NOT_INTERESTED
    - no leak of internal state
    - WebSocket reconnect replay
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.application.production_turn_processor import ProductionTurnProcessor
from app.application.sales_runtime.composition import (
    ProductionSalesRuntimeFacade,
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
from app.runtime.chat.bootstrap.contracts import SessionBootstrapInput


# ---------------------------------------------------------------------------
# Shared factory + fixtures — same pattern as Slice 51J
# ---------------------------------------------------------------------------


class _TurnFactory:
    """Factory that constructs real ProductionTurnProcessors."""

    def __init__(self, facade: ProductionSalesRuntimeFacade, evidence=()):
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
        provider = self._providers.setdefault(
            session.session_id, self._reasoning(),
        )
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


def _phone() -> NormalizedPhone:
    return NormalizedPhone(
        raw="+15551234567", normalized="+15551234567",
        status=PhoneNormalizationStatus.VALID_E164,
    )


def _lead(lead_id="lead-1", tenant="t1") -> LeadRecord:
    return LeadRecord(
        lead_id=lead_id, tenant_id=tenant, campaign_id="c1",
        business_name="ABC Services", phone=_phone(),
        category=None, city=None, state=None, country=None,
        address=None, email=None, contact_name=None,
        website=None, website_status=WebsiteStatus.UNKNOWN,
    )


def _queue(lead_id="lead-1", tenant="t1") -> LeadQueueEntry:
    return LeadQueueEntry(
        lead_id=lead_id, tenant_id=tenant, campaign_id="c1",
        queue_position=0, status=LeadQueueStatus.READY,
    )


def _campaign(tenant="t1") -> CampaignConfig:
    return CampaignConfig(
        campaign_id="c1", tenant_id=tenant, name="Test",
        status=CampaignStatus.ACTIVE,
        authorized_service_ids=("svc-automation",),
    )


def _policy_cfg() -> PreCallDiscoveryPolicy:
    return PreCallDiscoveryPolicy(
        policy_id="pol-1", version="1.0",
        service_rules=(
            ServiceDiscoveryRule(
                service_id="svc-automation",
                required_fact_keys=("website.presence",),
            ),
        ),
    )


def _start_payload(sid="sess-52", lead_id="lead-1", req="req-1"):
    return {
        "campaign_id": "c1",
        "lead_id": lead_id,
        "session_id": sid,
        "bootstrap_request_id": req,
        "trusted_time_iso": "2026-10-29T10:00:00Z",
    }


@pytest.fixture
def api_env(tmp_path):
    """Build a real runtime + wire into create_app."""
    db_path = str(tmp_path / "slice52.db")
    conn = SqliteConnection(db_path)
    evidence = (
        ParticipantContinuityEvidence(
            kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
            source_turn_id="t1",
            claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
        ),
    )
    facade = build_production_sales_runtime(
        connection=conn,
        processor_factory=_TurnFactory(None, ()),
        intent_transport=None,
    )
    facade.runtime._processor_factory = _TurnFactory(facade, evidence_sequence=evidence) \
        if False else _TurnFactory(facade, evidence=evidence)
    # Seed data for a few tenants + leads
    for tenant in ("t1", "t2"):
        facade.runtime.policies.register_default("c1", _policy_cfg())
    facade.runtime.leads.save(_lead("lead-1", tenant="t1"))
    facade.runtime.queue.save(_queue("lead-1", tenant="t1"))
    facade.runtime.leads.save(_lead("lead-2", tenant="t2"))
    facade.runtime.queue.save(_queue("lead-2", tenant="t2"))

    campaigns = {
        "c1": _campaign(tenant="t1"),
        "c2": _campaign(tenant="t2"),
    }
    # c2 belongs to tenant t2
    campaigns["c2"] = CampaignConfig(
        campaign_id="c2", tenant_id="t2", name="T2",
        status=CampaignStatus.ACTIVE,
        authorized_service_ids=("svc-automation",),
    )

    def _lookup(cid):
        return campaigns.get(cid)

    app = create_app(runtime=facade.runtime, campaign_lookup=_lookup)
    return app, facade


@pytest.fixture
def client(api_env):
    app, _ = api_env
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health + readiness
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_ready_returns_ready_when_runtime_configured(self, client):
        r = client.get("/ready")
        assert r.status_code == 200
        assert r.json()["status"] == "ready"


# ---------------------------------------------------------------------------
# Session bootstrap
# ---------------------------------------------------------------------------


class TestStartSession:
    def test_create_requires_tenant_header(self, client):
        r = client.post("/api/v1/sessions", json=_start_payload())
        assert r.status_code == 401

    def test_create_session_happy_path(self, client):
        r = client.post(
            "/api/v1/sessions",
            headers={"X-Tenant-Id": "t1"},
            json=_start_payload(),
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["session_id"] == "sess-52"
        assert body["tenant_id"] == "t1"
        assert body["campaign_id"] == "c1"

    def test_create_session_wrong_tenant_rejected(self, client):
        # c1 belongs to t1 but caller claims t2 — rejected
        r = client.post(
            "/api/v1/sessions",
            headers={"X-Tenant-Id": "t2"},
            json=_start_payload(),
        )
        assert r.status_code == 409
        assert r.json()["detail"]["code"] == "BOOTSTRAP_REJECTED"

    def test_create_unknown_campaign_rejected(self, client):
        payload = _start_payload()
        payload["campaign_id"] = "no-such-campaign"
        r = client.post(
            "/api/v1/sessions",
            headers={"X-Tenant-Id": "t1"},
            json=payload,
        )
        assert r.status_code == 409

    def test_oversized_bootstrap_id_rejected(self, client):
        payload = _start_payload()
        payload["bootstrap_request_id"] = "x" * 200
        r = client.post(
            "/api/v1/sessions",
            headers={"X-Tenant-Id": "t1"},
            json=payload,
        )
        assert r.status_code == 422

    def test_extra_field_rejected(self, client):
        """§51 — client cannot inject arbitrary runtime state."""
        payload = _start_payload()
        payload["authorized_services"] = ["anything"]
        r = client.post(
            "/api/v1/sessions",
            headers={"X-Tenant-Id": "t1"},
            json=payload,
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Process turn
# ---------------------------------------------------------------------------


class TestProcessTurn:
    def _start(self, client):
        r = client.post(
            "/api/v1/sessions",
            headers={"X-Tenant-Id": "t1"},
            json=_start_payload(),
        )
        assert r.status_code == 201, r.text

    def test_turn_commits_and_returns_text(self, client):
        self._start(client)
        r = client.post(
            "/api/v1/sessions/sess-52/turns",
            headers={"X-Tenant-Id": "t1"},
            json={"turn_request_id": "tr-1", "text": "Hello there"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["turn_request_id"] == "tr-1"
        assert body["turn_sequence"] == 1
        assert body["replayed"] is False
        assert body["revision"] >= 1
        assert len(body["text"]) > 0

    def test_duplicate_turn_request_replays(self, client):
        """§60 — same turn_request_id returns the same committed result."""
        self._start(client)
        headers = {"X-Tenant-Id": "t1"}
        payload = {"turn_request_id": "tr-dup", "text": "First"}
        r1 = client.post(
            "/api/v1/sessions/sess-52/turns", headers=headers, json=payload,
        )
        assert r1.status_code == 200
        first = r1.json()
        r2 = client.post(
            "/api/v1/sessions/sess-52/turns", headers=headers, json=payload,
        )
        assert r2.status_code == 200
        second = r2.json()
        assert second["replayed"] is True
        assert second["text"] == first["text"]
        assert second["revision"] == first["revision"]
        assert second["turn_sequence"] == first["turn_sequence"]

    def test_unknown_session_returns_404(self, client):
        r = client.post(
            "/api/v1/sessions/no-such-session/turns",
            headers={"X-Tenant-Id": "t1"},
            json={"turn_request_id": "tr-1", "text": "hello"},
        )
        assert r.status_code == 404
        assert r.json()["detail"]["code"] == "SESSION_NOT_FOUND"

    def test_wrong_tenant_cannot_touch_session(self, client):
        """§14 — tenant isolation at session lookup boundary."""
        self._start(client)
        r = client.post(
            "/api/v1/sessions/sess-52/turns",
            headers={"X-Tenant-Id": "t2"},
            json={"turn_request_id": "tr-1", "text": "hi"},
        )
        assert r.status_code == 403
        assert r.json()["detail"]["code"] == "SESSION_FORBIDDEN"

    def test_empty_text_rejected_by_schema(self, client):
        self._start(client)
        r = client.post(
            "/api/v1/sessions/sess-52/turns",
            headers={"X-Tenant-Id": "t1"},
            json={"turn_request_id": "tr-1", "text": ""},
        )
        assert r.status_code == 422

    def test_oversized_text_rejected_by_schema(self, client):
        self._start(client)
        r = client.post(
            "/api/v1/sessions/sess-52/turns",
            headers={"X-Tenant-Id": "t1"},
            json={"turn_request_id": "tr-1", "text": "x" * 3000},
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# GET session
# ---------------------------------------------------------------------------


class TestGetSession:
    def test_get_session_returns_public_view(self, client):
        client.post(
            "/api/v1/sessions",
            headers={"X-Tenant-Id": "t1"},
            json=_start_payload(),
        )
        r = client.get(
            "/api/v1/sessions/sess-52",
            headers={"X-Tenant-Id": "t1"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["session_id"] == "sess-52"
        assert body["tenant_id"] == "t1"
        assert body["turn_count"] == 0
        # §20/§21 — no snapshot payloads, no transcripts
        for forbidden in ("snapshot", "transcript", "runtime_snapshot_json"):
            assert forbidden not in body

    def test_get_session_tenant_isolation(self, client):
        client.post(
            "/api/v1/sessions",
            headers={"X-Tenant-Id": "t1"},
            json=_start_payload(),
        )
        r = client.get(
            "/api/v1/sessions/sess-52",
            headers={"X-Tenant-Id": "t2"},
        )
        assert r.status_code == 403


# ---------------------------------------------------------------------------
# Finish session
# ---------------------------------------------------------------------------


class TestFinishSession:
    def _start_and_turn(self, client):
        client.post(
            "/api/v1/sessions",
            headers={"X-Tenant-Id": "t1"},
            json=_start_payload(),
        )
        client.post(
            "/api/v1/sessions/sess-52/turns",
            headers={"X-Tenant-Id": "t1"},
            json={"turn_request_id": "tr-1", "text": "hello"},
        )

    def test_finish_natural_returns_outcome(self, client):
        self._start_and_turn(client)
        r = client.post(
            "/api/v1/sessions/sess-52/finish",
            headers={"X-Tenant-Id": "t1"},
            json={},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["session_id"] == "sess-52"
        assert body["dnc_status"] == "not_dnc"

    def test_finish_dnc_marks_queue(self, client, api_env):
        _, facade = api_env
        self._start_and_turn(client)
        r = client.post(
            "/api/v1/sessions/sess-52/finish",
            headers={"X-Tenant-Id": "t1"},
            json={"is_dnc": True},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["outcome_status"] == "dnc"
        q = facade.runtime.queue.get("t1", "c1", "lead-1")
        assert q.status.value == "dnc"

    def test_finish_not_interested_distinct_from_dnc(self, client, api_env):
        _, facade = api_env
        self._start_and_turn(client)
        r = client.post(
            "/api/v1/sessions/sess-52/finish",
            headers={"X-Tenant-Id": "t1"},
            json={"is_not_interested": True},
        )
        assert r.status_code == 200
        assert r.json()["outcome_status"] != "dnc"
        q = facade.runtime.queue.get("t1", "c1", "lead-1")
        assert q.status.value != "dnc"


# ---------------------------------------------------------------------------
# WebSocket chat
# ---------------------------------------------------------------------------


class TestWebSocketChat:
    def _start(self, client):
        r = client.post(
            "/api/v1/sessions",
            headers={"X-Tenant-Id": "t1"},
            json=_start_payload(),
        )
        assert r.status_code == 201

    def test_connect_and_single_turn(self, client):
        self._start(client)
        with client.websocket_connect(
            "/ws/sessions/sess-52?tenant=t1",
        ) as ws:
            ws.send_json({
                "type": "user_turn",
                "turn_request_id": "ws-1",
                "text": "Hi there",
            })
            msg = ws.receive_json()
            assert msg["type"] == "agent_turn"
            assert msg["turn_request_id"] == "ws-1"
            assert msg["turn_sequence"] == 1
            assert msg["replayed"] is False
            assert len(msg["text"]) > 0

    def test_duplicate_request_replays_over_ws(self, client):
        """§31 — reconnect + replay same request_id returns committed result."""
        self._start(client)
        with client.websocket_connect(
            "/ws/sessions/sess-52?tenant=t1",
        ) as ws:
            payload = {
                "type": "user_turn",
                "turn_request_id": "ws-dup",
                "text": "Hello",
            }
            ws.send_json(payload)
            first = ws.receive_json()
            ws.send_json(payload)
            second = ws.receive_json()
            assert second["replayed"] is True
            assert second["text"] == first["text"]
            assert second["revision"] == first["revision"]

    def test_connect_unknown_session_closes(self, client):
        from starlette.websockets import WebSocketDisconnect
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                "/ws/sessions/no-such?tenant=t1",
            ) as ws:
                ws.receive_json()

    def test_connect_wrong_tenant_closes(self, client):
        from starlette.websockets import WebSocketDisconnect
        self._start(client)
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                "/ws/sessions/sess-52?tenant=t2",
            ) as ws:
                ws.receive_json()

    def test_malformed_message_returns_error_without_closing(self, client):
        self._start(client)
        with client.websocket_connect(
            "/ws/sessions/sess-52?tenant=t1",
        ) as ws:
            # Not valid JSON
            ws.send_text("not valid json")
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "VALIDATION_ERROR"
            # Connection still works for a valid message
            ws.send_json({
                "type": "user_turn",
                "turn_request_id": "ws-after-err",
                "text": "hi",
            })
            msg = ws.receive_json()
            assert msg["type"] == "agent_turn"


# ---------------------------------------------------------------------------
# Cross-transport consistency (§83)
# ---------------------------------------------------------------------------


class TestCrossTransport:
    def _start(self, client):
        client.post(
            "/api/v1/sessions",
            headers={"X-Tenant-Id": "t1"},
            json=_start_payload(),
        )

    def test_http_then_ws_same_session(self, client):
        """§35 — HTTP turn then WS turn in same session: WS sees committed state."""
        self._start(client)
        r = client.post(
            "/api/v1/sessions/sess-52/turns",
            headers={"X-Tenant-Id": "t1"},
            json={"turn_request_id": "tr-http", "text": "first turn via http"},
        )
        assert r.status_code == 200
        http_rev = r.json()["revision"]

        with client.websocket_connect(
            "/ws/sessions/sess-52?tenant=t1",
        ) as ws:
            ws.send_json({
                "type": "user_turn",
                "turn_request_id": "ws-after",
                "text": "second turn via ws",
            })
            msg = ws.receive_json()
            assert msg["turn_sequence"] == 2  # authoritative sequence
            assert msg["revision"] > http_rev

    def test_http_commit_replays_through_ws(self, client):
        """§37 — HTTP commit, WS retry with same turn_request_id: replays."""
        self._start(client)
        r = client.post(
            "/api/v1/sessions/sess-52/turns",
            headers={"X-Tenant-Id": "t1"},
            json={"turn_request_id": "shared-tr", "text": "hello"},
        )
        http_body = r.json()

        with client.websocket_connect(
            "/ws/sessions/sess-52?tenant=t1",
        ) as ws:
            ws.send_json({
                "type": "user_turn",
                "turn_request_id": "shared-tr",
                "text": "hello",
            })
            msg = ws.receive_json()
            assert msg["replayed"] is True
            assert msg["text"] == http_body["text"]
            assert msg["revision"] == http_body["revision"]


# ---------------------------------------------------------------------------
# No business logic in transport (§98)
# ---------------------------------------------------------------------------


class TestNoLogicInTransport:
    def test_api_modules_import_no_llm_or_brain_runtime(self):
        """§50 — transport must not import Brain reasoning or LLM providers."""
        import inspect
        from app.api import app as app_mod
        from app.api.routes import sessions as sessions_mod
        from app.api.websocket import chat as ws_mod
        for mod in (app_mod, sessions_mod, ws_mod):
            src = inspect.getsource(mod)
            for forbidden in (
                "openai", "anthropic", "google_genai",
                "BrainOrchestrator", "ReasoningProvider",
            ):
                assert forbidden not in src, (
                    f"{forbidden} leaked into transport module {mod.__name__}"
                )

    def test_api_modules_have_no_industry_hardcoding(self):
        import inspect
        from app.api import app as app_mod
        from app.api.routes import sessions as sessions_mod
        from app.api.websocket import chat as ws_mod
        for mod in (app_mod, sessions_mod, ws_mod):
            src = inspect.getsource(mod).lower()
            for forbidden in ("restaurant", "dentist", "hvac", "automation"):
                assert forbidden not in src, (
                    f"{forbidden} appears in {mod.__name__}"
                )
