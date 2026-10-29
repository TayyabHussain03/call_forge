"""Real-browser E2E harness.

Spins up a real uvicorn server over the canonical production runtime
and a real headless Chromium via Playwright. Deliberately bounded:

    - uses the already-installed `playwright` Python package; adds no
      new dependency and does not require `pytest-playwright`
    - the browser process is real; the server is real; the SQLite DB
      is a temp file per test module
    - the semantic LLM intent provider is a scripted deterministic
      transport (same pattern as Slice 51J onward); no external LLM

Design rules:
    - one uvicorn server per test module
    - one Chromium browser instance per test module; a fresh context
      per test so cookies/storage/console state do not bleed
    - page.on("console") + page.on("pageerror") trap unexpected
      browser-side failures so a UI that silently throws fails tests
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from pathlib import Path
from typing import Iterator

import pytest
import uvicorn
from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

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
# Runtime wiring (same pattern as Slice 51J/52/54)
# ---------------------------------------------------------------------------


class _TurnFactory:
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


# ---------------------------------------------------------------------------
# Server fixture — real uvicorn in a background thread
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _UvicornThread(threading.Thread):
    def __init__(self, app, host, port):
        super().__init__(daemon=True)
        config = uvicorn.Config(
            app, host=host, port=port, log_level="warning",
            lifespan="off", loop="asyncio",
        )
        self.server = uvicorn.Server(config)

    def run(self):
        self.server.run()


@pytest.fixture
def server_env(tmp_path) -> Iterator[dict]:
    """Spawn one real uvicorn server per TEST — the backend lead queue
    is single-use (READY → IN_PROGRESS on bootstrap), so sharing a
    runtime across tests would fail subsequent bootstraps.

    Scripted intent sequence covers the scenarios in this module:
        turn 1 → none          (self-id)
        turn 2 → catalog       ("what services…")
        turn 3 → direct        ("what does automation do")
        turn 4+ → none
    """
    tmp_db = tmp_path / "slice55.db"
    conn = SqliteConnection(str(tmp_db))
    transport = ScriptedIntentTransport((
        '{"intent":"none"}',
        '{"intent":"catalog_request"}',
        '{"intent":"direct_service_question"}',
        '{"intent":"none"}',
        '{"intent":"none"}',
        '{"intent":"none"}',
    ))
    facade = build_production_sales_runtime(
        connection=conn,
        processor_factory=_TurnFactory(None, ()),
        intent_transport=transport,
    )
    facade.runtime.leads.save(_lead())
    facade.runtime.queue.save(_queue())
    # Second lead for isolation tests
    facade.runtime.leads.save(LeadRecord(
        lead_id="lead-2", tenant_id="t1", campaign_id="c1",
        business_name="XYZ Co", phone=_phone(),
        category=None, city=None, state=None, country=None,
        address=None, email=None, contact_name=None,
        website=None, website_status=WebsiteStatus.UNKNOWN,
    ))
    facade.runtime.queue.save(LeadQueueEntry(
        lead_id="lead-2", tenant_id="t1", campaign_id="c1",
        queue_position=1, status=LeadQueueStatus.READY,
    ))
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
    app = create_app(runtime=facade.runtime, campaign_lookup=campaigns.get)

    port = _free_port()
    thread = _UvicornThread(app, "127.0.0.1", port)
    thread.start()

    # Wait for server to accept connections
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        with contextlib.closing(socket.socket()) as s:
            s.settimeout(0.2)
            try:
                s.connect(("127.0.0.1", port))
                break
            except OSError:
                time.sleep(0.05)
    else:
        thread.server.should_exit = True
        raise RuntimeError("uvicorn did not become ready")

    yield {
        "base_url": f"http://127.0.0.1:{port}",
        "ws_base": f"ws://127.0.0.1:{port}",
        "facade": facade,
        "port": port,
    }

    thread.server.should_exit = True
    thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Playwright fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def playwright_browser() -> Iterator[Browser]:
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        try:
            yield b
        finally:
            b.close()


@pytest.fixture
def browser_context(playwright_browser: Browser) -> Iterator[BrowserContext]:
    ctx = playwright_browser.new_context()
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def page_with_console(browser_context: BrowserContext):
    """A Page that traps console errors and page errors.

    The returned object has `.page`, `.errors`, `.console_errors` so
    tests can assert that no unexpected error surfaced in the browser.
    """
    page = browser_context.new_page()
    errors: list = []
    console_errors: list = []

    page.on("pageerror", lambda e: errors.append(str(e)))

    def _on_console(msg):
        if msg.type == "error":
            console_errors.append(msg.text)
    page.on("console", _on_console)

    class _PageWrap:
        pass
    wrap = _PageWrap()
    wrap.page = page
    wrap.errors = errors
    wrap.console_errors = console_errors
    return wrap


def chat_url(server_env, *, tenant="t1", campaign="c1", lead="lead-1") -> str:
    base = server_env["base_url"]
    ws = server_env["ws_base"]
    return (
        f"{base}/static/chat.html"
        f"?api={base}&ws={ws}"
        f"&tenant={tenant}&campaign={campaign}&lead={lead}"
    )
