"""Slice 53 — typed catalog intent acceptance.

Catalog requests are a distinct semantic intent — "what services do you
provide?" (overview) rather than "what does automation do?" (concept
explanation). This slice proves:

    - CATALOG_REQUEST is a first-class ReExplanationIntent value
    - the intent-provider adapter parses it from scripted transport JSON
    - the processor short-circuits catalog outside ReExplanationPolicy
    - catalog turns do NOT advance per-concept coverage (§14, §15)
    - catalog turns do NOT require a concept id (short-circuit before
      concept provider is invoked)
    - participant isolation preserved
    - catalog request reaches the planner as `catalog_request=True`
    - catalog travels through HTTP and WebSocket identically
    - idempotent replay through canonical runtime
    - no keyword matching, no hardcoded service/industry logic
"""

from __future__ import annotations

import inspect

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
    ExplanationDepth,
    ParticipantContinuityEvidence,
    ParticipantEvidenceKind,
    ParticipantRoleHint,
)
from app.conversation.participant.intent_provider import (
    IntentClassificationInput,
    RaisingIntentTransport,
    ReExplanationIntentLLMProvider,
    ScriptedIntentTransport,
)
from app.conversation.participant.concept_provider import StaticConceptProvider
from app.conversation.participant.policy import ReExplanationIntent
from app.conversation.response_planning.contracts import ResponsePlanningInput
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
# Section 1 — intent classification (ReExplanationIntent + adapter)
# ---------------------------------------------------------------------------


class TestCatalogIntentClassification:
    def test_enum_contains_catalog_request(self):
        assert ReExplanationIntent.CATALOG_REQUEST.value == "catalog_request"
        # First-class member, not alias
        assert ReExplanationIntent("catalog_request") == ReExplanationIntent.CATALOG_REQUEST

    def test_allowed_intents_prompt_includes_catalog(self):
        """The semantic provider's prompt payload tells the transport
        which values are permitted. Catalog must appear there — otherwise
        the model has no way to propose it."""
        inp = IntentClassificationInput(utterance="any text")
        allowed = inp.to_prompt_dict()["allowed_intents"]
        assert "catalog_request" in allowed
        # Other distinct intents still enumerated
        assert "direct_service_question" in allowed
        assert "recap_request" in allowed

    def test_adapter_parses_scripted_catalog_response(self):
        transport = ScriptedIntentTransport(('{"intent":"catalog_request"}',))
        provider = ReExplanationIntentLLMProvider(transport)
        inp = IntentClassificationInput(utterance="what services do you provide?")
        assert provider.classify(inp) == ReExplanationIntent.CATALOG_REQUEST

    def test_adapter_degrades_catalog_on_malformed_response(self):
        """§7 — provider failure cannot fabricate CATALOG. Malformed →
        NONE, not false authority exposure."""
        transport = ScriptedIntentTransport(('{"intent":"catalog_reqeust"}',))  # typo
        provider = ReExplanationIntentLLMProvider(transport)
        inp = IntentClassificationInput(utterance="x")
        assert provider.classify(inp) == ReExplanationIntent.NONE

    def test_adapter_degrades_catalog_on_transport_exception(self):
        provider = ReExplanationIntentLLMProvider(RaisingIntentTransport())
        inp = IntentClassificationInput(utterance="x")
        assert provider.classify(inp) == ReExplanationIntent.NONE


# ---------------------------------------------------------------------------
# Section 2 — processor short-circuit routing
# ---------------------------------------------------------------------------


class _TurnFactory:
    def __init__(self, facade, evidence=(), concept_override=None,
                 planner_spy_container=None):
        self._facade = facade
        self._evidence = list(evidence)
        self._concept_override = concept_override
        self._planner_spy = planner_spy_container
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
        planner = _PlannerSpy() if self._planner_spy is not None else ResponsePlanner()
        if self._planner_spy is not None:
            self._planner_spy.append(planner)
        processor = ProductionTurnProcessor(
            orchestrator=orchestrator,
            response_planner=planner,
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
        if self._concept_override is not None:
            processor._explanation_concept_provider = self._concept_override
        else:
            processor._explanation_concept_provider = (
                self._facade.concept_callable_for(processor)
            )
        return processor

    def create_for_session(self, session, campaign, policy):
        return self._build(session, initial_state=ConversationState.NEW_CALL)

    def create_from_snapshot(self, session, campaign, policy, snapshot):
        return self._build(session, initial_state=snapshot.conversation_state)


class _PlannerSpy(ResponsePlanner):
    """Captures the last ResponsePlanningInput so tests can assert
    `catalog_request=True` reached the planner. Delegates to the real
    planner — no behavior change."""
    def __init__(self):
        super().__init__()
        self.last_input: ResponsePlanningInput | None = None

    def plan(self, inp):
        self.last_input = inp
        return super().plan(inp)


def _phone():
    return NormalizedPhone(
        raw="+15551234567", normalized="+15551234567",
        status=PhoneNormalizationStatus.VALID_E164,
    )


def _lead(lead_id="lead-1", tenant="t1"):
    return LeadRecord(
        lead_id=lead_id, tenant_id=tenant, campaign_id="c1",
        business_name="ABC Services", phone=_phone(),
        category=None, city=None, state=None, country=None,
        address=None, email=None, contact_name=None,
        website=None, website_status=WebsiteStatus.UNKNOWN,
    )


def _queue(lead_id="lead-1", tenant="t1"):
    return LeadQueueEntry(
        lead_id=lead_id, tenant_id=tenant, campaign_id="c1",
        queue_position=0, status=LeadQueueStatus.READY,
    )


def _campaign(tenant="t1", authorized=("svc-automation", "svc-web")):
    return CampaignConfig(
        campaign_id="c1", tenant_id=tenant, name="Test",
        status=CampaignStatus.ACTIVE,
        authorized_service_ids=authorized,
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


def _boot(sid="sess-53", lead_id="lead-1", req="req-1"):
    return SessionBootstrapInput(
        tenant_id="t1", campaign_id="c1", lead_id=lead_id,
        session_id=sid, bootstrap_request_id=req,
        trusted_time_iso="2026-10-29T10:00:00Z",
    )


@pytest.fixture
def catalog_runtime(tmp_path):
    """Runtime wired with a scripted catalog intent transport."""
    db_path = str(tmp_path / "slice53.db")
    conn = SqliteConnection(db_path)
    transport = ScriptedIntentTransport((
        '{"intent":"none"}',           # turn 1 — self-id
        '{"intent":"catalog_request"}',  # turn 2 — catalog
    ))
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
            source_turn_id="chat_sess-53_t1",
            claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
        ),
    )
    planner_spies: list = []
    facade.runtime._processor_factory = _TurnFactory(
        facade, evidence=evidence,
        concept_override=StaticConceptProvider("service:automation").as_callable(),
        planner_spy_container=planner_spies,
    )
    return facade, planner_spies


class TestProcessorShortCircuit:
    def test_catalog_request_populates_planner_catalog_flag(self, catalog_runtime):
        """§2 — the already-existing catalog_request planner field must
        now actually be set when provider returns CATALOG_REQUEST."""
        facade, planner_spies = catalog_runtime
        facade.runtime.start_session(_boot(), _campaign())
        facade.runtime.process_turn("sess-53", "hi, I handle marketing", 1, "tr-1")
        facade.runtime.process_turn("sess-53", "what services do you provide?", 2, "tr-2")

        # The second-turn planner input carries catalog_request=True
        catalog_turn = planner_spies[0].last_input
        assert catalog_turn is not None
        assert catalog_turn.catalog_request is True

    def test_catalog_bypasses_re_explanation_strategy(self, catalog_runtime):
        """§4, §18 — catalog is NOT a re-explanation of one concept."""
        facade, planner_spies = catalog_runtime
        facade.runtime.start_session(_boot(), _campaign())
        facade.runtime.process_turn("sess-53", "hi", 1, "tr-1")
        facade.runtime.process_turn("sess-53", "what services do you offer?", 2, "tr-2")

        catalog_turn = planner_spies[0].last_input
        # No strategy threaded, no concept id, no prior depth — catalog
        # does not flow through ReExplanationPolicy
        assert catalog_turn.re_explanation_strategy is None
        assert catalog_turn.explanation_concept_id is None
        assert catalog_turn.prior_explanation_depth is None

    def test_catalog_does_not_advance_concept_coverage(self, catalog_runtime):
        """§14, §15 — the critical correctness rule. A catalog overview
        is NOT equivalent to explaining every service."""
        facade, _ = catalog_runtime
        facade.runtime.start_session(_boot(), _campaign())
        facade.runtime.process_turn("sess-53", "hi", 1, "tr-1")
        proc = facade.runtime._processors["sess-53"]

        before = proc.discussion_coverage.entries
        facade.runtime.process_turn("sess-53", "what services do you provide?", 2, "tr-2")
        after = proc.discussion_coverage.entries

        # Catalog turn added zero coverage entries
        assert before == after

    def test_catalog_short_circuits_without_concept_provider(self, tmp_path):
        """§3 — catalog detection must not require concept resolution."""
        db_path = str(tmp_path / "slice53_noconcept.db")
        conn = SqliteConnection(db_path)
        transport = ScriptedIntentTransport((
            '{"intent":"none"}',
            '{"intent":"catalog_request"}',
        ))
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
        planner_spies: list = []
        # Concept provider returns None — proves catalog does not depend on it
        facade.runtime._processor_factory = _TurnFactory(
            facade, evidence=evidence,
            concept_override=StaticConceptProvider(None).as_callable(),
            planner_spy_container=planner_spies,
        )
        facade.runtime.start_session(_boot(), _campaign())
        facade.runtime.process_turn("sess-53", "hi", 1, "tr-1")
        facade.runtime.process_turn("sess-53", "what services do you offer?", 2, "tr-2")

        catalog_turn = planner_spies[0].last_input
        assert catalog_turn.catalog_request is True


# ---------------------------------------------------------------------------
# Section 3 — direct-service question stays distinct from catalog
# ---------------------------------------------------------------------------


class TestDirectVsCatalogDistinction:
    def test_direct_service_question_does_not_set_catalog_flag(self, tmp_path):
        """§17 — the two intents must produce different planner input."""
        db_path = str(tmp_path / "slice53_direct.db")
        conn = SqliteConnection(db_path)
        transport = ScriptedIntentTransport((
            '{"intent":"none"}',
            '{"intent":"direct_service_question"}',
        ))
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
        planner_spies: list = []
        facade.runtime._processor_factory = _TurnFactory(
            facade, evidence=evidence,
            concept_override=StaticConceptProvider("service:automation").as_callable(),
            planner_spy_container=planner_spies,
        )
        facade.runtime.start_session(_boot(), _campaign())
        facade.runtime.process_turn("sess-53", "hi", 1, "tr-1")
        facade.runtime.process_turn(
            "sess-53", "what does automation do?", 2, "tr-2",
        )

        direct_turn = planner_spies[0].last_input
        # Direct-question path: concept id is set, catalog flag is NOT
        assert direct_turn.explanation_concept_id == "service:automation"
        assert direct_turn.catalog_request is False


# ---------------------------------------------------------------------------
# Section 4 — later direct question after catalog gets fresh explanation
# ---------------------------------------------------------------------------


def _sequence_concept_provider(sequence):
    """Return a provider callable that returns concept_id by call order."""
    it = iter(sequence)
    def _p(turn, understanding):
        try:
            return next(it)
        except StopIteration:
            return None
    return _p


class TestCatalogThenDirectQuestion:
    def test_direct_question_after_catalog_is_treated_as_first_explanation(
        self, tmp_path,
    ):
        """§45 — catalog did not advance coverage, so the subsequent
        direct-service question is first-time explanation territory
        (NO_PRIOR_COVERAGE strategy).

        We use a sequence-aware concept provider: turn 1 and 2 return
        None (no concept resolved — nothing could be explained even in
        principle); turn 3 returns automation. If catalog advanced
        coverage, turn 3 would see prior_depth != NONE and route to
        CONCISE_RECAP. We assert NO_PRIOR_COVERAGE.
        """
        from app.conversation.participant.contracts import ReExplanationStrategy
        db_path = str(tmp_path / "slice53_mix.db")
        conn = SqliteConnection(db_path)
        transport = ScriptedIntentTransport((
            '{"intent":"none"}',
            '{"intent":"catalog_request"}',
            '{"intent":"direct_service_question"}',
        ))
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
        planner_spies: list = []
        facade.runtime._processor_factory = _TurnFactory(
            facade, evidence=evidence,
            # CATALOG_REQUEST short-circuits the concept provider,
            # so sequence has one fewer entry than number of turns.
            concept_override=_sequence_concept_provider(
                [None, "service:automation"],
            ),
            planner_spy_container=planner_spies,
        )
        facade.runtime.start_session(_boot(), _campaign())
        facade.runtime.process_turn("sess-53", "hi", 1, "tr-1")
        facade.runtime.process_turn("sess-53", "what services do you offer?", 2, "tr-2")
        facade.runtime.process_turn("sess-53", "what does automation do?", 3, "tr-3")

        third = planner_spies[0].last_input
        # First explanation of automation — NO_PRIOR_COVERAGE because
        # catalog did NOT mark it explained
        assert third.explanation_concept_id == "service:automation"
        assert third.catalog_request is False
        assert third.re_explanation_strategy == ReExplanationStrategy.NO_PRIOR_COVERAGE


# ---------------------------------------------------------------------------
# Section 5 — participant isolation (§19, §33)
# ---------------------------------------------------------------------------


class TestCatalogParticipantIsolation:
    def test_person_b_direct_question_after_person_a_catalog_is_fresh(
        self, tmp_path,
    ):
        """Person A asked catalog → does NOT mean Person B has heard
        anything explained. Person B's later direct question must be
        first-time explanation."""
        from app.conversation.participant.contracts import ReExplanationStrategy
        db_path = str(tmp_path / "slice53_participant.db")
        conn = SqliteConnection(db_path)
        transport = ScriptedIntentTransport((
            '{"intent":"none"}',             # t1 A hi
            '{"intent":"catalog_request"}',  # t2 A catalog
            '{"intent":"none"}',             # t3 handoff
            '{"intent":"direct_service_question"}',  # t4 B asks about automation
        ))
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
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.NONE,
                source_turn_id="t2",
                claimed_role_hint=ParticipantRoleHint.UNKNOWN,
            ),
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_HANDOFF,
                source_turn_id="t3",
                claimed_role_hint=ParticipantRoleHint.OWNER,
            ),
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.NONE,
                source_turn_id="t4",
                claimed_role_hint=ParticipantRoleHint.UNKNOWN,
            ),
        )
        planner_spies: list = []
        facade.runtime._processor_factory = _TurnFactory(
            facade, evidence=evidence,
            # CATALOG_REQUEST short-circuits the provider; sequence is
            # [turn1 none, turn3 handoff-none, turn4 direct].
            concept_override=_sequence_concept_provider(
                [None, None, "service:automation"],
            ),
            planner_spy_container=planner_spies,
        )
        facade.runtime.start_session(_boot(), _campaign())
        facade.runtime.process_turn("sess-53", "hi", 1, "tr-1")
        facade.runtime.process_turn("sess-53", "what services?", 2, "tr-2")
        facade.runtime.process_turn("sess-53", "owner here", 3, "tr-3")
        facade.runtime.process_turn("sess-53", "what does automation do?", 4, "tr-4")

        b_turn = planner_spies[0].last_input
        # Person B gets first-time explanation. The Slice 51C policy
        # branch for new-participant + concept heard-elsewhere =
        # FULL_RE_EXPLAIN_FOR_NEW_PARTICIPANT; here nothing was ever
        # explained (concept id None in A's turns) so the strategy is
        # the first-time path.
        assert b_turn.re_explanation_strategy in (
            ReExplanationStrategy.NO_PRIOR_COVERAGE,
            ReExplanationStrategy.FULL_RE_EXPLAIN_FOR_NEW_PARTICIPANT,
        )


# ---------------------------------------------------------------------------
# Section 6 — transport (HTTP + WS)
# ---------------------------------------------------------------------------


@pytest.fixture
def api_env(tmp_path):
    db_path = str(tmp_path / "slice53_api.db")
    conn = SqliteConnection(db_path)
    transport = ScriptedIntentTransport((
        '{"intent":"none"}',
        '{"intent":"catalog_request"}',
    ))
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
    facade.runtime._processor_factory = _TurnFactory(
        facade, evidence=evidence,
        concept_override=StaticConceptProvider("service:automation").as_callable(),
    )
    campaigns = {"c1": _campaign()}
    app = create_app(runtime=facade.runtime, campaign_lookup=campaigns.get)
    return app, facade


class TestCatalogThroughTransport:
    def _start(self, client):
        r = client.post(
            "/api/v1/sessions",
            headers={"X-Tenant-Id": "t1"},
            json={
                "campaign_id": "c1",
                "lead_id": "lead-1",
                "session_id": "sess-53",
                "bootstrap_request_id": "req-1",
                "trusted_time_iso": "2026-10-29T10:00:00Z",
            },
        )
        assert r.status_code == 201, r.text

    def test_catalog_through_http(self, api_env):
        app, _ = api_env
        client = TestClient(app)
        self._start(client)
        r1 = client.post(
            "/api/v1/sessions/sess-53/turns",
            headers={"X-Tenant-Id": "t1"},
            json={"turn_request_id": "tr-1", "text": "hi"},
        )
        assert r1.status_code == 200
        r2 = client.post(
            "/api/v1/sessions/sess-53/turns",
            headers={"X-Tenant-Id": "t1"},
            json={"turn_request_id": "tr-cat", "text": "what services do you provide?"},
        )
        assert r2.status_code == 200
        body = r2.json()
        assert body["turn_sequence"] == 2
        assert len(body["text"]) > 0

    def test_catalog_duplicate_turn_replays(self, api_env):
        """§30 — idempotent replay of a catalog turn returns committed."""
        app, _ = api_env
        client = TestClient(app)
        self._start(client)
        client.post(
            "/api/v1/sessions/sess-53/turns",
            headers={"X-Tenant-Id": "t1"},
            json={"turn_request_id": "tr-1", "text": "hi"},
        )
        payload = {"turn_request_id": "tr-cat", "text": "what services?"}
        a = client.post(
            "/api/v1/sessions/sess-53/turns",
            headers={"X-Tenant-Id": "t1"},
            json=payload,
        ).json()
        b = client.post(
            "/api/v1/sessions/sess-53/turns",
            headers={"X-Tenant-Id": "t1"},
            json=payload,
        ).json()
        assert b["replayed"] is True
        assert b["text"] == a["text"]
        assert b["revision"] == a["revision"]

    def test_catalog_through_websocket(self, api_env):
        app, _ = api_env
        client = TestClient(app)
        self._start(client)
        # First HTTP turn to establish participant
        client.post(
            "/api/v1/sessions/sess-53/turns",
            headers={"X-Tenant-Id": "t1"},
            json={"turn_request_id": "tr-1", "text": "hi"},
        )
        with client.websocket_connect("/ws/sessions/sess-53?tenant=t1") as ws:
            ws.send_json({
                "type": "user_turn",
                "turn_request_id": "ws-cat",
                "text": "what services do you provide?",
            })
            msg = ws.receive_json()
            assert msg["type"] == "agent_turn"
            assert msg["turn_sequence"] == 2
            assert msg["replayed"] is False


# ---------------------------------------------------------------------------
# Section 7 — scanners
# ---------------------------------------------------------------------------


class TestNoKeywordCatalogClassifier:
    def test_processor_has_no_catalog_phrase_matching(self):
        """§43 — no 'what services' / 'kya services' string in processor."""
        from app.application import production_turn_processor
        src = inspect.getsource(production_turn_processor).lower()
        forbidden = [
            '"what services"', "'what services'",
            '"kya services"', "'kya services'",
            '"services provide"', "'services provide'",
        ]
        for f in forbidden:
            assert f not in src, f"keyword {f} in processor"

    def test_participant_policy_has_no_service_phrase_matching(self):
        from app.conversation.participant import policy
        src = inspect.getsource(policy).lower()
        for f in ('"what services"', "'what services'",
                  '"kya services"', "'kya services'"):
            assert f not in src

    def test_intent_provider_has_no_service_phrase_matching(self):
        from app.conversation.participant import intent_provider
        src = inspect.getsource(intent_provider).lower()
        # The provider does prompt-build an "allowed_intents" list; that is
        # typed enum names, not phrase matching. Phrase strings must not
        # appear.
        for f in ('"what services"', "'what services'",
                  '"kya services"', "'kya services'"):
            assert f not in src

    def test_processor_still_has_no_industry_hardcoding(self):
        from app.application import production_turn_processor
        src = inspect.getsource(production_turn_processor).lower()
        for f in ("restaurant", "dentist", "hvac", "plumbing"):
            assert f not in src
