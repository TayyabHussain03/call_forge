"""Slice 51I — budget rollback consistency + canonical production builder.

Fixes two gaps:
    - Budget rollback (§1–§7): committed-turn Model B. _reasoning_calls
      must roll back when the turn commit fails/stales, matching the
      Slice 51B/51I rollback contract for discussion_coverage. Same
      logical history produces the same budget state regardless of
      process lifetime.
    - Canonical production builder (§8–§12): one entry point
      `build_production_sales_runtime(...)` that returns a facade
      carrying the ready runtime + pre-built intent callable + a
      bound concept-callable factory + renderer wrapper.
"""

from __future__ import annotations

import pytest

from app.application.sales_runtime.composition import (
    ProductionSalesRuntimeFacade,
    build_production_sales_runtime,
    wrap_renderer_strategy_aware,
)
from app.application.sales_runtime.persistent_runtime import PersistentSalesRuntime
from app.conversation.participant.contracts import (
    ParticipantContinuityEvidence,
    ParticipantEvidenceKind,
    ParticipantRoleHint,
)
from app.conversation.participant.intent_provider import (
    RaisingIntentTransport,
    ScriptedIntentTransport,
)
from app.conversation.participant.policy import ReExplanationIntent
from app.conversation.response_rendering.renderer import DeterministicResponseRenderer
from app.conversation.response_rendering.strategy_frame import (
    StrategyAwareResponseRenderer,
)
from app.persistence.errors import StaleRevisionError
from app.persistence.sqlite.connection import SqliteConnection


# ---------------------------------------------------------------------------
# Fixtures shared with the Slice 51B rollback test harness
# ---------------------------------------------------------------------------


from app.application.production_turn_processor import ProductionTurnProcessor
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
from app.conversation.response_planning.planner import ResponsePlanner
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
from app.precall.contracts import PreCallDiscoveryPolicy, ServiceDiscoveryRule
from app.runtime.chat.bootstrap.contracts import SessionBootstrapInput


class _Factory:
    def __init__(self, evidence_sequence=()) -> None:
        self._evidence_sequence = evidence_sequence
        self._providers = {}

    def _make_provider(self):
        proposals = tuple(
            BrainProposal(
                detected_intent=Intent.INTERESTED,
                proposed_action=a,
                topic_category=TopicCategory.QUALIFICATION,
            )
            for a in (
                AgentAction.GREET, AgentAction.ASK_IDENTITY,
                AgentAction.INTRODUCE_REASON, AgentAction.ANSWER_QUESTION,
            )
        )
        return MockReasoningProvider(scripted=proposals, default=proposals[-1])

    def _evidence_provider(self):
        seq = iter(self._evidence_sequence)
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
            session.session_id, self._make_provider(),
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
        return ProductionTurnProcessor(
            orchestrator=orchestrator,
            response_planner=ResponsePlanner(),
            response_renderer=DeterministicResponseRenderer(),
            initial_context=ConversationContext(session.session_id),
            budget_policy=load_budget_policies(
                "app/config/budget_policy.yaml")["standard"],
            scope_policy=load_scope_policies(
                "app/config/scope_policy.yaml")["business_general"],
            authority_policy=load_authority_policies(
                "app/config/authority_policy.yaml")["standard"],
            trusted_priority_provider=lambda _: TrustedPriorityOutcome.NONE,
            participant_evidence_provider=self._evidence_provider(),
            **kwargs,
        )

    def create_for_session(self, session, campaign, policy):
        return self._build(session, initial_state=ConversationState.NEW_CALL)

    def create_from_snapshot(self, session, campaign, policy, snapshot):
        # Slice 51I — restore reasoning-call budget from BUDGET_STATE_KEY
        # payload so restart cannot grant extra reasoning capacity.
        from app.application.sales_runtime.serializers import (
            BUDGET_STATE_KEY, BudgetStateSerializer,
        )
        budget_env = snapshot.payload(BUDGET_STATE_KEY)
        initial_calls = (
            BudgetStateSerializer().deserialize(budget_env) if budget_env else 0
        )
        return self._build(
            session,
            initial_state=snapshot.conversation_state,
            initial_reasoning_calls=initial_calls,
        )


def _phone() -> NormalizedPhone:
    return NormalizedPhone(
        raw="+15551234567", normalized="+15551234567",
        status=PhoneNormalizationStatus.VALID_E164,
    )


def _lead() -> LeadRecord:
    return LeadRecord(
        lead_id="lead-1", tenant_id="t1", campaign_id="c1",
        business_name="ABC", phone=_phone(),
        category=None, city=None, state=None, country=None,
        address=None, email=None, contact_name=None,
        website=None, website_status=WebsiteStatus.UNKNOWN,
    )


def _queue() -> LeadQueueEntry:
    return LeadQueueEntry(
        lead_id="lead-1", tenant_id="t1", campaign_id="c1",
        queue_position=0, status=LeadQueueStatus.READY,
    )


def _campaign() -> CampaignConfig:
    return CampaignConfig(
        campaign_id="c1", tenant_id="t1", name="Test",
        status=CampaignStatus.ACTIVE,
        authorized_service_ids=("svc-automation",),
    )


def _policy() -> PreCallDiscoveryPolicy:
    return PreCallDiscoveryPolicy(
        policy_id="pol-1", version="1.0",
        service_rules=(
            ServiceDiscoveryRule(
                service_id="svc-automation",
                required_fact_keys=("website.presence",),
            ),
        ),
    )


def _boot(sid="sess-51i") -> SessionBootstrapInput:
    return SessionBootstrapInput(
        tenant_id="t1", campaign_id="c1", lead_id="lead-1",
        session_id=sid, bootstrap_request_id=f"req-{sid}",
        trusted_time_iso="2026-10-20T10:00:00Z",
    )


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "slice51i.db")


def _make_runtime(db_path, factory=None):
    conn = SqliteConnection(db_path)
    f = factory or _Factory()
    rt = PersistentSalesRuntime(conn, f)
    if rt.leads.get("t1", "c1", "lead-1") is None:
        rt.leads.save(_lead())
    if rt.queue.get("t1", "c1", "lead-1") is None:
        rt.queue.save(_queue())
    if rt.policies.resolve("c1") is None:
        rt.policies.register_default("c1", _policy())
    return rt


# ---------------------------------------------------------------------------
# Budget rollback consistency (Model B — committed-turn semantics)
# ---------------------------------------------------------------------------


class TestBudgetRollsBackOnCommitFailure:
    """§1-§7: _reasoning_calls must roll back alongside _discussion_coverage
    when the turn commit fails. Same logical history → same budget state
    regardless of process lifetime."""

    def test_reasoning_calls_restored_on_commit_failure(self, db_path) -> None:
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51i_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        rt = _make_runtime(db_path, _Factory(evidence_sequence=evidence))
        rt.start_session(_boot(), _campaign())
        rt.process_turn("sess-51i", "hi", 1, "tr-1")
        proc = rt._processors["sess-51i"]
        pre_turn_calls = proc.reasoning_calls

        # Monkey-patch session.update to raise, simulating mid-txn failure
        original_update = rt.sessions.update

        def failing_update(session):
            # Simulate processor mutating budget during the processor call
            proc._reasoning_calls += 5
            raise StaleRevisionError("simulated stale revision")

        rt.sessions.update = failing_update
        try:
            with pytest.raises(StaleRevisionError):
                rt.process_turn("sess-51i", "another", 2, "tr-fail")
        finally:
            rt.sessions.update = original_update

        # Rollback restored _reasoning_calls to pre-turn state
        assert proc.reasoning_calls == pre_turn_calls


# ---------------------------------------------------------------------------
# Budget persistence survives restart (success path)
# ---------------------------------------------------------------------------


class TestBudgetSurvivesRestart:
    def test_committed_reasoning_calls_restored(self, db_path) -> None:
        """After a successful commit, reasoning_calls is persisted via
        BUDGET_STATE_KEY and restored on restart with the same value."""
        rt = _make_runtime(db_path)
        rt.start_session(_boot(), _campaign())
        rt.process_turn("sess-51i", "hi", 1, "tr-1")
        proc = rt._processors["sess-51i"]
        calls_before = proc.reasoning_calls

        # Force a non-zero counter to prove the persist/restore path
        proc._reasoning_calls = 3
        rt.process_turn("sess-51i", "more", 2, "tr-2")

        from app.application.sales_runtime.serializers import (
            BUDGET_STATE_KEY, BudgetStateSerializer,
        )
        snap = rt.snapshots.load("sess-51i")
        payload_keys = {k for k, _ in snap.persisted_contract_payloads}
        # Budget payload emitted (counter > 0)
        assert BUDGET_STATE_KEY in payload_keys
        env = snap.payload(BUDGET_STATE_KEY)
        assert BudgetStateSerializer().deserialize(env) >= 3


# ---------------------------------------------------------------------------
# Equivalence: no-restart vs restart budget state
# ---------------------------------------------------------------------------


class TestBudgetRestartEquivalence:
    """§4 budget invariant: identical logical execution with/without
    restart yields identical remaining allowance."""

    def test_committed_budget_equivalent_across_restart(self, tmp_path) -> None:
        # Run A: no restart
        db_a = str(tmp_path / "a.db")
        rt_a = _make_runtime(db_a)
        rt_a.start_session(_boot(sid="sess-A"), _campaign())
        rt_a.process_turn("sess-A", "hi", 1, "trA-1")
        proc_a = rt_a._processors["sess-A"]
        proc_a._reasoning_calls = 2
        rt_a.process_turn("sess-A", "more", 2, "trA-2")
        a_calls = proc_a.reasoning_calls

        # Run B: restart after same amount of work
        db_b = str(tmp_path / "b.db")
        rt_b = _make_runtime(db_b)
        rt_b.start_session(_boot(sid="sess-B"), _campaign())
        rt_b.process_turn("sess-B", "hi", 1, "trB-1")
        proc_b = rt_b._processors["sess-B"]
        proc_b._reasoning_calls = 2
        rt_b.process_turn("sess-B", "more", 2, "trB-2")

        # Hard restart
        rt_b2 = _make_runtime(db_b, _Factory())
        rt_b2.restore_processor_if_possible("sess-B", _campaign())
        proc_b2 = rt_b2._processors["sess-B"]
        b_calls = proc_b2.reasoning_calls

        # Equal state regardless of process lifetime
        assert a_calls == b_calls


# ---------------------------------------------------------------------------
# Canonical production builder
# ---------------------------------------------------------------------------


class TestCanonicalBuilder:
    def test_returns_facade_with_runtime(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        facade = build_production_sales_runtime(
            connection=conn,
            processor_factory=_Factory(),
            intent_transport=None,
        )
        assert isinstance(facade, ProductionSalesRuntimeFacade)
        assert facade.runtime is not None

    def test_none_transport_yields_none_intent_callable(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        facade = build_production_sales_runtime(
            connection=conn,
            processor_factory=_Factory(),
            intent_transport=None,
        )
        assert facade.intent_callable is None

    def test_transport_yields_working_intent_callable(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        transport = ScriptedIntentTransport(('{"intent":"recap_request"}',))
        facade = build_production_sales_runtime(
            connection=conn,
            processor_factory=_Factory(),
            intent_transport=transport,
        )
        assert facade.intent_callable is not None

        class _T:
            utterance = "remind me"
        class _U:
            class language_profile:
                primary_language = "en"
                secondary_language = None

        assert facade.intent_callable(_T(), _U()) == ReExplanationIntent.RECAP_REQUEST

    def test_raising_transport_degrades_safely(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        facade = build_production_sales_runtime(
            connection=conn,
            processor_factory=_Factory(),
            intent_transport=RaisingIntentTransport(),
        )

        class _T:
            utterance = "hi"

        assert facade.intent_callable(_T(), None) == ReExplanationIntent.NONE

    def test_concept_callable_for_processor(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        facade = build_production_sales_runtime(
            connection=conn,
            processor_factory=_Factory(),
            intent_transport=None,
        )

        class FakeGuidance:
            selected_service_id = "automation"

        class FakeProcessor:
            @property
            def playbook_guidance(self):
                return FakeGuidance()

        callable_ = facade.concept_callable_for(FakeProcessor())
        assert callable_(None, None) == "service:automation"

    def test_wrap_renderer_bound_on_facade(self) -> None:
        wrapped = ProductionSalesRuntimeFacade.wrap_renderer(
            DeterministicResponseRenderer(),
        )
        assert isinstance(wrapped, StrategyAwareResponseRenderer)

    def test_wrap_renderer_is_idempotent_safe(self) -> None:
        """Double-wrapping is harmless because the inner wrapper only
        reacts to a non-None strategy on the plan."""
        once = ProductionSalesRuntimeFacade.wrap_renderer(
            DeterministicResponseRenderer(),
        )
        twice = ProductionSalesRuntimeFacade.wrap_renderer(once)
        assert isinstance(twice, StrategyAwareResponseRenderer)


# ---------------------------------------------------------------------------
# E2E via canonical builder — basic smoke proving composition is real
# ---------------------------------------------------------------------------


class TestE2EViaCanonicalBuilder:
    def test_canonical_builder_runs_a_turn(self, db_path) -> None:
        """Prove the facade's runtime can accept a bootstrap + turn
        without any manual intent/concept/renderer wiring in the test."""
        conn = SqliteConnection(db_path)
        factory = _Factory()
        facade = build_production_sales_runtime(
            connection=conn,
            processor_factory=factory,
            intent_transport=None,
        )
        # Seed minimum state
        facade.runtime.leads.save(_lead())
        facade.runtime.queue.save(_queue())
        facade.runtime.policies.register_default("c1", _policy())

        boot_result = facade.runtime.start_session(_boot(), _campaign())
        from app.runtime.chat.bootstrap.contracts import BootstrapStatus
        assert boot_result.status == BootstrapStatus.CREATED

        turn = facade.runtime.process_turn("sess-51i", "hello", 1, "tr-1")
        assert turn.committed_revision == 1

    def test_canonical_builder_survives_restart(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        facade = build_production_sales_runtime(
            connection=conn,
            processor_factory=_Factory(),
            intent_transport=None,
        )
        facade.runtime.leads.save(_lead())
        facade.runtime.queue.save(_queue())
        facade.runtime.policies.register_default("c1", _policy())
        facade.runtime.start_session(_boot(), _campaign())
        facade.runtime.process_turn("sess-51i", "hi", 1, "tr-1")

        # Rebuild via canonical builder only
        conn2 = SqliteConnection(db_path)
        facade2 = build_production_sales_runtime(
            connection=conn2,
            processor_factory=_Factory(),
            intent_transport=None,
        )
        from app.application.sales_runtime.recovery import RecoveryStatus
        result = facade2.runtime.recover_session("sess-51i")
        assert result.status == RecoveryStatus.RECOVERED_ACTIVE


# ---------------------------------------------------------------------------
# No forbidden imports in canonical builder
# ---------------------------------------------------------------------------


class TestNoForbiddenImports:
    def test_composition_module_remains_clean(self) -> None:
        import inspect
        from app.application.sales_runtime import composition
        source = inspect.getsource(composition)
        for forbidden in (
            "BrainOrchestrator", "StateMachine", "ActionValidator",
            "HybridRetrievalEngine", "openai", "anthropic",
        ):
            assert forbidden not in source
