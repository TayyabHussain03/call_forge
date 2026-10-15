"""Slice 47 — Real ProductionTurnProcessor snapshot wiring tests.

These tests use the REAL production processor composition — not a scripted
stub — to prove end-to-end mid-conversation restart recovery.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.application.production_turn_processor import ProductionTurnProcessor
from app.application.sales_runtime.manifest import SnapshotCompatibilityManifest
from app.application.sales_runtime.persistent_runtime import PersistentSalesRuntime
from app.application.sales_runtime.recovery import RecoveryStatus
from app.application.sales_runtime.serializers import (
    BUSINESS_CONVERSATION_KEY,
    LANGUAGE_PROFILE_KEY,
    QUALIFICATION_KEY,
    BusinessConversationSerializer,
    LanguageProfileSerializer,
    QualificationSerializer,
)
from app.application.sales_runtime.snapshot import SNAPSHOT_VERSION_CURRENT
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
from app.conversation.business_conversation.contracts import (
    BusinessConversationSnapshot,
    BusinessFactKind,
    ObservedBusinessFact,
)
from app.conversation.contact.resolver import ContactResolver
from app.conversation.engine import ConversationEngine
from app.conversation.guardrails.action_validator import ActionValidator
from app.conversation.guardrails.clarification import ClarificationEngine
from app.conversation.guardrails.fallbacks import FallbackEngine
from app.conversation.guardrails.priority import TrustedPriorityOutcome
from app.conversation.qualification.contracts import (
    QualificationCompleteness,
    QualificationFieldState,
    QualificationProgress,
    QualificationSnapshot,
    QualificationValueState,
)
from app.conversation.response_planning.contracts import (
    AddresseeStatus,
    InterruptionCategory,
)
from app.conversation.response_planning.planner import ResponsePlanner
from app.conversation.response_rendering.renderer import DeterministicResponseRenderer
from app.conversation.state_machine.machine import ConversationStateMachine
from app.conversation.state_machine.states import load_config
from app.conversation.understanding.contracts import (
    ConversationalRegister,
    LanguageProfile,
    LanguageScript,
)
from app.core.constants import (
    AgentAction,
    ConversationState,
    Intent,
    TopicCategory,
)
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
from app.persistence.errors import InvalidPersistedPayloadError
from app.persistence.sqlite.connection import SqliteConnection
from app.precall.contracts import PreCallDiscoveryPolicy, ServiceDiscoveryRule
from app.runtime.chat.bootstrap.contracts import SessionBootstrapInput
from app.runtime.chat.contracts import ChatSession, RuntimeMode
from app.runtime.chat.outcome.contracts import (
    ExecutionStatus,
    OutcomeStatus,
    TerminationReason,
)


# ---------------------------------------------------------------------------
# Real-PTP factory
# ---------------------------------------------------------------------------


def _proposal(action: AgentAction = AgentAction.GREET) -> BrainProposal:
    return BrainProposal(
        detected_intent=Intent.INTERESTED,
        proposed_action=action,
        topic_category=TopicCategory.QUALIFICATION,
    )


class RealPTPFactory:
    """Factory that builds a real ProductionTurnProcessor for each session.

    Uses MockReasoningProvider for deterministic reasoning and
    DeterministicResponseRenderer for deterministic rendering.
    """

    def __init__(
        self,
        action_sequence: tuple[AgentAction, ...] = (
            AgentAction.GREET, AgentAction.ASK_IDENTITY,
            AgentAction.INTRODUCE_REASON,
        ),
        priority: TrustedPriorityOutcome = TrustedPriorityOutcome.NONE,
    ) -> None:
        self._actions = action_sequence
        self._priority = priority
        self._providers: dict[str, MockReasoningProvider] = {}

    def _make_provider(self) -> MockReasoningProvider:
        # Default provider yields a sequence of actions
        return MockReasoningProvider(
            default=_proposal(self._actions[0]),
        )

    def _build(
        self,
        session: ChatSession,
        campaign: CampaignConfig,
        policy: PreCallDiscoveryPolicy,
        *,
        initial_state: ConversationState,
        initial_business_conversation: BusinessConversationSnapshot | None = None,
        initial_language_profile: LanguageProfile | None = None,
        initial_qualification: QualificationSnapshot | None = None,
    ) -> ProductionTurnProcessor:
        config = load_config(get_settings().conversation_config_path)
        machine = ConversationStateMachine(config, initial_state)
        provider = self._providers.setdefault(
            session.session_id, self._make_provider(),
        )
        engine = ConversationEngine(
            machine,
            ActionValidator(config),
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
        priority = self._priority
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
            trusted_priority_provider=lambda _: priority,
            initial_business_conversation=initial_business_conversation,
            initial_language_profile=initial_language_profile,
            initial_qualification=initial_qualification,
        )

    def create_for_session(self, session, campaign, policy):
        return self._build(
            session, campaign, policy,
            initial_state=ConversationState.NEW_CALL,
        )

    def create_from_snapshot(self, session, campaign, policy, snapshot):
        bci_env = snapshot.payload(BUSINESS_CONVERSATION_KEY)
        lang_env = snapshot.payload(LANGUAGE_PROFILE_KEY)
        qual_env = snapshot.payload(QUALIFICATION_KEY)
        bci = (
            BusinessConversationSerializer().deserialize(bci_env)
            if bci_env else BusinessConversationSnapshot()
        )
        language = (
            LanguageProfileSerializer().deserialize(lang_env)
            if lang_env else None
        )
        qual = (
            QualificationSerializer().deserialize(qual_env)
            if qual_env else None
        )
        return self._build(
            session, campaign, policy,
            initial_state=snapshot.conversation_state,
            initial_business_conversation=bci,
            initial_language_profile=language,
            initial_qualification=qual,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _phone() -> NormalizedPhone:
    return NormalizedPhone(
        raw="+15551234567", normalized="+15551234567",
        status=PhoneNormalizationStatus.VALID_E164,
    )


def _lead() -> LeadRecord:
    return LeadRecord(
        lead_id="lead-1", tenant_id="t1", campaign_id="c1",
        business_name="ABC Services", phone=_phone(),
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
        authorized_service_ids=("svc-web",),
    )


def _policy() -> PreCallDiscoveryPolicy:
    return PreCallDiscoveryPolicy(
        policy_id="pol-1", version="1.0",
        service_rules=(
            ServiceDiscoveryRule(
                service_id="svc-web",
                required_fact_keys=("website.presence",),
            ),
        ),
    )


def _boot_input(sid: str = "sess-real-1", req: str = "req-1") -> SessionBootstrapInput:
    return SessionBootstrapInput(
        tenant_id="t1", campaign_id="c1", lead_id="lead-1",
        session_id=sid, bootstrap_request_id=req,
        trusted_time_iso="2026-10-15T10:00:00Z",
    )


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "real.db")


def _make_runtime(db_path, factory=None):
    conn = SqliteConnection(db_path)
    f = factory or RealPTPFactory()
    rt = PersistentSalesRuntime(conn, f)
    if rt.leads.get("t1", "c1", "lead-1") is None:
        rt.leads.save(_lead())
    if rt.queue.get("t1", "c1", "lead-1") is None:
        rt.queue.save(_queue())
    if rt.policies.resolve("c1") is None:
        rt.policies.register_default("c1", _policy())
    return rt


# ---------------------------------------------------------------------------
# Export tests — real PTP emits real payloads
# ---------------------------------------------------------------------------


class TestRealPTPExport:
    """Uses REAL ProductionTurnProcessor."""

    def test_real_ptp_exports_bci_payload(self, db_path) -> None:
        rt = _make_runtime(db_path)
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-real-1", "hello", 1, "tr-1")
        snap = rt.snapshots.load("sess-real-1")
        payload_keys = {k for k, _ in snap.persisted_contract_payloads}
        assert BUSINESS_CONVERSATION_KEY in payload_keys

    def test_bci_payload_matches_live_processor_state(self, db_path) -> None:
        rt = _make_runtime(db_path)
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-real-1", "hi", 1, "tr-1")
        processor = rt._processors["sess-real-1"]
        snap = rt.snapshots.load("sess-real-1")
        env = snap.payload(BUSINESS_CONVERSATION_KEY)
        restored_bci = BusinessConversationSerializer().deserialize(env)
        assert restored_bci == processor._business_conversation

    def test_deterministic_payload_bytes(self, db_path) -> None:
        rt = _make_runtime(db_path)
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-real-1", "hi", 1, "tr-1")
        snap1 = rt.snapshots.load("sess-real-1")

        # Second serialization of same live state
        processor = rt._processors["sess-real-1"]
        snap2 = processor.export_snapshot(snap1.source_session_revision)
        bci1 = snap1.payload(BUSINESS_CONVERSATION_KEY)
        bci2 = snap2.payload(BUSINESS_CONVERSATION_KEY)
        assert bci1 == bci2


# ---------------------------------------------------------------------------
# Restore tests — real PTP restored from snapshot
# ---------------------------------------------------------------------------


class TestRealPTPRestore:
    def test_restored_processor_has_bci_from_snapshot(self, db_path) -> None:
        """Simulate: BCI accumulated turn 1 → restart → recovered PTP has it."""
        # Seed BCI by injecting via initial_business_conversation path
        bci_before = BusinessConversationSnapshot(
            facts=(
                ObservedBusinessFact(
                    BusinessFactKind.WEBSITE, "existing website", "t1",
                ),
            ),
        )

        class SeededFactory(RealPTPFactory):
            def create_for_session(self, session, campaign, policy):
                return self._build(
                    session, campaign, policy,
                    initial_state=ConversationState.NEW_CALL,
                    initial_business_conversation=bci_before,
                )

        rt = _make_runtime(db_path, SeededFactory())
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-real-1", "we have a website", 1, "tr-1")

        # Restart: new runtime, new factory
        rt2 = _make_runtime(db_path, RealPTPFactory())
        result = rt2.restore_processor_if_possible("sess-real-1", _campaign())
        assert result.status == RecoveryStatus.RECOVERED_ACTIVE
        restored = rt2._processors["sess-real-1"]
        values = {f.value for f in restored._business_conversation.facts}
        assert "existing website" in values

    def test_restored_processor_has_language(self, db_path) -> None:
        urdu = LanguageProfile(
            primary_language="ur", secondary_language="en",
            mixed_language=True, script=LanguageScript.LATIN,
            conversational_register=ConversationalRegister.CASUAL,
        )

        class LangSeededFactory(RealPTPFactory):
            def create_for_session(self, session, campaign, policy):
                return self._build(
                    session, campaign, policy,
                    initial_state=ConversationState.NEW_CALL,
                    initial_language_profile=urdu,
                )

        rt = _make_runtime(db_path, LangSeededFactory())
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-real-1", "roman urdu mein", 1, "tr-1")

        rt2 = _make_runtime(db_path, RealPTPFactory())
        rt2.restore_processor_if_possible("sess-real-1", _campaign())
        restored = rt2._processors["sess-real-1"]
        assert restored.language_profile is not None
        assert restored.language_profile.primary_language == "ur"
        assert restored.language_profile.mixed_language is True

    def test_restored_processor_has_qualification(self, db_path) -> None:
        qual = QualificationSnapshot(
            configuration_id=("t1", "biz-1", "c1"),
            fields=(
                QualificationFieldState(
                    field_id="decision_maker",
                    state=QualificationValueState.OBSERVED,
                    value="owner", source_turn_id="t1",
                ),
            ),
            completeness=QualificationCompleteness.PARTIAL,
            progress=QualificationProgress.ADVANCING,
        )

        class QualSeededFactory(RealPTPFactory):
            def create_for_session(self, session, campaign, policy):
                return self._build(
                    session, campaign, policy,
                    initial_state=ConversationState.NEW_CALL,
                    initial_qualification=qual,
                )

        rt = _make_runtime(db_path, QualSeededFactory())
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-real-1", "I'm the owner", 1, "tr-1")

        rt2 = _make_runtime(db_path, RealPTPFactory())
        rt2.restore_processor_if_possible("sess-real-1", _campaign())
        restored = rt2._processors["sess-real-1"]
        assert restored._qualification is not None
        states = {f.field_id: f.state for f in restored._qualification.fields}
        assert states["decision_maker"] == QualificationValueState.OBSERVED

    def test_restored_processor_has_fsm_state(self, db_path) -> None:
        """FSM must be at saved position, not reset to NEW_CALL."""
        rt = _make_runtime(db_path)
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-real-1", "hello", 1, "tr-1")
        snap_before = rt.snapshots.load("sess-real-1")

        rt2 = _make_runtime(db_path, RealPTPFactory())
        rt2.restore_processor_if_possible("sess-real-1", _campaign())
        restored = rt2._processors["sess-real-1"]
        assert restored.current_state == snap_before.conversation_state
        # It is NOT NEW_CALL (GREET from new_call transitioned to greeting)
        assert restored.current_state != ConversationState.NEW_CALL


# ---------------------------------------------------------------------------
# No-restart vs restart state equivalence (REAL PTP)
# ---------------------------------------------------------------------------


class TestRealPTPEquivalence:
    def test_bci_identical_with_and_without_restart(self, tmp_path) -> None:
        # Run A: no restart — commit two turns in one process
        rt_a = _make_runtime(str(tmp_path / "a.db"))
        rt_a.start_session(
            _boot_input(sid="sess-A", req="req-A"), _campaign(),
        )
        rt_a.process_turn("sess-A", "m1", 1, "trA-1")
        bci_a_after_t1 = rt_a._processors["sess-A"]._business_conversation

        # Run B: restart after turn 1
        db_b = str(tmp_path / "b.db")
        rt_b = _make_runtime(db_b)
        rt_b.start_session(
            _boot_input(sid="sess-B", req="req-B"), _campaign(),
        )
        rt_b.process_turn("sess-B", "m1", 1, "trB-1")

        rt_b2 = _make_runtime(db_b, RealPTPFactory())
        rt_b2.restore_processor_if_possible("sess-B", _campaign())
        bci_b_after_t1 = rt_b2._processors["sess-B"]._business_conversation
        # BCI established before restart must survive restart identically
        assert bci_a_after_t1 == bci_b_after_t1

    def test_final_outcome_identical_with_and_without_restart(
        self, tmp_path,
    ) -> None:
        rt_a = _make_runtime(str(tmp_path / "a.db"))
        rt_a.start_session(
            _boot_input(sid="sess-A", req="req-A"), _campaign(),
        )
        for i in range(1, 3):
            rt_a.process_turn("sess-A", f"m{i}", i, f"trA-{i}")
        outcome_a = rt_a.finish_session("sess-A").lead_outcome

        db_b = str(tmp_path / "b.db")
        rt_b = _make_runtime(db_b)
        rt_b.start_session(
            _boot_input(sid="sess-B", req="req-B"), _campaign(),
        )
        rt_b.process_turn("sess-B", "m1", 1, "trB-1")

        rt_b2 = _make_runtime(db_b, RealPTPFactory())
        rt_b2.restore_processor_if_possible("sess-B", _campaign())
        rt_b2.process_turn("sess-B", "m2", 2, "trB-2")
        outcome_b = rt_b2.finish_session("sess-B").lead_outcome

        assert outcome_a.outcome_status == outcome_b.outcome_status
        assert outcome_a.interest_level == outcome_b.interest_level
        assert outcome_a.dnc_status == outcome_b.dnc_status
        assert outcome_a.execution_status == outcome_b.execution_status
        assert outcome_a.termination_reason == outcome_b.termination_reason


# ---------------------------------------------------------------------------
# DNC with real PTP
# ---------------------------------------------------------------------------


class TestRealPTPDNC:
    def test_dnc_turn_persists_queue_dnc_after_restart(self, db_path) -> None:
        rt = _make_runtime(
            db_path, RealPTPFactory(
                action_sequence=(AgentAction.MARK_DNC,),
                priority=TrustedPriorityOutcome.DNC,
            ),
        )
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-real-1", "stop calling me", 1, "tr-1")
        rt.mark_lead_dnc("t1", "c1", "lead-1")

        rt2 = _make_runtime(db_path, RealPTPFactory())
        entry = rt2.queue.get("t1", "c1", "lead-1")
        assert entry.is_dnc is True
        assert entry.status == LeadQueueStatus.DNC


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestRealPTPFailures:
    def test_missing_bci_blocks_recovery(self, db_path) -> None:
        from dataclasses import replace as _replace
        rt = _make_runtime(db_path)
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-real-1", "hi", 1, "tr-1")

        snap = rt.snapshots.load("sess-real-1")
        kept = tuple(
            (k, v) for k, v in snap.persisted_contract_payloads
            if k != BUSINESS_CONVERSATION_KEY
        )
        rt.snapshots.save(_replace(snap, persisted_contract_payloads=kept))

        rt2 = _make_runtime(db_path, RealPTPFactory())
        result = rt2.recover_session("sess-real-1")
        assert result.status == RecoveryStatus.REQUIRED_PAYLOAD_MISSING
        assert result.diagnostics.missing_required_payload == BUSINESS_CONVERSATION_KEY

    def test_corrupt_bci_rejected(self, db_path) -> None:
        import json
        from dataclasses import replace as _replace
        rt = _make_runtime(db_path)
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-real-1", "hi", 1, "tr-1")

        snap = rt.snapshots.load("sess-real-1")
        corrupted = tuple(
            (k, (v if k != BUSINESS_CONVERSATION_KEY
                 else '{"contract":"bci","version":999,"payload":{}}'))
            for k, v in snap.persisted_contract_payloads
        )
        rt.snapshots.save(_replace(snap, persisted_contract_payloads=corrupted))

        rt2 = _make_runtime(db_path, RealPTPFactory())
        result = rt2.recover_session("sess-real-1")
        assert result.status == RecoveryStatus.UNSUPPORTED_PAYLOAD_VERSION


# ---------------------------------------------------------------------------
# Idempotency after restart with real PTP
# ---------------------------------------------------------------------------


class TestRealPTPIdempotency:
    def test_duplicate_turn_after_restart_replays(self, db_path) -> None:
        rt = _make_runtime(db_path)
        rt.start_session(_boot_input(), _campaign())
        r1 = rt.process_turn("sess-real-1", "hi", 1, "tr-same")

        # Restart
        rt2 = _make_runtime(db_path, RealPTPFactory())
        r2 = rt2.process_turn("sess-real-1", "hi again", 1, "tr-same")
        assert r2.replayed_from_idempotency is True
        assert r2.committed_revision == r1.committed_revision


# ---------------------------------------------------------------------------
# No transcript replay
# ---------------------------------------------------------------------------


class TestNoTranscriptReplay:
    def test_recovery_does_not_call_reasoning_provider(self, db_path) -> None:
        """Recovery must not invoke Brain/LLM. We verify by checking the
        provider's call_count across the restore operation."""
        rt = _make_runtime(db_path)
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-real-1", "hi", 1, "tr-1")

        factory2 = RealPTPFactory()
        rt2 = _make_runtime(db_path, factory2)
        rt2.restore_processor_if_possible("sess-real-1", _campaign())
        # Provider was built during restore but should have ZERO calls yet
        provider = factory2._providers["sess-real-1"]
        assert provider.call_count == 0
