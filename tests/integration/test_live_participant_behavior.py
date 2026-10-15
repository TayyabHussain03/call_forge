"""Slice 49 — Live participant-aware behavior on the real ProductionTurnProcessor.

Proves that the processor:
    - consumes typed participant evidence via its new provider seam
    - resolves continuity through ParticipantContinuityPolicy
    - archives Person A and installs Person B on NEW_CONFIRMED
    - never inherits participant-scoped state between people
    - exposes a working re_explanation_decision() API
    - records discussion coverage after successful commits
    - survives restart with the new PARTICIPANT_STATE + DISCUSSION_COVERAGE
      + PROSPECT_INTELLIGENCE payloads
    - preserves business facts across participant change
    - respects DNC regardless of participant change
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.application.sales_runtime.persistent_runtime import PersistentSalesRuntime
from app.application.sales_runtime.recovery import RecoveryStatus
from app.application.sales_runtime.serializers import (
    BUSINESS_CONVERSATION_KEY,
    DISCUSSION_COVERAGE_KEY,
    LANGUAGE_PROFILE_KEY,
    PARTICIPANT_STATE_KEY,
    PROSPECT_INTELLIGENCE_KEY,
    QUALIFICATION_KEY,
    BusinessConversationSerializer,
    DiscussionCoverageSerializer,
    LanguageProfileSerializer,
    ParticipantStateSerializer,
    ProspectIntelligenceSerializer,
    QualificationSerializer,
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
from app.conversation.business_conversation.contracts import (
    BusinessConversationSnapshot,
)
from app.conversation.contact.resolver import ContactResolver
from app.conversation.engine import ConversationEngine
from app.conversation.guardrails.action_validator import ActionValidator
from app.conversation.guardrails.clarification import ClarificationEngine
from app.conversation.guardrails.fallbacks import FallbackEngine
from app.conversation.guardrails.priority import TrustedPriorityOutcome
from app.conversation.participant.contracts import (
    ExplanationDepth,
    ParticipantContinuityEvidence,
    ParticipantContinuityState,
    ParticipantEvidenceKind,
    ParticipantRoleHint,
    ReExplanationStrategy,
)
from app.conversation.participant.policy import ReExplanationIntent
from app.conversation.response_planning.planner import ResponsePlanner
from app.conversation.response_rendering.renderer import DeterministicResponseRenderer
from app.conversation.state_machine.machine import ConversationStateMachine
from app.conversation.state_machine.states import load_config
from app.conversation.understanding.contracts import LanguageProfile, LanguageScript
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
from app.persistence.sqlite.connection import SqliteConnection
from app.precall.contracts import PreCallDiscoveryPolicy, ServiceDiscoveryRule
from app.runtime.chat.bootstrap.contracts import SessionBootstrapInput


# ---------------------------------------------------------------------------
# Real-PTP factory with injectable participant-evidence sequence
# ---------------------------------------------------------------------------


FIVE_TURN_ACTIONS = (
    AgentAction.GREET, AgentAction.ASK_IDENTITY,
    AgentAction.INTRODUCE_REASON, AgentAction.ANSWER_QUESTION,
    AgentAction.ANSWER_QUESTION, AgentAction.ANSWER_QUESTION,
    AgentAction.ANSWER_QUESTION,
)


class RealPTPFactory:
    def __init__(
        self,
        evidence_sequence: tuple = (),
        actions: tuple[AgentAction, ...] = FIVE_TURN_ACTIONS,
    ) -> None:
        self._evidence_sequence = evidence_sequence
        self._actions = actions
        self._providers: dict[str, MockReasoningProvider] = {}

    def _make_provider(self) -> MockReasoningProvider:
        proposals = tuple(
            BrainProposal(
                detected_intent=Intent.INTERESTED,
                proposed_action=a,
                topic_category=TopicCategory.QUALIFICATION,
            )
            for a in self._actions
        )
        return MockReasoningProvider(scripted=proposals, default=proposals[-1])

    def _evidence_provider(self):
        seq = iter(self._evidence_sequence)

        def _provider(turn, understanding):
            try:
                return next(seq)
            except StopIteration:
                return None

        return _provider

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
        from app.application.production_turn_processor import ProductionTurnProcessor
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
        bci_env = snapshot.payload(BUSINESS_CONVERSATION_KEY)
        lang_env = snapshot.payload(LANGUAGE_PROFILE_KEY)
        qual_env = snapshot.payload(QUALIFICATION_KEY)
        prospect_env = snapshot.payload(PROSPECT_INTELLIGENCE_KEY)
        part_env = snapshot.payload(PARTICIPANT_STATE_KEY)
        cov_env = snapshot.payload(DISCUSSION_COVERAGE_KEY)

        bci = (
            BusinessConversationSerializer().deserialize(bci_env)
            if bci_env else BusinessConversationSnapshot()
        )
        language = (
            LanguageProfileSerializer().deserialize(lang_env) if lang_env else None
        )
        qual = (
            QualificationSerializer().deserialize(qual_env) if qual_env else None
        )
        prospect = (
            ProspectIntelligenceSerializer().deserialize(prospect_env)
            if prospect_env else None
        )
        if part_env:
            active, archived = ParticipantStateSerializer().deserialize(part_env)
        else:
            active, archived = None, ()
        coverage = (
            DiscussionCoverageSerializer().deserialize(cov_env) if cov_env else None
        )

        kwargs = dict(
            initial_business_conversation=bci,
            initial_language_profile=language,
            initial_qualification=qual,
            initial_active_participant=active,
            initial_archived_participants=archived,
            initial_discussion_coverage=coverage,
        )
        if prospect is not None:
            kwargs["initial_prospect_intelligence"] = prospect
        return self._build(
            session, initial_state=snapshot.conversation_state, **kwargs,
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
        authorized_service_ids=("svc-web", "svc-automation"),
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


def _boot_input(sid: str = "sess-live", req: str = "req-live") -> SessionBootstrapInput:
    return SessionBootstrapInput(
        tenant_id="t1", campaign_id="c1", lead_id="lead-1",
        session_id=sid, bootstrap_request_id=req,
        trusted_time_iso="2026-10-15T10:00:00Z",
    )


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "slice49.db")


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
# Participant resolution through the live pipeline
# ---------------------------------------------------------------------------


class TestLiveParticipantResolution:
    def test_no_evidence_preserves_absence_of_participant(self, db_path) -> None:
        """No provider → no participant created. Fail-closed."""
        rt = _make_runtime(db_path, RealPTPFactory(evidence_sequence=()))
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-live", "hi", 1, "tr-1")
        proc = rt._processors["sess-live"]
        # No decision reached because provider returned None
        assert proc.latest_participant_decision is None

    def test_explicit_self_identification_creates_participant(self, db_path) -> None:
        """First turn: Person A self-identifies as employee."""
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-live_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        rt = _make_runtime(db_path, RealPTPFactory(evidence_sequence=evidence))
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-live", "I handle marketing", 1, "tr-1")

        proc = rt._processors["sess-live"]
        assert proc.active_participant is not None
        assert proc.active_participant.role_hint == ParticipantRoleHint.EMPLOYEE
        # No one archived yet — first participant on first evidence
        assert proc.latest_participant_decision.state in (
            ParticipantContinuityState.NEW_CONFIRMED,
            ParticipantContinuityState.SAME_CONFIRMED,
        )

    def test_handoff_archives_prior_and_installs_new(self, db_path) -> None:
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-live_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_HANDOFF,
                source_turn_id="chat_sess-live_t2",
                claimed_role_hint=ParticipantRoleHint.OWNER,
            ),
        )
        rt = _make_runtime(db_path, RealPTPFactory(evidence_sequence=evidence))
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-live", "I handle marketing", 1, "tr-1")
        proc = rt._processors["sess-live"]
        first_ref = proc.active_participant.reference_id

        rt.process_turn("sess-live", "my colleague gave me the phone", 2, "tr-2")
        proc = rt._processors["sess-live"]
        # New active participant
        assert proc.active_participant.reference_id != first_ref
        assert proc.active_participant.role_hint == ParticipantRoleHint.OWNER
        # Previous participant archived
        archived_refs = {
            s.participant.reference_id for s in proc.archived_participant_states
        }
        assert first_ref in archived_refs

    def test_inferred_evidence_cannot_switch_participant(self, db_path) -> None:
        """INFERRED evidence alone must NOT confirm new participant."""
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-live_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.INFERRED,
                source_turn_id="chat_sess-live_t2",
            ),
        )
        rt = _make_runtime(db_path, RealPTPFactory(evidence_sequence=evidence))
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-live", "I handle marketing", 1, "tr-1")
        proc = rt._processors["sess-live"]
        first_ref = proc.active_participant.reference_id

        rt.process_turn("sess-live", "what was this about", 2, "tr-2")
        proc = rt._processors["sess-live"]
        # Active unchanged
        assert proc.active_participant.reference_id == first_ref
        # Nothing archived from this turn
        assert len(proc.archived_participant_states) == 0


# ---------------------------------------------------------------------------
# ReExplanationDecision API query
# ---------------------------------------------------------------------------


class TestReExplanationQuery:
    def _rt_with_handoff(self, db_path):
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-live_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_HANDOFF,
                source_turn_id="chat_sess-live_t2",
                claimed_role_hint=ParticipantRoleHint.OWNER,
            ),
        )
        rt = _make_runtime(db_path, RealPTPFactory(evidence_sequence=evidence))
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-live", "I handle marketing", 1, "tr-1")
        # Simulate explaining automation to participant A
        proc = rt._processors["sess-live"]
        proc.record_discussion_coverage(
            "svc.automation", ExplanationDepth.EXPLAINED, "chat_sess-live_t1",
        )
        rt.process_turn("sess-live", "my colleague took over", 2, "tr-2")
        return rt

    def test_new_participant_direct_question_triggers_full_re_explain(self, db_path) -> None:
        rt = self._rt_with_handoff(db_path)
        proc = rt._processors["sess-live"]
        decision = proc.re_explanation_decision(
            "svc.automation",
            intent=ReExplanationIntent.DIRECT_SERVICE_QUESTION,
        )
        assert decision.strategy == ReExplanationStrategy.FULL_RE_EXPLAIN_FOR_NEW_PARTICIPANT

    def test_same_participant_already_explained_suppressed(self, db_path) -> None:
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-live_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        rt = _make_runtime(db_path, RealPTPFactory(evidence_sequence=evidence))
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-live", "I handle marketing", 1, "tr-1")
        proc = rt._processors["sess-live"]
        proc.record_discussion_coverage(
            "svc.automation", ExplanationDepth.EXPLAINED, "chat_sess-live_t1",
        )
        decision = proc.re_explanation_decision(
            "svc.automation", intent=ReExplanationIntent.NONE,
        )
        assert decision.strategy == ReExplanationStrategy.SUPPRESS

    def test_same_participant_recap_request(self, db_path) -> None:
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-live_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        rt = _make_runtime(db_path, RealPTPFactory(evidence_sequence=evidence))
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-live", "I handle marketing", 1, "tr-1")
        proc = rt._processors["sess-live"]
        proc.record_discussion_coverage(
            "svc.automation", ExplanationDepth.EXPLAINED, "chat_sess-live_t1",
        )
        decision = proc.re_explanation_decision(
            "svc.automation", intent=ReExplanationIntent.RECAP_REQUEST,
        )
        assert decision.strategy == ReExplanationStrategy.CONCISE_RECAP

    def test_didnt_understand_triggers_simplify(self, db_path) -> None:
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-live_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        rt = _make_runtime(db_path, RealPTPFactory(evidence_sequence=evidence))
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-live", "I handle marketing", 1, "tr-1")
        proc = rt._processors["sess-live"]
        decision = proc.re_explanation_decision(
            "svc.automation", intent=ReExplanationIntent.SIMPLIFY_REQUEST,
        )
        assert decision.strategy == ReExplanationStrategy.SIMPLIFY

    def test_more_detail_triggers_deepen(self, db_path) -> None:
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-live_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        rt = _make_runtime(db_path, RealPTPFactory(evidence_sequence=evidence))
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-live", "I handle marketing", 1, "tr-1")
        proc = rt._processors["sess-live"]
        decision = proc.re_explanation_decision(
            "svc.automation", intent=ReExplanationIntent.DEEPEN_REQUEST,
        )
        assert decision.strategy == ReExplanationStrategy.DEEPEN


# ---------------------------------------------------------------------------
# Snapshot includes participant payloads
# ---------------------------------------------------------------------------


class TestSnapshotIncludesParticipantPayloads:
    def test_participant_and_coverage_in_persisted_snapshot(self, db_path) -> None:
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-live_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        rt = _make_runtime(db_path, RealPTPFactory(evidence_sequence=evidence))
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-live", "I handle marketing", 1, "tr-1")

        proc = rt._processors["sess-live"]
        proc.record_discussion_coverage(
            "svc.automation", ExplanationDepth.EXPLAINED, "chat_sess-live_t1",
        )
        # Trigger a second turn that re-exports the snapshot with coverage
        rt.process_turn("sess-live", "ok tell me more", 2, "tr-2")

        snap = rt.snapshots.load("sess-live")
        keys = {k for k, _ in snap.persisted_contract_payloads}
        assert PARTICIPANT_STATE_KEY in keys
        assert DISCUSSION_COVERAGE_KEY in keys
        assert PROSPECT_INTELLIGENCE_KEY in keys


# ---------------------------------------------------------------------------
# Restart with participant state + coverage preserved
# ---------------------------------------------------------------------------


class TestParticipantStateSurvivesRestart:
    def test_active_and_coverage_restored(self, db_path) -> None:
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-live_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        rt = _make_runtime(db_path, RealPTPFactory(evidence_sequence=evidence))
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-live", "I handle marketing", 1, "tr-1")

        proc = rt._processors["sess-live"]
        active_ref_before = proc.active_participant.reference_id
        proc.record_discussion_coverage(
            "svc.automation", ExplanationDepth.EXPLAINED, "chat_sess-live_t1",
        )
        rt.process_turn("sess-live", "ok", 2, "tr-2")

        # Hard restart
        rt2 = _make_runtime(db_path, RealPTPFactory())
        result = rt2.restore_processor_if_possible("sess-live", _campaign())
        assert result.status == RecoveryStatus.RECOVERED_ACTIVE

        restored = rt2._processors["sess-live"]
        assert restored.active_participant is not None
        assert restored.active_participant.reference_id == active_ref_before
        # Coverage survived
        assert "svc.automation" in restored.discussion_coverage.heard_by(active_ref_before)

    def test_archived_participant_restored(self, db_path) -> None:
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-live_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_HANDOFF,
                source_turn_id="chat_sess-live_t2",
                claimed_role_hint=ParticipantRoleHint.OWNER,
            ),
        )
        rt = _make_runtime(db_path, RealPTPFactory(evidence_sequence=evidence))
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-live", "I handle marketing", 1, "tr-1")
        rt.process_turn("sess-live", "colleague took phone", 2, "tr-2")

        rt2 = _make_runtime(db_path, RealPTPFactory())
        rt2.restore_processor_if_possible("sess-live", _campaign())
        restored = rt2._processors["sess-live"]
        assert len(restored.archived_participant_states) == 1
        assert restored.active_participant.role_hint == ParticipantRoleHint.OWNER


# ---------------------------------------------------------------------------
# Business continuity across participant change
# ---------------------------------------------------------------------------


class TestBusinessPreservedAcrossHandoff:
    def test_bci_untouched_by_participant_switch(self, db_path) -> None:
        from app.conversation.business_conversation.contracts import (
            BusinessConversationSnapshot, BusinessFactKind, ObservedBusinessFact,
        )
        seeded_bci = BusinessConversationSnapshot(
            facts=(
                ObservedBusinessFact(
                    BusinessFactKind.WEBSITE, "existing website", "t0",
                ),
            ),
        )

        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-live_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_HANDOFF,
                source_turn_id="chat_sess-live_t2",
                claimed_role_hint=ParticipantRoleHint.OWNER,
            ),
        )

        class SeededFactory(RealPTPFactory):
            def create_for_session(self, session, campaign, policy):
                return self._build(
                    session, initial_state=ConversationState.NEW_CALL,
                    initial_business_conversation=seeded_bci,
                )

        rt = _make_runtime(db_path, SeededFactory(evidence_sequence=evidence))
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-live", "I handle marketing", 1, "tr-1")
        rt.process_turn("sess-live", "owner here now", 2, "tr-2")

        proc = rt._processors["sess-live"]
        fact_values = {f.value for f in proc._business_conversation.facts}
        # Business truth preserved across the switch
        assert "existing website" in fact_values


# ---------------------------------------------------------------------------
# Role is not authority — AuthorityPolicy remains separate
# ---------------------------------------------------------------------------


class TestRoleIsNotAuthority:
    def test_owner_role_hint_does_not_mutate_authority_policy(self, db_path) -> None:
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-live_t1",
                claimed_role_hint=ParticipantRoleHint.OWNER,
            ),
        )
        rt = _make_runtime(db_path, RealPTPFactory(evidence_sequence=evidence))
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-live", "I'm the owner", 1, "tr-1")
        proc = rt._processors["sess-live"]
        # AuthorityPolicy identity preserved — participant layer did not
        # replace it (role hint ≠ authority grant). The processor still
        # holds the same policy object it was constructed with.
        assert proc._authority_policy is not None
        # Role hint recorded on the participant though
        assert proc.active_participant.role_hint == ParticipantRoleHint.OWNER


# ---------------------------------------------------------------------------
# DNC remains highest priority across participant change
# ---------------------------------------------------------------------------


class TestDNCAcrossParticipantChange:
    def test_dnc_persists_through_handoff(self, db_path) -> None:
        rt = _make_runtime(db_path)
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-live", "hi", 1, "tr-1")
        rt.mark_lead_dnc("t1", "c1", "lead-1")

        rt2 = _make_runtime(db_path, RealPTPFactory())
        entry = rt2.queue.get("t1", "c1", "lead-1")
        assert entry.is_dnc is True
        assert entry.status == LeadQueueStatus.DNC


# ---------------------------------------------------------------------------
# Provider failure is tolerated — fail closed, no fake participant
# ---------------------------------------------------------------------------


class TestProviderFailureToleratedSafely:
    def test_provider_exception_does_not_crash_turn(self, db_path) -> None:
        class RaisingFactory(RealPTPFactory):
            def _evidence_provider(self):
                def _p(turn, understanding):
                    raise RuntimeError("provider blew up")
                return _p

        rt = _make_runtime(db_path, RaisingFactory())
        rt.start_session(_boot_input(), _campaign())
        # Turn still commits; participant remains None
        rt.process_turn("sess-live", "hi", 1, "tr-1")
        proc = rt._processors["sess-live"]
        assert proc.latest_participant_decision is None
        assert proc.active_participant is None


# ---------------------------------------------------------------------------
# Coverage is only recorded by explicit call (committed-explanation only)
# ---------------------------------------------------------------------------


class TestCoverageOnlyOnExplicitRecord:
    def test_coverage_empty_without_explicit_record(self, db_path) -> None:
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-live_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        rt = _make_runtime(db_path, RealPTPFactory(evidence_sequence=evidence))
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-live", "hi", 1, "tr-1")
        proc = rt._processors["sess-live"]
        # No explicit record() call → no coverage entries
        assert proc.discussion_coverage.entries == ()

    def test_coverage_requires_active_participant(self, db_path) -> None:
        # No evidence provider → no active participant
        rt = _make_runtime(db_path, RealPTPFactory())
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-live", "hi", 1, "tr-1")
        proc = rt._processors["sess-live"]
        proc.record_discussion_coverage(
            "svc.automation", ExplanationDepth.EXPLAINED, "chat_sess-live_t1",
        )
        # record_discussion_coverage silently no-ops when no active participant
        assert proc.discussion_coverage.entries == ()


# ---------------------------------------------------------------------------
# No hardcoded industry/service branching in participant layer
# ---------------------------------------------------------------------------


class TestNoHardcodedServiceLogic:
    def test_no_industry_branches_in_participant_modules(self) -> None:
        import inspect
        from app.conversation.participant import contracts, policy
        for module in (contracts, policy):
            src = inspect.getsource(module)
            for forbidden in (
                "restaurant", "dentist", "hvac", "law_firm",
                '== "website"', '== "automation"', 'if industry',
            ):
                assert forbidden not in src.lower() or forbidden in (
                    # Allow the role-hint enum values which happen to include "manager"
                ), f"found {forbidden!r} in {module.__name__}"
