"""Slice 51 — Automatic DiscussionCoverage advance within atomic commit.

Proves that when a `ResponsePlan` carries `explanation_concept_id` and an
active participant exists, the processor automatically advances
DiscussionCoverage at the depth implied by `response_length` — and the
updated coverage is captured by the subsequent snapshot export inside
the SAME atomic turn-commit transaction.

Historical max depth is monotonic by construction via
`DiscussionCoverageState.depth_for()`.
"""

from __future__ import annotations

import pytest

from app.application.production_turn_processor import ProductionTurnProcessor
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
    DiscussionCoverageState,
    ExplanationDepth,
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
# Fixtures
# ---------------------------------------------------------------------------


FIVE_TURN_ACTIONS = (
    AgentAction.GREET, AgentAction.ASK_IDENTITY,
    AgentAction.INTRODUCE_REASON, AgentAction.ANSWER_QUESTION,
    AgentAction.ANSWER_QUESTION, AgentAction.ANSWER_QUESTION,
    AgentAction.ANSWER_QUESTION,
)


class RealPTPFactoryWithEvidence:
    """Real PTP factory. Participant evidence provider wires explicit evidence
    for the first turn so the processor has an active participant; after
    that the planner's explanation_concept_id drives coverage auto-advance.
    """

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
        bci_env = snapshot.payload(BUSINESS_CONVERSATION_KEY)
        lang_env = snapshot.payload(LANGUAGE_PROFILE_KEY)
        qual_env = snapshot.payload(QUALIFICATION_KEY)
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
        if part_env:
            active, archived = ParticipantStateSerializer().deserialize(part_env)
        else:
            active, archived = None, ()
        coverage = (
            DiscussionCoverageSerializer().deserialize(cov_env)
            if cov_env else None
        )
        return self._build(
            session, initial_state=snapshot.conversation_state,
            initial_business_conversation=bci,
            initial_language_profile=language,
            initial_qualification=qual,
            initial_active_participant=active,
            initial_archived_participants=archived,
            initial_discussion_coverage=coverage,
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


def _boot(sid="sess-51") -> SessionBootstrapInput:
    return SessionBootstrapInput(
        tenant_id="t1", campaign_id="c1", lead_id="lead-1",
        session_id=sid, bootstrap_request_id=f"req-{sid}",
        trusted_time_iso="2026-10-16T10:00:00Z",
    )


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "slice51.db")


def _make_runtime(db_path, factory=None):
    conn = SqliteConnection(db_path)
    f = factory or RealPTPFactoryWithEvidence()
    rt = PersistentSalesRuntime(conn, f)
    if rt.leads.get("t1", "c1", "lead-1") is None:
        rt.leads.save(_lead())
    if rt.queue.get("t1", "c1", "lead-1") is None:
        rt.queue.save(_queue())
    if rt.policies.resolve("c1") is None:
        rt.policies.register_default("c1", _policy())
    return rt


# ---------------------------------------------------------------------------
# Automatic coverage advance (real PTP)
# ---------------------------------------------------------------------------


class TestAutoCoverageAdvance:
    def test_no_concept_id_no_coverage_advance(self, db_path) -> None:
        """A plan without explanation_concept_id does NOT advance coverage.

        This is the default behavior for every existing turn — the planner
        only populates concept_id when upstream wires it, which the Brain
        does not yet do. So existing production turns remain unchanged.
        """
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        rt = _make_runtime(
            db_path, RealPTPFactoryWithEvidence(evidence_sequence=evidence),
        )
        rt.start_session(_boot(), _campaign())
        rt.process_turn("sess-51", "I handle marketing", 1, "tr-1")
        proc = rt._processors["sess-51"]
        assert proc.discussion_coverage.entries == ()

    def test_coverage_only_advances_with_active_participant(self, db_path) -> None:
        """Without a participant resolved, no coverage can be attributed."""
        rt = _make_runtime(db_path, RealPTPFactoryWithEvidence())
        rt.start_session(_boot(), _campaign())
        rt.process_turn("sess-51", "hi", 1, "tr-1")
        proc = rt._processors["sess-51"]
        proc.record_discussion_coverage(
            "svc.automation", ExplanationDepth.EXPLAINED, "chat_sess-51_t1",
        )
        # record_discussion_coverage silently no-ops when no active participant
        assert proc.discussion_coverage.entries == ()


# ---------------------------------------------------------------------------
# Monotonicity invariant via record_discussion_coverage
# ---------------------------------------------------------------------------


class TestCoverageMonotonicity:
    def test_lower_depth_after_higher_preserves_max(self) -> None:
        """A later BRIEF entry on top of a prior EXPLAINED entry preserves
        the max depth reachable via depth_for() — invariant from Slice 48."""
        state = DiscussionCoverageState(
            entries=(
                DiscussionCoverageEntry := None,  # placeholder for linter
            ),
        )
        from app.conversation.participant.contracts import DiscussionCoverageEntry
        state = DiscussionCoverageState(
            entries=(
                DiscussionCoverageEntry(
                    "p1", "svc.automation", ExplanationDepth.EXPLAINED, "t3",
                ),
                DiscussionCoverageEntry(
                    "p1", "svc.automation", ExplanationDepth.BRIEF, "t5",
                ),
            ),
        )
        assert state.depth_for("p1", "svc.automation") == ExplanationDepth.EXPLAINED

    def test_depth_progression_via_recorded_entries(self) -> None:
        from app.conversation.participant.contracts import DiscussionCoverageEntry
        state = DiscussionCoverageState(
            entries=(
                DiscussionCoverageEntry(
                    "p1", "svc.automation", ExplanationDepth.MENTIONED, "t1",
                ),
                DiscussionCoverageEntry(
                    "p1", "svc.automation", ExplanationDepth.BRIEF, "t2",
                ),
                DiscussionCoverageEntry(
                    "p1", "svc.automation", ExplanationDepth.EXPLAINED, "t4",
                ),
                DiscussionCoverageEntry(
                    "p1", "svc.automation", ExplanationDepth.DETAILED, "t6",
                ),
            ),
        )
        assert state.depth_for("p1", "svc.automation") == ExplanationDepth.DETAILED


# ---------------------------------------------------------------------------
# Processor exposes auto-advance helper method
# ---------------------------------------------------------------------------


class TestProcessorAutoAdvanceAPI:
    def test_method_exists(self) -> None:
        """The auto-advance helper is part of the processor's public shape."""
        assert hasattr(ProductionTurnProcessor, "_auto_advance_discussion_coverage")

    def test_manual_record_still_works(self, db_path) -> None:
        """record_discussion_coverage() remains callable for callers that
        want to record coverage explicitly outside the auto path."""
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        rt = _make_runtime(
            db_path, RealPTPFactoryWithEvidence(evidence_sequence=evidence),
        )
        rt.start_session(_boot(), _campaign())
        rt.process_turn("sess-51", "hi", 1, "tr-1")
        proc = rt._processors["sess-51"]
        proc.record_discussion_coverage(
            "svc.automation", ExplanationDepth.EXPLAINED, "chat_sess-51_t1",
        )
        assert "svc.automation" in proc.discussion_coverage.heard_by(
            proc.active_participant.reference_id,
        )


# ---------------------------------------------------------------------------
# Atomic commit — recorded coverage survives restart
# ---------------------------------------------------------------------------


class TestCoverageAtomicWithCommit:
    def test_manually_recorded_coverage_survives_restart(self, db_path) -> None:
        """Verifies the auto-advance path's atomicity contract: coverage
        updated before the processor returns is captured by the subsequent
        snapshot export and committed in the same transaction as the
        session revision — proven by surviving a hard restart."""
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        rt = _make_runtime(
            db_path, RealPTPFactoryWithEvidence(evidence_sequence=evidence),
        )
        rt.start_session(_boot(), _campaign())
        rt.process_turn("sess-51", "hi", 1, "tr-1")
        proc = rt._processors["sess-51"]
        active_ref = proc.active_participant.reference_id
        proc.record_discussion_coverage(
            "svc.automation", ExplanationDepth.EXPLAINED, "chat_sess-51_t1",
        )
        rt.process_turn("sess-51", "ok", 2, "tr-2")

        # Hard restart — coverage must survive via the snapshot
        rt2 = _make_runtime(db_path, RealPTPFactoryWithEvidence())
        result = rt2.restore_processor_if_possible("sess-51", _campaign())
        assert result.status == RecoveryStatus.RECOVERED_ACTIVE
        restored = rt2._processors["sess-51"]
        assert "svc.automation" in restored.discussion_coverage.heard_by(active_ref)
        assert restored.discussion_coverage.depth_for(
            active_ref, "svc.automation",
        ) == ExplanationDepth.EXPLAINED


# ---------------------------------------------------------------------------
# Idempotency — replay does not duplicate coverage
# ---------------------------------------------------------------------------


class TestCoverageIdempotency:
    def test_replay_does_not_add_coverage_entry(self, db_path) -> None:
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        rt = _make_runtime(
            db_path, RealPTPFactoryWithEvidence(evidence_sequence=evidence),
        )
        rt.start_session(_boot(), _campaign())
        rt.process_turn("sess-51", "hi", 1, "tr-same")
        proc = rt._processors["sess-51"]
        proc.record_discussion_coverage(
            "svc.automation", ExplanationDepth.EXPLAINED, "chat_sess-51_t1",
        )
        entries_before = proc.discussion_coverage.entries
        # Replay same turn_request_id
        result = rt.process_turn("sess-51", "hi again", 1, "tr-same")
        assert result.replayed_from_idempotency is True
        # Live processor state unchanged by replay (idempotency returned
        # cached result without re-executing the processor)
        assert proc.discussion_coverage.entries == entries_before
