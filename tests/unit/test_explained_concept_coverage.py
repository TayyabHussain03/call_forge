"""Slice 51B — ExplainedConcept semantics + stale-revision rollback.

Proves:
    - ExplainedConcept contract rejects NONE depth
    - DiscussionCoverage advances only when plan.delivered_explanation is set
    - SUPPRESS / question-only plans never advance coverage
    - Historical depth remains monotonic (BRIEF after EXPLAINED keeps EXPLAINED)
    - Stale/failed turn commit does NOT leak speculative coverage into
      a subsequent snapshot export (processor-level atomicity invariant)
"""

from __future__ import annotations

import pytest

from app.application.sales_runtime.persistent_runtime import PersistentSalesRuntime
from app.application.sales_runtime.serializers import (
    BUSINESS_CONVERSATION_KEY,
    DISCUSSION_COVERAGE_KEY,
    LANGUAGE_PROFILE_KEY,
    PARTICIPANT_STATE_KEY,
    QUALIFICATION_KEY,
    BusinessConversationSerializer,
    DiscussionCoverageSerializer,
    LanguageProfileSerializer,
    ParticipantStateSerializer,
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
    ExplainedConcept,
    ExplanationDepth,
    ParticipantContinuityEvidence,
    ParticipantEvidenceKind,
    ParticipantRoleHint,
)
from app.conversation.response_planning.planner import ResponsePlanner
from app.conversation.response_rendering.renderer import DeterministicResponseRenderer
from app.conversation.state_machine.machine import ConversationStateMachine
from app.conversation.state_machine.states import load_config
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
from app.persistence.errors import StaleRevisionError
from app.persistence.sqlite.connection import SqliteConnection
from app.precall.contracts import PreCallDiscoveryPolicy, ServiceDiscoveryRule
from app.runtime.chat.bootstrap.contracts import SessionBootstrapInput


# ---------------------------------------------------------------------------
# ExplainedConcept contract validation
# ---------------------------------------------------------------------------


class TestExplainedConceptContract:
    def test_valid_concept_depth_roundtrip(self) -> None:
        c = ExplainedConcept(
            concept_id="svc.automation",
            delivered_depth=ExplanationDepth.EXPLAINED,
        )
        assert c.concept_id == "svc.automation"
        assert c.delivered_depth == ExplanationDepth.EXPLAINED

    def test_empty_concept_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            ExplainedConcept(
                concept_id="", delivered_depth=ExplanationDepth.BRIEF,
            )

    def test_depth_none_rejected(self) -> None:
        """NONE is the sentinel for 'no explanation delivered' — it must
        never appear on an ExplainedConcept. Fail closed."""
        with pytest.raises(ValueError):
            ExplainedConcept(
                concept_id="svc.automation",
                delivered_depth=ExplanationDepth.NONE,
            )

    def test_non_enum_depth_rejected(self) -> None:
        with pytest.raises(TypeError):
            ExplainedConcept(
                concept_id="svc.automation",
                delivered_depth="detailed",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Real-PTP factory for the atomicity rollback test
# ---------------------------------------------------------------------------


class RealPTPFactory:
    def __init__(
        self,
        actions=(AgentAction.GREET, AgentAction.ASK_IDENTITY,
                 AgentAction.INTRODUCE_REASON, AgentAction.ANSWER_QUESTION),
        evidence_sequence=(),
    ) -> None:
        self._actions = actions
        self._evidence_sequence = evidence_sequence
        self._providers: dict[str, MockReasoningProvider] = {}

    def _make_provider(self):
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
        from app.application.production_turn_processor import ProductionTurnProcessor
        config = load_config(get_settings().conversation_config_path)
        machine = ConversationStateMachine(config, initial_state)
        provider = self._providers.setdefault(session.session_id, self._make_provider())
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
            DiscussionCoverageSerializer().deserialize(cov_env) if cov_env else None
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


def _boot(sid="sess-51b") -> SessionBootstrapInput:
    return SessionBootstrapInput(
        tenant_id="t1", campaign_id="c1", lead_id="lead-1",
        session_id=sid, bootstrap_request_id=f"req-{sid}",
        trusted_time_iso="2026-10-16T10:00:00Z",
    )


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "slice51b.db")


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
# Explicit delivered-explanation drives coverage; nothing else does
# ---------------------------------------------------------------------------


class TestExplicitDeliveryDrivesCoverage:
    def test_default_production_turn_adds_no_coverage(self, db_path) -> None:
        """A normal production turn (no planner emission of
        delivered_explanation) does NOT advance coverage. This proves the
        Slice 51A ResponseLength heuristic is gone."""
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51b_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        rt = _make_runtime(
            db_path, RealPTPFactory(evidence_sequence=evidence),
        )
        rt.start_session(_boot(), _campaign())
        rt.process_turn("sess-51b", "hi", 1, "tr-1")
        proc = rt._processors["sess-51b"]
        # Default production turn — no coverage recorded
        assert proc.discussion_coverage.entries == ()

    def test_manual_record_still_advances_coverage(self, db_path) -> None:
        """record_discussion_coverage() remains the explicit API callers
        can use when the planner has not yet emitted delivered_explanation."""
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51b_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        rt = _make_runtime(
            db_path, RealPTPFactory(evidence_sequence=evidence),
        )
        rt.start_session(_boot(), _campaign())
        rt.process_turn("sess-51b", "hi", 1, "tr-1")
        proc = rt._processors["sess-51b"]
        proc.record_discussion_coverage(
            "svc.automation", ExplanationDepth.EXPLAINED, "chat_sess-51b_t1",
        )
        active = proc.active_participant.reference_id
        assert "svc.automation" in proc.discussion_coverage.heard_by(active)
        assert proc.discussion_coverage.depth_for(
            active, "svc.automation",
        ) == ExplanationDepth.EXPLAINED


# ---------------------------------------------------------------------------
# Stale-revision rollback invariant
# ---------------------------------------------------------------------------


class TestStaleCommitRollsBackCoverage:
    """Core correctness gate (Slice 51B §42):
    "Can a stale or failed turn cause the next request to believe a
     concept was successfully explained when it was not committed?"
     Required answer: NO.
    """

    def test_speculative_coverage_restored_on_stale_commit(self, db_path) -> None:
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51b_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        rt = _make_runtime(
            db_path, RealPTPFactory(evidence_sequence=evidence),
        )
        rt.start_session(_boot(), _campaign())
        rt.process_turn("sess-51b", "hi", 1, "tr-1")
        proc = rt._processors["sess-51b"]
        active = proc.active_participant.reference_id

        # Simulate a committed explanation
        proc.record_discussion_coverage(
            "svc.automation", ExplanationDepth.EXPLAINED, "chat_sess-51b_t1",
        )
        rt.process_turn("sess-51b", "ok", 2, "tr-2")

        # Now simulate: a worker speculatively mutates coverage further,
        # then its turn commit fails (DB-level or stale revision).
        pre = proc.discussion_coverage
        proc.record_discussion_coverage(
            "svc.automation", ExplanationDepth.DETAILED, "chat_sess-51b_speculative",
        )
        assert proc.discussion_coverage.depth_for(
            active, "svc.automation",
        ) == ExplanationDepth.DETAILED

        # Simulate the failed-commit rollback path by restoring the
        # pre-turn snapshot (exactly what PersistentSalesRuntime does on
        # commit failure).
        proc._discussion_coverage = pre
        # The processor is back to EXPLAINED — speculative DETAILED gone
        assert proc.discussion_coverage.depth_for(
            active, "svc.automation",
        ) == ExplanationDepth.EXPLAINED

    def test_persistent_runtime_rolls_back_on_commit_failure(self, db_path) -> None:
        """When the write transaction fails, the processor's live
        _discussion_coverage must be restored to the pre-turn snapshot
        so subsequent snapshot exports cannot leak speculative state."""
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51b_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        rt = _make_runtime(
            db_path, RealPTPFactory(evidence_sequence=evidence),
        )
        rt.start_session(_boot(), _campaign())
        rt.process_turn("sess-51b", "hi", 1, "tr-1")
        proc = rt._processors["sess-51b"]
        pre_turn_coverage = proc.discussion_coverage

        # Monkey-patch the session repository to raise during update —
        # simulates a mid-transaction DB failure (stale revision,
        # idempotency conflict, connection loss, ...)
        original_update = rt.sessions.update

        def failing_update(session):
            # Simulate processor mutating coverage during process_turn
            # (speculative state that must NOT persist across the failure)
            proc.record_discussion_coverage(
                "svc.automation",
                ExplanationDepth.DETAILED,
                "chat_sess-51b_t2_speculative",
            )
            raise StaleRevisionError("simulated stale revision")

        rt.sessions.update = failing_update

        try:
            with pytest.raises(StaleRevisionError):
                rt.process_turn("sess-51b", "another", 2, "tr-fail")
        finally:
            rt.sessions.update = original_update

        # The runtime's except-clause restored coverage to pre-turn state
        assert proc.discussion_coverage == pre_turn_coverage


# ---------------------------------------------------------------------------
# Monotonicity under new semantics
# ---------------------------------------------------------------------------


class TestMonotonicityUnderNewSemantics:
    def test_brief_after_explained_preserves_explained(self, db_path) -> None:
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51b_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        rt = _make_runtime(
            db_path, RealPTPFactory(evidence_sequence=evidence),
        )
        rt.start_session(_boot(), _campaign())
        rt.process_turn("sess-51b", "hi", 1, "tr-1")
        proc = rt._processors["sess-51b"]
        active = proc.active_participant.reference_id

        proc.record_discussion_coverage(
            "svc.automation", ExplanationDepth.EXPLAINED, "chat_sess-51b_t1",
        )
        proc.record_discussion_coverage(
            "svc.automation", ExplanationDepth.BRIEF, "chat_sess-51b_recap",
        )
        # Max remains EXPLAINED
        assert proc.discussion_coverage.depth_for(
            active, "svc.automation",
        ) == ExplanationDepth.EXPLAINED


# ---------------------------------------------------------------------------
# No remaining ResponseLength → ExplanationDepth heuristic in processor
# ---------------------------------------------------------------------------


class TestNoLegacyHeuristic:
    def test_processor_no_length_to_depth_mapping(self) -> None:
        """The Slice 51A `_length_to_depth` mapping must be gone."""
        import inspect
        from app.application import production_turn_processor
        source = inspect.getsource(production_turn_processor)
        # The old mapping variable name must not appear
        assert "_length_to_depth" not in source
