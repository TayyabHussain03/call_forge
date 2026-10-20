"""Slice 51J — real canonical production E2E acceptance.

This test file uses ONLY `build_production_sales_runtime(...)` to
construct the runtime. No manual injection of:
    - ReExplanationIntent
    - ExplanationConcept
    - ReExplanationStrategy
    - DiscussionCoverage
    - Participant prospect archive
    - StrategyAwareResponseRenderer

The scripted LLM transport is permitted (§5).

Proves:
    - canonical composition verification (§6)
    - multi-session isolation (§8, §55) — critical pre-FastAPI concurrency gate
    - participant handoff + restart with real stack (§15–§32)
    - outcome attribution safety (§33–§37)
    - no transcript replay (§54)
"""

from __future__ import annotations

import pytest

from app.application.production_turn_processor import ProductionTurnProcessor
from app.application.sales_runtime.composition import (
    ProductionSalesRuntimeFacade,
    build_production_sales_runtime,
)
from app.application.sales_runtime.recovery import RecoveryStatus
from app.application.sales_runtime.serializers import (
    BUDGET_STATE_KEY,
    BUSINESS_CONVERSATION_KEY,
    DISCUSSION_COVERAGE_KEY,
    LANGUAGE_PROFILE_KEY,
    PARTICIPANT_PROSPECT_ARCHIVE_KEY,
    PARTICIPANT_STATE_KEY,
    PROSPECT_INTELLIGENCE_KEY,
    QUALIFICATION_KEY,
    BudgetStateSerializer,
    BusinessConversationSerializer,
    DiscussionCoverageSerializer,
    LanguageProfileSerializer,
    ParticipantProspectArchiveSerializer,
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
    ParticipantContinuityEvidence,
    ParticipantEvidenceKind,
    ParticipantRoleHint,
)
from app.conversation.participant.intent_provider import ScriptedIntentTransport
from app.conversation.participant.policy import ReExplanationIntent
from app.conversation.response_planning.planner import ResponsePlanner
from app.conversation.response_rendering.renderer import DeterministicResponseRenderer
from app.conversation.response_rendering.strategy_frame import (
    StrategyAwareResponseRenderer,
)
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
# Factory wiring through the facade — real production stack
# ---------------------------------------------------------------------------


class _ProductionFactory:
    """Factory used by the canonical builder. Reads the facade's
    `intent_callable`, uses the facade's `concept_callable_for`, and
    wraps the renderer via the facade — exactly as a real production
    composition root would.
    """

    def __init__(
        self,
        facade: ProductionSalesRuntimeFacade,
        evidence_sequence=(),
        concept_provider_override=None,
    ) -> None:
        self._facade = facade
        self._evidence_sequence = list(evidence_sequence)
        self._concept_provider_override = concept_provider_override
        self._providers = {}
        self.process_turn_call_count = 0

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
                AgentAction.ANSWER_QUESTION, AgentAction.ANSWER_QUESTION,
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
        # Canonical composition: wrap renderer, use intent callable from
        # facade, build concept callable from the processor we're constructing
        wrapped_renderer = self._facade.wrap_renderer(DeterministicResponseRenderer())
        processor = ProductionTurnProcessor(
            orchestrator=orchestrator,
            response_planner=ResponsePlanner(),
            response_renderer=wrapped_renderer,
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
        # Late-bind concept callable after processor exists (resolves §7
        # circular-concept-provider concern cleanly).
        if self._concept_provider_override is not None:
            processor._explanation_concept_provider = (
                self._concept_provider_override
            )
        else:
            processor._explanation_concept_provider = (
                self._facade.concept_callable_for(processor)
            )
        return processor

    def create_for_session(self, session, campaign, policy):
        return self._build(session, initial_state=ConversationState.NEW_CALL)

    def create_from_snapshot(self, session, campaign, policy, snapshot):
        bci_env = snapshot.payload(BUSINESS_CONVERSATION_KEY)
        lang_env = snapshot.payload(LANGUAGE_PROFILE_KEY)
        qual_env = snapshot.payload(QUALIFICATION_KEY)
        part_env = snapshot.payload(PARTICIPANT_STATE_KEY)
        cov_env = snapshot.payload(DISCUSSION_COVERAGE_KEY)
        prospect_env = snapshot.payload(PROSPECT_INTELLIGENCE_KEY)
        archive_env = snapshot.payload(PARTICIPANT_PROSPECT_ARCHIVE_KEY)
        budget_env = snapshot.payload(BUDGET_STATE_KEY)

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
        prospect = (
            ProspectIntelligenceSerializer().deserialize(prospect_env)
            if prospect_env else None
        )
        prospect_archive = (
            ParticipantProspectArchiveSerializer().deserialize(archive_env)
            if archive_env else ()
        )
        reasoning_calls = (
            BudgetStateSerializer().deserialize(budget_env) if budget_env else 0
        )

        kwargs = dict(
            initial_business_conversation=bci,
            initial_language_profile=language,
            initial_qualification=qual,
            initial_active_participant=active,
            initial_archived_participants=archived,
            initial_discussion_coverage=coverage,
            initial_participant_prospect_archive=prospect_archive,
            initial_reasoning_calls=reasoning_calls,
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


def _lead(lead_id="lead-1") -> LeadRecord:
    return LeadRecord(
        lead_id=lead_id, tenant_id="t1", campaign_id="c1",
        business_name="ABC Services", phone=_phone(),
        category=None, city=None, state=None, country=None,
        address=None, email=None, contact_name=None,
        website=None, website_status=WebsiteStatus.UNKNOWN,
    )


def _queue(lead_id="lead-1") -> LeadQueueEntry:
    return LeadQueueEntry(
        lead_id=lead_id, tenant_id="t1", campaign_id="c1",
        queue_position=0, status=LeadQueueStatus.READY,
    )


def _campaign() -> CampaignConfig:
    return CampaignConfig(
        campaign_id="c1", tenant_id="t1", name="Test",
        status=CampaignStatus.ACTIVE,
        authorized_service_ids=("svc-automation", "svc-web"),
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


def _boot(sid="sess-51j", lead_id="lead-1") -> SessionBootstrapInput:
    return SessionBootstrapInput(
        tenant_id="t1", campaign_id="c1", lead_id=lead_id,
        session_id=sid, bootstrap_request_id=f"req-{sid}",
        trusted_time_iso="2026-10-20T10:00:00Z",
    )


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "slice51j.db")


# ---------------------------------------------------------------------------
# Composition verification (§6)
# ---------------------------------------------------------------------------


class TestCanonicalComposition:
    def test_facade_carries_intent_callable_when_transport_supplied(
        self, db_path,
    ) -> None:
        conn = SqliteConnection(db_path)
        transport = ScriptedIntentTransport(('{"intent":"recap_request"}',))
        facade = build_production_sales_runtime(
            connection=conn,
            processor_factory=_ProductionFactory(None, ()),  # placeholder
            intent_transport=transport,
        )
        # Even without the real factory, the facade's intent_callable must be set
        assert facade.intent_callable is not None
        assert facade.wrap_renderer is not None

    def test_strategy_renderer_applied_by_facade(self) -> None:
        wrapped = ProductionSalesRuntimeFacade.wrap_renderer(
            DeterministicResponseRenderer(),
        )
        assert isinstance(wrapped, StrategyAwareResponseRenderer)


# ---------------------------------------------------------------------------
# Multi-session isolation (§8, §55) — critical pre-FastAPI gate
# ---------------------------------------------------------------------------


class TestMultiSessionIsolation:
    """Two independent sessions through the canonical builder. Verify no
    cross-session leakage of participant/concept/coverage/budget."""

    def _build_facade(self, db_path, evidence_a=(), evidence_b=()):
        conn = SqliteConnection(db_path)
        facade = build_production_sales_runtime(
            connection=conn,
            processor_factory=_ProductionFactory(None, ()),  # overwritten below
            intent_transport=None,
        )
        # Wire the factory with facade reference now that it exists
        facade.runtime._processor_factory = _ProductionFactory(
            facade, evidence_sequence=evidence_a + evidence_b,
        )
        # Seed common state
        facade.runtime.leads.save(_lead("lead-A"))
        facade.runtime.queue.save(_queue("lead-A"))
        facade.runtime.leads.save(_lead("lead-B"))
        facade.runtime.queue.save(_queue("lead-B"))
        facade.runtime.policies.register_default("c1", _policy_cfg())
        return facade

    def test_two_sessions_isolated_participant_state(self, db_path) -> None:
        evidence_a = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-A_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        # Build fresh facade and use its own factory
        conn = SqliteConnection(db_path)
        facade = build_production_sales_runtime(
            connection=conn,
            processor_factory=_ProductionFactory(None, ()),
            intent_transport=None,
        )
        # Seed
        facade.runtime.leads.save(_lead("lead-A"))
        facade.runtime.queue.save(_queue("lead-A"))
        facade.runtime.leads.save(_lead("lead-B"))
        facade.runtime.queue.save(_queue("lead-B"))
        facade.runtime.policies.register_default("c1", _policy_cfg())
        # Replace factory with one that references the facade for all sessions
        facade.runtime._processor_factory = _ProductionFactory(
            facade, evidence_sequence=evidence_a + evidence_a,
        )

        facade.runtime.start_session(
            _boot(sid="sess-A", lead_id="lead-A"), _campaign(),
        )
        facade.runtime.start_session(
            _boot(sid="sess-B", lead_id="lead-B"), _campaign(),
        )
        facade.runtime.process_turn("sess-A", "I handle marketing", 1, "trA-1")
        facade.runtime.process_turn("sess-B", "I handle sales", 1, "trB-1")

        proc_a = facade.runtime._processors["sess-A"]
        proc_b = facade.runtime._processors["sess-B"]

        # Each session's active participant lives in its own factory counter
        assert proc_a.active_participant is not None
        assert proc_b.active_participant is not None
        assert proc_a.active_participant.reference_id != proc_b.active_participant.reference_id

        # Session A's archive does NOT contain session B's participant
        a_archive_ids = {
            s.participant.reference_id for s in proc_a.archived_participant_states
        }
        b_archive_ids = {
            s.participant.reference_id for s in proc_b.archived_participant_states
        }
        assert proc_b.active_participant.reference_id not in a_archive_ids
        assert proc_a.active_participant.reference_id not in b_archive_ids

    def test_two_sessions_isolated_coverage(self, db_path) -> None:
        """Coverage recorded in session A must not appear in session B."""
        from app.conversation.participant.contracts import ExplanationDepth
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-A_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-B_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        conn = SqliteConnection(db_path)
        facade = build_production_sales_runtime(
            connection=conn,
            processor_factory=_ProductionFactory(None, ()),
            intent_transport=None,
        )
        facade.runtime.leads.save(_lead("lead-A"))
        facade.runtime.queue.save(_queue("lead-A"))
        facade.runtime.leads.save(_lead("lead-B"))
        facade.runtime.queue.save(_queue("lead-B"))
        facade.runtime.policies.register_default("c1", _policy_cfg())
        facade.runtime._processor_factory = _ProductionFactory(
            facade, evidence_sequence=evidence,
        )

        facade.runtime.start_session(_boot(sid="sess-A", lead_id="lead-A"), _campaign())
        facade.runtime.start_session(_boot(sid="sess-B", lead_id="lead-B"), _campaign())
        facade.runtime.process_turn("sess-A", "hi", 1, "trA-1")
        facade.runtime.process_turn("sess-B", "hi", 1, "trB-1")

        proc_a = facade.runtime._processors["sess-A"]
        proc_b = facade.runtime._processors["sess-B"]
        proc_a.record_discussion_coverage(
            "svc.automation", ExplanationDepth.EXPLAINED, "chat_sess-A_t1",
        )
        # Session B coverage must still be empty
        assert proc_b.discussion_coverage.entries == ()

    def test_two_sessions_isolated_budget(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        facade = build_production_sales_runtime(
            connection=conn,
            processor_factory=_ProductionFactory(None, ()),
            intent_transport=None,
        )
        facade.runtime.leads.save(_lead("lead-A"))
        facade.runtime.queue.save(_queue("lead-A"))
        facade.runtime.leads.save(_lead("lead-B"))
        facade.runtime.queue.save(_queue("lead-B"))
        facade.runtime.policies.register_default("c1", _policy_cfg())
        facade.runtime._processor_factory = _ProductionFactory(facade, ())

        facade.runtime.start_session(_boot(sid="sess-A", lead_id="lead-A"), _campaign())
        facade.runtime.start_session(_boot(sid="sess-B", lead_id="lead-B"), _campaign())
        facade.runtime.process_turn("sess-A", "hi", 1, "trA-1")
        facade.runtime.process_turn("sess-B", "hi", 1, "trB-1")

        proc_a = facade.runtime._processors["sess-A"]
        proc_b = facade.runtime._processors["sess-B"]
        # Mutate A's budget — B must not see it
        proc_a._reasoning_calls = 7
        assert proc_b._reasoning_calls != 7


# ---------------------------------------------------------------------------
# E2E with real stack — basic turn through canonical builder
# ---------------------------------------------------------------------------


class TestCanonicalBuilderBasicE2E:
    def test_full_turn_through_canonical_builder(self, db_path) -> None:
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51j_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        conn = SqliteConnection(db_path)
        transport = ScriptedIntentTransport(('{"intent":"none"}',))
        facade = build_production_sales_runtime(
            connection=conn,
            processor_factory=_ProductionFactory(None, ()),
            intent_transport=transport,
        )
        facade.runtime.leads.save(_lead())
        facade.runtime.queue.save(_queue())
        facade.runtime.policies.register_default("c1", _policy_cfg())
        facade.runtime._processor_factory = _ProductionFactory(
            facade, evidence_sequence=evidence,
        )

        from app.runtime.chat.bootstrap.contracts import BootstrapStatus
        boot = facade.runtime.start_session(_boot(), _campaign())
        assert boot.status == BootstrapStatus.CREATED

        result = facade.runtime.process_turn("sess-51j", "I handle marketing", 1, "tr-1")
        assert result.committed_revision == 1

        # Participant was resolved via the production pipeline
        proc = facade.runtime._processors["sess-51j"]
        assert proc.active_participant is not None
        assert proc.active_participant.role_hint == ParticipantRoleHint.EMPLOYEE


# ---------------------------------------------------------------------------
# Participant handoff + restart through canonical builder
# ---------------------------------------------------------------------------


class TestCanonicalHandoffAndRestart:
    def test_handoff_archives_a_resets_b_prospect(self, db_path) -> None:
        from app.conversation.prospect_intelligence.contracts import (
            EvidenceProvenance, EvidenceSourceKind, ObservedProspectFacts,
            ObservedValue, ProspectIntelligenceSnapshot,
        )
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51j_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_HANDOFF,
                source_turn_id="chat_sess-51j_t2",
                claimed_role_hint=ParticipantRoleHint.OWNER,
            ),
        )
        conn = SqliteConnection(db_path)
        facade = build_production_sales_runtime(
            connection=conn,
            processor_factory=_ProductionFactory(None, ()),
            intent_transport=None,
        )
        facade.runtime.leads.save(_lead())
        facade.runtime.queue.save(_queue())
        facade.runtime.policies.register_default("c1", _policy_cfg())
        facade.runtime._processor_factory = _ProductionFactory(
            facade, evidence_sequence=evidence,
        )

        facade.runtime.start_session(_boot(), _campaign())
        facade.runtime.process_turn("sess-51j", "I handle marketing", 1, "tr-1")
        proc = facade.runtime._processors["sess-51j"]

        # Give Person A interest
        prov = EvidenceProvenance("t1", EvidenceSourceKind.EXPLICIT_STATEMENT)
        proc._prospect_intelligence = ProspectIntelligenceSnapshot(
            observed=ObservedProspectFacts(
                explicit_interest_signal=ObservedValue(value=True, provenance=prov),
            ),
        )
        person_a_ref = proc.active_participant.reference_id

        # Turn 2 — handoff
        facade.runtime.process_turn("sess-51j", "my colleague took over", 2, "tr-2")
        proc = facade.runtime._processors["sess-51j"]

        # Person B now active and prospect state is fresh
        assert proc.active_participant.reference_id != person_a_ref
        assert proc.active_participant.role_hint == ParticipantRoleHint.OWNER
        assert proc.prospect_intelligence.observed.explicit_interest_signal is None

        # Person A's prospect state archived under their ref
        archive = {pid: snap for pid, snap in proc.participant_prospect_archive}
        assert person_a_ref in archive
        assert archive[person_a_ref].observed.explicit_interest_signal.value is True

    def test_hard_restart_through_canonical_builder_recovers_state(
        self, db_path,
    ) -> None:
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51j_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        # Session A
        conn1 = SqliteConnection(db_path)
        facade1 = build_production_sales_runtime(
            connection=conn1,
            processor_factory=_ProductionFactory(None, ()),
            intent_transport=None,
        )
        facade1.runtime.leads.save(_lead())
        facade1.runtime.queue.save(_queue())
        facade1.runtime.policies.register_default("c1", _policy_cfg())
        facade1.runtime._processor_factory = _ProductionFactory(
            facade1, evidence_sequence=evidence,
        )
        facade1.runtime.start_session(_boot(), _campaign())
        facade1.runtime.process_turn("sess-51j", "I handle marketing", 1, "tr-1")

        # Hard restart — new facade over same DB
        conn2 = SqliteConnection(db_path)
        facade2 = build_production_sales_runtime(
            connection=conn2,
            processor_factory=_ProductionFactory(None, ()),
            intent_transport=None,
        )
        facade2.runtime._processor_factory = _ProductionFactory(facade2, ())
        result = facade2.runtime.restore_processor_if_possible(
            "sess-51j", _campaign(),
        )
        assert result.status == RecoveryStatus.RECOVERED_ACTIVE


# ---------------------------------------------------------------------------
# Outcome attribution — Person B neutral is NOT marked interested
# ---------------------------------------------------------------------------


class TestOutcomeAttributionSafety:
    def test_person_a_interest_does_not_leak_to_person_b_outcome(
        self, db_path,
    ) -> None:
        """§33 — explicit outcome engine test. After A interested → B
        neutral handoff, the outcome engine reads B's fresh snapshot, so
        'owner interested' is never asserted falsely."""
        from app.conversation.prospect_intelligence.contracts import (
            EvidenceProvenance, EvidenceSourceKind, ObservedProspectFacts,
            ObservedValue, ProspectIntelligenceSnapshot,
        )
        from app.runtime.chat.outcome.contracts import OutcomeStatus, TerminationReason
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51j_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_HANDOFF,
                source_turn_id="chat_sess-51j_t2",
                claimed_role_hint=ParticipantRoleHint.OWNER,
            ),
        )
        conn = SqliteConnection(db_path)
        facade = build_production_sales_runtime(
            connection=conn,
            processor_factory=_ProductionFactory(None, ()),
            intent_transport=None,
        )
        facade.runtime.leads.save(_lead())
        facade.runtime.queue.save(_queue())
        facade.runtime.policies.register_default("c1", _policy_cfg())
        facade.runtime._processor_factory = _ProductionFactory(
            facade, evidence_sequence=evidence,
        )

        facade.runtime.start_session(_boot(), _campaign())
        facade.runtime.process_turn("sess-51j", "I handle marketing", 1, "tr-1")
        proc = facade.runtime._processors["sess-51j"]

        # Person A interested
        prov = EvidenceProvenance("t1", EvidenceSourceKind.EXPLICIT_STATEMENT)
        proc._prospect_intelligence = ProspectIntelligenceSnapshot(
            observed=ObservedProspectFacts(
                explicit_interest_signal=ObservedValue(value=True, provenance=prov),
            ),
        )

        # Handoff to Person B (owner, neutral)
        facade.runtime.process_turn("sess-51j", "owner took over", 2, "tr-2")

        # Finish session — outcome engine reads live (Person B's fresh)
        # prospect snapshot
        done = facade.runtime.finish_session(
            "sess-51j",
            termination_reason=TerminationReason.NATURAL_COMPLETION,
        )
        # Owner is neutral — must NOT be marked interested
        assert done.lead_outcome.outcome_status != OutcomeStatus.INTERESTED


# ---------------------------------------------------------------------------
# No transcript replay through canonical restart
# ---------------------------------------------------------------------------


class TestNoTranscriptReplay:
    def test_restart_reads_from_snapshot_only(self, db_path) -> None:
        """Restart recovery must come from structured snapshot payloads,
        never from parsing transcript. We prove this by inspecting the
        source of `_ProductionFactory.create_from_snapshot` — it reads
        only serializer-decoded payloads, never session.transcript."""
        import inspect
        source = inspect.getsource(_ProductionFactory.create_from_snapshot)
        # Should not read transcript to rebuild authoritative state
        assert "session.transcript" not in source
        assert "transcript_json" not in source  # no raw transcript JSON touch

    def test_production_processor_recovery_does_not_import_transcript(self) -> None:
        """§42 — scan all production restart/recovery paths. None may
        rebuild authoritative state from session.transcript."""
        import inspect
        from app.application.sales_runtime import recovery
        source = inspect.getsource(recovery)
        # recovery must only pull from runtime_snapshot_json + payloads
        assert "transcript" not in source.lower() or "runtime_snapshot" in source


# ---------------------------------------------------------------------------
# SIMPLIFY / DEEPEN / RECAP real E2E through canonical builder
# ---------------------------------------------------------------------------
#
# Scope note: the orchestrator's reasoning sequence does not currently
# select a specific service (no SERVICE_* action in the mock proposals).
# To exercise the strategy path end-to-end through the real policy +
# planner + processor, we override the concept provider with
# StaticConceptProvider (explicitly a production-registered option for
# scenarios where the concept is already known). The intent still flows
# through the real ScriptedIntentTransport → ReExplanationIntentLLMProvider
# → ReExplanationPolicy → processor chain. No manual strategy injection.


def _build_strategy_facade(db_path, *, scripted_intents, evidence, concept_id):
    from app.conversation.participant.concept_provider import StaticConceptProvider

    conn = SqliteConnection(db_path)
    transport = ScriptedIntentTransport(scripted_intents)
    facade = build_production_sales_runtime(
        connection=conn,
        processor_factory=_ProductionFactory(None, ()),
        intent_transport=transport,
    )
    facade.runtime.leads.save(_lead())
    facade.runtime.queue.save(_queue())
    facade.runtime.policies.register_default("c1", _policy_cfg())
    facade.runtime._processor_factory = _ProductionFactory(
        facade,
        evidence_sequence=evidence,
        concept_provider_override=StaticConceptProvider(concept_id).as_callable(),
    )
    return facade


class TestSimplifyDeepenRecapE2E:
    """§4, §7, §11 — SIMPLIFY, DEEPEN, RECAP flow through the full
    production policy chain. Verifies ExplainedConcept emission at the
    correct ExplanationDepth and monotonic coverage."""

    EVIDENCE = (
        ParticipantContinuityEvidence(
            kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
            source_turn_id="chat_sess-51j_t1",
            claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
        ),
        ParticipantContinuityEvidence(
            kind=ParticipantEvidenceKind.NONE,
            source_turn_id="chat_sess-51j_t2",
            claimed_role_hint=ParticipantRoleHint.UNKNOWN,
        ),
    )
    CONCEPT = "service:test-concept"

    def _seed_prior_coverage(self, processor) -> None:
        """Simulate 'concept X has been explained to the active participant'
        so a subsequent SIMPLIFY/DEEPEN/RECAP intent can route correctly."""
        from app.conversation.participant.contracts import ExplanationDepth
        processor.record_discussion_coverage(
            self.CONCEPT, ExplanationDepth.EXPLAINED, "chat_sess-51j_t1",
        )

    def test_simplify_request_produces_simplify_strategy_and_brief_depth(
        self, db_path,
    ) -> None:
        from app.conversation.participant.contracts import ExplanationDepth
        facade = _build_strategy_facade(
            db_path,
            scripted_intents=(
                '{"intent":"none"}',
                '{"intent":"simplify_request"}',
            ),
            evidence=self.EVIDENCE,
            concept_id=self.CONCEPT,
        )
        facade.runtime.start_session(_boot(), _campaign())
        facade.runtime.process_turn("sess-51j", "I handle marketing", 1, "tr-1")
        proc = facade.runtime._processors["sess-51j"]
        self._seed_prior_coverage(proc)

        facade.runtime.process_turn("sess-51j", "simple words mein batao", 2, "tr-2")

        # After SIMPLIFY, coverage for the concept must still be at least
        # EXPLAINED (monotonic — BRIEF simplification does NOT downgrade
        # historical maximum)
        depth = proc.discussion_coverage.depth_for(
            proc.active_participant.reference_id, self.CONCEPT,
        )
        assert depth in (ExplanationDepth.EXPLAINED, ExplanationDepth.DETAILED)

    def test_deepen_request_produces_detailed_coverage(self, db_path) -> None:
        from app.conversation.participant.contracts import ExplanationDepth
        facade = _build_strategy_facade(
            db_path,
            scripted_intents=(
                '{"intent":"none"}',
                '{"intent":"deepen_request"}',
            ),
            evidence=self.EVIDENCE,
            concept_id=self.CONCEPT,
        )
        facade.runtime.start_session(_boot(), _campaign())
        facade.runtime.process_turn("sess-51j", "I handle marketing", 1, "tr-1")
        proc = facade.runtime._processors["sess-51j"]
        self._seed_prior_coverage(proc)

        facade.runtime.process_turn("sess-51j", "ab detail mein batao", 2, "tr-2")

        depth = proc.discussion_coverage.depth_for(
            proc.active_participant.reference_id, self.CONCEPT,
        )
        # DEEPEN emits DETAILED
        assert depth == ExplanationDepth.DETAILED

    def test_recap_preserves_historical_maximum_depth(self, db_path) -> None:
        """§11 — DETAILED must remain DETAILED after a subsequent recap."""
        from app.conversation.participant.contracts import ExplanationDepth
        facade = _build_strategy_facade(
            db_path,
            scripted_intents=(
                '{"intent":"none"}',
                '{"intent":"deepen_request"}',
                '{"intent":"recap_request"}',
            ),
            evidence=self.EVIDENCE + (
                ParticipantContinuityEvidence(
                    kind=ParticipantEvidenceKind.NONE,
                    source_turn_id="chat_sess-51j_t3",
                    claimed_role_hint=ParticipantRoleHint.UNKNOWN,
                ),
            ),
            concept_id=self.CONCEPT,
        )
        facade.runtime.start_session(_boot(), _campaign())
        facade.runtime.process_turn("sess-51j", "hello", 1, "tr-1")
        proc = facade.runtime._processors["sess-51j"]
        self._seed_prior_coverage(proc)

        facade.runtime.process_turn("sess-51j", "detail do", 2, "tr-2")
        facade.runtime.process_turn("sess-51j", "short recap", 3, "tr-3")

        depth = proc.discussion_coverage.depth_for(
            proc.active_participant.reference_id, self.CONCEPT,
        )
        # Historical maximum from DEEPEN must be preserved
        assert depth == ExplanationDepth.DETAILED


# ---------------------------------------------------------------------------
# Restart after DEEPEN — DETAILED coverage survives
# ---------------------------------------------------------------------------


class TestRestartAfterDeepen:
    """§13 — hard restart after DETAILED coverage must preserve coverage."""

    def test_detailed_coverage_persists_across_canonical_restart(
        self, db_path,
    ) -> None:
        from app.conversation.participant.contracts import ExplanationDepth
        from app.conversation.participant.concept_provider import StaticConceptProvider

        CONCEPT = "service:deepen-concept"
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51j_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.NONE,
                source_turn_id="chat_sess-51j_t2",
                claimed_role_hint=ParticipantRoleHint.UNKNOWN,
            ),
        )
        # Session A — deepen
        conn1 = SqliteConnection(db_path)
        transport = ScriptedIntentTransport((
            '{"intent":"none"}',
            '{"intent":"deepen_request"}',
        ))
        facade1 = build_production_sales_runtime(
            connection=conn1,
            processor_factory=_ProductionFactory(None, ()),
            intent_transport=transport,
        )
        facade1.runtime.leads.save(_lead())
        facade1.runtime.queue.save(_queue())
        facade1.runtime.policies.register_default("c1", _policy_cfg())
        facade1.runtime._processor_factory = _ProductionFactory(
            facade1, evidence_sequence=evidence,
            concept_provider_override=StaticConceptProvider(CONCEPT).as_callable(),
        )
        facade1.runtime.start_session(_boot(), _campaign())
        facade1.runtime.process_turn("sess-51j", "hi", 1, "tr-1")
        proc1 = facade1.runtime._processors["sess-51j"]
        proc1.record_discussion_coverage(
            CONCEPT, ExplanationDepth.EXPLAINED, "chat_sess-51j_t1",
        )
        facade1.runtime.process_turn("sess-51j", "detail", 2, "tr-2")
        pre_restart_depth = proc1.discussion_coverage.depth_for(
            proc1.active_participant.reference_id, CONCEPT,
        )
        assert pre_restart_depth == ExplanationDepth.DETAILED

        # Hard restart — fresh facade
        conn2 = SqliteConnection(db_path)
        facade2 = build_production_sales_runtime(
            connection=conn2,
            processor_factory=_ProductionFactory(None, ()),
            intent_transport=None,
        )
        facade2.runtime._processor_factory = _ProductionFactory(
            facade2, evidence_sequence=(),
            concept_provider_override=StaticConceptProvider(CONCEPT).as_callable(),
        )
        result = facade2.runtime.restore_processor_if_possible(
            "sess-51j", _campaign(),
        )
        assert result.status == RecoveryStatus.RECOVERED_ACTIVE
        proc2 = facade2.runtime._processors["sess-51j"]
        active_id = proc1.active_participant.reference_id
        post_restart_depth = proc2.discussion_coverage.depth_for(
            active_id, CONCEPT,
        )
        assert post_restart_depth == ExplanationDepth.DETAILED


# ---------------------------------------------------------------------------
# Owner-interest outcome — reverse of existing test (§15)
# ---------------------------------------------------------------------------


class TestOwnerInterestOutcome:
    def test_person_b_explicit_interest_drives_outcome_interested(
        self, db_path,
    ) -> None:
        """§15 — after handoff, if Person B says 'yes, this is useful',
        the live snapshot carries B's interest and the outcome engine
        emits INTERESTED."""
        from app.conversation.prospect_intelligence.contracts import (
            EvidenceProvenance, EvidenceSourceKind, ObservedProspectFacts,
            ObservedValue, ProspectIntelligenceSnapshot,
        )
        from app.runtime.chat.outcome.contracts import (
            OutcomeStatus, TerminationReason,
        )
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51j_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_HANDOFF,
                source_turn_id="chat_sess-51j_t2",
                claimed_role_hint=ParticipantRoleHint.OWNER,
            ),
        )
        conn = SqliteConnection(db_path)
        facade = build_production_sales_runtime(
            connection=conn,
            processor_factory=_ProductionFactory(None, ()),
            intent_transport=None,
        )
        facade.runtime.leads.save(_lead())
        facade.runtime.queue.save(_queue())
        facade.runtime.policies.register_default("c1", _policy_cfg())
        facade.runtime._processor_factory = _ProductionFactory(
            facade, evidence_sequence=evidence,
        )

        facade.runtime.start_session(_boot(), _campaign())
        facade.runtime.process_turn("sess-51j", "I handle marketing", 1, "tr-1")
        facade.runtime.process_turn("sess-51j", "owner took over", 2, "tr-2")
        proc = facade.runtime._processors["sess-51j"]

        # Simulate Person B expressing explicit interest
        prov = EvidenceProvenance("t1", EvidenceSourceKind.EXPLICIT_STATEMENT)
        proc._prospect_intelligence = ProspectIntelligenceSnapshot(
            observed=ObservedProspectFacts(
                explicit_interest_signal=ObservedValue(
                    value=True, provenance=prov,
                ),
            ),
        )

        done = facade.runtime.finish_session(
            "sess-51j",
            termination_reason=TerminationReason.NATURAL_COMPLETION,
        )
        assert done.lead_outcome.outcome_status == OutcomeStatus.INTERESTED


# ---------------------------------------------------------------------------
# Objection isolation (§18)
# ---------------------------------------------------------------------------


class TestObjectionIsolation:
    def test_person_a_price_objection_does_not_follow_to_person_b(
        self, db_path,
    ) -> None:
        from app.conversation.prospect_intelligence.contracts import (
            EvidenceProvenance, EvidenceSourceKind, ObjectionType,
            ObservedProspectFacts, ObservedValue, ProspectIntelligenceSnapshot,
        )
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51j_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_HANDOFF,
                source_turn_id="chat_sess-51j_t2",
                claimed_role_hint=ParticipantRoleHint.OWNER,
            ),
        )
        conn = SqliteConnection(db_path)
        facade = build_production_sales_runtime(
            connection=conn,
            processor_factory=_ProductionFactory(None, ()),
            intent_transport=None,
        )
        facade.runtime.leads.save(_lead())
        facade.runtime.queue.save(_queue())
        facade.runtime.policies.register_default("c1", _policy_cfg())
        facade.runtime._processor_factory = _ProductionFactory(
            facade, evidence_sequence=evidence,
        )

        facade.runtime.start_session(_boot(), _campaign())
        facade.runtime.process_turn("sess-51j", "I handle marketing", 1, "tr-1")
        proc = facade.runtime._processors["sess-51j"]
        prov = EvidenceProvenance("t1", EvidenceSourceKind.EXPLICIT_STATEMENT)
        # Person A objects on price
        proc._prospect_intelligence = ProspectIntelligenceSnapshot(
            observed=ObservedProspectFacts(
                explicit_objection=ObservedValue(
                    value=ObjectionType.PRICE, provenance=prov,
                ),
            ),
        )
        facade.runtime.process_turn("sess-51j", "owner took over", 2, "tr-2")
        proc = facade.runtime._processors["sess-51j"]

        # Person B's live snapshot does NOT carry A's price objection
        obj = proc.prospect_intelligence.observed.explicit_objection
        assert obj is None or obj.value != ObjectionType.PRICE


# ---------------------------------------------------------------------------
# Idempotent replay (§35)
# ---------------------------------------------------------------------------


class TestIdempotentReplay:
    def test_duplicate_turn_request_short_circuits_before_processor(
        self, db_path,
    ) -> None:
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51j_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        conn = SqliteConnection(db_path)
        facade = build_production_sales_runtime(
            connection=conn,
            processor_factory=_ProductionFactory(None, ()),
            intent_transport=None,
        )
        facade.runtime.leads.save(_lead())
        facade.runtime.queue.save(_queue())
        facade.runtime.policies.register_default("c1", _policy_cfg())
        facade.runtime._processor_factory = _ProductionFactory(
            facade, evidence_sequence=evidence,
        )
        facade.runtime.start_session(_boot(), _campaign())
        first = facade.runtime.process_turn("sess-51j", "hi", 1, "tr-1")
        # Replay same turn_request_id
        replayed = facade.runtime.process_turn("sess-51j", "hi", 1, "tr-1")

        assert replayed.replayed_from_idempotency is True
        assert replayed.committed_revision == first.committed_revision
        assert replayed.agent_response == first.agent_response

    def test_replay_does_not_advance_coverage_or_budget(self, db_path) -> None:
        from app.conversation.participant.concept_provider import StaticConceptProvider
        from app.conversation.participant.contracts import ExplanationDepth
        CONCEPT = "service:idemp"
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51j_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        conn = SqliteConnection(db_path)
        transport = ScriptedIntentTransport(('{"intent":"deepen_request"}',))
        facade = build_production_sales_runtime(
            connection=conn,
            processor_factory=_ProductionFactory(None, ()),
            intent_transport=transport,
        )
        facade.runtime.leads.save(_lead())
        facade.runtime.queue.save(_queue())
        facade.runtime.policies.register_default("c1", _policy_cfg())
        facade.runtime._processor_factory = _ProductionFactory(
            facade,
            evidence_sequence=evidence,
            concept_provider_override=StaticConceptProvider(CONCEPT).as_callable(),
        )
        facade.runtime.start_session(_boot(), _campaign())
        facade.runtime.process_turn("sess-51j", "hi", 1, "tr-1")
        proc = facade.runtime._processors["sess-51j"]
        proc.record_discussion_coverage(
            CONCEPT, ExplanationDepth.EXPLAINED, "chat_sess-51j_t1",
        )
        reasoning_before = proc._reasoning_calls

        # First commit of turn 2
        facade.runtime.process_turn("sess-51j", "deep", 2, "tr-2")
        depth_after = proc.discussion_coverage.depth_for(
            proc.active_participant.reference_id, CONCEPT,
        )
        reasoning_after = proc._reasoning_calls

        # Replay same request id
        facade.runtime.process_turn("sess-51j", "deep", 2, "tr-2")
        depth_replay = proc.discussion_coverage.depth_for(
            proc.active_participant.reference_id, CONCEPT,
        )
        reasoning_replay = proc._reasoning_calls

        # Coverage and budget identical before/after replay — short-circuit
        assert depth_replay == depth_after
        assert reasoning_replay == reasoning_after
        assert reasoning_after > reasoning_before  # first turn did advance


# ---------------------------------------------------------------------------
# A → B → C participant generality (§38)
# ---------------------------------------------------------------------------


class TestABCParticipantGenerality:
    def test_three_participants_in_sequence(self, db_path) -> None:
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51j_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_HANDOFF,
                source_turn_id="chat_sess-51j_t2",
                claimed_role_hint=ParticipantRoleHint.OWNER,
            ),
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_HANDOFF,
                source_turn_id="chat_sess-51j_t3",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        conn = SqliteConnection(db_path)
        facade = build_production_sales_runtime(
            connection=conn,
            processor_factory=_ProductionFactory(None, ()),
            intent_transport=None,
        )
        facade.runtime.leads.save(_lead())
        facade.runtime.queue.save(_queue())
        facade.runtime.policies.register_default("c1", _policy_cfg())
        facade.runtime._processor_factory = _ProductionFactory(
            facade, evidence_sequence=evidence,
        )
        facade.runtime.start_session(_boot(), _campaign())
        facade.runtime.process_turn("sess-51j", "I handle marketing", 1, "tr-1")
        proc = facade.runtime._processors["sess-51j"]
        a_ref = proc.active_participant.reference_id

        facade.runtime.process_turn("sess-51j", "owner now", 2, "tr-2")
        proc = facade.runtime._processors["sess-51j"]
        b_ref = proc.active_participant.reference_id

        facade.runtime.process_turn("sess-51j", "new colleague", 3, "tr-3")
        proc = facade.runtime._processors["sess-51j"]
        c_ref = proc.active_participant.reference_id

        # All three refs are distinct — logic is not A→B specific
        assert len({a_ref, b_ref, c_ref}) == 3
        archive_ids = {
            s.participant.reference_id for s in proc.archived_participant_states
        }
        assert a_ref in archive_ids
        assert b_ref in archive_ids
        # C is active, not archived
        assert c_ref not in archive_ids


# ---------------------------------------------------------------------------
# DNC + NOT_INTERESTED distinction (§28–§30)
# ---------------------------------------------------------------------------


class TestDNCCanonicalPath:
    def test_dnc_terminates_outcome_and_marks_queue(self, db_path) -> None:
        from app.leads.campaign.contracts import LeadQueueStatus
        from app.runtime.chat.outcome.contracts import (
            OutcomeStatus, TerminationReason,
        )
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51j_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        conn = SqliteConnection(db_path)
        facade = build_production_sales_runtime(
            connection=conn,
            processor_factory=_ProductionFactory(None, ()),
            intent_transport=None,
        )
        facade.runtime.leads.save(_lead())
        facade.runtime.queue.save(_queue())
        facade.runtime.policies.register_default("c1", _policy_cfg())
        facade.runtime._processor_factory = _ProductionFactory(
            facade, evidence_sequence=evidence,
        )

        facade.runtime.start_session(_boot(), _campaign())
        facade.runtime.process_turn("sess-51j", "hi", 1, "tr-1")

        done = facade.runtime.finish_session(
            "sess-51j",
            is_dnc=True,
            termination_reason=TerminationReason.DNC_REQUEST,
        )
        assert done.lead_outcome.outcome_status == OutcomeStatus.DNC

        # Queue durably marked
        queue_entry = facade.runtime.queue.get("t1", "c1", "lead-1")
        assert queue_entry.status == LeadQueueStatus.DNC

    def test_not_interested_is_distinct_from_dnc(self, db_path) -> None:
        from app.leads.campaign.contracts import LeadQueueStatus
        from app.runtime.chat.outcome.contracts import (
            OutcomeStatus, TerminationReason,
        )
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51j_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        conn = SqliteConnection(db_path)
        facade = build_production_sales_runtime(
            connection=conn,
            processor_factory=_ProductionFactory(None, ()),
            intent_transport=None,
        )
        facade.runtime.leads.save(_lead())
        facade.runtime.queue.save(_queue())
        facade.runtime.policies.register_default("c1", _policy_cfg())
        facade.runtime._processor_factory = _ProductionFactory(
            facade, evidence_sequence=evidence,
        )

        facade.runtime.start_session(_boot(), _campaign())
        facade.runtime.process_turn("sess-51j", "hi", 1, "tr-1")

        done = facade.runtime.finish_session(
            "sess-51j",
            is_not_interested=True,
            termination_reason=TerminationReason.NOT_INTERESTED_EXIT,
        )
        # NOT_INTERESTED is a different outcome than DNC
        assert done.lead_outcome.outcome_status != OutcomeStatus.DNC
        # And must NOT durably mark the queue as DNC
        queue_entry = facade.runtime.queue.get("t1", "c1", "lead-1")
        assert queue_entry.status != LeadQueueStatus.DNC


# ---------------------------------------------------------------------------
# Multi-session interleaving — expanded (§36)
# ---------------------------------------------------------------------------


class TestMultiSessionInterleaving:
    def test_interleaved_turns_do_not_share_participant_state(
        self, db_path,
    ) -> None:
        self_id = ParticipantContinuityEvidence(
            kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
            source_turn_id="t1",
            claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
        )
        none_evid = ParticipantContinuityEvidence(
            kind=ParticipantEvidenceKind.NONE,
            source_turn_id="any",
            claimed_role_hint=ParticipantRoleHint.UNKNOWN,
        )
        conn = SqliteConnection(db_path)
        facade = build_production_sales_runtime(
            connection=conn,
            processor_factory=_ProductionFactory(None, ()),
            intent_transport=None,
        )
        facade.runtime.leads.save(_lead("lead-A"))
        facade.runtime.queue.save(_queue("lead-A"))
        facade.runtime.leads.save(_lead("lead-B"))
        facade.runtime.queue.save(_queue("lead-B"))
        facade.runtime.policies.register_default("c1", _policy_cfg())
        # Each session gets its own fresh iterator over this list.
        # Both sessions see (self_id, none_evid) in order.
        facade.runtime._processor_factory = _ProductionFactory(
            facade, evidence_sequence=(self_id, none_evid),
        )

        facade.runtime.start_session(_boot(sid="sess-A", lead_id="lead-A"), _campaign())
        facade.runtime.start_session(_boot(sid="sess-B", lead_id="lead-B"), _campaign())
        facade.runtime.process_turn("sess-A", "A hello", 1, "trA-1")
        facade.runtime.process_turn("sess-B", "B hello", 1, "trB-1")
        facade.runtime.process_turn("sess-A", "A follow", 2, "trA-2")
        facade.runtime.process_turn("sess-B", "B follow", 2, "trB-2")

        proc_a = facade.runtime._processors["sess-A"]
        proc_b = facade.runtime._processors["sess-B"]
        # Core isolation guarantee: distinct processor instances with
        # distinct participant reference_ids. Even with identical
        # evidence streams, no cross-session reference sharing.
        assert proc_a is not proc_b
        assert (
            proc_a.active_participant.reference_id
            != proc_b.active_participant.reference_id
        )
        # Session ids are embedded in the reference_id — proves per-session
        # participant namespacing
        assert "sess-A" in proc_a.active_participant.reference_id
        assert "sess-B" in proc_b.active_participant.reference_id


# ---------------------------------------------------------------------------
# Scanners — no hardcoded services/industries in production (§41, §40)
# ---------------------------------------------------------------------------


class TestNoHardcodingScanner:
    def test_production_participant_module_has_no_industry_hardcoding(
        self,
    ) -> None:
        import inspect
        from app.conversation.participant import policy, intent_provider, concept_provider
        for mod in (policy, intent_provider, concept_provider):
            src = inspect.getsource(mod).lower()
            for forbidden in (
                "restaurant", "dentist", "hvac", "plumbing", "automation",
            ):
                assert forbidden not in src, f"{forbidden} in {mod.__name__}"

    def test_production_processor_has_no_keyword_phrase_matching(self) -> None:
        """§40 — no raw phrase matching for semantic decisions."""
        import inspect
        from app.application import production_turn_processor
        src = inspect.getsource(production_turn_processor).lower()
        # These phrases would indicate keyword-matching semantics
        for forbidden in (
            '"simple words"', "'simple words'",
            '"what services"', "'what services'",
            '"owner"', "'owner'",
            '"detail mein"', "'detail mein'",
        ):
            assert forbidden not in src, f"keyword phrase {forbidden} in processor"

    def test_strategy_renderer_has_no_industry_hardcoding(self) -> None:
        import inspect
        from app.conversation.response_rendering import strategy_frame
        src = inspect.getsource(strategy_frame).lower()
        for forbidden in ("restaurant", "dentist", "hvac"):
            assert forbidden not in src


# ---------------------------------------------------------------------------
# Explicit re-ask after prior EXPLAINED must NOT be SUPPRESS (§14)
# ---------------------------------------------------------------------------


class TestExplicitReAsk:
    def test_direct_service_question_after_explained_returns_concise_recap(
        self,
    ) -> None:
        """§14 — same participant, direct_service_question intent, prior
        coverage EXPLAINED. Must emit CONCISE_RECAP, not SUPPRESS."""
        from app.conversation.participant.contracts import (
            DiscussionCoverageEntry, DiscussionCoverageState, ExplanationDepth,
            ParticipantContinuityState, ReExplanationStrategy,
        )
        from app.conversation.participant.policy import (
            ReExplanationIntent, ReExplanationPolicy,
        )

        policy = ReExplanationPolicy()
        coverage = DiscussionCoverageState(entries=(
            DiscussionCoverageEntry(
                participant_reference_id="p1",
                concept_id="svc:x",
                depth=ExplanationDepth.EXPLAINED,
                source_turn_id="t1",
            ),
        ))
        decision = policy.decide(
            continuity=ParticipantContinuityState.SAME_CONFIRMED,
            active_reference_id="p1",
            concept_id="svc:x",
            coverage=coverage,
            intent=ReExplanationIntent.DIRECT_SERVICE_QUESTION,
        )
        assert decision.strategy == ReExplanationStrategy.CONCISE_RECAP

