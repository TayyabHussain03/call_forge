"""Slice 51D — Per-participant Prospect Intelligence archive.

Proves:
    - Serializer round-trip for the archive payload
    - On NEW_CONFIRMED: Person A's live ProspectIntelligenceSnapshot is
      archived under their reference ID; the live snapshot is reset to a
      fresh default for Person B — NO inheritance.
    - Archive survives restart via the new PARTICIPANT_PROSPECT_ARCHIVE_KEY
      payload.
    - Business/BCI state remains shared across the switch.
"""

from __future__ import annotations

import pytest

from app.application.sales_runtime.persistent_runtime import PersistentSalesRuntime
from app.application.sales_runtime.recovery import RecoveryStatus
from app.application.sales_runtime.serializers import (
    BUSINESS_CONVERSATION_KEY,
    DISCUSSION_COVERAGE_KEY,
    LANGUAGE_PROFILE_KEY,
    PARTICIPANT_PROSPECT_ARCHIVE_KEY,
    PARTICIPANT_STATE_KEY,
    PROSPECT_INTELLIGENCE_KEY,
    QUALIFICATION_KEY,
    BusinessConversationSerializer,
    DiscussionCoverageSerializer,
    LanguageProfileSerializer,
    ParticipantProspectArchiveSerializer,
    ParticipantStateSerializer,
    ProspectIntelligenceSerializer,
    QualificationSerializer,
)
from app.application.sales_runtime.manifest import default_manifest
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
from app.conversation.prospect_intelligence.contracts import (
    EvidenceProvenance,
    EvidenceSourceKind,
    ObservedProspectFacts,
    ObservedValue,
    ProspectIntelligenceSnapshot,
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
from app.persistence.errors import InvalidPersistedPayloadError
from app.persistence.sqlite.connection import SqliteConnection
from app.precall.contracts import PreCallDiscoveryPolicy, ServiceDiscoveryRule
from app.runtime.chat.bootstrap.contracts import SessionBootstrapInput


# ---------------------------------------------------------------------------
# Serializer round-trip
# ---------------------------------------------------------------------------


class TestArchiveSerializer:
    def test_empty_archive_roundtrip(self) -> None:
        ser = ParticipantProspectArchiveSerializer()
        env = ser.serialize(())
        assert ser.deserialize(env) == ()

    def test_populated_archive_roundtrip(self) -> None:
        ser = ParticipantProspectArchiveSerializer()
        prov = EvidenceProvenance("t1", EvidenceSourceKind.EXPLICIT_STATEMENT)
        snap_a = ProspectIntelligenceSnapshot(
            observed=ObservedProspectFacts(
                explicit_interest_signal=ObservedValue(value=True, provenance=prov),
            ),
        )
        snap_b = ProspectIntelligenceSnapshot(
            observed=ObservedProspectFacts(
                explicit_busy_signal=ObservedValue(value=True, provenance=prov),
            ),
        )
        archive = (("p1", snap_a), ("p2", snap_b))
        env = ser.serialize(archive)
        back = ser.deserialize(env)
        assert back == archive
        # Verify interest state preserved and isolated
        assert back[0][1].observed.explicit_interest_signal.value is True
        assert back[1][1].observed.explicit_busy_signal.value is True

    def test_deterministic_output(self) -> None:
        ser = ParticipantProspectArchiveSerializer()
        # Order shouldn't matter — sorted by participant_reference_id
        a = ser.serialize((("p2", ProspectIntelligenceSnapshot()),
                           ("p1", ProspectIntelligenceSnapshot())))
        b = ser.serialize((("p1", ProspectIntelligenceSnapshot()),
                           ("p2", ProspectIntelligenceSnapshot())))
        assert a == b

    def test_malformed_envelope_rejected(self) -> None:
        ser = ParticipantProspectArchiveSerializer()
        with pytest.raises(InvalidPersistedPayloadError):
            ser.deserialize("{not valid json")


# ---------------------------------------------------------------------------
# Manifest knows the new payload
# ---------------------------------------------------------------------------


class TestManifestKnowsArchive:
    def test_archive_key_is_optional_derived(self) -> None:
        m = default_manifest()
        assert not m.is_required(PARTICIPANT_PROSPECT_ARCHIVE_KEY)
        assert m.is_supported(PARTICIPANT_PROSPECT_ARCHIVE_KEY, 1)

    def test_archive_key_in_optional_set(self) -> None:
        m = default_manifest()
        assert PARTICIPANT_PROSPECT_ARCHIVE_KEY in m.optional_derived


# ---------------------------------------------------------------------------
# Real PTP: NEW_CONFIRMED archives Person A, resets for Person B
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
        archive_env = snapshot.payload(PARTICIPANT_PROSPECT_ARCHIVE_KEY)
        prospect_env = snapshot.payload(PROSPECT_INTELLIGENCE_KEY)
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
        prospect_archive = (
            ParticipantProspectArchiveSerializer().deserialize(archive_env)
            if archive_env else ()
        )
        prospect = (
            ProspectIntelligenceSerializer().deserialize(prospect_env)
            if prospect_env else None
        )
        kwargs = dict(
            initial_business_conversation=bci,
            initial_language_profile=language,
            initial_qualification=qual,
            initial_active_participant=active,
            initial_archived_participants=archived,
            initial_discussion_coverage=coverage,
            initial_participant_prospect_archive=prospect_archive,
        )
        if prospect is not None:
            kwargs["initial_prospect_intelligence"] = prospect
        return self._build(
            session, initial_state=snapshot.conversation_state, **kwargs,
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


def _boot(sid="sess-51d") -> SessionBootstrapInput:
    return SessionBootstrapInput(
        tenant_id="t1", campaign_id="c1", lead_id="lead-1",
        session_id=sid, bootstrap_request_id=f"req-{sid}",
        trusted_time_iso="2026-10-19T10:00:00Z",
    )


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "slice51d.db")


def _make_runtime(db_path, factory=None):
    conn = SqliteConnection(db_path)
    f = factory or RealPTPFactory()
    rt = PersistentSalesRuntime(conn, f)
    if rt.leads.get("t1", "c1", "lead-1") is None:
        rt.leads.save(_lead())
    if rt.queue.get("t1", "c1", "lead-1") is None:
        rt.queue.save(_queue())
    if rt.policies.resolve("c1") is None:
        rt.policies.register_default("c1", _policy_cfg())
    return rt


# ---------------------------------------------------------------------------
# Core correctness: Person B does NOT inherit Person A's prospect state
# ---------------------------------------------------------------------------


class TestProspectIntelligenceIsolation:
    def _run_handoff_scenario(self, db_path, interest=True, busy=False):
        """Set up Person A → explicit handoff → Person B, with Person A
        carrying some prospect-state signals before the handoff."""
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51d_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_HANDOFF,
                source_turn_id="chat_sess-51d_t2",
                claimed_role_hint=ParticipantRoleHint.OWNER,
            ),
        )
        rt = _make_runtime(db_path, RealPTPFactory(evidence_sequence=evidence))
        rt.start_session(_boot(), _campaign())
        rt.process_turn("sess-51d", "I handle marketing", 1, "tr-1")

        # Simulate Person A establishing prospect state
        proc = rt._processors["sess-51d"]
        prov = EvidenceProvenance("t1", EvidenceSourceKind.EXPLICIT_STATEMENT)
        proc._prospect_intelligence = ProspectIntelligenceSnapshot(
            observed=ObservedProspectFacts(
                explicit_interest_signal=ObservedValue(value=interest, provenance=prov),
                explicit_busy_signal=ObservedValue(value=busy, provenance=prov),
            ),
        )
        return rt, proc

    def test_person_a_interest_not_inherited_by_b(self, db_path) -> None:
        rt, proc = self._run_handoff_scenario(db_path, interest=True)
        person_a_ref = proc.active_participant.reference_id

        # Handoff — Person B becomes active
        rt.process_turn("sess-51d", "my colleague took over", 2, "tr-2")
        proc = rt._processors["sess-51d"]
        person_b_ref = proc.active_participant.reference_id
        assert person_b_ref != person_a_ref

        # Person B's live prospect intelligence must be FRESH — no interest signal
        assert proc.prospect_intelligence.observed.explicit_interest_signal is None
        assert proc.prospect_intelligence.observed.explicit_busy_signal is None

    def test_person_a_prospect_archived_under_their_ref(self, db_path) -> None:
        rt, proc = self._run_handoff_scenario(db_path, interest=True)
        person_a_ref = proc.active_participant.reference_id

        rt.process_turn("sess-51d", "my colleague took over", 2, "tr-2")
        proc = rt._processors["sess-51d"]

        # Archive carries Person A's state under their reference_id
        archive = {pid: snap for pid, snap in proc.participant_prospect_archive}
        assert person_a_ref in archive
        assert archive[person_a_ref].observed.explicit_interest_signal is not None
        assert archive[person_a_ref].observed.explicit_interest_signal.value is True

    def test_person_a_objections_not_inherited_by_b(self, db_path) -> None:
        from app.conversation.prospect_intelligence.contracts import ObjectionType
        rt, proc = self._run_handoff_scenario(db_path)
        prov = EvidenceProvenance("t1", EvidenceSourceKind.EXPLICIT_STATEMENT)
        # Add an objection to Person A
        proc._prospect_intelligence = ProspectIntelligenceSnapshot(
            observed=ObservedProspectFacts(
                explicit_objection=ObservedValue(
                    value=ObjectionType.PRICE, provenance=prov,
                ),
            ),
        )

        rt.process_turn("sess-51d", "my colleague took over", 2, "tr-2")
        proc = rt._processors["sess-51d"]
        # Person B has NO inherited objection
        assert proc.prospect_intelligence.observed.explicit_objection is None


# ---------------------------------------------------------------------------
# Restart preserves archive
# ---------------------------------------------------------------------------


class TestArchiveSurvivesRestart:
    def test_archive_survives_restart(self, db_path) -> None:
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51d_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_HANDOFF,
                source_turn_id="chat_sess-51d_t2",
                claimed_role_hint=ParticipantRoleHint.OWNER,
            ),
        )
        rt = _make_runtime(db_path, RealPTPFactory(evidence_sequence=evidence))
        rt.start_session(_boot(), _campaign())
        rt.process_turn("sess-51d", "I handle marketing", 1, "tr-1")

        # Give Person A some state before handoff
        proc = rt._processors["sess-51d"]
        prov = EvidenceProvenance("t1", EvidenceSourceKind.EXPLICIT_STATEMENT)
        proc._prospect_intelligence = ProspectIntelligenceSnapshot(
            observed=ObservedProspectFacts(
                explicit_interest_signal=ObservedValue(value=True, provenance=prov),
            ),
        )
        person_a_ref = proc.active_participant.reference_id

        rt.process_turn("sess-51d", "my colleague took over", 2, "tr-2")

        # Snapshot should carry the archive payload
        snap = rt.snapshots.load("sess-51d")
        payload_keys = {k for k, _ in snap.persisted_contract_payloads}
        assert PARTICIPANT_PROSPECT_ARCHIVE_KEY in payload_keys

        # Hard restart
        rt2 = _make_runtime(db_path, RealPTPFactory())
        result = rt2.restore_processor_if_possible("sess-51d", _campaign())
        assert result.status == RecoveryStatus.RECOVERED_ACTIVE
        restored = rt2._processors["sess-51d"]

        # Archive restored
        archive = {pid: snap for pid, snap in restored.participant_prospect_archive}
        assert person_a_ref in archive
        assert archive[person_a_ref].observed.explicit_interest_signal.value is True

        # Live prospect intelligence (Person B) is still fresh
        assert restored.prospect_intelligence.observed.explicit_interest_signal is None


# ---------------------------------------------------------------------------
# Business state remains shared across the switch
# ---------------------------------------------------------------------------


class TestBusinessStatePreservedAcrossSwitch:
    def test_bci_unchanged_by_prospect_switch(self, db_path) -> None:
        from app.conversation.business_conversation.contracts import (
            BusinessFactKind, ObservedBusinessFact,
        )
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51d_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_HANDOFF,
                source_turn_id="chat_sess-51d_t2",
                claimed_role_hint=ParticipantRoleHint.OWNER,
            ),
        )
        seeded_bci = BusinessConversationSnapshot(
            facts=(
                ObservedBusinessFact(
                    BusinessFactKind.WEBSITE, "existing website", "t0",
                ),
            ),
        )

        class SeededFactory(RealPTPFactory):
            def create_for_session(self, session, campaign, policy):
                return self._build(
                    session, initial_state=ConversationState.NEW_CALL,
                    initial_business_conversation=seeded_bci,
                )

        rt = _make_runtime(
            db_path, SeededFactory(evidence_sequence=evidence),
        )
        rt.start_session(_boot(), _campaign())
        rt.process_turn("sess-51d", "I handle marketing", 1, "tr-1")
        rt.process_turn("sess-51d", "owner joined", 2, "tr-2")

        proc = rt._processors["sess-51d"]
        fact_values = {f.value for f in proc._business_conversation.facts}
        # Business truth preserved despite handoff + prospect reset
        assert "existing website" in fact_values


# ---------------------------------------------------------------------------
# No reasoning imports in archive serializer
# ---------------------------------------------------------------------------


class TestNoReasoningInArchiveSerializer:
    def test_no_forbidden_imports(self) -> None:
        import inspect
        from app.application.sales_runtime.serializers import participant_prospect
        source = inspect.getsource(participant_prospect)
        for forbidden in (
            "BrainOrchestrator", "ReasoningProvider", "StateMachine",
            "ActionValidator", "ResponseRenderer",
        ):
            assert forbidden not in source
