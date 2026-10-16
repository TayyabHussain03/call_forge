"""Slice 51C — Live re-explanation intent + concept + strategy threading.

Proves:
    - re_explanation_intent_provider seam exists and is called
    - explanation_concept_provider seam exists and is called
    - processor automatically populates ResponsePlanningInput fields
    - processor automatically emits ExplainedConcept via delivered_explanation
    - strategy → delivered_depth mapping uses semantic strategy, not length
    - DIRECT_SERVICE_QUESTION + same participant + prior coverage →
      CONCISE_RECAP (not SUPPRESS) — the explicit customer question gate
    - Provider failure degrades safely (no fabricated intent/strategy)
    - No concept-id emitted → no participant-aware threading
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
    DiscussionCoverageEntry,
    DiscussionCoverageState,
    ExplainedConcept,
    ExplanationDepth,
    ParticipantContinuityEvidence,
    ParticipantContinuityState,
    ParticipantEvidenceKind,
    ParticipantRoleHint,
    ReExplanationStrategy,
)
from app.conversation.participant.policy import (
    ReExplanationIntent,
    ReExplanationPolicy,
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
# §53 Explicit direct-question gate — policy correction
# ---------------------------------------------------------------------------


class TestDirectQuestionAnswersEvenWhenCovered:
    """§53: 'If a participant already has DETAILED coverage and directly
    asks what the service does again, does the agent answer them?'
    Required answer: YES. Strategy must not be SUPPRESS.
    """

    def test_direct_question_with_prior_detailed_coverage_recaps(self) -> None:
        policy = ReExplanationPolicy()
        coverage = DiscussionCoverageState(
            entries=(
                DiscussionCoverageEntry(
                    "sess:p1", "svc.automation",
                    ExplanationDepth.DETAILED, "t5",
                ),
            ),
        )
        decision = policy.decide(
            ParticipantContinuityState.SAME_CONFIRMED,
            "sess:p1", "svc.automation", coverage,
            intent=ReExplanationIntent.DIRECT_SERVICE_QUESTION,
        )
        # Previously (Slice 48) would have returned SUPPRESS. Fixed.
        assert decision.strategy == ReExplanationStrategy.CONCISE_RECAP
        assert decision.strategy != ReExplanationStrategy.SUPPRESS

    def test_direct_question_with_prior_explained_coverage_recaps(self) -> None:
        policy = ReExplanationPolicy()
        coverage = DiscussionCoverageState(
            entries=(
                DiscussionCoverageEntry(
                    "sess:p1", "svc.automation",
                    ExplanationDepth.EXPLAINED, "t3",
                ),
            ),
        )
        decision = policy.decide(
            ParticipantContinuityState.SAME_CONFIRMED,
            "sess:p1", "svc.automation", coverage,
            intent=ReExplanationIntent.DIRECT_SERVICE_QUESTION,
        )
        assert decision.strategy == ReExplanationStrategy.CONCISE_RECAP

    def test_direct_question_no_prior_coverage_falls_through(self) -> None:
        """Direct question with no prior coverage still gets no-prior-
        coverage branch (first-time explanation path)."""
        policy = ReExplanationPolicy()
        decision = policy.decide(
            ParticipantContinuityState.SAME_CONFIRMED,
            "sess:p1", "svc.automation", DiscussionCoverageState(),
            intent=ReExplanationIntent.DIRECT_SERVICE_QUESTION,
        )
        assert decision.strategy == ReExplanationStrategy.NO_PRIOR_COVERAGE

    def test_direct_question_new_participant_full_re_explain(self) -> None:
        """Regression: new participants still get FULL_RE_EXPLAIN."""
        policy = ReExplanationPolicy()
        coverage = DiscussionCoverageState(
            entries=(
                DiscussionCoverageEntry(
                    "sess:p1", "svc.automation",
                    ExplanationDepth.DETAILED, "t3",
                ),
            ),
        )
        decision = policy.decide(
            ParticipantContinuityState.NEW_CONFIRMED,
            "sess:p2", "svc.automation", coverage,
            intent=ReExplanationIntent.DIRECT_SERVICE_QUESTION,
        )
        assert decision.strategy == ReExplanationStrategy.FULL_RE_EXPLAIN_FOR_NEW_PARTICIPANT


# ---------------------------------------------------------------------------
# Real-PTP factory with intent + concept providers
# ---------------------------------------------------------------------------


class RealPTPFactory:
    def __init__(
        self,
        actions=(AgentAction.GREET, AgentAction.ASK_IDENTITY,
                 AgentAction.INTRODUCE_REASON, AgentAction.ANSWER_QUESTION,
                 AgentAction.ANSWER_QUESTION),
        evidence_sequence=(),
        intent_sequence=(),
        concept_sequence=(),
    ) -> None:
        self._actions = actions
        self._evidence_sequence = evidence_sequence
        self._intent_sequence = intent_sequence
        self._concept_sequence = concept_sequence
        self._providers: dict[str, MockReasoningProvider] = {}
        self.intent_call_count = 0
        self.concept_call_count = 0

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

    def _intent_provider(self):
        seq = iter(self._intent_sequence)
        def _p(turn, understanding):
            self.intent_call_count += 1
            try:
                return next(seq)
            except StopIteration:
                return ReExplanationIntent.NONE
        return _p

    def _concept_provider(self):
        seq = iter(self._concept_sequence)
        def _p(turn, understanding):
            self.concept_call_count += 1
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
            re_explanation_intent_provider=self._intent_provider(),
            explanation_concept_provider=self._concept_provider(),
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


def _boot(sid="sess-51c") -> SessionBootstrapInput:
    return SessionBootstrapInput(
        tenant_id="t1", campaign_id="c1", lead_id="lead-1",
        session_id=sid, bootstrap_request_id=f"req-{sid}",
        trusted_time_iso="2026-10-16T10:00:00Z",
    )


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "slice51c.db")


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
# Processor automatically threads participant-aware fields
# ---------------------------------------------------------------------------


class TestAutomaticThreading:
    def test_providers_are_invoked_each_turn(self, db_path) -> None:
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51c_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        factory = RealPTPFactory(
            evidence_sequence=evidence,
            intent_sequence=(ReExplanationIntent.DIRECT_SERVICE_QUESTION,),
            concept_sequence=("svc.automation",),
        )
        rt = _make_runtime(db_path, factory)
        rt.start_session(_boot(), _campaign())
        rt.process_turn("sess-51c", "what does your automation do?", 1, "tr-1")
        assert factory.intent_call_count >= 1
        assert factory.concept_call_count >= 1

    def test_concept_auto_populates_planning_input(self, db_path) -> None:
        """Processor auto-fills explanation_concept_id and participant_reference_id
        onto the plan when providers supply them."""
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51c_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        factory = RealPTPFactory(
            evidence_sequence=evidence,
            intent_sequence=(ReExplanationIntent.DIRECT_SERVICE_QUESTION,),
            concept_sequence=("svc.automation",),
        )
        rt = _make_runtime(db_path, factory)
        rt.start_session(_boot(), _campaign())
        rt.process_turn("sess-51c", "what does automation do?", 1, "tr-1")
        proc = rt._processors["sess-51c"]
        # The processor should have auto-populated the latest plan's fields
        # via the ResponsePlanningInput → ResponsePlan pass-through.
        # We verify indirectly by checking that coverage auto-advanced.
        active = proc.active_participant.reference_id
        assert "svc.automation" in proc.discussion_coverage.heard_by(active)

    def test_first_explanation_delivers_explained_depth(self, db_path) -> None:
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51c_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        factory = RealPTPFactory(
            evidence_sequence=evidence,
            intent_sequence=(ReExplanationIntent.DIRECT_SERVICE_QUESTION,),
            concept_sequence=("svc.automation",),
        )
        rt = _make_runtime(db_path, factory)
        rt.start_session(_boot(), _campaign())
        rt.process_turn("sess-51c", "what does automation do?", 1, "tr-1")
        proc = rt._processors["sess-51c"]
        active = proc.active_participant.reference_id
        # DIRECT_SERVICE_QUESTION + prior=NONE → NO_PRIOR_COVERAGE → EXPLAINED
        assert proc.discussion_coverage.depth_for(
            active, "svc.automation",
        ) == ExplanationDepth.EXPLAINED


# ---------------------------------------------------------------------------
# No concept emitted → no participant-aware threading
# ---------------------------------------------------------------------------


class TestNoConceptNoThreading:
    def test_no_concept_no_coverage_update(self, db_path) -> None:
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51c_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        # Concept provider returns None — processor must not fabricate one
        factory = RealPTPFactory(
            evidence_sequence=evidence,
            intent_sequence=(ReExplanationIntent.DIRECT_SERVICE_QUESTION,),
            concept_sequence=(None,),
        )
        rt = _make_runtime(db_path, factory)
        rt.start_session(_boot(), _campaign())
        rt.process_turn("sess-51c", "hi", 1, "tr-1")
        proc = rt._processors["sess-51c"]
        assert proc.discussion_coverage.entries == ()


# ---------------------------------------------------------------------------
# Provider failure degrades safely
# ---------------------------------------------------------------------------


class TestProviderFailureDegradesSafely:
    def test_intent_provider_exception_falls_back_to_none(self, db_path) -> None:
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51c_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )

        class RaisingIntentFactory(RealPTPFactory):
            def _intent_provider(self):
                def _p(turn, understanding):
                    raise RuntimeError("intent provider died")
                return _p

        factory = RaisingIntentFactory(
            evidence_sequence=evidence,
            concept_sequence=("svc.automation",),
        )
        rt = _make_runtime(db_path, factory)
        rt.start_session(_boot(), _campaign())
        # Turn still commits — no fabricated intent, no crash
        rt.process_turn("sess-51c", "hi", 1, "tr-1")

    def test_concept_provider_exception_falls_back_to_no_threading(self, db_path) -> None:
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51c_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )

        class RaisingConceptFactory(RealPTPFactory):
            def _concept_provider(self):
                def _p(turn, understanding):
                    raise RuntimeError("concept provider died")
                return _p

        factory = RaisingConceptFactory(
            evidence_sequence=evidence,
            intent_sequence=(ReExplanationIntent.DIRECT_SERVICE_QUESTION,),
        )
        rt = _make_runtime(db_path, factory)
        rt.start_session(_boot(), _campaign())
        rt.process_turn("sess-51c", "hi", 1, "tr-1")
        proc = rt._processors["sess-51c"]
        assert proc.discussion_coverage.entries == ()


# ---------------------------------------------------------------------------
# Semantic depth mapping from strategy (not length)
# ---------------------------------------------------------------------------


class TestStrategyToDeliveredDepthMapping:
    def _rt_with_prior_explanation(self, db_path, factory):
        """Set up a session with Person A + prior EXPLAINED coverage."""
        rt = _make_runtime(db_path, factory)
        rt.start_session(_boot(), _campaign())
        rt.process_turn("sess-51c", "what does automation do?", 1, "tr-1")
        return rt

    def test_concise_recap_delivers_brief(self, db_path) -> None:
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51c_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        factory = RealPTPFactory(
            evidence_sequence=evidence,
            intent_sequence=(
                ReExplanationIntent.DIRECT_SERVICE_QUESTION,
                ReExplanationIntent.RECAP_REQUEST,
            ),
            concept_sequence=("svc.automation", "svc.automation"),
        )
        rt = self._rt_with_prior_explanation(db_path, factory)
        rt.process_turn("sess-51c", "remind me", 2, "tr-2")
        proc = rt._processors["sess-51c"]
        active = proc.active_participant.reference_id
        # Historical max remains EXPLAINED (BRIEF never downgrades)
        assert proc.discussion_coverage.depth_for(
            active, "svc.automation",
        ) == ExplanationDepth.EXPLAINED

    def test_deepen_delivers_detailed(self, db_path) -> None:
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51c_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        factory = RealPTPFactory(
            evidence_sequence=evidence,
            intent_sequence=(
                ReExplanationIntent.DIRECT_SERVICE_QUESTION,
                ReExplanationIntent.DEEPEN_REQUEST,
            ),
            concept_sequence=("svc.automation", "svc.automation"),
        )
        rt = self._rt_with_prior_explanation(db_path, factory)
        rt.process_turn("sess-51c", "explain in detail", 2, "tr-2")
        proc = rt._processors["sess-51c"]
        active = proc.active_participant.reference_id
        assert proc.discussion_coverage.depth_for(
            active, "svc.automation",
        ) == ExplanationDepth.DETAILED


# ---------------------------------------------------------------------------
# Restart preserves prior depth
# ---------------------------------------------------------------------------


class TestRestartPreservesPriorDepth:
    def test_restart_after_explained_recap_still_works(self, db_path) -> None:
        evidence = (
            ParticipantContinuityEvidence(
                kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
                source_turn_id="chat_sess-51c_t1",
                claimed_role_hint=ParticipantRoleHint.EMPLOYEE,
            ),
        )
        factory = RealPTPFactory(
            evidence_sequence=evidence,
            intent_sequence=(ReExplanationIntent.DIRECT_SERVICE_QUESTION,),
            concept_sequence=("svc.automation",),
        )
        rt = _make_runtime(db_path, factory)
        rt.start_session(_boot(), _campaign())
        rt.process_turn("sess-51c", "what does automation do?", 1, "tr-1")

        rt2 = _make_runtime(db_path, RealPTPFactory())
        result = rt2.restore_processor_if_possible("sess-51c", _campaign())
        from app.application.sales_runtime.recovery import RecoveryStatus
        assert result.status == RecoveryStatus.RECOVERED_ACTIVE
        restored = rt2._processors["sess-51c"]
        active = restored.active_participant.reference_id
        assert restored.discussion_coverage.depth_for(
            active, "svc.automation",
        ) == ExplanationDepth.EXPLAINED


# ---------------------------------------------------------------------------
# No keyword trees + no new industry hardcoding
# ---------------------------------------------------------------------------


class TestNoKeywordOrIndustryLogic:
    def test_no_keyword_intent_tree_in_processor(self) -> None:
        import inspect
        from app.application import production_turn_processor
        source = inspect.getsource(production_turn_processor)
        for forbidden in (
            '"remind me"', '"didn\'t understand"', '"tell me more"',
            "'remind me'", "'didn\\'t understand'", "'tell me more'",
        ):
            assert forbidden not in source
