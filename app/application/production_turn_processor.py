"""Compose the existing authoritative and communication pipelines per call."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from app.brain.authority.models import AuthorityPolicy
from app.brain.budget.evaluator import TrustedExitSignals
from app.brain.budget.models import BudgetPolicy
from app.brain.contracts import BudgetState, BusinessIntelligenceSnapshot
from app.brain.orchestrator.orchestrator import (
    BrainOrchestrator,
    BrainSnapshotInput,
    ConversationTerminationStatus,
    TurnResult,
    TurnStageOutcome,
)
from app.brain.scope.models import ScopePolicy
from app.contracts.conversation_context import ConversationContext
from app.conversation.guardrails.priority import TrustedPriorityOutcome
from app.conversation.context.builder import LeanContextBuildInput, LeanContextBuilder
from app.conversation.context.contracts import (
    ApprovedEvidenceItem,
    LeanTurnContext,
    RecentTurn,
    TurnSpeaker,
)
from app.conversation.consultative.contracts import (
    ConsultativeConversationDecision,
    ConsultativeDecisionInput,
    ConsultativeTurnSignals,
    ProblemEvidence,
    ProblemField,
    ProspectProblem,
    ServiceAnswerContext,
    ServiceFitDecision,
    ServiceFitStatus,
)
from app.conversation.consultative.engine import ConsultativeDecisionEngine
from app.conversation.consultative.problem import ProblemModelUpdater
from app.conversation.consultative.service_relevance import ServiceRelevanceResolver
from app.conversation.escalation.contracts import (
    AvailableEscalationCapabilities,
    EscalationDecision,
    EscalationRequest,
    GracefulEscalationInput,
    KnowledgeRequestKind,
)
from app.conversation.escalation.policy import GracefulEscalationPolicy
from app.conversation.prospect_intelligence.contracts import (
    ProspectEvidence,
    ProspectIntelligenceSnapshot,
)
from app.conversation.prospect_intelligence.updater import ProspectIntelligenceUpdater
from app.conversation.response_planning.contracts import (
    AddresseeStatus,
    AuthoritativeResultKind,
    ExplanationNeed,
    InterruptionCategory,
    QuestionStrategy,
    ResponsePlanningInput,
)
from app.conversation.response_planning.planner import ResponsePlanner
from app.conversation.response_rendering.contracts import (
    ContactConfirmationStatus,
    ResponseRenderInput,
    ResponseRenderingBudget,
    TrustedRenderingContext,
)
from app.conversation.response_rendering.renderer import ResponseRenderer
from app.conversation.realization.contracts import LeanContextView
from app.conversation.sales_cognition.contracts import (
    CognitionSignals,
    SalesCognitionInput,
    SalesConversationGuidance,
)
from app.conversation.sales_cognition.engine import HumanSalesCognitionEngine
from app.conversation.sales_playbook.contracts import (
    BusinessContext,
    OpportunityGuidance,
    SalesPlaybookInput,
    ServicePlaybook,
)
from app.conversation.sales_playbook.engine import SalesPlaybookIntelligenceEngine
from app.conversation.business_conversation.contracts import (
    BusinessConversationEvidence,
    BusinessConversationSnapshot,
)
from app.conversation.business_conversation.engine import (
    BusinessConversationIntelligenceEngine,
)
from app.conversation.business_diagnostic.contracts import (
    BusinessDiagnosticInput,
    BusinessDiagnosticSnapshot,
)
from app.conversation.business_diagnostic.engine import BusinessDiagnosticEngine
from app.conversation.conversation_steering.contracts import (
    ConversationPrioritySnapshot,
    ConversationSteeringInput,
)
from app.conversation.conversation_steering.engine import ConversationSteeringEngine
from app.conversation.qualification.contracts import QualificationConfiguration, QualificationEvidence, QualificationSnapshot
from app.conversation.qualification.engine import ProgressiveQualificationEngine
from app.conversation.business_memory.contracts import BusinessMemoryScope, BusinessMentalModelSnapshot
from app.conversation.business_memory.engine import BusinessMentalModelEngine
from app.knowledge.retrieval_planning.contracts import KnowledgeNeed, KnowledgeRetrievalPolicy, RetrievalPlan, RetrievalPlanningContext
from app.knowledge.retrieval_planning.planner import KnowledgeRetrievalPlanner
from app.knowledge.hybrid_retrieval.contracts import RetrievalCandidateSet
from app.knowledge.hybrid_retrieval.engine import HybridRetrievalEngine
from app.knowledge.evidence_validation.contracts import CommercialDisclosurePolicy, EvidenceValidationPolicy, EvidenceValidationResult
from app.knowledge.evidence_validation.engine import EvidenceValidationEngine
from app.conversation.strategy.contracts import (
    ConversationStrategy,
    ConversationStrategyHint,
    ConversationStrategyInput,
    SalesStage,
)
from app.conversation.strategy.engine import ConversationStrategyEngine
from app.conversation.supervisor.buffer import (
    BufferWriteOutcome,
    StrategyBuffer,
    StrategyBufferSnapshot,
)
from app.conversation.supervisor.contracts import SupervisorInput, SupervisorInsight
from app.conversation.understanding.contracts import (
    FreeTextUnderstanding,
    FreeTextUnderstandingInput,
    LanguageProfile,
)
from app.conversation.understanding.mapper import UnderstandingEvidenceMapper
from app.conversation.understanding.provider import (
    FreeTextUnderstandingProvider,
    UnderstandingError,
)
from app.core.constants import AgentAction, ConversationState
from app.llm.providers.contact_understanding_provider import (
    ContactUnderstandingError,
    ContactUnderstandingProvider,
    UnderstandingInput,
)
from app.runtime.contracts import (
    CoordinatedTurnOutput,
    CoordinatedUserTurn,
    TurnProcessor,
)

TrustedPriorityProvider = Callable[[CoordinatedUserTurn], TrustedPriorityOutcome]
RenderingContextProvider = Callable[
    [CoordinatedUserTurn, AuthoritativeResultKind, ConversationContext],
    TrustedRenderingContext,
]
StrategyInputProvider = Callable[
    [
        CoordinatedUserTurn,
        ConversationContext,
        SalesStage,
        ConversationState,
        ProspectIntelligenceSnapshot,
        ConversationStrategyHint | None,
    ],
    ConversationStrategyInput,
]
ProspectEvidenceProvider = Callable[
    [CoordinatedUserTurn, ProspectIntelligenceSnapshot], ProspectEvidence | None
]
EscalationRequestProvider = Callable[
    [CoordinatedUserTurn, LeanTurnContext], EscalationRequest
]
ProblemEvidenceProvider = Callable[
    [CoordinatedUserTurn, ProspectProblem, ProspectIntelligenceSnapshot],
    ProblemEvidence | None,
]
ConsultativeSignalProvider = Callable[
    [CoordinatedUserTurn, LeanTurnContext], ConsultativeTurnSignals
]
CognitionSignalProvider = Callable[
    [CoordinatedUserTurn, LeanTurnContext], CognitionSignals
]
PlaybookContextProvider = Callable[
    [CoordinatedUserTurn, LeanTurnContext], BusinessContext
]
BusinessConversationEvidenceProvider = Callable[
    [CoordinatedUserTurn, LeanTurnContext, FreeTextUnderstanding | None],
    BusinessConversationEvidence | None,
]


class ProductionTurnProcessor(TurnProcessor):
    """Call-scoped composition; delegates every decision to existing layers."""

    _CONTACT_ACTIONS = {
        AgentAction.ASK_EMAIL,
        AgentAction.CONFIRM_CONTACT,
        AgentAction.CLARIFY_CONTACT,
    }

    def __init__(
        self,
        *,
        orchestrator: BrainOrchestrator,
        response_planner: ResponsePlanner,
        response_renderer: ResponseRenderer,
        initial_context: ConversationContext,
        budget_policy: BudgetPolicy,
        scope_policy: ScopePolicy,
        authority_policy: AuthorityPolicy,
        trusted_priority_provider: TrustedPriorityProvider | None = None,
        contact_understanding_provider: ContactUnderstandingProvider | None = None,
        rendering_context_provider: RenderingContextProvider | None = None,
        strategy_engine: ConversationStrategyEngine | None = None,
        strategy_input_provider: StrategyInputProvider | None = None,
        initial_prospect_intelligence: ProspectIntelligenceSnapshot | None = None,
        prospect_intelligence_updater: ProspectIntelligenceUpdater | None = None,
        prospect_evidence_provider: ProspectEvidenceProvider | None = None,
        strategy_buffer: StrategyBuffer | None = None,
        lean_context_builder: LeanContextBuilder | None = None,
        approved_evidence: tuple[ApprovedEvidenceItem, ...] = (),
        business_intelligence: BusinessIntelligenceSnapshot | None = None,
        relevant_business_fields: tuple[str, ...] = (),
        campaign_goal: str | None = None,
        discovery_priorities: tuple[str, ...] = (),
        escalation_policy: GracefulEscalationPolicy | None = None,
        escalation_capabilities: AvailableEscalationCapabilities = (
            AvailableEscalationCapabilities()
        ),
        escalation_request_provider: EscalationRequestProvider | None = None,
        free_text_understanding_provider: FreeTextUnderstandingProvider | None = None,
        understanding_evidence_mapper: UnderstandingEvidenceMapper | None = None,
        initial_problem: ProspectProblem = ProspectProblem(),
        problem_evidence_provider: ProblemEvidenceProvider | None = None,
        problem_model_updater: ProblemModelUpdater | None = None,
        service_relevance_resolver: ServiceRelevanceResolver | None = None,
        consultative_decision_engine: ConsultativeDecisionEngine | None = None,
        consultative_signal_provider: ConsultativeSignalProvider | None = None,
        sales_cognition_engine: HumanSalesCognitionEngine | None = None,
        cognition_signal_provider: CognitionSignalProvider | None = None,
        sales_playbook_engine: SalesPlaybookIntelligenceEngine | None = None,
        service_playbooks: tuple[ServicePlaybook, ...] = (),
        playbook_context_provider: PlaybookContextProvider | None = None,
        business_conversation_engine: BusinessConversationIntelligenceEngine | None = None,
        business_conversation_evidence_provider: BusinessConversationEvidenceProvider | None = None,
        initial_business_conversation: BusinessConversationSnapshot | None = None,
        business_diagnostic_engine: BusinessDiagnosticEngine | None = None,
        conversation_steering_engine: ConversationSteeringEngine | None = None,
        qualification_configuration: QualificationConfiguration | None = None,
        qualification_evidence_provider: Callable[[CoordinatedUserTurn, LeanTurnContext], QualificationEvidence | None] | None = None,
        initial_qualification: QualificationSnapshot | None = None,
        initial_language_profile: LanguageProfile | None = None,
        # Slice 49 — participant-aware wiring. All additive and optional.
        participant_evidence_provider=None,
        initial_active_participant=None,
        initial_archived_participants=(),
        initial_discussion_coverage=None,
        # Slice 51C — live re-explanation intent + concept provider seams.
        # Both are optional/additive. When supplied, the processor
        # automatically resolves ReExplanationStrategy and emits
        # delivered_explanation on the plan — no manual caller injection.
        re_explanation_intent_provider=None,
        explanation_concept_provider=None,
        # Slice 51D — per-participant Prospect Intelligence archive.
        # Keyed by participant_reference_id; empty by default. Carries
        # participant-scoped prospect state across restart so Person B
        # cannot inherit Person A's interest/objection/busy/role.
        initial_participant_prospect_archive=(),
        # Slice 51H — persisted reasoning-call counter so restart cannot
        # grant extra Brain reasoning capacity. Counts committed calls.
        initial_reasoning_calls: int = 0,
        business_memory_scope: BusinessMemoryScope | None = None,
        knowledge_retrieval_policy: KnowledgeRetrievalPolicy | None = None,
        knowledge_need_provider: Callable[[CoordinatedUserTurn, LeanTurnContext], tuple[KnowledgeNeed, ...]] | None = None,
        hybrid_retrieval_engine: HybridRetrievalEngine | None = None,
        evidence_validation_policy: EvidenceValidationPolicy | None = None,
        commercial_disclosure_policy: CommercialDisclosurePolicy = CommercialDisclosurePolicy(),
        active_knowledge_base_id: str | None = None,
        # Slice 58B.3 — dev-only typed trace sink. When supplied, the
        # processor emits a bounded structured record per turn (brain
        # proposal fields, scope/authority classification, plan goal,
        # realization-called flag, final text). Opt-in; never exposed
        # over HTTP/WS. The sink is responsible for its own output
        # discipline — must not log secrets, raw prompts, or
        # chain-of-thought.
        dev_trace_sink: Callable[[dict], None] | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._planner = response_planner
        self._renderer = response_renderer
        self._dev_trace_sink = dev_trace_sink
        self._context = initial_context
        self._budget_policy = budget_policy
        self._scope_policy = scope_policy
        self._authority_policy = authority_policy
        self._priority = trusted_priority_provider or _no_priority
        self._contact = contact_understanding_provider
        self._rendering_context = rendering_context_provider
        self._strategy_engine = strategy_engine or ConversationStrategyEngine()
        self._strategy_input = strategy_input_provider or _default_strategy_input
        self._sales_stage = SalesStage.OPENING
        self._prospect_intelligence = (
            initial_prospect_intelligence or ProspectIntelligenceSnapshot()
        )
        self._prospect_updater = (
            prospect_intelligence_updater or ProspectIntelligenceUpdater()
        )
        self._prospect_evidence = prospect_evidence_provider or _no_prospect_evidence
        self._strategy_buffer = strategy_buffer or StrategyBuffer(initial_context.call_id)
        if self._strategy_buffer.call_id != initial_context.call_id:
            raise ValueError("strategy buffer must belong to the processor call")
        self._latest_strategy: ConversationStrategy | None = None
        self._latest_strategy_source_sequence: int | None = None
        self._context_builder = lean_context_builder or LeanContextBuilder()
        self._approved_evidence = tuple(approved_evidence)
        self._business_intelligence = business_intelligence
        self._relevant_business_fields = tuple(relevant_business_fields)
        self._campaign_goal = campaign_goal
        self._discovery_priorities = tuple(discovery_priorities)
        self._escalation_policy = escalation_policy or GracefulEscalationPolicy()
        self._escalation_capabilities = escalation_capabilities
        self._escalation_request = (
            escalation_request_provider or _default_escalation_request
        )
        self._free_text_understanding = free_text_understanding_provider
        self._understanding_mapper = (
            understanding_evidence_mapper or UnderstandingEvidenceMapper()
        )
        self._problem = initial_problem
        self._problem_evidence = problem_evidence_provider
        self._problem_updater = problem_model_updater or ProblemModelUpdater()
        self._service_relevance = service_relevance_resolver
        self._consultative_engine = (
            consultative_decision_engine or ConsultativeDecisionEngine()
        )
        self._consultative_signals = (
            consultative_signal_provider or _default_consultative_signals
        )
        self._sales_cognition = sales_cognition_engine or HumanSalesCognitionEngine()
        self._cognition_signals = cognition_signal_provider or _default_cognition_signals
        self._latest_sales_guidance: SalesConversationGuidance | None = None
        self._sales_playbook = sales_playbook_engine or SalesPlaybookIntelligenceEngine()
        self._service_playbooks = tuple(service_playbooks)
        self._playbook_context = playbook_context_provider
        self._latest_playbook_guidance: OpportunityGuidance | None = None
        self._business_conversation = initial_business_conversation or BusinessConversationSnapshot()
        self._business_conversation_engine = business_conversation_engine or BusinessConversationIntelligenceEngine()
        self._business_conversation_evidence = business_conversation_evidence_provider
        self._business_diagnostic = business_diagnostic_engine or BusinessDiagnosticEngine()
        self._latest_business_diagnostic: BusinessDiagnosticSnapshot | None = None
        self._conversation_steering = conversation_steering_engine or ConversationSteeringEngine()
        self._latest_conversation_priority: ConversationPrioritySnapshot | None = None
        self._qualification_configuration = qualification_configuration
        self._qualification_evidence = qualification_evidence_provider
        self._qualification_engine = ProgressiveQualificationEngine()
        self._qualification: QualificationSnapshot | None = initial_qualification
        # Last observed language profile — survives per-turn until the next
        # understanding provider overrides it. Carries recovery continuity so
        # a Roman Urdu switch before restart is not forgotten.
        self._last_language_profile: LanguageProfile | None = initial_language_profile

        # Slice 49 — participant-aware state. The processor OWNS participant
        # resolution via the deterministic policy; providers only propose
        # typed evidence. Business/BCI state remains shared; participant-
        # scoped state (role, interest, objections, language, discussion
        # exposure) is isolated per ParticipantReference.
        from app.conversation.participant.contracts import (
            DiscussionCoverageState,
        )
        from app.conversation.participant.policy import (
            ParticipantContinuityPolicy,
            deterministic_participant_factory,
        )
        self._participant_evidence_provider = participant_evidence_provider
        self._active_participant = initial_active_participant
        self._archived_participant_states: tuple = tuple(initial_archived_participants)
        self._discussion_coverage = (
            initial_discussion_coverage or DiscussionCoverageState()
        )
        self._participant_policy = ParticipantContinuityPolicy()
        _factory_start = len(self._archived_participant_states) + (
            1 if self._active_participant is not None else 0
        )
        self._participant_factory = deterministic_participant_factory(
            initial_context.call_id, start=_factory_start,
        )
        self._latest_participant_decision = None
        self._re_explanation_intent_provider = re_explanation_intent_provider
        self._explanation_concept_provider = explanation_concept_provider
        self._participant_prospect_archive: tuple = tuple(
            initial_participant_prospect_archive,
        )
        self._business_memory_scope = business_memory_scope
        self._business_memory_engine = BusinessMentalModelEngine()
        self._business_memory: BusinessMentalModelSnapshot | None = None
        self._knowledge_retrieval_policy = knowledge_retrieval_policy
        self._knowledge_need = knowledge_need_provider
        self._knowledge_retrieval_planner = KnowledgeRetrievalPlanner()
        self._retrieval_plan: RetrievalPlan | None = None
        self._hybrid_retrieval = hybrid_retrieval_engine
        self._retrieval_candidates: RetrievalCandidateSet | None = None
        self._evidence_validation_policy = evidence_validation_policy
        self._commercial_disclosure_policy = commercial_disclosure_policy
        self._evidence_validator = EvidenceValidationEngine()
        self._evidence_validation: EvidenceValidationResult | None = None
        self._recent_question_concepts: tuple[ProblemField, ...] = ()
        if active_knowledge_base_id is not None and (
            not active_knowledge_base_id.strip()
            or len(active_knowledge_base_id) > 100
        ):
            raise ValueError("active knowledge base id must be bounded")
        self._active_knowledge_base_id = active_knowledge_base_id
        self._consultative_enabled = any(
            value is not None
            for value in (
                problem_evidence_provider,
                service_relevance_resolver,
                consultative_signal_provider,
            )
        )
        self._recent_turns: tuple[RecentTurn, ...] = ()
        self._turn_count = 0
        if not isinstance(initial_reasoning_calls, int) or isinstance(
            initial_reasoning_calls, bool,
        ) or initial_reasoning_calls < 0:
            raise ValueError("initial_reasoning_calls must be a non-negative int")
        self._reasoning_calls = initial_reasoning_calls

    @property
    def context(self) -> ConversationContext:
        """Return the current authoritative domain context."""
        return self._context

    @property
    def current_state(self) -> ConversationState:
        """Return the authoritative state without exposing engine internals."""
        return self._orchestrator.current_state

    @property
    def sales_stage(self) -> SalesStage:
        """Return advisory sales progression, separate from authoritative state."""
        return self._sales_stage

    @property
    def prospect_intelligence(self) -> ProspectIntelligenceSnapshot:
        """Return the current immutable, call-scoped person snapshot."""
        return self._prospect_intelligence

    @property
    def prospect_problem(self) -> ProspectProblem:
        """Return the bounded current problem model without persistence."""
        return self._problem

    @property
    def sales_guidance(self) -> SalesConversationGuidance | None:
        """Return current advisory sales cognition without domain authority."""
        return self._latest_sales_guidance

    @property
    def active_knowledge_base_id(self) -> str | None:
        """Return the configured reference without loading knowledge."""
        return self._active_knowledge_base_id

    @property
    def playbook_guidance(self) -> OpportunityGuidance | None:
        """Return the latest advisory opportunity without service authority."""
        return self._latest_playbook_guidance

    @property
    def business_conversation(self) -> BusinessConversationSnapshot:
        """Return call-scoped advisory understanding, never domain authority."""
        return self._business_conversation

    @property
    def business_diagnostic(self) -> BusinessDiagnosticSnapshot | None:
        """Return latest non-commercial diagnostic without execution authority."""
        return self._latest_business_diagnostic

    @property
    def conversation_priority(self) -> ConversationPrioritySnapshot | None:
        """Return latest advisory discovery priority without decision authority."""
        return self._latest_conversation_priority

    @property
    def qualification(self) -> QualificationSnapshot | None:
        return self._qualification

    @property
    def business_memory(self) -> BusinessMentalModelSnapshot | None:
        return self._business_memory

    @property
    def retrieval_plan(self) -> RetrievalPlan | None:
        return self._retrieval_plan

    @property
    def retrieval_candidates(self) -> RetrievalCandidateSet | None:
        """Return unapproved candidates; never expose them as response evidence."""
        return self._retrieval_candidates

    @property
    def evidence_validation(self) -> EvidenceValidationResult | None:
        """Return validation output; no evidence is consumed downstream yet."""
        return self._evidence_validation

    @property
    def strategy_buffer(self) -> StrategyBufferSnapshot:
        """Return an immutable advisory-buffer snapshot."""
        return self._strategy_buffer.snapshot()

    def record_supervisor_result(
        self, insight: SupervisorInsight
    ) -> BufferWriteOutcome:
        """Record an externally completed advisory result without executing it."""
        return self._strategy_buffer.record(insight)

    @property
    def language_profile(self):
        """Most recent authoritative language profile; survives per-turn and restart."""
        return self._last_language_profile

    @property
    def active_participant(self):
        """Session-scoped current participant reference, if any."""
        return self._active_participant

    @property
    def archived_participant_states(self):
        """Tuple of archived participant conversation states."""
        return self._archived_participant_states

    @property
    def discussion_coverage(self):
        """Participant-aware per-concept exposure state."""
        return self._discussion_coverage

    @property
    def participant_prospect_archive(self):
        """Per-participant ProspectIntelligenceSnapshot archive.

        Tuple of (participant_reference_id, snapshot). Person B must never
        inherit from any entry here — these are archived snapshots keyed
        by the retiring participant's reference ID.
        """
        return self._participant_prospect_archive

    @property
    def reasoning_calls(self) -> int:
        """Committed reasoning-call counter. Persisted so restart cannot
        grant extra Brain reasoning capacity."""
        return self._reasoning_calls

    @property
    def latest_participant_decision(self):
        """Policy decision made on the most recent turn carrying evidence."""
        return self._latest_participant_decision

    def re_explanation_decision(self, concept_id: str, intent=None):
        """Query the participant-aware ReExplanationPolicy for a concept.

        Returns a ReExplanationDecision which downstream planning can
        consult to choose SUPPRESS / CONCISE_RECAP /
        FULL_RE_EXPLAIN_FOR_NEW_PARTICIPANT / DEEPEN / SIMPLIFY /
        NO_PRIOR_COVERAGE.
        """
        from app.conversation.participant.contracts import (
            ParticipantContinuityState,
        )
        from app.conversation.participant.policy import (
            ReExplanationIntent, ReExplanationPolicy,
        )
        policy = ReExplanationPolicy()
        state = (
            self._latest_participant_decision.state
            if self._latest_participant_decision is not None
            else ParticipantContinuityState.UNKNOWN
        )
        active_ref = (
            self._active_participant.reference_id
            if self._active_participant is not None else "unknown"
        )
        return policy.decide(
            state, active_ref, concept_id, self._discussion_coverage,
            intent=(intent or ReExplanationIntent.NONE),
        )

    def record_discussion_coverage(
        self, concept_id: str, depth, source_turn_id: str,
    ) -> None:
        """Append a coverage entry for the current active participant.

        Call this ONLY after a response explaining the concept has
        committed successfully. The persistence layer serializes the
        resulting DiscussionCoverageState with the turn's snapshot.
        """
        from app.conversation.participant.contracts import (
            DiscussionCoverageEntry, DiscussionCoverageState, ExplanationDepth,
        )
        if self._active_participant is None:
            return
        if not isinstance(depth, ExplanationDepth):
            raise TypeError("depth must be an ExplanationDepth")
        new_entry = DiscussionCoverageEntry(
            participant_reference_id=self._active_participant.reference_id,
            concept_id=concept_id,
            depth=depth,
            source_turn_id=source_turn_id,
        )
        self._discussion_coverage = DiscussionCoverageState(
            entries=self._discussion_coverage.entries + (new_entry,),
            asked_question_concepts=self._discussion_coverage.asked_question_concepts,
            handled_objection_concepts=self._discussion_coverage.handled_objection_concepts,
            discussed_evidence_ids=self._discussion_coverage.discussed_evidence_ids,
        )

    def _apply_participant_decision(self, decision) -> None:
        """Apply a resolved participant decision to processor state.

        NEW_CONFIRMED: archive the current active-participant snapshot (as
        a ParticipantConversationState capturing per-participant exposure
        + role hint + language preference) and install the new active.
        SAME_CONFIRMED: refine the active participant (role hint only).
        LIKELY_NEW / UNCERTAIN / UNKNOWN: do NOT switch — fail-closed.
        """
        from app.conversation.participant.contracts import (
            ParticipantContinuityState, ParticipantConversationState,
        )
        if decision.state == ParticipantContinuityState.NEW_CONFIRMED:
            if self._active_participant is not None:
                heard = self._discussion_coverage.heard_by(
                    self._active_participant.reference_id,
                )
                archived = ParticipantConversationState(
                    participant=self._active_participant,
                    heard_explanation_concepts=heard,
                    role_hint=self._active_participant.role_hint,
                    language_preference=(
                        self._last_language_profile.primary_language
                        if self._last_language_profile else None
                    ),
                )
                self._archived_participant_states = (
                    *self._archived_participant_states, archived,
                )
                # Slice 51D — archive the retiring participant's Prospect
                # Intelligence under their reference_id so Person B does
                # NOT inherit Person A's interest/objection/busy/role.
                self._participant_prospect_archive = (
                    *self._participant_prospect_archive,
                    (self._active_participant.reference_id, self._prospect_intelligence),
                )
            self._active_participant = decision.active_participant
            # New participant: fresh ProspectIntelligence — no inheritance
            # of Person A's personal signals. Business/BCI state remains
            # shared via its own channel.
            self._prospect_intelligence = ProspectIntelligenceSnapshot()
            # New participant: reset per-turn language snapshot so provider
            # of next turn can refresh it from their utterance.
            self._last_language_profile = None
            return
        if decision.state == ParticipantContinuityState.SAME_CONFIRMED:
            self._active_participant = decision.active_participant
            return
        # LIKELY_NEW, UNCERTAIN, UNKNOWN: preserve active, do not switch.

    def export_snapshot(self, session_revision: int):
        """Export the authoritative runtime state as a ConversationRuntimeSnapshot.

        Emits REAL payloads for BCI, LanguageProfile (when observed), and
        Qualification (when configured) using the Slice 46 serializers. The
        snapshot carries FSM state + sales_stage + active_knowledge_base_id
        + observed DNC/NOT_INTERESTED markers. One committed turn's snapshot
        is enough to rebuild a semantically equivalent processor.
        """
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
        from app.application.sales_runtime.snapshot import (
            SNAPSHOT_VERSION_CURRENT,
            ConversationRuntimeSnapshot,
        )

        payloads: list[tuple[str, str]] = []
        payloads.append((
            BUSINESS_CONVERSATION_KEY,
            BusinessConversationSerializer().serialize(self._business_conversation),
        ))
        if self._last_language_profile is not None:
            payloads.append((
                LANGUAGE_PROFILE_KEY,
                LanguageProfileSerializer().serialize(self._last_language_profile),
            ))
        if self._qualification is not None:
            payloads.append((
                QUALIFICATION_KEY,
                QualificationSerializer().serialize(self._qualification),
            ))
        payloads.append((
            PROSPECT_INTELLIGENCE_KEY,
            ProspectIntelligenceSerializer().serialize(self._prospect_intelligence),
        ))
        if self._active_participant is not None or self._archived_participant_states:
            payloads.append((
                PARTICIPANT_STATE_KEY,
                ParticipantStateSerializer().serialize(
                    self._active_participant, self._archived_participant_states,
                ),
            ))
        if self._participant_prospect_archive:
            payloads.append((
                PARTICIPANT_PROSPECT_ARCHIVE_KEY,
                ParticipantProspectArchiveSerializer().serialize(
                    self._participant_prospect_archive,
                ),
            ))
        # Slice 51H — persist reasoning_calls so restart cannot grant
        # extra reasoning capacity. Emit whenever counter > 0.
        if self._reasoning_calls > 0:
            from app.application.sales_runtime.serializers import (
                BUDGET_STATE_KEY, BudgetStateSerializer,
            )
            payloads.append((
                BUDGET_STATE_KEY,
                BudgetStateSerializer().serialize(self._reasoning_calls),
            ))
        # Only emit coverage when non-empty — avoids empty-payload churn
        if (
            self._discussion_coverage.entries
            or self._discussion_coverage.asked_question_concepts
            or self._discussion_coverage.handled_objection_concepts
            or self._discussion_coverage.discussed_evidence_ids
        ):
            payloads.append((
                DISCUSSION_COVERAGE_KEY,
                DiscussionCoverageSerializer().serialize(self._discussion_coverage),
            ))

        return ConversationRuntimeSnapshot(
            snapshot_version=SNAPSHOT_VERSION_CURRENT,
            session_id=self._context.call_id,
            source_session_revision=session_revision,
            conversation_state=self.current_state,
            sales_stage=self._sales_stage.value if self._sales_stage else None,
            active_knowledge_base_id=self._active_knowledge_base_id,
            last_pipeline_outcome=None,
            persisted_contract_payloads=tuple(payloads),
            is_dnc_observed=False,
            is_not_interested_observed=False,
        )

    def build_supervisor_input(
        self,
        turn: CoordinatedUserTurn,
        *,
        recent_context: tuple[str, ...] = (),
        campaign_context_summary: str | None = None,
    ) -> SupervisorInput:
        """Build bounded input for optional analysis after this turn completes."""
        if (
            self._latest_strategy is None
            or self._latest_strategy_source_sequence != turn.sequence_number
        ):
            raise ValueError("supervisor input requires the latest completed normal turn")
        extra_recent = tuple(
            RecentTurn(f"provided-{index}", TurnSpeaker.USER, item[:300])
            for index, item in enumerate(recent_context)
            if item.strip()
        )
        lean = self._build_lean_context(
            turn,
            self._latest_strategy,
            None,
            recent_turns=(*self._recent_turns, *extra_recent),
            campaign_goal=campaign_context_summary or self._campaign_goal,
        )
        return SupervisorInput(
            self._context.call_id,
            turn.turn_id,
            turn.sequence_number,
            self._context_builder.for_supervisor(lean),
        )

    def process_turn(self, turn: CoordinatedUserTurn) -> CoordinatedTurnOutput:
        """Run each existing layer once, in its established authority order."""
        priority = self._priority(turn)
        budget = self._budget_state()
        strategy = None
        supervisor_insight = None
        # Slice 51C — ensure understanding is defined on the priority path
        # too so participant-aware auto-threading can safely reference it.
        understanding = None
        # Seed language with last observed value so recovery/mid-conversation
        # restart retains language preference even if this turn's understanding
        # provider yields no language signal.
        language_profile = self._last_language_profile
        if priority == TrustedPriorityOutcome.NONE:
            supervisor_insight = self._strategy_buffer.consume_for(
                turn.sequence_number
            )
            if (
                supervisor_insight is not None
                and supervisor_insight.prospect_evidence is not None
            ):
                self._prospect_intelligence = self._prospect_updater.update(
                    self._prospect_intelligence,
                    supervisor_insight.prospect_evidence,
                )
            preliminary_context = self._build_lean_context(
                turn,
                self._latest_strategy,
                supervisor_insight,
            )
            understanding, evidence = self._understanding_evidence(
                turn, preliminary_context
            )
            if understanding is not None:
                if understanding.language_profile is not None:
                    language_profile = understanding.language_profile
                    self._last_language_profile = language_profile
            # Participant resolution — MUST happen before any person-scoped
            # evidence from this turn is attributed. Provider proposes; the
            # deterministic policy decides. Fail-closed on absent provider.
            if self._participant_evidence_provider is not None:
                try:
                    participant_evidence = self._participant_evidence_provider(
                        turn, understanding,
                    )
                except Exception:
                    participant_evidence = None
                if participant_evidence is not None:
                    self._latest_participant_decision = (
                        self._participant_policy.decide(
                            self._active_participant,
                            participant_evidence,
                            self._participant_factory,
                        )
                    )
                    self._apply_participant_decision(self._latest_participant_decision)
            if self._business_conversation_evidence is not None:
                business_evidence = self._business_conversation_evidence(
                    turn, preliminary_context, understanding
                )
                if business_evidence is not None:
                    self._business_conversation = self._business_conversation_engine.update(
                        self._business_conversation, business_evidence
                    )
            if evidence is not None:
                self._prospect_intelligence = self._prospect_updater.update(
                    self._prospect_intelligence, evidence
                )
            evidence = self._prospect_evidence(turn, self._prospect_intelligence)
            if evidence is not None:
                self._prospect_intelligence = self._prospect_updater.update(
                    self._prospect_intelligence, evidence
                )
            if self._problem_evidence is not None:
                problem_evidence = self._problem_evidence(
                    turn,
                    self._problem,
                    self._prospect_intelligence,
                )
                if problem_evidence is not None:
                    self._problem = self._problem_updater.update(
                        self._problem, problem_evidence
                    )
            strategy = self._strategy_engine.recommend(
                self._strategy_input(
                    turn,
                    self._context,
                    self._sales_stage,
                    self._orchestrator.current_state,
                    self._prospect_intelligence,
                    (
                        supervisor_insight.recommended_strategy_hint
                        if supervisor_insight is not None
                        else None
                    ),
                )
            )
        lean_context = self._build_lean_context(turn, strategy, supervisor_insight)
        slice_two = self._orchestrator.process_turn(
            self._context,
            priority,
            budget,
            self._budget_policy,
            TrustedExitSignals(),
            BrainSnapshotInput(
                current_utterance=turn.utterance,
                lean_context=self._context_builder.for_brain(lean_context),
            ),
            self._scope_policy,
            self._authority_policy,
        )
        if strategy is not None:
            self._sales_stage = strategy.sales_stage
        self._turn_count += 1
        if slice_two.trace.provider_name is not None:
            self._reasoning_calls += 1
        self._context = slice_two.updated_context

        slice_three = None
        if slice_two.outcome == TurnStageOutcome.AUTHORITY_APPROVED:
            understanding = self._contact_understanding(turn, slice_two)
            slice_three = self._orchestrator.execute_slice_three(
                slice_two, understanding
            )
            if slice_three.execution is not None:
                self._context = slice_three.execution.updated_context

        authoritative_result = _result_kind(slice_two, slice_three)
        terminal = (
            slice_three.conversation_status
            if slice_three is not None
            else slice_two.conversation_status
        ) == ConversationTerminationStatus.TERMINAL
        execution = (
            slice_three.execution if slice_three is not None else slice_two.execution
        )
        rendering_context = self._trusted_rendering_context(
            turn, priority, authoritative_result
        )
        escalation = self._escalation_decision(
            turn,
            lean_context,
            slice_two,
            priority,
        )
        consultative, service_answer = self._consultative_decision(
            turn,
            lean_context,
            language_profile,
            priority,
        )
        sales_guidance = self._sales_conversation_guidance(
            turn,
            lean_context,
            consultative,
            priority,
        )
        diagnostic = self._business_diagnostic.diagnose(
            BusinessDiagnosticInput(
                self._business_conversation,
                lean_context.prospect,
                lean_context.strategy,
            )
        ) if priority == TrustedPriorityOutcome.NONE else None
        steering = self._conversation_steering.steer(
            ConversationSteeringInput(
                self._business_conversation, diagnostic, lean_context.prospect,
                lean_context.strategy, self._orchestrator.current_state,
                self._latest_conversation_priority,
            )
        ) if diagnostic is not None else None
        if priority == TrustedPriorityOutcome.NONE and self._qualification_configuration and self._qualification_evidence:
            qualification_evidence = self._qualification_evidence(turn, lean_context)
            if qualification_evidence is not None:
                self._qualification = self._qualification_engine.update(self._qualification_configuration, qualification_evidence, self._qualification)
        if diagnostic is not None and self._business_memory_scope is not None:
            self._business_memory = self._business_memory_engine.project(
                self._business_memory_scope, self._business_conversation,
                self._prospect_intelligence, self._qualification, diagnostic,
                self._business_memory,
            )
        if priority == TrustedPriorityOutcome.NONE and self._knowledge_retrieval_policy is not None:
            needs = self._knowledge_need(turn, lean_context) if self._knowledge_need else ()
            policy = self._knowledge_retrieval_policy
            self._retrieval_plan = self._knowledge_retrieval_planner.plan(
                needs, policy, tenant_id=policy.tenant_id,
                business_id=policy.business_id, campaign_id=policy.campaign_id,
                context=RetrievalPlanningContext(self._business_memory, diagnostic, self._qualification, steering),
                preferred_language=language_profile.preferred_response_language if language_profile else None,
                fallback_language=language_profile.primary_language if language_profile else None,
            )
            if self._hybrid_retrieval is not None:
                self._retrieval_candidates = self._hybrid_retrieval.retrieve(
                    self._retrieval_plan
                )
                if self._evidence_validation_policy is not None:
                    self._evidence_validation = self._evidence_validator.validate(
                        self._retrieval_candidates,
                        self._evidence_validation_policy,
                        self._commercial_disclosure_policy,
                    )
        playbook_guidance = self._playbook_opportunity_guidance(
            turn,
            lean_context,
            consultative,
            sales_guidance,
            diagnostic,
            steering,
            priority,
        )
        if escalation is not None and escalation.evidence_ids:
            rendering_context = replace(
                rendering_context,
                approved_evidence=tuple(
                    item
                    for item in lean_context.approved_evidence
                    if item.evidence_id in escalation.evidence_ids
                ),
            )
        # Slice 51C — auto-resolve participant-aware re-explanation metadata
        # BEFORE the planner runs so the planner's input is already enriched.
        participant_fields = self._resolve_participant_aware_planner_fields(
            turn, understanding,
        )
        # Slice 51G — strategy shapes explanation_need so the planner's
        # own length/complexity calculation genuinely differs across
        # SIMPLIFY / DEEPEN / CONCISE_RECAP, not merely the prefix. Fallback
        # to the category-driven default when no strategy is supplied.
        base_need = _explanation_need(turn.conversation_category)
        effective_need = _strategy_shaped_need(
            base_need, participant_fields["strategy"],
        )
        plan = self._planner.plan(
            ResponsePlanningInput(
                authoritative_state=self._orchestrator.current_state,
                authoritative_result=authoritative_result,
                current_prospect_message=turn.utterance,
                approved_action=(
                    execution.approved_action if execution is not None else None
                ),
                conversation_category=turn.conversation_category,
                addressee_status=turn.addressee_status,
                interruption=turn.interruption,
                explanation_need=effective_need,
                escalation_decision=escalation,
                language_profile=language_profile,
                consultative_decision=consultative,
                service_answer_context=service_answer,
                sales_guidance=sales_guidance,
                playbook_guidance=playbook_guidance,
                business_conversation=self._business_conversation,
                business_diagnostic=diagnostic,
                conversation_priority=steering,
                qualification=self._qualification,
                business_memory=self._business_memory,
                retrieval_plan=self._retrieval_plan,
                re_explanation_strategy=participant_fields["strategy"],
                explanation_concept_id=participant_fields["concept_id"],
                prior_explanation_depth=participant_fields["prior_depth"],
                participant_reference_id=participant_fields["participant_ref"],
                catalog_request=participant_fields.get("catalog_request", False),
            )
        )
        # Attach delivered_explanation if the strategy indicates an
        # explanation is being delivered. ResponsePlanner currently passes
        # through delivered_explanation from input; emission authority
        # lives here until planner internals learn to decide depth.
        delivered = self._compute_delivered_explanation(
            plan, participant_fields,
        )
        if delivered is not None:
            from dataclasses import replace as _replace
            plan = _replace(plan, delivered_explanation=delivered)
        rendered = self._renderer.render(
            ResponseRenderInput(
                plan=plan,
                authoritative_result=authoritative_result,
                budget=ResponseRenderingBudget(
                    self._budget_policy.limits.max_response_sentences
                ),
                trusted_context=rendering_context,
                variation_seed=turn.turn_id,
                current_prospect_message=turn.utterance[:300],
                lean_context_view=_lean_context_view(turn, self._problem, lean_context),
            )
        )
        unfinished = (
            plan.pending_intent.summary
            if plan.pending_intent is not None
            else rendering_context.primary_fact
        )
        if strategy is not None:
            self._latest_strategy = strategy
            self._latest_strategy_source_sequence = turn.sequence_number
        if (
            sales_guidance is not None
            and sales_guidance.question_focus is not None
            and plan.question_strategy != QuestionStrategy.NONE
        ):
            self._recent_question_concepts = (
                *self._recent_question_concepts,
                sales_guidance.question_focus,
            )[-4:]
        self._latest_sales_guidance = sales_guidance
        self._latest_playbook_guidance = playbook_guidance
        self._latest_business_diagnostic = diagnostic
        self._latest_conversation_priority = steering
        self._recent_turns = (
            *self._recent_turns,
            RecentTurn(turn.turn_id, TurnSpeaker.USER, turn.utterance[:300]),
            RecentTurn(turn.turn_id, TurnSpeaker.AGENT, rendered.text[:300]),
        )[-8:]
        # Slice 51 — automatic participant-aware DiscussionCoverage advance.
        # If the plan carries an explanation_concept_id and an active
        # participant exists, record coverage at the depth implied by the
        # plan's response_length. This happens BEFORE the processor returns,
        # so the subsequent snapshot export (done by PersistentSalesRuntime
        # inside the atomic transaction) captures the updated coverage.
        # Monotonicity is preserved by DiscussionCoverageState.depth_for().
        self._auto_advance_discussion_coverage(turn, plan)
        # Slice 58B.5 — terminal/clarification consistency invariant (§25).
        # A terminal turn must not ask the user a follow-up question — the
        # session has ended, so clarification is semantically impossible.
        # If the planner emitted a safe-recovery plan AND execution landed
        # terminal (e.g. unexpected FallbackEngine terminal path), clear
        # the clarification request so the renderer does not produce a
        # question appended to a terminal response.
        if terminal and (
            plan.clarification_required
            or plan.question_strategy != QuestionStrategy.NONE
        ):
            from dataclasses import replace as _replace
            plan = _replace(
                plan,
                clarification_required=False,
                question_strategy=QuestionStrategy.NONE,
            )
        output = CoordinatedTurnOutput(
            turn.turn_id,
            plan,
            rendered,
            unfinished,
            authoritative_result,
            terminal,
        )
        self._emit_dev_trace_safely(
            turn=turn, slice_two=slice_two, slice_three=slice_three,
            plan=plan, output=output,
            realization_metrics=_extract_realization_metrics(self._renderer),
        )
        return output

    def _emit_dev_trace_safely(
        self, *, turn, slice_two, slice_three, plan, output,
        realization_metrics=None,
    ) -> None:
        """Opt-in trace emission that NEVER corrupts the live turn.

        Slice 58B.3b — the sink is wrapped so a builder/sink exception
        does not break the turn. When the sink IS configured (dev mode
        explicitly on), a failure also emits a bounded
        `[dev-trace-error]` record so the operator sees that tracing
        broke rather than silently getting no output.
        """
        sink = self._dev_trace_sink
        if sink is None:
            return
        try:
            record = _build_dev_trace(
                turn=turn,
                slice_two=slice_two,
                slice_three=slice_three,
                plan=plan,
                output=output,
                reasoning_calls=self._reasoning_calls,
                realization_metrics=realization_metrics,
            )
        except Exception as exc:
            self._emit_dev_trace_error_safely(sink, exc, phase="build")
            return
        try:
            sink(record)
        except Exception as exc:
            self._emit_dev_trace_error_safely(sink, exc, phase="sink")

    def _emit_dev_trace_error_safely(self, sink, exc, *, phase: str) -> None:
        """Bounded error record so silent trace failure is not possible.

        Includes only the exception type name + a bounded, truncated
        message — no traceback, no credentials, no raw provider output.
        If the error reporter itself fails, swallow — never recurse.
        """
        try:
            sink({
                "__dev_trace_error__": True,
                "phase": phase,
                "error_type": type(exc).__name__,
                "message": str(exc)[:200],
            })
        except Exception:
            pass

    def _unknown_participant_state_default(self):
        from app.conversation.participant.contracts import (
            ParticipantContinuityState,
        )
        return ParticipantContinuityState.UNKNOWN

    def _resolve_participant_aware_planner_fields(self, turn, understanding) -> dict:
        """Auto-derive the four participant-aware ResponsePlanningInput
        fields when both an intent provider and a concept provider are
        wired. Fail-closed on provider errors — degrade to no strategy.

        Slice 53 — CATALOG_REQUEST short-circuits:
            - intent resolved first so catalog is detectable without a concept
            - catalog does NOT flow through ReExplanationPolicy
            - strategy stays None → no delivered_explanation →
              no per-concept coverage advancement (§14, §15)
        """
        from app.conversation.participant.policy import (
            ReExplanationIntent, ReExplanationPolicy,
        )
        fields = {
            "strategy": None,
            "concept_id": None,
            "prior_depth": None,
            "participant_ref": None,
            "catalog_request": False,
        }
        if self._active_participant is None:
            return fields
        fields["participant_ref"] = self._active_participant.reference_id

        # Resolve intent FIRST so CATALOG_REQUEST can short-circuit without
        # requiring a concept provider output.
        intent = ReExplanationIntent.NONE
        if self._re_explanation_intent_provider is not None:
            try:
                proposed = self._re_explanation_intent_provider(
                    turn, understanding,
                )
                if isinstance(proposed, ReExplanationIntent):
                    intent = proposed
            except Exception:
                intent = ReExplanationIntent.NONE

        if intent == ReExplanationIntent.CATALOG_REQUEST:
            fields["catalog_request"] = True
            # Deliberately leave strategy=None, concept_id=None, prior_depth=None.
            # Catalog is NOT a re-explanation of one concept and must not
            # advance per-concept coverage.
            return fields

        # Resolve concept id via provider (re-explanation path)
        concept_id = None
        if self._explanation_concept_provider is not None:
            try:
                concept_id = self._explanation_concept_provider(
                    turn, understanding,
                )
            except Exception:
                concept_id = None
        if not concept_id:
            return fields
        fields["concept_id"] = concept_id
        # Compute prior depth and strategy via existing APIs
        fields["prior_depth"] = self._discussion_coverage.depth_for(
            self._active_participant.reference_id, concept_id,
        )
        decision = ReExplanationPolicy().decide(
            self._latest_participant_decision.state
            if self._latest_participant_decision is not None
            else self._unknown_participant_state_default(),
            self._active_participant.reference_id,
            concept_id,
            self._discussion_coverage,
            intent=intent,
        )
        fields["strategy"] = decision.strategy
        return fields

    def _compute_delivered_explanation(self, plan, fields: dict):
        """Compute ExplainedConcept the plan actually delivers, mapped
        from the authoritative strategy — NOT from ResponseLength.

        Rules (deliberate, documented):
            NO_PRIOR_COVERAGE                → EXPLAINED
            FULL_RE_EXPLAIN_FOR_NEW_PARTICIPANT → EXPLAINED
            CONCISE_RECAP                    → BRIEF
            DEEPEN                           → DETAILED
            SIMPLIFY                         → BRIEF
            SUPPRESS                         → None (no delivery)
        """
        from app.conversation.participant.contracts import (
            ExplainedConcept, ExplanationDepth, ReExplanationStrategy,
        )
        concept_id = fields.get("concept_id")
        strategy = fields.get("strategy")
        if not concept_id or strategy is None:
            return None
        mapping = {
            ReExplanationStrategy.NO_PRIOR_COVERAGE: ExplanationDepth.EXPLAINED,
            ReExplanationStrategy.FULL_RE_EXPLAIN_FOR_NEW_PARTICIPANT: ExplanationDepth.EXPLAINED,
            ReExplanationStrategy.CONCISE_RECAP: ExplanationDepth.BRIEF,
            ReExplanationStrategy.DEEPEN: ExplanationDepth.DETAILED,
            ReExplanationStrategy.SIMPLIFY: ExplanationDepth.BRIEF,
        }
        depth = mapping.get(strategy)
        if depth is None:
            return None  # SUPPRESS or unknown — no delivery event
        return ExplainedConcept(concept_id=concept_id, delivered_depth=depth)

    def _auto_advance_discussion_coverage(self, turn, plan) -> None:
        """Record coverage from the plan's EXPLICIT delivered-explanation
        metadata. Slice 51A used a ResponseLength → ExplanationDepth
        heuristic that was semantically unsafe (a long response could be a
        question or clarification, not an explanation). Slice 51B replaces
        that with an explicit `ExplainedConcept` field the planner emits
        ONLY when it is genuinely delivering an explanation. Suppress
        plans, question-only turns, greetings, acknowledgements, and
        failed re-explanations never carry this field and therefore never
        advance coverage.
        """
        from app.conversation.participant.contracts import ExplainedConcept
        delivered = getattr(plan, "delivered_explanation", None)
        if delivered is None or not isinstance(delivered, ExplainedConcept):
            return
        if self._active_participant is None:
            return
        self.record_discussion_coverage(
            delivered.concept_id, delivered.delivered_depth, turn.turn_id,
        )

    def _consultative_decision(
        self,
        turn: CoordinatedUserTurn,
        lean_context: LeanTurnContext,
        language_profile: LanguageProfile | None,
        priority: TrustedPriorityOutcome,
    ) -> tuple[
        ConsultativeConversationDecision | None,
        ServiceAnswerContext | None,
    ]:
        if not self._consultative_enabled or priority != TrustedPriorityOutcome.NONE:
            return None, None
        fit = (
            self._service_relevance.resolve(
                self._problem,
                eligible_service_ids=lean_context.eligible_service_ids,
                offered_service_ids=lean_context.offered_service_ids,
                approved_evidence=lean_context.approved_evidence,
            )
            if self._service_relevance is not None
            else ServiceFitDecision(
                ServiceFitStatus.INSUFFICIENT_CONTEXT,
            )
        )
        decision = self._consultative_engine.decide(
            ConsultativeDecisionInput(
                self._problem,
                fit,
                lean_context.prospect,
                lean_context.strategy,
                language_profile,
                self._consultative_signals(turn, lean_context),
                lean_context.conversation_category,
                lean_context.addressee_status,
                lean_context.pending_intent,
            )
        )
        answer = (
            self._service_relevance.build_answer_context(
                fit,
                self._problem,
                lean_context.approved_evidence,
            )
            if self._service_relevance is not None
            else None
        )
        return decision, answer

    def _sales_conversation_guidance(
        self,
        turn: CoordinatedUserTurn,
        lean_context: LeanTurnContext,
        decision: ConsultativeConversationDecision | None,
        priority: TrustedPriorityOutcome,
    ) -> SalesConversationGuidance | None:
        """Derive advisory cognition only on the normal consultative path."""
        if decision is None or priority != TrustedPriorityOutcome.NONE:
            return None
        fit = decision.service_fit or ServiceFitDecision(
            ServiceFitStatus.INSUFFICIENT_CONTEXT
        )
        return self._sales_cognition.evaluate(
            SalesCognitionInput(
                prospect=lean_context.prospect,
                problem=self._problem,
                service_fit=fit,
                strategy=lean_context.strategy,
                consultative_decision=decision,
                signals=self._cognition_signals(turn, lean_context),
                recent_question_concepts=self._recent_question_concepts,
                prior_guidance=self._latest_sales_guidance,
                returning_discussion=(
                    self._context.previous_conversation_exists or self._turn_count > 1
                ),
            )
        )

    def _playbook_opportunity_guidance(
        self,
        turn: CoordinatedUserTurn,
        lean_context: LeanTurnContext,
        decision: ConsultativeConversationDecision | None,
        sales_guidance: SalesConversationGuidance | None,
        diagnostic: BusinessDiagnosticSnapshot | None,
        steering: ConversationPrioritySnapshot | None,
        priority: TrustedPriorityOutcome,
    ) -> OpportunityGuidance | None:
        """Evaluate structured playbooks only on the normal advisory path."""
        if (
            priority != TrustedPriorityOutcome.NONE
            or decision is None
            or self._playbook_context is None
            or not self._service_playbooks
        ):
            return None
        return self._sales_playbook.evaluate(
            SalesPlaybookInput(
                context=self._playbook_context(turn, lean_context),
                playbooks=self._service_playbooks,
                eligible_service_ids=frozenset(lean_context.eligible_service_ids),
                prospect=lean_context.prospect,
                strategy=lean_context.strategy,
                problem=self._problem,
                consultative_decision=decision,
                sales_guidance=sales_guidance,
                already_discussed_service_ids=frozenset(
                    lean_context.offered_service_ids
                ),
                business_conversation=self._business_conversation,
                business_diagnostic=diagnostic,
                conversation_priority=steering,
                qualification=self._qualification,
                business_memory=self._business_memory,
                retrieval_plan=self._retrieval_plan,
            )
        )

    def _understanding_evidence(
        self,
        turn: CoordinatedUserTurn,
        lean_context: LeanTurnContext,
    ) -> tuple[FreeTextUnderstanding | None, ProspectEvidence | None]:
        """Run one advisory interpretation and map it without side effects."""
        if self._free_text_understanding is None:
            return None, None
        try:
            understanding = self._free_text_understanding.understand(
                FreeTextUnderstandingInput(
                    current_turn_id=turn.turn_id,
                    current_user_message=lean_context.untrusted_user_input.message,
                    current_state=lean_context.current_state,
                    prospect_summary=lean_context.prospect,
                    current_strategy=lean_context.strategy,
                    interruption=lean_context.interruption,
                    addressee_status=lean_context.addressee_status,
                    conversation_category=lean_context.conversation_category,
                )
            )
            if not isinstance(understanding, FreeTextUnderstanding):
                raise TypeError("understanding provider returned an invalid contract")
            if turn.addressee_status != AddresseeStatus.ADDRESSED_TO_AGENT:
                return None, None
            evidence = self._understanding_mapper.map(understanding, turn.turn_id)
            return understanding, evidence
        except (UnderstandingError, TypeError, ValueError):
            return None, None

    def _escalation_decision(
        self,
        turn: CoordinatedUserTurn,
        lean_context: LeanTurnContext,
        slice_two: TurnResult,
        priority: TrustedPriorityOutcome,
    ) -> EscalationDecision | None:
        """Evaluate communication recovery only on the normal post-Brain path."""
        if priority != TrustedPriorityOutcome.NONE:
            return None
        return self._escalation_policy.evaluate(
            GracefulEscalationInput(
                context=lean_context,
                request=self._escalation_request(turn, lean_context),
                capabilities=self._escalation_capabilities,
                scope_category=slice_two.trace.scope_category,
                authority_tier=slice_two.trace.authority_tier,
            )
        )

    def _build_lean_context(
        self,
        turn: CoordinatedUserTurn,
        strategy: ConversationStrategy | None,
        supervisor_insight: SupervisorInsight | None,
        *,
        recent_turns: tuple[RecentTurn, ...] | None = None,
        campaign_goal: str | None = None,
    ) -> LeanTurnContext:
        """Build a fresh canonical view from current typed state."""
        return self._context_builder.build(
            LeanContextBuildInput(
                call_id=self._context.call_id,
                current_turn_id=turn.turn_id,
                current_turn_sequence=turn.sequence_number,
                current_user_message=turn.utterance,
                current_state=self._orchestrator.current_state,
                conversation_context=self._context,
                strategy=strategy,
                recent_turns=(
                    self._recent_turns if recent_turns is None else recent_turns
                ),
                prospect_intelligence=self._prospect_intelligence,
                pending_intent=turn.interruption.previous_intent,
                approved_evidence=self._approved_evidence,
                authority_policy=self._authority_policy,
                business_intelligence=self._business_intelligence,
                relevant_business_fields=self._relevant_business_fields,
                campaign_goal=(
                    self._campaign_goal if campaign_goal is None else campaign_goal
                ),
                discovery_priorities=self._discovery_priorities,
                interruption=turn.interruption,
                conversation_category=turn.conversation_category,
                addressee_status=turn.addressee_status,
                supervisor_insight=supervisor_insight,
            )
        )

    def _contact_understanding(
        self, turn: CoordinatedUserTurn, slice_two: TurnResult
    ):
        if self._contact is None or slice_two.action not in self._CONTACT_ACTIONS:
            return None
        try:
            return self._contact.understand(
                UnderstandingInput(
                    client_utterance=turn.utterance,
                    current_state=self._orchestrator.current_state,
                )
            )
        except ContactUnderstandingError:
            return None

    def _trusted_rendering_context(
        self,
        turn: CoordinatedUserTurn,
        priority: TrustedPriorityOutcome,
        result: AuthoritativeResultKind,
    ) -> TrustedRenderingContext:
        if self._rendering_context is not None:
            return self._rendering_context(turn, result, self._context)
        if priority == TrustedPriorityOutcome.DNC:
            return TrustedRenderingContext(
                primary_fact="Your do-not-call request has been recorded"
            )
        if priority == TrustedPriorityOutcome.NOT_INTERESTED:
            return TrustedRenderingContext(
                primary_fact="We will not continue this sales conversation"
            )
        contact_status = (
            ContactConfirmationStatus.CONFIRMED
            if self._context.contact_confirmed
            else (
                ContactConfirmationStatus.UNCONFIRMED
                if self._context.contact_candidate is not None
                else ContactConfirmationStatus.NONE
            )
        )
        return TrustedRenderingContext(contact_status=contact_status)

    def _budget_state(self) -> BudgetState:
        limits = self._budget_policy.limits
        return BudgetState(
            turns_remaining=max(limits.max_turns - self._turn_count, 0),
            seconds_remaining=limits.max_call_seconds,
            reasoning_calls_remaining=max(
                limits.max_reasoning_calls - self._reasoning_calls, 0
            ),
            max_response_sentences=limits.max_response_sentences,
        )


def _no_priority(turn: CoordinatedUserTurn) -> TrustedPriorityOutcome:
    return TrustedPriorityOutcome.NONE


def _default_strategy_input(
    turn: CoordinatedUserTurn,
    context: ConversationContext,
    current_stage: SalesStage,
    current_state: ConversationState,
    prospect_intelligence: ProspectIntelligenceSnapshot,
    strategy_hint: ConversationStrategyHint | None,
) -> ConversationStrategyInput:
    """Map existing typed metadata only; never classify the raw utterance here."""
    return ConversationStrategyInput(
        current_stage=current_stage,
        current_state=current_state,
        campaign_context_label=context.campaign_id,
        objection_present=turn.conversation_category == InterruptionCategory.OBJECTION,
        question_present=turn.conversation_category == InterruptionCategory.QUESTION,
        clarification_needed=(
            turn.conversation_category == InterruptionCategory.CLARIFICATION
        ),
        interruption=turn.interruption,
        prospect_intelligence=prospect_intelligence,
        strategy_hint=strategy_hint,
    )


def _no_prospect_evidence(
    turn: CoordinatedUserTurn,
    previous: ProspectIntelligenceSnapshot,
) -> ProspectEvidence | None:
    return None


def _default_escalation_request(
    turn: CoordinatedUserTurn,
    context: LeanTurnContext,
) -> EscalationRequest:
    """Map existing typed metadata only; never classify raw utterance text."""
    if turn.conversation_category == InterruptionCategory.CLARIFICATION:
        return EscalationRequest(KnowledgeRequestKind.AMBIGUOUS)
    if turn.conversation_category == InterruptionCategory.QUESTION:
        return EscalationRequest(KnowledgeRequestKind.FACT)
    return EscalationRequest()


def _default_consultative_signals(
    turn: CoordinatedUserTurn,
    context: LeanTurnContext,
) -> ConsultativeTurnSignals:
    """Use existing typed turn metadata only; never classify raw utterance text."""
    return ConsultativeTurnSignals(
        direct_question=turn.conversation_category == InterruptionCategory.QUESTION,
        correction=turn.conversation_category == InterruptionCategory.CORRECTION,
    )


def _default_cognition_signals(
    turn: CoordinatedUserTurn,
    context: LeanTurnContext,
) -> CognitionSignals:
    """Map typed turn metadata only; never infer cognition from raw text."""
    return CognitionSignals(
        confused=turn.conversation_category == InterruptionCategory.CLARIFICATION,
        skeptical=turn.conversation_category == InterruptionCategory.OBJECTION,
        correction=turn.conversation_category == InterruptionCategory.CORRECTION,
    )


def _lean_context_view(
    turn: CoordinatedUserTurn,
    problem: ProspectProblem,
    context: LeanTurnContext,
) -> LeanContextView:
    """Project only wording-safe current-turn context for a realizer provider."""
    role = (
        context.prospect.explicit_role.value.value
        if context.prospect is not None and context.prospect.explicit_role is not None
        else None
    )
    return LeanContextView(
        turn.utterance[:300],
        problem.explicit_description or problem.friction,
        role,
    )


def _result_kind(slice_two, slice_three) -> AuthoritativeResultKind:
    if slice_three is not None:
        return AuthoritativeResultKind(slice_three.outcome.value)
    return {
        TurnStageOutcome.FALLBACK: AuthoritativeResultKind.FALLBACK,
        TurnStageOutcome.REDIRECT: AuthoritativeResultKind.REDIRECT,
        TurnStageOutcome.ESCALATE: AuthoritativeResultKind.ESCALATE,
        TurnStageOutcome.PIPELINE_STOPPED: AuthoritativeResultKind.PIPELINE_STOPPED,
        TurnStageOutcome.AUTHORITY_APPROVED: AuthoritativeResultKind.AUTHORITY_APPROVED,
    }[slice_two.outcome]


def _explanation_need(category: InterruptionCategory) -> ExplanationNeed:
    if category in {InterruptionCategory.QUESTION, InterruptionCategory.OBJECTION}:
        return ExplanationNeed.STANDARD
    return ExplanationNeed.SIMPLE


def _strategy_shaped_need(base: ExplanationNeed, strategy):
    """Slice 51G — strategy overrides ExplanationNeed so the planner's own
    length calculation genuinely differs across simplify/deepen/recap.

    Mapping (semantic, not merely length):
        SIMPLIFY         → ExplanationNeed.SIMPLE   (less content, simpler shape)
        CONCISE_RECAP    → ExplanationNeed.SIMPLE   (recap is intentionally brief)
        DEEPEN           → ExplanationNeed.COMPLEX  (planner may use detailed length)
        FULL_RE_EXPLAIN  → ExplanationNeed.STANDARD (orient the new person)
        SUPPRESS         → base (don't re-explain the concept, but keep normal
                                 behavior for whatever the planner is now doing)
        NO_PRIOR_COVERAGE → base (first-time explanation uses category default)
        None              → base

    This runs BEFORE the planner so planner.plan() sees the adjusted need
    and makes a semantically different choice. Combined with Slice 50's
    _apply_re_explanation_length, both the ExplanationNeed-driven base
    length AND the strategy-aware post-mapping agree on the final length.
    """
    try:
        from app.conversation.participant.contracts import ReExplanationStrategy
    except ImportError:
        return base
    if not isinstance(strategy, ReExplanationStrategy):
        return base
    if strategy == ReExplanationStrategy.SIMPLIFY:
        return ExplanationNeed.SIMPLE
    if strategy == ReExplanationStrategy.CONCISE_RECAP:
        return ExplanationNeed.SIMPLE
    if strategy == ReExplanationStrategy.DEEPEN:
        return ExplanationNeed.COMPLEX
    if strategy == ReExplanationStrategy.FULL_RE_EXPLAIN_FOR_NEW_PARTICIPANT:
        return ExplanationNeed.STANDARD
    return base


def _extract_realization_metrics(renderer) -> dict | None:
    """Walk one level of renderer wrapping to find GuardedConversation
    Realizer's `last_metrics`. StrategyAwareResponseRenderer wraps the
    realizer in dev composition; the realizer wraps the deterministic
    fallback. Return None when no live realizer is in the chain.
    """
    candidate = renderer
    for _ in range(4):
        if candidate is None:
            return None
        last_metrics = getattr(candidate, "last_metrics", None)
        if isinstance(last_metrics, dict):
            # Copy to avoid handing the live mutable dict to the sink
            return dict(last_metrics)
        inner = getattr(candidate, "_inner", None) or getattr(candidate, "_wrapped", None)
        if inner is None:
            inner = getattr(candidate, "_renderer", None)
        candidate = inner
    return None


def _build_dev_trace(
    *,
    turn,
    slice_two,
    slice_three,
    plan,
    output,
    reasoning_calls: int,
    realization_metrics: dict | None = None,
) -> dict:
    """Build a bounded structured dev-trace record for one turn.

    Captures the fields a diagnostician needs to identify the first
    divergent layer per Slice 58B.2 §4. Deliberately excludes raw
    prompts, chain-of-thought, API credentials, Gemini token counts,
    and any secret. Enum values are serialized as their `.value` strings.
    """
    def _enum(v):
        return v.value if hasattr(v, "value") else (str(v) if v is not None else None)

    s2 = slice_two
    t = s2.trace if s2 is not None else None
    s3 = slice_three
    plan_fields = {
        "communicative_goal": _enum(plan.communicative_goal),
        "response_length": _enum(plan.response_length),
        "question_strategy": _enum(plan.question_strategy),
        "clarification_required": bool(plan.clarification_required),
        "addressee_status": _enum(plan.addressee_status),
        "catalog_request": bool(getattr(plan, "catalog_request", False)),
        "re_explanation_strategy": _enum(getattr(plan, "re_explanation_strategy", None)),
        "has_escalation_decision": plan.escalation_decision is not None,
    }
    return {
        "turn_id": turn.turn_id,
        "turn_sequence": turn.sequence_number,
        "utterance": turn.utterance[:200],
        "reasoning_calls_total": reasoning_calls,
        "slice_two": {
            "outcome": _enum(s2.outcome) if s2 is not None else None,
            "conversation_status": _enum(
                s2.conversation_status if s2 is not None else None,
            ),
            "stage_outcome": _enum(t.stage_outcome) if t is not None else None,
            "provider_name": t.provider_name if t is not None else None,
            "proposal_status": _enum(t.proposal_status) if t is not None else None,
            "scope_category": _enum(t.scope_category) if t is not None else None,
            "authority_tier": _enum(t.authority_tier) if t is not None else None,
            "action_confidence": t.action_confidence if t is not None else None,
            "notes": t.notes if t is not None else None,
            "priority_action": _enum(t.priority_action) if t is not None else None,
            # Slice 58B.5 — typed BrainProposal fields.
            "proposed_action": _enum(getattr(t, "proposed_action", None)) if t is not None else None,
            "detected_intent": _enum(getattr(t, "detected_intent", None)) if t is not None else None,
            "topic_category": _enum(getattr(t, "topic_category", None)) if t is not None else None,
        },
        "slice_three": {
            "outcome": _enum(s3.outcome) if s3 is not None else None,
            "conversation_status": _enum(
                s3.conversation_status if s3 is not None else None,
            ),
            "approved_action": _enum(
                s3.execution.approved_action
                if s3 is not None and s3.execution is not None
                else None,
            ),
            "is_terminal": (
                bool(s3.execution.is_terminal)
                if s3 is not None and s3.execution is not None
                else None
            ),
            "fallback_used": (
                bool(s3.execution.fallback_used)
                if s3 is not None and s3.execution is not None
                else None
            ),
        },
        "plan": plan_fields,
        # Slice 58B.3b — CoordinatedTurnOutput exposes
        # `pipeline_outcome: AuthoritativeResultKind | None`, not
        # `authoritative_result`. Previous builder tripped
        # AttributeError here and was swallowed.
        "authoritative_result": _enum(output.pipeline_outcome),
        "conversation_terminal": bool(output.conversation_terminal),
        "rendered_text": output.rendered_response.text[:300],
        # Slice 58B.4 — realization pass diagnostics so an operator can
        # tell live-Gemini-wording from deterministic-fallback-wording
        # at a glance.
        "realization": realization_metrics,
    }
