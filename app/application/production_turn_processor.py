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
    ) -> None:
        self._orchestrator = orchestrator
        self._planner = response_planner
        self._renderer = response_renderer
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
        self._reasoning_calls = 0

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
    def strategy_buffer(self) -> StrategyBufferSnapshot:
        """Return an immutable advisory-buffer snapshot."""
        return self._strategy_buffer.snapshot()

    def record_supervisor_result(
        self, insight: SupervisorInsight
    ) -> BufferWriteOutcome:
        """Record an externally completed advisory result without executing it."""
        return self._strategy_buffer.record(insight)

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
        language_profile = None
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
                language_profile = understanding.language_profile
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
        if escalation is not None and escalation.evidence_ids:
            rendering_context = replace(
                rendering_context,
                approved_evidence=tuple(
                    item
                    for item in lean_context.approved_evidence
                    if item.evidence_id in escalation.evidence_ids
                ),
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
                explanation_need=_explanation_need(turn.conversation_category),
                escalation_decision=escalation,
                language_profile=language_profile,
                consultative_decision=consultative,
                service_answer_context=service_answer,
            )
        )
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
        self._recent_turns = (
            *self._recent_turns,
            RecentTurn(turn.turn_id, TurnSpeaker.USER, turn.utterance[:300]),
            RecentTurn(turn.turn_id, TurnSpeaker.AGENT, rendered.text[:300]),
        )[-8:]
        return CoordinatedTurnOutput(
            turn.turn_id,
            plan,
            rendered,
            unfinished,
            authoritative_result,
            terminal,
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
