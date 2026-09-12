"""Compose the existing authoritative and communication pipelines per call."""

from __future__ import annotations

from collections.abc import Callable

from app.brain.authority.models import AuthorityPolicy
from app.brain.budget.evaluator import TrustedExitSignals
from app.brain.budget.models import BudgetPolicy
from app.brain.contracts import BudgetState
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
from app.conversation.response_planning.contracts import (
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
from app.conversation.strategy.contracts import (
    ConversationStrategyInput,
    SalesStage,
)
from app.conversation.strategy.engine import ConversationStrategyEngine
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
    [CoordinatedUserTurn, ConversationContext, SalesStage, ConversationState],
    ConversationStrategyInput,
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

    def process_turn(self, turn: CoordinatedUserTurn) -> CoordinatedTurnOutput:
        """Run each existing layer once, in its established authority order."""
        priority = self._priority(turn)
        budget = self._budget_state()
        strategy = None
        if priority == TrustedPriorityOutcome.NONE:
            strategy = self._strategy_engine.recommend(
                self._strategy_input(
                    turn,
                    self._context,
                    self._sales_stage,
                    self._orchestrator.current_state,
                )
            )
        slice_two = self._orchestrator.process_turn(
            self._context,
            priority,
            budget,
            self._budget_policy,
            TrustedExitSignals(),
            BrainSnapshotInput(
                current_utterance=turn.utterance,
                conversation_strategy=strategy,
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
            )
        )
        unfinished = (
            plan.pending_intent.summary
            if plan.pending_intent is not None
            else rendering_context.primary_fact
        )
        return CoordinatedTurnOutput(
            turn.turn_id,
            plan,
            rendered,
            unfinished,
            authoritative_result,
            terminal,
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
