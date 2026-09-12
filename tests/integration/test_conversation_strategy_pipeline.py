"""Slice 11 integration through Brain, policy, and deterministic execution."""

from __future__ import annotations

from app.brain.authority.models import load_authority_policies
from app.brain.authority.validator import AuthorityPolicyValidator
from app.brain.budget.evaluator import BudgetPolicyEvaluator, TrustedExitSignals
from app.brain.budget.models import load_budget_policies
from app.brain.contracts import BrainInput, BrainProposal, BudgetState, CommercialRequest
from app.brain.orchestrator.orchestrator import (
    BrainOrchestrator,
    BrainSnapshotInput,
    SliceThreeOutcome,
    TurnStageOutcome,
)
from app.brain.scope.models import load_scope_policies
from app.brain.scope.validator import ScopePolicyValidator
from app.config.settings import get_settings
from app.contracts.conversation_context import ConversationContext
from app.conversation.engine import ConversationEngine
from app.conversation.guardrails.action_validator import ActionValidator
from app.conversation.guardrails.priority import TrustedPriorityOutcome
from app.conversation.state_machine.machine import ConversationStateMachine
from app.conversation.state_machine.states import load_config
from app.conversation.strategy.contracts import (
    ConversationStrategy,
    ConversationStrategyInput,
    MicroCommitment,
    SalesStage,
    StrategyType,
)
from app.conversation.strategy.engine import ConversationStrategyEngine
from app.core.constants import (
    AgentAction,
    CommercialRequestKind,
    ConversationState,
    Intent,
    TopicCategory,
)
from app.llm.providers.reasoning_provider import ReasoningProvider


class CapturingProvider(ReasoningProvider):
    def __init__(self, proposal: BrainProposal) -> None:
        self.proposal = proposal
        self.inputs: list[BrainInput] = []

    @property
    def name(self) -> str:
        return "capturing"

    def reason(self, brain_input: BrainInput) -> BrainProposal:
        self.inputs.append(brain_input)
        return self.proposal


def _orchestrator(provider: ReasoningProvider) -> BrainOrchestrator:
    config = load_config(get_settings().conversation_config_path)
    engine = ConversationEngine(
        ConversationStateMachine(config),
        ActionValidator(config),
    )
    return BrainOrchestrator(
        engine,
        BudgetPolicyEvaluator(),
        reasoning_provider=provider,
        scope_validator=ScopePolicyValidator(),
        authority_validator=AuthorityPolicyValidator(),
    )


def _run(
    orchestrator: BrainOrchestrator, proposal_strategy: ConversationStrategy
):
    policies = load_budget_policies("app/config/budget_policy.yaml")
    return orchestrator.process_turn(
        ConversationContext("call"),
        TrustedPriorityOutcome.NONE,
        BudgetState(5, 120, 5, 2),
        policies["standard"],
        TrustedExitSignals(),
        BrainSnapshotInput("prospect turn", conversation_strategy=proposal_strategy),
        load_scope_policies("app/config/scope_policy.yaml")["business_general"],
        load_authority_policies("app/config/authority_policy.yaml")["standard"],
    )


def _next_step_strategy() -> ConversationStrategy:
    return ConversationStrategyEngine().recommend(
        ConversationStrategyInput(
            current_stage=SalesStage.DISCOVERY,
            current_state=ConversationState.NEW_CALL,
            explicit_next_step=MicroCommitment.DEMO,
        )
    )


def test_strategy_reaches_bounded_brain_input_but_cannot_override_authority() -> None:
    provider = CapturingProvider(
        BrainProposal(
            detected_intent=Intent.INTERESTED,
            proposed_action=AgentAction.GREET,
            topic_category=TopicCategory.COMMERCIAL_REQUEST,
            commercial_request=CommercialRequest(CommercialRequestKind.GUARANTEE),
        )
    )
    orchestrator = _orchestrator(provider)
    strategy = _next_step_strategy()

    classified = _run(orchestrator, strategy)
    execution = orchestrator.execute_slice_three(classified)

    assert len(provider.inputs) == 1
    assert provider.inputs[0].conversation_strategy == strategy
    assert strategy.strategy_type == StrategyType.PROPOSE_NEXT_STEP
    assert classified.outcome == TurnStageOutcome.REDIRECT
    assert execution.outcome == SliceThreeOutcome.REDIRECT
    assert execution.execution is None
    assert orchestrator.current_state == ConversationState.NEW_CALL


def test_fsm_executes_only_policy_approved_brain_action_not_strategy_direction() -> None:
    provider = CapturingProvider(
        BrainProposal(
            detected_intent=Intent.INTERESTED,
            proposed_action=AgentAction.GREET,
            topic_category=TopicCategory.QUALIFICATION,
        )
    )
    orchestrator = _orchestrator(provider)
    strategy = _next_step_strategy()

    classified = _run(orchestrator, strategy)
    execution = orchestrator.execute_slice_three(classified)

    assert strategy.sales_stage == SalesStage.NEXT_STEP
    assert classified.outcome == TurnStageOutcome.AUTHORITY_APPROVED
    assert execution.outcome == SliceThreeOutcome.EXECUTED
    assert execution.execution is not None
    assert execution.execution.approved_action == AgentAction.GREET
    assert orchestrator.current_state == ConversationState.GREETING
