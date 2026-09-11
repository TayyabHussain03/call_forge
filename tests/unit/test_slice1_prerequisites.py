"""Focused regression tests for the trustworthy Slice-1 boundary."""

from __future__ import annotations

import phonenumbers
import pytest

from app.brain.budget.evaluator import BudgetPolicyEvaluator, TrustedExitSignals
from app.brain.budget.models import BudgetLimits, BudgetPolicy
from app.brain.contracts import BudgetState
from app.brain.orchestrator.orchestrator import (
    BrainOrchestrator,
    ConversationTerminationStatus,
    TurnStageOutcome,
)
from app.config.settings import get_settings
from app.contracts.conversation import ProposedConversationDecision
from app.contracts.conversation_context import ConversationContext
from app.contracts.intent import IntentSignal
from app.conversation.engine import ConversationEngine
from app.conversation.guardrails.action_validator import ActionValidator
from app.conversation.guardrails.priority import TrustedPriorityOutcome
from app.conversation.state_machine.machine import ConversationStateMachine
from app.conversation.state_machine.states import load_config
from app.core.constants import (
    AgentAction,
    ConversationState,
    Intent,
)


def _engine(state: ConversationState = ConversationState.NEW_CALL) -> ConversationEngine:
    config = load_config(get_settings().conversation_config_path)
    machine = ConversationStateMachine(config, initial_state=state)
    return ConversationEngine(machine, ActionValidator(config))


def _orchestrator(state: ConversationState = ConversationState.NEW_CALL) -> BrainOrchestrator:
    return BrainOrchestrator(_engine(state), BudgetPolicyEvaluator())


def _healthy_budget() -> tuple[BudgetState, BudgetPolicy]:
    state = BudgetState(5, 120, 5, 2)
    policy = BudgetPolicy("test", BudgetLimits(5, 10, 180, 2))
    return state, policy


def _proposal(intent: Intent) -> ProposedConversationDecision:
    return ProposedConversationDecision(
        detected_intent=IntentSignal(intent=intent, intent_confidence=1.0),
        response_text="proposal",
        proposed_action=AgentAction.GREET,
        proposed_next_state=ConversationState.GREETING,
        turn_confidence=1.0,
    )


def test_untrusted_intent_cannot_authorize_dnc() -> None:
    engine = _engine()
    result = engine.process_turn(ConversationContext("call"), _proposal(Intent.DO_NOT_CALL))

    assert result.approved_action == AgentAction.GREET
    assert result.outcome is None
    assert not result.updated_context.dnc_pending


def test_trusted_dnc_triggers_deterministic_dnc_path() -> None:
    state, policy = _healthy_budget()
    result = _orchestrator().process_turn(
        ConversationContext("call"),
        TrustedPriorityOutcome.DNC,
        state,
        policy,
        TrustedExitSignals(),
    )

    assert result.outcome == TurnStageOutcome.PIPELINE_STOPPED
    assert result.conversation_status == ConversationTerminationStatus.TERMINAL
    assert result.execution is not None
    assert result.execution.approved_action == AgentAction.MARK_DNC
    assert result.execution.outcome == "do_not_call"
    assert result.updated_context.dnc_pending


@pytest.mark.parametrize(
    "state",
    [
        state
        for state in ConversationState
        if state not in {ConversationState.END_CALL, ConversationState.CHECK_HISTORY}
    ],
)
def test_dnc_is_structurally_available_from_every_live_state(
    state: ConversationState,
) -> None:
    engine = _engine(state)
    result = engine.execute_approved_action(ConversationContext("call"), AgentAction.MARK_DNC)

    assert result.is_terminal
    assert result.outcome == "do_not_call"
    assert result.updated_context.dnc_pending


def test_not_interested_remains_distinct_from_dnc() -> None:
    state, policy = _healthy_budget()
    result = _orchestrator(ConversationState.LISTEN).process_turn(
        ConversationContext("call"),
        TrustedPriorityOutcome.NOT_INTERESTED,
        state,
        policy,
        TrustedExitSignals(),
    )

    assert result.execution is not None
    assert result.execution.approved_action == AgentAction.END_CALL
    assert result.execution.outcome == "not_interested"
    assert not result.updated_context.dnc_pending


def test_untrusted_priority_enum_fails_closed() -> None:
    state, policy = _healthy_budget()
    result = _orchestrator().process_turn(
        ConversationContext("call"),
        Intent.DO_NOT_CALL,  # type: ignore[arg-type]
        state,
        policy,
        TrustedExitSignals(),
    )

    assert result.outcome == TurnStageOutcome.FALLBACK
    assert result.conversation_status == ConversationTerminationStatus.NOT_TERMINAL
    assert result.execution is None


def test_budget_wind_down_stops_pipeline_without_claiming_terminal_state() -> None:
    _, policy = _healthy_budget()
    result = _orchestrator().process_turn(
        ConversationContext("call"),
        TrustedPriorityOutcome.NONE,
        BudgetState(0, 120, 5, 2),
        policy,
        TrustedExitSignals(),
    )

    assert result.outcome == TurnStageOutcome.PIPELINE_STOPPED
    assert result.conversation_status == ConversationTerminationStatus.NOT_TERMINAL
    assert result.execution is None
    assert result.updated_context == ConversationContext("call")


def test_declared_phone_dependency_and_orchestrator_import() -> None:
    assert phonenumbers.__version__
    assert BrainOrchestrator is not None
