"""Focused behavioral tests for Slice 11 conversation strategy."""

from __future__ import annotations

from dataclasses import fields, replace

import pytest

from app.config.settings import get_settings
from app.conversation.response_planning.contracts import (
    InterruptionCategory,
    InterruptionContext,
    PendingConversationIntent,
)
from app.conversation.state_machine.machine import ConversationStateMachine
from app.conversation.state_machine.states import load_config
from app.conversation.strategy.contracts import (
    ConversationMode,
    ConversationStrategy,
    ConversationStrategyInput,
    InformationGap,
    MicroCommitment,
    SalesStage,
    StrategyType,
)
from app.conversation.strategy.engine import ConversationStrategyEngine
from app.core.constants import ConversationState


def _input(**changes: object) -> ConversationStrategyInput:
    base = ConversationStrategyInput(
        current_stage=SalesStage.OPENING,
        current_state=ConversationState.NEW_CALL,
    )
    return replace(base, **changes)


def test_initial_opening_earns_permission_to_continue() -> None:
    result = ConversationStrategyEngine().recommend(_input())

    assert result.sales_stage == SalesStage.OPENING
    assert result.strategy_type == StrategyType.OPEN_CONVERSATION
    assert result.micro_commitment == MicroCommitment.PERMISSION_TO_CONTINUE


def test_opening_recommends_discovery_after_introductory_states() -> None:
    result = ConversationStrategyEngine().recommend(
        _input(current_state=ConversationState.REASON_FOR_CALL)
    )

    assert result.sales_stage == SalesStage.DISCOVERY
    assert result.strategy_type == StrategyType.DISCOVER_NEED


def test_discovery_with_pain_recommends_value_exploration() -> None:
    result = ConversationStrategyEngine().recommend(
        _input(current_stage=SalesStage.DISCOVERY, pain_present=True)
    )

    assert result.sales_stage == SalesStage.VALUE_EXPLORATION
    assert result.strategy_type == StrategyType.EXPLAIN_VALUE


def test_discovery_without_enough_information_stays_in_discovery() -> None:
    result = ConversationStrategyEngine().recommend(
        _input(current_stage=SalesStage.DISCOVERY)
    )

    assert result.sales_stage == SalesStage.DISCOVERY
    assert result.strategy_type == StrategyType.DISCOVER_NEED


def test_explicit_demo_request_may_accelerate_to_next_step() -> None:
    result = ConversationStrategyEngine().recommend(
        _input(
            current_stage=SalesStage.DISCOVERY,
            explicit_next_step=MicroCommitment.DEMO,
        )
    )

    assert result.sales_stage == SalesStage.NEXT_STEP
    assert result.strategy_type == StrategyType.PROPOSE_NEXT_STEP
    assert result.micro_commitment == MicroCommitment.DEMO


def test_pricing_question_creates_question_detour_without_pricing_authority() -> None:
    result = ConversationStrategyEngine().recommend(
        _input(current_stage=SalesStage.DISCOVERY, question_present=True)
    )

    assert result.conversation_mode == ConversationMode.QUESTION_DETOUR
    assert result.strategy_type == StrategyType.HANDLE_QUESTION
    assert "price" not in {field.name for field in fields(result)}


def test_question_detour_preserves_previous_meaningful_stage() -> None:
    result = ConversationStrategyEngine().recommend(
        _input(current_stage=SalesStage.VALUE_EXPLORATION, question_present=True)
    )

    assert result.sales_stage == SalesStage.VALUE_EXPLORATION


def test_busy_signal_prefers_micro_commitment_over_full_pitch() -> None:
    result = ConversationStrategyEngine().recommend(
        _input(current_stage=SalesStage.DISCOVERY, busy=True, pain_present=True)
    )

    assert result.conversation_mode == ConversationMode.BUSY
    assert result.strategy_type == StrategyType.ASK_MICRO_COMMITMENT
    assert result.micro_commitment == MicroCommitment.CALLBACK


def test_current_solution_recommends_understanding_satisfaction() -> None:
    result = ConversationStrategyEngine().recommend(
        _input(current_stage=SalesStage.DISCOVERY, current_solution_present=True)
    )

    assert result.strategy_type == StrategyType.UNDERSTAND_CURRENT_SOLUTION
    assert result.information_gaps[0] == InformationGap.SATISFACTION
    assert len(set(result.information_gaps)) == len(result.information_gaps)


def test_interest_progresses_only_after_relevant_gaps_are_satisfied() -> None:
    engine = ConversationStrategyEngine()
    unresolved = engine.recommend(
        _input(
            current_stage=SalesStage.VALUE_EXPLORATION,
            meaningful_interest=True,
        )
    )
    resolved = engine.recommend(
        _input(
            current_stage=SalesStage.VALUE_EXPLORATION,
            meaningful_interest=True,
            satisfied_information=frozenset(InformationGap),
        )
    )

    assert unresolved.sales_stage == SalesStage.VALUE_EXPLORATION
    assert resolved.sales_stage == SalesStage.NEXT_STEP


def test_meaningful_interest_can_move_discovery_toward_value_without_fabricating_pain() -> None:
    result = ConversationStrategyEngine().recommend(
        _input(current_stage=SalesStage.DISCOVERY, meaningful_interest=True)
    )

    assert result.sales_stage == SalesStage.VALUE_EXPLORATION
    assert InformationGap.PAIN_POINT in result.information_gaps


def test_objection_mode_is_advisory_and_contains_no_action() -> None:
    result = ConversationStrategyEngine().recommend(
        _input(current_stage=SalesStage.DISCOVERY, objection_present=True)
    )

    assert result.conversation_mode == ConversationMode.OBJECTION
    assert result.strategy_type == StrategyType.HANDLE_OBJECTION
    assert "action" not in {field.name for field in fields(result)}


def test_micro_commitment_recommendation_is_not_confirmation() -> None:
    result = ConversationStrategyEngine().recommend(_input(busy=True))

    assert result.micro_commitment == MicroCommitment.CALLBACK
    assert "confirmed" not in {field.name for field in fields(result)}


def test_information_gaps_are_typed_unique_and_bounded() -> None:
    result = ConversationStrategyEngine().recommend(_input())

    assert len(result.information_gaps) <= 3
    assert len(set(result.information_gaps)) == len(result.information_gaps)
    assert all(isinstance(gap, InformationGap) for gap in result.information_gaps)


def test_unknown_information_is_not_fabricated() -> None:
    result = ConversationStrategyEngine().recommend(
        _input(current_stage=SalesStage.DISCOVERY)
    )

    assert result.strategy_type != StrategyType.EXPLAIN_VALUE
    assert InformationGap.PAIN_POINT in result.information_gaps


def test_interrupted_question_preserves_relevant_previous_goal() -> None:
    pending = PendingConversationIntent("explain value", "relevant value point")
    result = ConversationStrategyEngine().recommend(
        _input(
            current_stage=SalesStage.DISCOVERY,
            question_present=True,
            interruption=InterruptionContext(
                True, InterruptionCategory.QUESTION, pending
            ),
        )
    )

    assert result.resume_previous_goal


def test_objection_discards_stale_interrupted_pitch_goal() -> None:
    pending = PendingConversationIntent("continue pitch", "unfinished pitch")
    result = ConversationStrategyEngine().recommend(
        _input(
            current_stage=SalesStage.DISCOVERY,
            objection_present=True,
            interruption=InterruptionContext(
                True, InterruptionCategory.OBJECTION, pending
            ),
        )
    )

    assert not result.resume_previous_goal


def test_same_structured_input_has_deterministic_replay() -> None:
    engine = ConversationStrategyEngine()
    value = _input(current_stage=SalesStage.DISCOVERY, question_present=True)

    assert engine.recommend(value) == engine.recommend(value)


def test_campaign_business_label_does_not_change_generic_strategy() -> None:
    engine = ConversationStrategyEngine()
    first = engine.recommend(_input(campaign_context_label="plumbing"))
    second = engine.recommend(_input(campaign_context_label="software"))

    assert first == second


def test_configured_campaign_goal_guides_without_creating_campaign_strategy_type() -> None:
    result = ConversationStrategyEngine().recommend(
        _input(campaign_goal="understand whether a follow-up is useful")
    )

    assert "follow-up" in result.communication_goal
    assert result.strategy_type == StrategyType.OPEN_CONVERSATION


def test_strategy_contract_contains_no_execution_authority_fields() -> None:
    names = {field.name for field in fields(ConversationStrategy)}

    assert names.isdisjoint(
        {
            "next_state",
            "execute_action",
            "service_authorization",
            "confirmed_contact",
            "discount",
            "pricing_authority",
            "persist_contact",
            "human_approval",
        }
    )


def test_strategy_recommendation_cannot_transition_authoritative_fsm() -> None:
    machine = ConversationStateMachine(
        load_config(get_settings().conversation_config_path)
    )
    before = machine.current_state

    result = ConversationStrategyEngine().recommend(
        _input(explicit_next_step=MicroCommitment.DEMO)
    )

    assert result.sales_stage == SalesStage.NEXT_STEP
    assert machine.current_state == before == ConversationState.NEW_CALL


@pytest.mark.parametrize(
    "changes",
    (
        {"campaign_goal": "x" * 201},
        {"campaign_context_label": "x" * 101},
    ),
)
def test_strategy_input_rejects_unbounded_metadata(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _input(**changes)


def test_strategy_output_rejects_duplicate_or_unbounded_gaps() -> None:
    with pytest.raises(ValueError):
        ConversationStrategy(
            SalesStage.DISCOVERY,
            ConversationMode.NORMAL,
            "learn",
            StrategyType.DISCOVER_NEED,
            (InformationGap.PAIN_POINT, InformationGap.PAIN_POINT),
        )
