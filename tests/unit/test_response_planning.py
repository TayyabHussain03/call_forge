"""Focused Slice 6 tests for deterministic conversational response planning."""

from __future__ import annotations

from dataclasses import fields

import pytest

from app.conversation.response_planning.contracts import (
    AddresseeStatus,
    AcknowledgementKind,
    AuthoritativeResultKind,
    ConversationMove,
    ExplanationNeed,
    InterruptionCategory,
    InterruptionContext,
    InterruptionHandling,
    PendingConversationIntent,
    QuestionStrategy,
    ResponseLength,
    ResponsePlan,
    ResponsePlanningInput,
    VoiceActivityMetadata,
)
from app.conversation.response_planning.planner import ResponsePlanner
from app.core.constants import AgentAction, ConversationState


def _input(**changes: object) -> ResponsePlanningInput:
    values: dict[str, object] = {
        "authoritative_state": ConversationState.LISTEN,
        "authoritative_result": AuthoritativeResultKind.EXECUTED,
        "current_prospect_message": "Tell me more.",
        "approved_action": AgentAction.ANSWER_QUESTION,
        "explanation_need": ExplanationNeed.SIMPLE,
    }
    values.update(changes)
    return ResponsePlanningInput(**values)  # type: ignore[arg-type]


def _pending() -> PendingConversationIntent:
    return PendingConversationIntent(
        goal="explain website value",
        summary="Explain how an owned website complements existing social presence.",
    )


def test_simple_approved_turn_gets_short_plan() -> None:
    plan = ResponsePlanner().plan(_input())

    assert plan.communicative_goal == ConversationMove.COMMUNICATE_RESULT
    assert plan.response_length == ResponseLength.SHORT
    assert plan.question_strategy == QuestionStrategy.NONE


def test_service_question_allows_moderate_detail() -> None:
    plan = ResponsePlanner().plan(
        _input(
            conversation_category=InterruptionCategory.QUESTION,
            explanation_need=ExplanationNeed.STANDARD,
        )
    )

    assert plan.communicative_goal == ConversationMove.ANSWER_CURRENT_QUESTION
    assert plan.response_length == ResponseLength.MODERATE
    assert plan.question_strategy == QuestionStrategy.ANSWER_THEN_FOLLOW_UP


def test_complex_question_can_request_detailed_response_without_sentence_cap() -> None:
    plan = ResponsePlanner().plan(
        _input(
            conversation_category=InterruptionCategory.QUESTION,
            explanation_need=ExplanationNeed.COMPLEX,
        )
    )
    assert plan.response_length == ResponseLength.DETAILED
    assert not hasattr(plan, "max_sentences")


def test_interrupted_question_becomes_primary_without_restarting_old_point() -> None:
    plan = ResponsePlanner().plan(
        _input(
            conversation_category=InterruptionCategory.QUESTION,
            interruption=InterruptionContext(
                was_interrupted=True,
                category=InterruptionCategory.QUESTION,
                previous_intent=_pending(),
                voice_activity=VoiceActivityMetadata(barge_in_detected=True),
            ),
        )
    )

    assert plan.communicative_goal == ConversationMove.ANSWER_CURRENT_QUESTION
    assert plan.interruption_handling == InterruptionHandling.INTEGRATE_IF_USEFUL
    assert plan.resume_previous_point
    assert plan.pending_intent == _pending()


@pytest.mark.parametrize(
    ("category", "goal"),
    [
        (InterruptionCategory.TOPIC_SHIFT, ConversationMove.FOLLOW_NEW_DIRECTION),
        (InterruptionCategory.CORRECTION, ConversationMove.INCORPORATE_CORRECTION),
    ],
)
def test_changed_direction_discards_stale_pending_intent(
    category: InterruptionCategory, goal: ConversationMove
) -> None:
    plan = ResponsePlanner().plan(
        _input(
            conversation_category=category,
            interruption=InterruptionContext(True, category, _pending()),
        )
    )

    assert plan.communicative_goal == goal
    assert plan.interruption_handling == InterruptionHandling.DROP_STALE_POINT
    assert not plan.resume_previous_point
    assert plan.pending_intent is None


def test_objection_adapts_and_explores_instead_of_resuming_pitch() -> None:
    plan = ResponsePlanner().plan(
        _input(
            conversation_category=InterruptionCategory.OBJECTION,
            interruption=InterruptionContext(
                True, InterruptionCategory.OBJECTION, _pending()
            ),
        )
    )

    assert plan.communicative_goal == ConversationMove.EXPLORE_OBJECTION
    assert plan.acknowledgement != AcknowledgementKind.NONE
    assert plan.question_strategy == QuestionStrategy.EXPLORE_WITH_ONE_QUESTION
    assert plan.pending_intent is None


def test_uncertain_addressee_forces_short_clarification_and_holds_context() -> None:
    plan = ResponsePlanner().plan(
        _input(
            authoritative_result=AuthoritativeResultKind.AUTHORITY_APPROVED,
            approved_action=AgentAction.CONFIRM_CONTACT,
            addressee_status=AddresseeStatus.ADDRESSEE_UNCERTAIN,
            interruption=InterruptionContext(
                True,
                InterruptionCategory.ADDRESSEE_UNCERTAIN,
                _pending(),
                VoiceActivityMetadata(overlapping_speech=True),
            ),
        )
    )

    assert plan.communicative_goal == ConversationMove.CLARIFY_ADDRESSEE
    assert plan.response_length == ResponseLength.SHORT
    assert plan.clarification_required
    assert plan.question_strategy == QuestionStrategy.CONFIRM_ADDRESSEE
    assert plan.pending_intent == _pending()
    assert not hasattr(plan, "approved_action")


def test_confirmed_not_for_agent_ignores_utterance_and_preserves_prior_context() -> None:
    plan = ResponsePlanner().plan(
        _input(
            current_prospect_message="That was for my colleague.",
            addressee_status=AddresseeStatus.NOT_ADDRESSED_TO_AGENT,
            interruption=InterruptionContext(True, previous_intent=_pending()),
        )
    )

    assert plan.communicative_goal == ConversationMove.CONTINUE_PRIOR_CONTEXT
    assert plan.interruption_handling == InterruptionHandling.HOLD_PRIOR_CONTEXT
    assert plan.resume_previous_point
    assert plan.pending_intent == _pending()


def test_confirmed_for_agent_resumes_normal_processing() -> None:
    plan = ResponsePlanner().plan(
        _input(
            addressee_status=AddresseeStatus.ADDRESSED_TO_AGENT,
            conversation_category=InterruptionCategory.QUESTION,
        )
    )
    assert plan.communicative_goal == ConversationMove.ANSWER_CURRENT_QUESTION
    assert not plan.clarification_required


def test_multiple_interruptions_do_not_resurface_discarded_point() -> None:
    planner = ResponsePlanner()
    objection = planner.plan(
        _input(
            conversation_category=InterruptionCategory.OBJECTION,
            interruption=InterruptionContext(True, previous_intent=_pending()),
        )
    )
    next_plan = planner.plan(
        _input(
            conversation_category=InterruptionCategory.QUESTION,
            interruption=InterruptionContext(True, previous_intent=objection.pending_intent),
        )
    )

    assert objection.pending_intent is None
    assert next_plan.pending_intent is None
    assert not next_plan.resume_previous_point


def test_acknowledgement_is_not_forced_and_does_not_repeat() -> None:
    planner = ResponsePlanner()
    ordinary = planner.plan(_input())
    first = planner.plan(_input(conversation_category=InterruptionCategory.OBJECTION))
    second = planner.plan(
        _input(
            conversation_category=InterruptionCategory.OBJECTION,
            previous_acknowledgement=first.acknowledgement,
        )
    )

    assert ordinary.acknowledgement == AcknowledgementKind.NONE
    assert first.acknowledgement != second.acknowledgement


@pytest.mark.parametrize(
    "result",
    [
        AuthoritativeResultKind.FALLBACK,
        AuthoritativeResultKind.REDIRECT,
        AuthoritativeResultKind.ESCALATE,
    ],
)
def test_safety_results_have_deterministic_communication_plans(
    result: AuthoritativeResultKind,
) -> None:
    plan = ResponsePlanner().plan(_input(authoritative_result=result))
    assert plan.communicative_goal in {
        ConversationMove.SAFE_RECOVERY,
        ConversationMove.REDIRECT_SAFELY,
        ConversationMove.ACKNOWLEDGE_ESCALATION,
    }


def test_response_plan_cannot_encode_execution_authority() -> None:
    forbidden = {
        "next_state",
        "authorized_service",
        "execute_action",
        "persist_contact",
        "approve_discount",
        "human_approval_granted",
        "contact_confirmed",
    }
    assert forbidden.isdisjoint({field.name for field in fields(ResponsePlan)})

    with pytest.raises(TypeError):
        ResponsePlan(  # type: ignore[call-arg]
            communicative_goal=ConversationMove.COMMUNICATE_RESULT,
            response_length=ResponseLength.SHORT,
            tone=_input().tone,
            acknowledgement=AcknowledgementKind.NONE,
            question_strategy=QuestionStrategy.NONE,
            clarification_required=False,
            interruption_handling=InterruptionHandling.NONE,
            resume_previous_point=False,
            pending_intent=None,
            addressee_status=AddresseeStatus.ADDRESSED_TO_AGENT,
            approve_discount=True,
        )


def test_planner_is_side_effect_free_and_deterministic() -> None:
    planning_input = _input(
        conversation_category=InterruptionCategory.QUESTION,
        interruption=InterruptionContext(True, previous_intent=_pending()),
        trusted_context_summary=("website already exists",),
    )
    planner = ResponsePlanner()

    assert planner.plan(planning_input) == planner.plan(planning_input)
    assert planning_input.interruption.previous_intent == _pending()
    assert planning_input.authoritative_state == ConversationState.LISTEN
