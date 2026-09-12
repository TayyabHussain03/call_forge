"""Slice 6 multi-turn continuity scenarios over the offline simulation runner."""

from __future__ import annotations

from app.brain.contracts import BrainProposal
from app.contracts.conversation_context import ConversationContext
from app.conversation.response_planning.contracts import (
    AddresseeStatus,
    ConversationMove,
    InterruptionCategory,
    InterruptionContext,
    InterruptionHandling,
    PendingConversationIntent,
    QuestionStrategy,
)
from app.core.constants import AgentAction, Intent, TopicCategory
from app.simulations.contracts import SimulationScenario, SimulationTurn
from app.simulations.runner import run_scenario


def _proposal() -> BrainProposal:
    return BrainProposal(
        detected_intent=Intent.ASKING_QUESTION,
        topic_category=TopicCategory.BUSINESS_QUESTION,
        proposed_action=AgentAction.GREET,
    )


def _pending() -> PendingConversationIntent:
    return PendingConversationIntent(
        "explain website value",
        "Explain how a website can complement existing social presence.",
    )


def test_scenario_q_interrupted_value_explanation() -> None:
    result = run_scenario(
        SimulationScenario(
            "q_interrupted_value_explanation",
            ConversationContext("q"),
            turns=(
                SimulationTurn(
                    "How is that different from Facebook?",
                    proposal=_proposal(),
                    conversation_category=InterruptionCategory.QUESTION,
                    interruption=InterruptionContext(
                        True, InterruptionCategory.QUESTION, _pending()
                    ),
                ),
            ),
        )
    )
    plan = result.turns[0].response_plan
    assert plan is not None
    assert plan.communicative_goal == ConversationMove.ANSWER_CURRENT_QUESTION
    assert plan.resume_previous_point


def test_scenario_r_objection_interrupts_pitch() -> None:
    result = run_scenario(
        SimulationScenario(
            "r_objection_interrupts_pitch",
            ConversationContext("r"),
            turns=(
                SimulationTurn(
                    "We tried that before and it did not help.",
                    proposal=_proposal(),
                    conversation_category=InterruptionCategory.OBJECTION,
                    interruption=InterruptionContext(
                        True, InterruptionCategory.OBJECTION, _pending()
                    ),
                ),
            ),
        )
    )
    plan = result.turns[0].response_plan
    assert plan is not None
    assert plan.communicative_goal == ConversationMove.EXPLORE_OBJECTION
    assert plan.interruption_handling == InterruptionHandling.DROP_STALE_POINT
    assert plan.question_strategy == QuestionStrategy.EXPLORE_WITH_ONE_QUESTION


def test_scenario_s_possible_background_conversation() -> None:
    result = run_scenario(
        SimulationScenario(
            "s_possible_background_conversation",
            ConversationContext("s"),
            turns=(
                SimulationTurn(
                    "Unrelated overlapping speech.",
                    provider_failure=True,
                    addressee_status=AddresseeStatus.ADDRESSEE_UNCERTAIN,
                    conversation_category=InterruptionCategory.ADDRESSEE_UNCERTAIN,
                    interruption=InterruptionContext(
                        True, InterruptionCategory.ADDRESSEE_UNCERTAIN, _pending()
                    ),
                ),
            ),
        )
    )
    plan = result.turns[0].response_plan
    assert plan is not None
    assert plan.communicative_goal == ConversationMove.CLARIFY_ADDRESSEE
    assert plan.clarification_required
    assert result.final_state.value == "new_call"


def test_scenario_t_confirms_background_speech_was_not_for_agent() -> None:
    result = run_scenario(
        SimulationScenario(
            "t_background_speech_not_for_agent",
            ConversationContext("t"),
            turns=(
                SimulationTurn(
                    "Background speech.",
                    provider_failure=True,
                    addressee_status=AddresseeStatus.ADDRESSEE_UNCERTAIN,
                    interruption=InterruptionContext(True, previous_intent=_pending()),
                ),
                SimulationTurn(
                    "Sorry, that was for someone here.",
                    provider_failure=True,
                    addressee_status=AddresseeStatus.NOT_ADDRESSED_TO_AGENT,
                    interruption=InterruptionContext(True, previous_intent=_pending()),
                ),
            ),
        )
    )
    plan = result.turns[1].response_plan
    assert plan is not None
    assert plan.communicative_goal == ConversationMove.CONTINUE_PRIOR_CONTEXT
    assert plan.pending_intent == _pending()
    assert result.final_state.value == "new_call"
