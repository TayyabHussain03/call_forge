"""Slice 6 multi-turn continuity scenarios over the offline simulation runner."""

from __future__ import annotations

import re

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
    ResponseLength,
)
from app.conversation.response_rendering.contracts import TrustedRenderingContext
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
                    trusted_rendering_context=TrustedRenderingContext(
                        primary_fact="A website can complement an existing social presence"
                    ),
                ),
            ),
        )
    )
    plan = result.turns[0].response_plan
    assert plan is not None
    assert plan.communicative_goal == ConversationMove.ANSWER_CURRENT_QUESTION
    assert plan.resume_previous_point
    rendered = result.turns[0].rendered_response
    assert rendered is not None
    assert rendered.text.startswith("A website can complement")
    assert _sentence_count(rendered.text) <= 2


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
    rendered = result.turns[0].rendered_response
    assert rendered is not None
    assert "what did not work" in rendered.text
    assert "complement existing social" not in rendered.text


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
    rendered = result.turns[0].rendered_response
    assert rendered is not None
    assert rendered.clarification_required
    assert rendered.text.endswith("?")
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
    rendered = result.turns[1].rendered_response
    assert rendered is not None
    assert "background speech" not in rendered.text.lower()
    assert _pending().summary in rendered.text
    assert result.final_state.value == "new_call"


def test_normal_business_question_renders_moderate_grounded_explanation() -> None:
    result = run_scenario(
        SimulationScenario(
            "normal_moderate_explanation",
            ConversationContext("moderate"),
            turns=(
                SimulationTurn(
                    "What does that service help with?",
                    proposal=_proposal(),
                    conversation_category=InterruptionCategory.QUESTION,
                    trusted_rendering_context=TrustedRenderingContext(
                        service_name="Approved directory service",
                        service_facts=("helps keep listed business details accurate",),
                    ),
                ),
            ),
        )
    )
    plan = result.turns[0].response_plan
    rendered = result.turns[0].rendered_response
    assert plan is not None and plan.response_length == ResponseLength.MODERATE
    assert rendered is not None
    assert "listed business details accurate" in rendered.text
    assert _sentence_count(rendered.text) <= 2


def _sentence_count(text: str) -> int:
    return len(re.split(r"(?<=[.!?])\s+", text))
