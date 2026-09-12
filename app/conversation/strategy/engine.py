"""Pure deterministic conversation-strategy recommendations."""

from __future__ import annotations

from app.conversation.strategy.contracts import (
    ConversationMode,
    ConversationStrategy,
    ConversationStrategyInput,
    InformationGap,
    MicroCommitment,
    SalesStage,
    StrategyType,
)
from app.core.constants import ConversationState

_GAP_ORDER = (
    InformationGap.CURRENT_WORKFLOW,
    InformationGap.PAIN_POINT,
    InformationGap.CURRENT_SOLUTION,
    InformationGap.SATISFACTION,
    InformationGap.DECISION_AUTHORITY,
    InformationGap.URGENCY,
    InformationGap.NEXT_STEP_PREFERENCE,
)


class ConversationStrategyEngine:
    """Recommend sales direction without selecting actions or transitions."""

    def recommend(self, strategy_input: ConversationStrategyInput) -> ConversationStrategy:
        """Return the same bounded recommendation for the same structured input."""
        gaps = _remaining_gaps(strategy_input)
        pending = strategy_input.interruption.previous_intent
        resumable = bool(pending is not None and pending.active)

        if strategy_input.objection_present:
            return _strategy(
                strategy_input.current_stage,
                ConversationMode.OBJECTION,
                "understand and address the prospect's objection before progressing",
                StrategyType.HANDLE_OBJECTION,
                gaps,
            )
        if strategy_input.question_present:
            return _strategy(
                strategy_input.current_stage,
                ConversationMode.QUESTION_DETOUR,
                "answer the prospect's question safely before returning to the prior goal",
                StrategyType.HANDLE_QUESTION,
                gaps,
                MicroCommitment.ANSWER_ONE_QUESTION,
                resumable,
            )
        if strategy_input.clarification_needed:
            return _strategy(
                strategy_input.current_stage,
                ConversationMode.CLARIFICATION,
                "clarify the current meaning before continuing",
                StrategyType.CLARIFY,
                gaps,
            )
        if strategy_input.busy:
            return _strategy(
                strategy_input.current_stage,
                ConversationMode.BUSY,
                "reduce pressure and ask for the smallest reasonable next step",
                StrategyType.ASK_MICRO_COMMITMENT,
                gaps,
                MicroCommitment.CALLBACK,
            )
        if strategy_input.explicit_next_step not in (None, MicroCommitment.NONE):
            return _strategy(
                SalesStage.NEXT_STEP,
                ConversationMode.NORMAL,
                "respond to the prospect's explicit request for a next step",
                StrategyType.PROPOSE_NEXT_STEP,
                gaps,
                strategy_input.explicit_next_step,
            )
        if strategy_input.current_solution_present and (
            InformationGap.SATISFACTION not in strategy_input.satisfied_information
        ):
            satisfaction_gaps = (
                InformationGap.SATISFACTION,
                *(gap for gap in gaps if gap != InformationGap.SATISFACTION),
            )[:3]
            return _strategy(
                strategy_input.current_stage,
                ConversationMode.NORMAL,
                "understand satisfaction and any gap in the current solution",
                StrategyType.UNDERSTAND_CURRENT_SOLUTION,
                satisfaction_gaps,
            )
        if strategy_input.current_stage == SalesStage.OPENING:
            if strategy_input.current_state in {
                ConversationState.NEW_CALL,
                ConversationState.GREETING,
            }:
                return _strategy(
                    SalesStage.OPENING,
                    ConversationMode.NORMAL,
                    _opening_goal(strategy_input.campaign_goal),
                    StrategyType.OPEN_CONVERSATION,
                    gaps,
                    MicroCommitment.PERMISSION_TO_CONTINUE,
                )
            return _strategy(
                SalesStage.DISCOVERY,
                ConversationMode.NORMAL,
                "understand the prospect's current workflow and unmet need",
                StrategyType.DISCOVER_NEED,
                gaps,
            )
        if strategy_input.current_stage == SalesStage.DISCOVERY:
            if strategy_input.pain_present:
                return _strategy(
                    SalesStage.VALUE_EXPLORATION,
                    ConversationMode.NORMAL,
                    "connect the established need to relevant value without overclaiming",
                    StrategyType.EXPLAIN_VALUE,
                    gaps,
                )
            if strategy_input.meaningful_interest:
                if not gaps:
                    return _strategy(
                        SalesStage.NEXT_STEP,
                        ConversationMode.NORMAL,
                        "ask which reasonable next step the prospect prefers",
                        StrategyType.ASK_MICRO_COMMITMENT,
                        (),
                        MicroCommitment.ANSWER_ONE_QUESTION,
                    )
                return _strategy(
                    SalesStage.VALUE_EXPLORATION,
                    ConversationMode.NORMAL,
                    "explore relevant value while resolving the highest-priority gap",
                    StrategyType.EXPLAIN_VALUE,
                    gaps,
                )
            return _strategy(
                SalesStage.DISCOVERY,
                ConversationMode.NORMAL,
                "learn the highest-priority missing information",
                StrategyType.DISCOVER_NEED,
                gaps,
            )
        if strategy_input.current_stage == SalesStage.VALUE_EXPLORATION:
            if strategy_input.meaningful_interest and not gaps:
                return _strategy(
                    SalesStage.NEXT_STEP,
                    ConversationMode.NORMAL,
                    "ask which reasonable next step the prospect prefers",
                    StrategyType.ASK_MICRO_COMMITMENT,
                    (),
                    MicroCommitment.ANSWER_ONE_QUESTION,
                )
            return _strategy(
                SalesStage.VALUE_EXPLORATION,
                ConversationMode.NORMAL,
                "explain relevant value while resolving the remaining information gap",
                StrategyType.EXPLAIN_VALUE,
                gaps,
            )
        return _strategy(
            strategy_input.current_stage,
            ConversationMode.NORMAL,
            "qualify the next missing point without forcing a scripted sequence",
            StrategyType.QUALIFY_NEED,
            gaps,
        )


def _remaining_gaps(value: ConversationStrategyInput) -> tuple[InformationGap, ...]:
    satisfied = set(value.satisfied_information)
    if value.pain_present:
        satisfied.add(InformationGap.PAIN_POINT)
    if value.current_solution_present:
        satisfied.add(InformationGap.CURRENT_SOLUTION)
    if value.explicit_next_step not in (None, MicroCommitment.NONE):
        satisfied.add(InformationGap.NEXT_STEP_PREFERENCE)
    return tuple(gap for gap in _GAP_ORDER if gap not in satisfied)[:3]


def _opening_goal(campaign_goal: str | None) -> str:
    if campaign_goal is None or not campaign_goal.strip():
        return "open the conversation and earn permission to continue"
    return f"earn permission to explore the configured goal: {campaign_goal}"[:240]


def _strategy(
    stage: SalesStage,
    mode: ConversationMode,
    goal: str,
    strategy_type: StrategyType,
    gaps: tuple[InformationGap, ...],
    micro: MicroCommitment = MicroCommitment.NONE,
    resume: bool = False,
) -> ConversationStrategy:
    return ConversationStrategy(stage, mode, goal, strategy_type, gaps, micro, resume)
