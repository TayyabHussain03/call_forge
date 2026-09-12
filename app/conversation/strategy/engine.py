"""Pure deterministic conversation-strategy recommendations."""

from __future__ import annotations

from app.conversation.prospect_intelligence.contracts import (
    DecisionAuthority,
    ObjectionType,
    PreferredNextStep,
    ProspectRole,
    SolutionSatisfaction,
)
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
    InformationGap.ROLE,
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
        role = _role(strategy_input)
        authority = _decision_authority(strategy_input)
        objection = _objection(strategy_input)
        current_solution = _has_current_solution(strategy_input)
        satisfaction_known = _satisfaction_known(strategy_input)
        busy = strategy_input.busy or _observed_bool(strategy_input, "busy")
        interested = strategy_input.meaningful_interest or _observed_bool(
            strategy_input, "interested"
        )
        pain_present = strategy_input.pain_present or _has_observed_pain(strategy_input)
        next_step = strategy_input.explicit_next_step
        if next_step in (None, MicroCommitment.NONE):
            next_step = _preferred_micro_commitment(strategy_input)

        if (
            objection == ObjectionType.ALREADY_HAVE_PROVIDER
            and current_solution
            and not satisfaction_known
        ):
            return _strategy(
                strategy_input.current_stage,
                ConversationMode.OBJECTION,
                "understand satisfaction and any gap in the current solution",
                StrategyType.UNDERSTAND_CURRENT_SOLUTION,
                _satisfaction_first(gaps),
            )

        if strategy_input.objection_present or objection not in {
            ObjectionType.NONE,
            ObjectionType.UNKNOWN,
        }:
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
        if busy:
            return _strategy(
                strategy_input.current_stage,
                ConversationMode.BUSY,
                "reduce pressure and ask for the smallest reasonable next step",
                StrategyType.ASK_MICRO_COMMITMENT,
                gaps,
                MicroCommitment.CALLBACK,
            )
        if next_step not in (None, MicroCommitment.NONE):
            return _strategy(
                SalesStage.NEXT_STEP,
                ConversationMode.NORMAL,
                (
                    "respond to the prospect's explicit request for a next step"
                    if _has_explicit_next_step(strategy_input)
                    else "explore the likely preferred next step without assuming commitment"
                ),
                StrategyType.PROPOSE_NEXT_STEP,
                gaps,
                next_step,
            )
        if current_solution and not satisfaction_known:
            return _strategy(
                strategy_input.current_stage,
                ConversationMode.NORMAL,
                "understand satisfaction and any gap in the current solution",
                StrategyType.UNDERSTAND_CURRENT_SOLUTION,
                _satisfaction_first(gaps),
            )
        if role in {ProspectRole.RECEPTIONIST, ProspectRole.GATEKEEPER}:
            return _strategy(
                strategy_input.current_stage,
                ConversationMode.ROLE_ROUTING,
                "state the purpose briefly and respectfully identify the appropriate person",
                StrategyType.ROUTE_TO_DECISION_MAKER,
                gaps,
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
            if pain_present:
                return _strategy(
                    SalesStage.VALUE_EXPLORATION,
                    ConversationMode.NORMAL,
                    "connect the established need to relevant value without overclaiming",
                    StrategyType.EXPLAIN_VALUE,
                    gaps,
                )
            if interested:
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
            if role in {ProspectRole.OWNER, ProspectRole.DECISION_MAKER} or authority in {
                DecisionAuthority.HIGH,
                DecisionAuthority.FINAL,
            }:
                return _strategy(
                    SalesStage.DISCOVERY,
                    ConversationMode.NORMAL,
                    "understand the business impact and highest-priority unmet need",
                    StrategyType.DISCOVER_BUSINESS_IMPACT,
                    gaps,
                )
            if role in {ProspectRole.MANAGER, ProspectRole.ASSISTANT_MANAGER}:
                if (
                    strategy_input.strategy_hint is not None
                    and strategy_input.strategy_hint.strategy_type
                    == StrategyType.DISCOVER_BUSINESS_IMPACT
                ):
                    return _strategy(
                        SalesStage.DISCOVERY,
                        ConversationMode.NORMAL,
                        "understand operational business impact and the current workflow",
                        StrategyType.DISCOVER_BUSINESS_IMPACT,
                        gaps,
                    )
                return _strategy(
                    SalesStage.DISCOVERY,
                    ConversationMode.NORMAL,
                    "understand the current workflow and operational impact",
                    StrategyType.DISCOVER_NEED,
                    gaps,
                )
            if strategy_input.prospect_intelligence is not None and role == ProspectRole.UNKNOWN:
                return _strategy(
                    SalesStage.QUALIFICATION,
                    ConversationMode.NORMAL,
                    "lightly clarify the prospect's role and current workflow",
                    StrategyType.QUALIFY_NEED,
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
            if interested and not gaps:
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
    intelligence = value.prospect_intelligence
    if intelligence is not None:
        observed = intelligence.observed
        if (
            observed.explicit_role is not None
            and observed.explicit_role.value != ProspectRole.UNKNOWN
        ):
            satisfied.add(InformationGap.ROLE)
        if observed.explicit_pain_points:
            satisfied.add(InformationGap.PAIN_POINT)
        if observed.explicit_current_solution is not None:
            satisfied.add(InformationGap.CURRENT_SOLUTION)
            if (
                observed.explicit_current_solution.satisfaction
                != SolutionSatisfaction.UNKNOWN
            ):
                satisfied.add(InformationGap.SATISFACTION)
        if (
            observed.explicit_decision_authority_statement is not None
            and observed.explicit_decision_authority_statement.value
            != DecisionAuthority.UNKNOWN
        ):
            satisfied.add(InformationGap.DECISION_AUTHORITY)
        if observed.explicit_timing_preference is not None:
            satisfied.add(InformationGap.URGENCY)
        if (
            observed.explicit_next_step_request is not None
            and observed.explicit_next_step_request.value
            != PreferredNextStep.UNKNOWN
        ):
            satisfied.add(InformationGap.NEXT_STEP_PREFERENCE)
    if value.pain_present:
        satisfied.add(InformationGap.PAIN_POINT)
    if value.current_solution_present:
        satisfied.add(InformationGap.CURRENT_SOLUTION)
    if value.explicit_next_step not in (None, MicroCommitment.NONE):
        satisfied.add(InformationGap.NEXT_STEP_PREFERENCE)
    return tuple(gap for gap in _GAP_ORDER if gap not in satisfied)[:3]


def _role(value: ConversationStrategyInput) -> ProspectRole:
    intelligence = value.prospect_intelligence
    if intelligence is None:
        return ProspectRole.UNKNOWN
    if intelligence.observed.explicit_role is not None:
        return intelligence.observed.explicit_role.value
    if intelligence.inferred.likely_role is not None:
        return intelligence.inferred.likely_role.value
    return ProspectRole.UNKNOWN


def _decision_authority(value: ConversationStrategyInput) -> DecisionAuthority:
    intelligence = value.prospect_intelligence
    if intelligence is None:
        return DecisionAuthority.UNKNOWN
    observed = intelligence.observed.explicit_decision_authority_statement
    if observed is not None:
        return observed.value
    inferred = intelligence.inferred.decision_authority
    return inferred.value if inferred is not None else DecisionAuthority.UNKNOWN


def _objection(value: ConversationStrategyInput) -> ObjectionType:
    intelligence = value.prospect_intelligence
    if intelligence is None:
        return ObjectionType.NONE
    observed = intelligence.observed.explicit_objection
    if observed is not None:
        return observed.value
    inferred = intelligence.inferred.objection_type
    return inferred.value if inferred is not None else ObjectionType.NONE


def _has_current_solution(value: ConversationStrategyInput) -> bool:
    intelligence = value.prospect_intelligence
    return value.current_solution_present or bool(
        intelligence is not None
        and intelligence.observed.explicit_current_solution is not None
    )


def _has_observed_pain(value: ConversationStrategyInput) -> bool:
    intelligence = value.prospect_intelligence
    return bool(
        intelligence is not None and intelligence.observed.explicit_pain_points
    )


def _satisfaction_known(value: ConversationStrategyInput) -> bool:
    if InformationGap.SATISFACTION in value.satisfied_information:
        return True
    intelligence = value.prospect_intelligence
    if intelligence is None:
        return False
    solution = intelligence.observed.explicit_current_solution
    return bool(
        solution is not None
        and solution.satisfaction != SolutionSatisfaction.UNKNOWN
    )


def _observed_bool(value: ConversationStrategyInput, field: str) -> bool:
    intelligence = value.prospect_intelligence
    if intelligence is None:
        return False
    observed = intelligence.observed
    signal = (
        observed.explicit_busy_signal
        if field == "busy"
        else observed.explicit_interest_signal
    )
    return bool(signal is not None and signal.value)


def _preferred_micro_commitment(
    value: ConversationStrategyInput,
) -> MicroCommitment | None:
    intelligence = value.prospect_intelligence
    if intelligence is None:
        return None
    observed = intelligence.observed
    if observed.explicit_human_request is not None and observed.explicit_human_request.value:
        return MicroCommitment.HUMAN_FOLLOW_UP
    preference = observed.explicit_next_step_request
    preferred = (
        preference.value
        if preference is not None
        else (
            intelligence.inferred.preferred_next_step.value
            if intelligence.inferred.preferred_next_step is not None
            else PreferredNextStep.UNKNOWN
        )
    )
    return {
        PreferredNextStep.SEND_INFORMATION: MicroCommitment.SEND_INFORMATION,
        PreferredNextStep.CALLBACK: MicroCommitment.CALLBACK,
        PreferredNextStep.DEMO: MicroCommitment.DEMO,
        PreferredNextStep.HUMAN_FOLLOW_UP: MicroCommitment.HUMAN_FOLLOW_UP,
        PreferredNextStep.EMAIL: MicroCommitment.SHARE_CONTACT,
    }.get(preferred)


def _has_explicit_next_step(value: ConversationStrategyInput) -> bool:
    if value.explicit_next_step not in (None, MicroCommitment.NONE):
        return True
    intelligence = value.prospect_intelligence
    if intelligence is None:
        return False
    observed = intelligence.observed
    return bool(
        observed.explicit_next_step_request is not None
        or (
            observed.explicit_human_request is not None
            and observed.explicit_human_request.value
        )
    )


def _satisfaction_first(
    gaps: tuple[InformationGap, ...],
) -> tuple[InformationGap, ...]:
    return (
        InformationGap.SATISFACTION,
        *(gap for gap in gaps if gap != InformationGap.SATISFACTION),
    )[:3]


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
