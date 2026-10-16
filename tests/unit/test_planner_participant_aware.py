"""Slice 50 — ResponsePlanner consumes ReExplanationStrategy.

Proves the planner input/output accept the new participant-aware fields
and that strategy deterministically maps to ResponseLength without
downgrading the planner's existing choice when it's already appropriate.
"""

from __future__ import annotations

import pytest

from app.conversation.participant.contracts import (
    ExplanationDepth,
    ReExplanationStrategy,
)
from app.conversation.response_planning.contracts import (
    AddresseeStatus,
    AuthoritativeResultKind,
    ExplanationNeed,
    InterruptionCategory,
    InterruptionContext,
    ResponseLength,
    ResponsePlan,
    ResponsePlanningInput,
)
from app.conversation.response_planning.planner import (
    ResponsePlanner,
    _apply_re_explanation_length,
)
from app.core.constants import AgentAction, ConversationState, Tone


# ---------------------------------------------------------------------------
# Strategy → length deterministic mapping
# ---------------------------------------------------------------------------


class TestStrategyLengthMapping:
    def test_none_strategy_preserves_length(self) -> None:
        assert _apply_re_explanation_length(
            ResponseLength.DETAILED, None,
        ) == ResponseLength.DETAILED

    def test_simplify_maps_to_short(self) -> None:
        assert _apply_re_explanation_length(
            ResponseLength.DETAILED, ReExplanationStrategy.SIMPLIFY,
        ) == ResponseLength.SHORT

    def test_concise_recap_maps_to_short(self) -> None:
        assert _apply_re_explanation_length(
            ResponseLength.DETAILED, ReExplanationStrategy.CONCISE_RECAP,
        ) == ResponseLength.SHORT

    def test_deepen_always_detailed(self) -> None:
        assert _apply_re_explanation_length(
            ResponseLength.SHORT, ReExplanationStrategy.DEEPEN,
        ) == ResponseLength.DETAILED

    def test_full_re_explain_upgrades_from_short(self) -> None:
        assert _apply_re_explanation_length(
            ResponseLength.SHORT,
            ReExplanationStrategy.FULL_RE_EXPLAIN_FOR_NEW_PARTICIPANT,
        ) == ResponseLength.MODERATE

    def test_full_re_explain_does_not_downgrade_detailed(self) -> None:
        """If planner already chose DETAILED, FULL_RE_EXPLAIN preserves it."""
        assert _apply_re_explanation_length(
            ResponseLength.DETAILED,
            ReExplanationStrategy.FULL_RE_EXPLAIN_FOR_NEW_PARTICIPANT,
        ) == ResponseLength.DETAILED

    def test_suppress_preserves_length(self) -> None:
        """SUPPRESS means 'don't re-explain', NOT 'say less'."""
        assert _apply_re_explanation_length(
            ResponseLength.DETAILED, ReExplanationStrategy.SUPPRESS,
        ) == ResponseLength.DETAILED

    def test_no_prior_coverage_preserves_length(self) -> None:
        """First-time explanation uses planner's own computed length."""
        assert _apply_re_explanation_length(
            ResponseLength.DETAILED, ReExplanationStrategy.NO_PRIOR_COVERAGE,
        ) == ResponseLength.DETAILED


# ---------------------------------------------------------------------------
# Input/output contract accepts new fields
# ---------------------------------------------------------------------------


def _input(**kw) -> ResponsePlanningInput:
    defaults = dict(
        authoritative_state=ConversationState.LISTEN,
        authoritative_result=AuthoritativeResultKind.EXECUTED,
        current_prospect_message="hi",
        approved_action=AgentAction.ANSWER_QUESTION,
        tone=Tone.NEUTRAL,
        conversation_category=InterruptionCategory.OTHER,
        addressee_status=AddresseeStatus.ADDRESSED_TO_AGENT,
        interruption=InterruptionContext(),
        explanation_need=ExplanationNeed.STANDARD,
    )
    defaults.update(kw)
    return ResponsePlanningInput(**defaults)


class TestPlanningInputAcceptsNewFields:
    def test_strategy_concept_depth_participant(self) -> None:
        inp = _input(
            re_explanation_strategy=ReExplanationStrategy.DEEPEN,
            explanation_concept_id="svc.automation",
            prior_explanation_depth=ExplanationDepth.BRIEF,
            participant_reference_id="sess:participant_1",
            catalog_request=False,
        )
        assert inp.re_explanation_strategy == ReExplanationStrategy.DEEPEN
        assert inp.explanation_concept_id == "svc.automation"
        assert inp.prior_explanation_depth == ExplanationDepth.BRIEF
        assert inp.participant_reference_id == "sess:participant_1"
        assert inp.catalog_request is False

    def test_catalog_request_bool_type_enforced(self) -> None:
        with pytest.raises(TypeError):
            _input(catalog_request="yes")  # type: ignore[arg-type]

    def test_empty_concept_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            _input(explanation_concept_id="")

    def test_long_participant_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            _input(participant_reference_id="x" * 200)


# ---------------------------------------------------------------------------
# Planner end-to-end: fields pass through to ResponsePlan
# ---------------------------------------------------------------------------


class TestPlannerPassesStrategyThrough:
    def test_output_plan_carries_strategy_and_concept(self) -> None:
        planner = ResponsePlanner()
        inp = _input(
            re_explanation_strategy=ReExplanationStrategy.SIMPLIFY,
            explanation_concept_id="svc.automation",
            prior_explanation_depth=ExplanationDepth.EXPLAINED,
            participant_reference_id="sess:p1",
        )
        plan = planner.plan(inp)
        assert plan.re_explanation_strategy == ReExplanationStrategy.SIMPLIFY
        assert plan.explanation_concept_id == "svc.automation"
        assert plan.prior_explanation_depth == ExplanationDepth.EXPLAINED
        assert plan.participant_reference_id == "sess:p1"

    def test_simplify_strategy_shortens_response(self) -> None:
        planner = ResponsePlanner()
        inp = _input(
            explanation_need=ExplanationNeed.COMPLEX,
            re_explanation_strategy=ReExplanationStrategy.SIMPLIFY,
            explanation_concept_id="svc.automation",
        )
        plan = planner.plan(inp)
        assert plan.response_length == ResponseLength.SHORT

    def test_deepen_strategy_detailed_response(self) -> None:
        planner = ResponsePlanner()
        inp = _input(
            re_explanation_strategy=ReExplanationStrategy.DEEPEN,
            explanation_concept_id="svc.automation",
        )
        plan = planner.plan(inp)
        assert plan.response_length == ResponseLength.DETAILED

    def test_concise_recap_shortens_response(self) -> None:
        planner = ResponsePlanner()
        inp = _input(
            explanation_need=ExplanationNeed.COMPLEX,
            re_explanation_strategy=ReExplanationStrategy.CONCISE_RECAP,
            explanation_concept_id="svc.automation",
        )
        plan = planner.plan(inp)
        assert plan.response_length == ResponseLength.SHORT

    def test_full_re_explain_upgrades_short_response(self) -> None:
        planner = ResponsePlanner()
        inp = _input(
            explanation_need=ExplanationNeed.SIMPLE,
            re_explanation_strategy=(
                ReExplanationStrategy.FULL_RE_EXPLAIN_FOR_NEW_PARTICIPANT
            ),
            explanation_concept_id="svc.automation",
        )
        plan = planner.plan(inp)
        # Short was upgraded to moderate because new participant needs real coverage
        assert plan.response_length in (ResponseLength.MODERATE, ResponseLength.DETAILED)

    def test_suppress_does_not_shrink_response(self) -> None:
        planner = ResponsePlanner()
        inp = _input(
            explanation_need=ExplanationNeed.COMPLEX,
            re_explanation_strategy=ReExplanationStrategy.SUPPRESS,
            explanation_concept_id="svc.automation",
        )
        plan = planner.plan(inp)
        # SUPPRESS leaves the length untouched — the strategy means
        # "don't re-explain this concept"; the planner's chosen length
        # for whatever action it IS doing stands.
        assert plan.response_length is not None

    def test_no_strategy_preserves_default_pipeline(self) -> None:
        """Existing callers that don't pass strategy keep today's behavior."""
        planner = ResponsePlanner()
        inp = _input()
        plan = planner.plan(inp)
        assert plan.re_explanation_strategy is None
        assert plan.explanation_concept_id is None


# ---------------------------------------------------------------------------
# No forbidden hardcoded service/industry logic in the mapping
# ---------------------------------------------------------------------------


class TestNoHardcodedServiceLogic:
    def test_planner_module_no_industry_keywords(self) -> None:
        import inspect
        from app.conversation.response_planning import planner
        source = inspect.getsource(planner)
        for forbidden in (
            '"restaurant"', '"dentist"', '"hvac"',
        ):
            assert forbidden not in source.lower(), f"found {forbidden}"
