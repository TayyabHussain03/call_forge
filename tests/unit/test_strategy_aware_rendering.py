"""Slice 51F — GRC strategy-aware response framing.

Proves `StrategyAwareResponseRenderer` applies deterministic framings
derived from `ResponsePlan.re_explanation_strategy`:
    - CONCISE_RECAP:   "Quick recap: ..."
    - SIMPLIFY:         "In simpler terms: ..."
    - DEEPEN:           "Going deeper: ..."
    - FULL_RE_EXPLAIN:  "To bring you up to speed: ..."
    - SUPPRESS / NO_PRIOR_COVERAGE / None: unchanged base output
"""

from __future__ import annotations

import pytest

from app.conversation.participant.contracts import ReExplanationStrategy
from app.conversation.response_planning.contracts import (
    AcknowledgementKind,
    AddresseeStatus,
    ConversationMove,
    InterruptionHandling,
    QuestionStrategy,
    ResponseLength,
    ResponsePlan,
)
from app.conversation.response_rendering.contracts import (
    RenderedResponse,
    ResponseRenderInput,
)
from app.conversation.response_rendering.renderer import ResponseRenderer
from app.conversation.response_rendering.strategy_frame import (
    STRATEGY_FRAMINGS,
    StrategyAwareResponseRenderer,
    framing_for,
)
from app.core.constants import Tone


def _plan(strategy=None, concept_id=None, length=ResponseLength.MODERATE) -> ResponsePlan:
    return ResponsePlan(
        communicative_goal=ConversationMove.COMMUNICATE_RESULT,
        response_length=length,
        tone=Tone.NEUTRAL,
        acknowledgement=AcknowledgementKind.NONE,
        question_strategy=QuestionStrategy.NONE,
        clarification_required=False,
        interruption_handling=InterruptionHandling.NONE,
        resume_previous_point=False,
        pending_intent=None,
        addressee_status=AddresseeStatus.ADDRESSED_TO_AGENT,
        re_explanation_strategy=strategy,
        explanation_concept_id=concept_id,
    )


class _FixedRenderer(ResponseRenderer):
    """Returns a fixed rendered response — isolates the wrapper behavior."""

    def __init__(self, text: str = "This is the base response.") -> None:
        self._text = text

    def render(self, render_input: ResponseRenderInput) -> RenderedResponse:
        return RenderedResponse(
            text=self._text,
            communicative_goal=render_input.plan.communicative_goal,
            length_class=render_input.plan.response_length,
            clarification_required=render_input.plan.clarification_required,
        )


def _render_input(plan: ResponsePlan) -> ResponseRenderInput:
    from app.conversation.response_planning.contracts import AuthoritativeResultKind
    from app.conversation.response_rendering.contracts import ResponseRenderingBudget
    return ResponseRenderInput(
        plan=plan,
        authoritative_result=AuthoritativeResultKind.EXECUTED,
        budget=ResponseRenderingBudget(5),
        trusted_context=(),
        variation_seed="deterministic-seed",
        current_prospect_message="hi",
    )


# ---------------------------------------------------------------------------
# Deterministic framing map
# ---------------------------------------------------------------------------


class TestStrategyFramingMap:
    def test_concise_recap_framing(self) -> None:
        assert framing_for(ReExplanationStrategy.CONCISE_RECAP) == "Quick recap:"

    def test_simplify_framing(self) -> None:
        assert framing_for(ReExplanationStrategy.SIMPLIFY) == "In simpler terms:"

    def test_deepen_framing(self) -> None:
        assert framing_for(ReExplanationStrategy.DEEPEN) == "Going deeper:"

    def test_full_re_explain_framing(self) -> None:
        assert framing_for(
            ReExplanationStrategy.FULL_RE_EXPLAIN_FOR_NEW_PARTICIPANT,
        ) == "To bring you up to speed:"

    def test_suppress_no_framing(self) -> None:
        """SUPPRESS means do not re-explain — do not add a repeat prefix."""
        assert framing_for(ReExplanationStrategy.SUPPRESS) is None

    def test_no_prior_coverage_no_framing(self) -> None:
        """First-time explanation uses base renderer output as-is."""
        assert framing_for(ReExplanationStrategy.NO_PRIOR_COVERAGE) is None


# ---------------------------------------------------------------------------
# Wrapper applies framings to real plans
# ---------------------------------------------------------------------------


class TestWrapperAppliesFraming:
    def _wrapper(self):
        return StrategyAwareResponseRenderer(_FixedRenderer("Here is the answer."))

    def test_concise_recap_prepends_quick_recap(self) -> None:
        plan = _plan(strategy=ReExplanationStrategy.CONCISE_RECAP)
        result = self._wrapper().render(_render_input(plan))
        assert result.text.startswith("Quick recap:")

    def test_simplify_prepends_simpler_terms(self) -> None:
        plan = _plan(strategy=ReExplanationStrategy.SIMPLIFY)
        result = self._wrapper().render(_render_input(plan))
        assert result.text.startswith("In simpler terms:")

    def test_deepen_prepends_going_deeper(self) -> None:
        plan = _plan(strategy=ReExplanationStrategy.DEEPEN)
        result = self._wrapper().render(_render_input(plan))
        assert result.text.startswith("Going deeper:")

    def test_full_re_explain_prepends_bring_up_to_speed(self) -> None:
        plan = _plan(
            strategy=ReExplanationStrategy.FULL_RE_EXPLAIN_FOR_NEW_PARTICIPANT,
        )
        result = self._wrapper().render(_render_input(plan))
        assert result.text.startswith("To bring you up to speed:")

    def test_suppress_leaves_base_text_unchanged(self) -> None:
        """SUPPRESS must not create a repeated-concept framing."""
        plan = _plan(strategy=ReExplanationStrategy.SUPPRESS)
        result = self._wrapper().render(_render_input(plan))
        assert result.text == "Here is the answer."

    def test_no_prior_coverage_leaves_base_text_unchanged(self) -> None:
        plan = _plan(strategy=ReExplanationStrategy.NO_PRIOR_COVERAGE)
        result = self._wrapper().render(_render_input(plan))
        assert result.text == "Here is the answer."

    def test_no_strategy_leaves_base_text_unchanged(self) -> None:
        """When plan.re_explanation_strategy is None, wrapper is a no-op."""
        plan = _plan(strategy=None)
        result = self._wrapper().render(_render_input(plan))
        assert result.text == "Here is the answer."


# ---------------------------------------------------------------------------
# Framings don't contaminate base metadata
# ---------------------------------------------------------------------------


class TestFramingPreservesBaseMetadata:
    def test_communicative_goal_preserved(self) -> None:
        wrapper = StrategyAwareResponseRenderer(_FixedRenderer("base"))
        plan = _plan(strategy=ReExplanationStrategy.CONCISE_RECAP)
        result = wrapper.render(_render_input(plan))
        assert result.communicative_goal == plan.communicative_goal

    def test_length_class_preserved(self) -> None:
        wrapper = StrategyAwareResponseRenderer(_FixedRenderer("base"))
        plan = _plan(strategy=ReExplanationStrategy.DEEPEN,
                     length=ResponseLength.DETAILED)
        result = wrapper.render(_render_input(plan))
        assert result.length_class == ResponseLength.DETAILED

    def test_clarification_required_preserved(self) -> None:
        wrapper = StrategyAwareResponseRenderer(_FixedRenderer("base"))
        plan = _plan(strategy=ReExplanationStrategy.SIMPLIFY)
        result = wrapper.render(_render_input(plan))
        assert result.clarification_required == plan.clarification_required


# ---------------------------------------------------------------------------
# Visible structural differences across strategies
# ---------------------------------------------------------------------------


class TestVisibleStructuralDifferences:
    """Prose comparison across strategies — the behavioral acceptance gate."""

    def _render(self, strategy):
        wrapper = StrategyAwareResponseRenderer(
            _FixedRenderer("Our automation handles the manual Excel workflow.")
        )
        plan = _plan(strategy=strategy, concept_id="service:automation")
        return wrapper.render(_render_input(plan)).text

    def test_simplify_and_deepen_produce_different_prose(self) -> None:
        simplify = self._render(ReExplanationStrategy.SIMPLIFY)
        deepen = self._render(ReExplanationStrategy.DEEPEN)
        assert simplify != deepen

    def test_recap_and_full_re_explain_produce_different_prose(self) -> None:
        recap = self._render(ReExplanationStrategy.CONCISE_RECAP)
        full = self._render(ReExplanationStrategy.FULL_RE_EXPLAIN_FOR_NEW_PARTICIPANT)
        assert recap != full

    def test_suppress_matches_no_strategy_output(self) -> None:
        suppress = self._render(ReExplanationStrategy.SUPPRESS)
        no_strategy = self._render(None)
        assert suppress == no_strategy


# ---------------------------------------------------------------------------
# No service/industry hardcoding in the wrapper
# ---------------------------------------------------------------------------


class TestNoHardcoding:
    def test_wrapper_module_no_industry_keywords(self) -> None:
        import inspect
        from app.conversation.response_rendering import strategy_frame
        source = inspect.getsource(strategy_frame)
        for forbidden in (
            '"restaurant"', '"dentist"', '"hvac"', '"excel"',
            'if industry', 'if service ==',
        ):
            assert forbidden not in source.lower()


# ---------------------------------------------------------------------------
# Determinism — same input, same output
# ---------------------------------------------------------------------------


class TestDeterministic:
    def test_same_strategy_same_output(self) -> None:
        wrapper = StrategyAwareResponseRenderer(_FixedRenderer("deterministic base"))
        plan = _plan(strategy=ReExplanationStrategy.DEEPEN)
        a = wrapper.render(_render_input(plan)).text
        b = wrapper.render(_render_input(plan)).text
        assert a == b
