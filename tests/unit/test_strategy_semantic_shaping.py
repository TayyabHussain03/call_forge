"""Slice 51G — strategy-shaped ExplanationNeed + language-adaptive framings.

Proves (per Slice 51G §5-§11, §42-§44):
    - SIMPLIFY / CONCISE_RECAP override ExplanationNeed → SIMPLE
    - DEEPEN → ExplanationNeed.COMPLEX (planner produces longer base)
    - FULL_RE_EXPLAIN → ExplanationNeed.STANDARD
    - SUPPRESS / NO_PRIOR_COVERAGE / None → preserve base need
    - Framings adapt to the plan's LanguageProfile (English vs Roman Urdu)
    - Primary acceptance: prefix-only strategy difference is NOT enough —
      the planner now produces different ResponseLength values for
      SIMPLIFY vs DEEPEN vs baseline
"""

from __future__ import annotations

import pytest

from app.application.production_turn_processor import _strategy_shaped_need
from app.conversation.participant.contracts import ReExplanationStrategy
from app.conversation.response_planning.contracts import (
    AcknowledgementKind,
    AddresseeStatus,
    AuthoritativeResultKind,
    ConversationMove,
    ExplanationNeed,
    InterruptionCategory,
    InterruptionContext,
    InterruptionHandling,
    QuestionStrategy,
    ResponseLength,
    ResponsePlan,
    ResponsePlanningInput,
)
from app.conversation.response_planning.planner import ResponsePlanner
from app.conversation.response_rendering.contracts import (
    RenderedResponse,
    ResponseRenderInput,
    ResponseRenderingBudget,
)
from app.conversation.response_rendering.renderer import ResponseRenderer
from app.conversation.response_rendering.strategy_frame import (
    STRATEGY_FRAMINGS,
    STRATEGY_FRAMINGS_URDU,
    StrategyAwareResponseRenderer,
    framing_for,
)
from app.conversation.understanding.contracts import (
    ConversationalRegister,
    LanguageProfile,
    LanguageScript,
)
from app.core.constants import AgentAction, ConversationState, Tone


# ---------------------------------------------------------------------------
# Strategy-shaped ExplanationNeed
# ---------------------------------------------------------------------------


class TestStrategyShapedNeed:
    def test_simplify_shapes_to_simple(self) -> None:
        assert _strategy_shaped_need(
            ExplanationNeed.COMPLEX, ReExplanationStrategy.SIMPLIFY,
        ) == ExplanationNeed.SIMPLE

    def test_concise_recap_shapes_to_simple(self) -> None:
        assert _strategy_shaped_need(
            ExplanationNeed.COMPLEX, ReExplanationStrategy.CONCISE_RECAP,
        ) == ExplanationNeed.SIMPLE

    def test_deepen_shapes_to_complex(self) -> None:
        assert _strategy_shaped_need(
            ExplanationNeed.SIMPLE, ReExplanationStrategy.DEEPEN,
        ) == ExplanationNeed.COMPLEX

    def test_full_re_explain_shapes_to_standard(self) -> None:
        assert _strategy_shaped_need(
            ExplanationNeed.SIMPLE,
            ReExplanationStrategy.FULL_RE_EXPLAIN_FOR_NEW_PARTICIPANT,
        ) == ExplanationNeed.STANDARD

    def test_suppress_preserves_base(self) -> None:
        assert _strategy_shaped_need(
            ExplanationNeed.STANDARD, ReExplanationStrategy.SUPPRESS,
        ) == ExplanationNeed.STANDARD

    def test_no_prior_coverage_preserves_base(self) -> None:
        assert _strategy_shaped_need(
            ExplanationNeed.STANDARD, ReExplanationStrategy.NO_PRIOR_COVERAGE,
        ) == ExplanationNeed.STANDARD

    def test_none_strategy_preserves_base(self) -> None:
        assert _strategy_shaped_need(ExplanationNeed.STANDARD, None) == ExplanationNeed.STANDARD


# ---------------------------------------------------------------------------
# Planner length agreement — ExplanationNeed + strategy produce different lengths
# ---------------------------------------------------------------------------


def _input(need: ExplanationNeed, strategy=None) -> ResponsePlanningInput:
    return ResponsePlanningInput(
        authoritative_state=ConversationState.LISTEN,
        authoritative_result=AuthoritativeResultKind.EXECUTED,
        current_prospect_message="hi",
        approved_action=AgentAction.ANSWER_QUESTION,
        tone=Tone.NEUTRAL,
        conversation_category=InterruptionCategory.OTHER,
        addressee_status=AddresseeStatus.ADDRESSED_TO_AGENT,
        interruption=InterruptionContext(),
        explanation_need=need,
        re_explanation_strategy=strategy,
        explanation_concept_id="svc.automation",
    )


class TestPlannerLengthAgreement:
    """§6-§11: prefix-only differences are not enough. The planner's own
    length calculation must materially change across strategies."""

    def test_simplify_produces_short(self) -> None:
        planner = ResponsePlanner()
        # Simulate what the processor does: shape need via strategy, pass to planner
        shaped = _strategy_shaped_need(
            ExplanationNeed.COMPLEX, ReExplanationStrategy.SIMPLIFY,
        )
        plan = planner.plan(_input(shaped, ReExplanationStrategy.SIMPLIFY))
        assert plan.response_length == ResponseLength.SHORT

    def test_deepen_produces_detailed(self) -> None:
        planner = ResponsePlanner()
        shaped = _strategy_shaped_need(
            ExplanationNeed.SIMPLE, ReExplanationStrategy.DEEPEN,
        )
        plan = planner.plan(_input(shaped, ReExplanationStrategy.DEEPEN))
        assert plan.response_length == ResponseLength.DETAILED

    def test_simplify_and_deepen_lengths_differ(self) -> None:
        planner = ResponsePlanner()
        s_input = _input(
            _strategy_shaped_need(
                ExplanationNeed.STANDARD, ReExplanationStrategy.SIMPLIFY,
            ),
            ReExplanationStrategy.SIMPLIFY,
        )
        d_input = _input(
            _strategy_shaped_need(
                ExplanationNeed.STANDARD, ReExplanationStrategy.DEEPEN,
            ),
            ReExplanationStrategy.DEEPEN,
        )
        simplify_plan = planner.plan(s_input)
        deepen_plan = planner.plan(d_input)
        assert simplify_plan.response_length != deepen_plan.response_length


# ---------------------------------------------------------------------------
# Language-adaptive framings
# ---------------------------------------------------------------------------


class TestLanguageAdaptiveFramings:
    def test_english_framing_default(self) -> None:
        assert framing_for(
            ReExplanationStrategy.SIMPLIFY, primary_language="en",
        ) == "In simpler terms:"

    def test_english_framing_none_language(self) -> None:
        """No language → default to English."""
        assert framing_for(
            ReExplanationStrategy.DEEPEN, primary_language=None,
        ) == "Going deeper:"

    def test_roman_urdu_simplify_framing(self) -> None:
        assert framing_for(
            ReExplanationStrategy.SIMPLIFY, primary_language="ur",
        ) == "Simple words mein:"

    def test_roman_urdu_deepen_framing(self) -> None:
        assert framing_for(
            ReExplanationStrategy.DEEPEN, primary_language="ur",
        ) == "Thora detail mein:"

    def test_roman_urdu_recap_framing(self) -> None:
        assert framing_for(
            ReExplanationStrategy.CONCISE_RECAP, primary_language="ur",
        ) == "Short recap:"

    def test_roman_urdu_new_participant_framing(self) -> None:
        assert framing_for(
            ReExplanationStrategy.FULL_RE_EXPLAIN_FOR_NEW_PARTICIPANT,
            primary_language="ur",
        ) == "Aapko context de doon:"

    def test_case_insensitive_language_matching(self) -> None:
        assert framing_for(
            ReExplanationStrategy.SIMPLIFY, primary_language="UR",
        ) == "Simple words mein:"
        assert framing_for(
            ReExplanationStrategy.SIMPLIFY, primary_language="Urdu",
        ) == "Simple words mein:"

    def test_english_and_urdu_framings_differ(self) -> None:
        for strategy in STRATEGY_FRAMINGS:
            if strategy in STRATEGY_FRAMINGS_URDU:
                assert framing_for(strategy, "en") != framing_for(strategy, "ur")


# ---------------------------------------------------------------------------
# Wrapper applies language-correct framing
# ---------------------------------------------------------------------------


class _FixedRenderer(ResponseRenderer):
    def __init__(self, text: str = "base text") -> None:
        self._text = text

    def render(self, render_input: ResponseRenderInput) -> RenderedResponse:
        return RenderedResponse(
            text=self._text,
            communicative_goal=render_input.plan.communicative_goal,
            length_class=render_input.plan.response_length,
            clarification_required=render_input.plan.clarification_required,
        )


def _plan(strategy, language_profile=None) -> ResponsePlan:
    return ResponsePlan(
        communicative_goal=ConversationMove.COMMUNICATE_RESULT,
        response_length=ResponseLength.MODERATE,
        tone=Tone.NEUTRAL,
        acknowledgement=AcknowledgementKind.NONE,
        question_strategy=QuestionStrategy.NONE,
        clarification_required=False,
        interruption_handling=InterruptionHandling.NONE,
        resume_previous_point=False,
        pending_intent=None,
        addressee_status=AddresseeStatus.ADDRESSED_TO_AGENT,
        language_profile=language_profile,
        re_explanation_strategy=strategy,
        explanation_concept_id="svc.automation",
    )


def _render_input(plan: ResponsePlan) -> ResponseRenderInput:
    return ResponseRenderInput(
        plan=plan,
        authoritative_result=AuthoritativeResultKind.EXECUTED,
        budget=ResponseRenderingBudget(5),
        trusted_context=(),
        variation_seed="seed",
        current_prospect_message="hi",
    )


class TestWrapperAppliesLanguageFraming:
    def _wrapper(self):
        return StrategyAwareResponseRenderer(_FixedRenderer("ok"))

    def test_english_simplify_uses_english_prefix(self) -> None:
        lp = LanguageProfile(primary_language="en")
        plan = _plan(ReExplanationStrategy.SIMPLIFY, language_profile=lp)
        result = self._wrapper().render(_render_input(plan))
        assert result.text.startswith("In simpler terms:")

    def test_roman_urdu_simplify_uses_urdu_prefix(self) -> None:
        lp = LanguageProfile(
            primary_language="ur", secondary_language="en", mixed_language=True,
        )
        plan = _plan(ReExplanationStrategy.SIMPLIFY, language_profile=lp)
        result = self._wrapper().render(_render_input(plan))
        assert result.text.startswith("Simple words mein:")

    def test_roman_urdu_deepen_uses_urdu_prefix(self) -> None:
        lp = LanguageProfile(
            primary_language="ur", secondary_language="en", mixed_language=True,
        )
        plan = _plan(ReExplanationStrategy.DEEPEN, language_profile=lp)
        result = self._wrapper().render(_render_input(plan))
        assert result.text.startswith("Thora detail mein:")

    def test_no_language_profile_defaults_to_english(self) -> None:
        plan = _plan(ReExplanationStrategy.CONCISE_RECAP, language_profile=None)
        result = self._wrapper().render(_render_input(plan))
        assert result.text.startswith("Quick recap:")


# ---------------------------------------------------------------------------
# Prefix-only acceptance guard (§5)
# ---------------------------------------------------------------------------


class TestPrefixOnlyIsNotEnough:
    """§5: prefix-only strategy handling is NOT enough. The planner's own
    structure (ExplanationNeed → ResponseLength) must also change.
    This guard ensures that if someone adds a strategy to a plan whose
    ExplanationNeed is also strategy-shaped, the plan length differs."""

    def test_strategy_changes_length_not_just_prefix(self) -> None:
        planner = ResponsePlanner()
        # Same base need; one with no strategy, one with DEEPEN
        baseline = planner.plan(_input(ExplanationNeed.STANDARD, None))
        shaped_deepen_need = _strategy_shaped_need(
            ExplanationNeed.STANDARD, ReExplanationStrategy.DEEPEN,
        )
        deep = planner.plan(_input(shaped_deepen_need, ReExplanationStrategy.DEEPEN))
        # DEEPEN produced DETAILED; baseline STANDARD need produced a shorter length
        assert deep.response_length == ResponseLength.DETAILED
        assert baseline.response_length != deep.response_length


# ---------------------------------------------------------------------------
# No language inference from role or name (§43)
# ---------------------------------------------------------------------------


class TestLanguageNotInferredFromRole:
    def test_framing_sourced_only_from_language_profile(self) -> None:
        """The wrapper reads primary_language from the plan's
        LanguageProfile — nothing else influences the framing choice."""
        import inspect
        from app.conversation.response_rendering import strategy_frame
        source = inspect.getsource(strategy_frame)
        for forbidden in (
            "role_hint", "business_name", "contact_name", "campaign_id",
            '"pakistan"', '"india"', '"usa"',
        ):
            assert forbidden not in source.lower()


# ---------------------------------------------------------------------------
# No hardcoded service/industry logic in semantic shaping
# ---------------------------------------------------------------------------


class TestNoHardcoding:
    def test_strategy_shaped_need_no_industry_branches(self) -> None:
        import inspect
        from app.application import production_turn_processor
        source = inspect.getsource(production_turn_processor)
        # Scan just the strategy-shaping helper for hardcoded industry logic
        start = source.find("def _strategy_shaped_need")
        end = source.find("\ndef ", start + 1)
        helper_source = source[start:end]
        for forbidden in (
            '"restaurant"', '"dentist"', '"hvac"',
            '"excel"', '"automation"', '"website"',
            "if industry", "if service ==",
        ):
            assert forbidden not in helper_source.lower()
