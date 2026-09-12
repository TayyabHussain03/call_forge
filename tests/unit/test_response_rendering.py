"""Focused Slice 7 tests for bounded grounded final-response rendering."""

from __future__ import annotations

from dataclasses import fields

import pytest

from app.conversation.response_planning.contracts import (
    AddresseeStatus,
    AcknowledgementKind,
    AuthoritativeResultKind,
    ConversationMove,
    InterruptionHandling,
    PendingConversationIntent,
    QuestionStrategy,
    ResponseLength,
    ResponsePlan,
)
from app.conversation.response_rendering import renderer as renderer_module
from app.conversation.response_rendering.contracts import (
    ContactConfirmationStatus,
    RenderedResponse,
    ResponseRenderInput,
    ResponseRenderingBudget,
    TrustedRenderingContext,
)
from app.conversation.response_rendering.renderer import (
    DeterministicResponseRenderer,
    RenderValidationError,
    validate_rendered_text,
)
from app.core.constants import Tone


def _plan(
    *,
    goal: ConversationMove = ConversationMove.COMMUNICATE_RESULT,
    length: ResponseLength = ResponseLength.SHORT,
    addressee: AddresseeStatus = AddresseeStatus.ADDRESSED_TO_AGENT,
    question: QuestionStrategy = QuestionStrategy.NONE,
    resume: bool = False,
    pending: PendingConversationIntent | None = None,
    handling: InterruptionHandling = InterruptionHandling.NONE,
    clarification: bool = False,
    acknowledgement: AcknowledgementKind = AcknowledgementKind.NONE,
) -> ResponsePlan:
    return ResponsePlan(
        communicative_goal=goal,
        response_length=length,
        tone=Tone.NEUTRAL,
        acknowledgement=acknowledgement,
        question_strategy=question,
        clarification_required=clarification,
        interruption_handling=handling,
        resume_previous_point=resume,
        pending_intent=pending,
        addressee_status=addressee,
    )


def _input(
    plan: ResponsePlan,
    *,
    result: AuthoritativeResultKind = AuthoritativeResultKind.EXECUTED,
    sentences: int = 5,
    characters: int = 800,
    context: TrustedRenderingContext = TrustedRenderingContext(),
    seed: str = "test",
) -> ResponseRenderInput:
    return ResponseRenderInput(
        plan,
        result,
        ResponseRenderingBudget(sentences, characters),
        context,
        seed,
    )


def _pending() -> PendingConversationIntent:
    return PendingConversationIntent(
        "explain value", "an owned site can complement the current social presence."
    )


def test_uncertain_addressee_renders_short_natural_clarification() -> None:
    rendered = DeterministicResponseRenderer().render(
        _input(
            _plan(
                goal=ConversationMove.CLARIFY_ADDRESSEE,
                addressee=AddresseeStatus.ADDRESSEE_UNCERTAIN,
                clarification=True,
            ),
            seed="call-1",
        )
    )
    assert rendered.clarification_required
    assert rendered.text.endswith("?")
    assert len(rendered.text.split()) < 12


def test_addressee_clarification_varies_deterministically() -> None:
    render_input = _input(
        _plan(addressee=AddresseeStatus.ADDRESSEE_UNCERTAIN), seed="same"
    )
    renderer = DeterministicResponseRenderer()
    assert renderer.render(render_input) == renderer.render(render_input)
    variants = {
        renderer.render(
            _input(_plan(addressee=AddresseeStatus.ADDRESSEE_UNCERTAIN), seed=str(i))
        ).text
        for i in range(12)
    }
    assert len(variants) > 1


def test_moderate_service_answer_is_grounded_and_not_artificially_clipped() -> None:
    rendered = DeterministicResponseRenderer().render(
        _input(
            _plan(
                goal=ConversationMove.ANSWER_CURRENT_QUESTION,
                length=ResponseLength.MODERATE,
                question=QuestionStrategy.ANSWER_THEN_FOLLOW_UP,
            ),
            context=TrustedRenderingContext(
                service_name="Local search support",
                service_facts=("helps maintain accurate directory information",),
            ),
        )
    )
    assert "Local search support" in rendered.text
    assert "accurate directory information" in rendered.text
    assert rendered.text.endswith("?")


def test_detailed_response_is_voice_appropriate_and_bounded() -> None:
    rendered = DeterministicResponseRenderer().render(
        _input(
            _plan(
                goal=ConversationMove.ANSWER_CURRENT_QUESTION,
                length=ResponseLength.DETAILED,
                question=QuestionStrategy.ANSWER_THEN_FOLLOW_UP,
            ),
            sentences=4,
            context=TrustedRenderingContext(
                service_name="Website service",
                service_facts=("presents services clearly", "provides a contact route"),
            ),
        )
    )
    assert len(rendered.text.split()) < 80
    assert renderer_module._sentence_count(rendered.text) <= 4


def test_authoritative_budget_constrains_detail_without_mutating_plan() -> None:
    plan = _plan(
        goal=ConversationMove.ANSWER_CURRENT_QUESTION,
        length=ResponseLength.DETAILED,
        question=QuestionStrategy.ANSWER_THEN_FOLLOW_UP,
    )
    render_input = _input(
        plan,
        sentences=2,
        context=TrustedRenderingContext(
            primary_fact="The approved option supports an online contact route",
            service_name="Website service",
            service_facts=("presents services clearly", "supports inquiry collection"),
        ),
    )
    rendered = DeterministicResponseRenderer().render(render_input)

    assert renderer_module._sentence_count(rendered.text) <= 2
    assert render_input.plan is plan
    assert plan.response_length == ResponseLength.DETAILED


def test_interrupted_question_answers_first_and_does_not_restart() -> None:
    pending = _pending()
    rendered = DeterministicResponseRenderer().render(
        _input(
            _plan(
                goal=ConversationMove.ANSWER_CURRENT_QUESTION,
                length=ResponseLength.MODERATE,
                question=QuestionStrategy.ANSWER_THEN_FOLLOW_UP,
                resume=True,
                pending=pending,
                handling=InterruptionHandling.INTEGRATE_IF_USEFUL,
            ),
            context=TrustedRenderingContext(primary_fact="A website need not replace social media"),
        )
    )
    assert rendered.text.startswith("A website need not replace social media")
    assert not rendered.text.startswith(pending.summary)


def test_objection_is_addressed_without_resuming_stale_pitch() -> None:
    rendered = DeterministicResponseRenderer().render(
        _input(
            _plan(
                goal=ConversationMove.EXPLORE_OBJECTION,
                length=ResponseLength.MODERATE,
                question=QuestionStrategy.EXPLORE_WITH_ONE_QUESTION,
                handling=InterruptionHandling.DROP_STALE_POINT,
                acknowledgement=AcknowledgementKind.MAKES_SENSE,
            ),
            sentences=2,
        )
    )
    assert "what did not work" in rendered.text
    assert "owned site" not in rendered.text
    assert rendered.text.endswith("?")


def test_topic_shift_omits_pending_point() -> None:
    rendered = DeterministicResponseRenderer().render(
        _input(
            _plan(
                goal=ConversationMove.FOLLOW_NEW_DIRECTION,
                length=ResponseLength.MODERATE,
                pending=_pending(),
                handling=InterruptionHandling.DROP_STALE_POINT,
            ),
            context=TrustedRenderingContext(primary_fact="Callback timing is the current topic"),
        )
    )
    assert "Callback timing" in rendered.text
    assert "owned site" not in rendered.text


def test_background_speech_is_not_answered() -> None:
    rendered = DeterministicResponseRenderer().render(
        _input(_plan(addressee=AddresseeStatus.ADDRESSEE_UNCERTAIN))
    )
    assert (
        "was that for me" in rendered.text.lower()
        or "saying that to me" in rendered.text.lower()
    )
    assert "discount" not in rendered.text.lower()


def test_not_addressed_can_resume_only_relevant_previous_context() -> None:
    rendered = DeterministicResponseRenderer().render(
        _input(
            _plan(
                goal=ConversationMove.CONTINUE_PRIOR_CONTEXT,
                addressee=AddresseeStatus.NOT_ADDRESSED_TO_AGENT,
                resume=True,
                pending=_pending(),
                handling=InterruptionHandling.HOLD_PRIOR_CONTEXT,
            )
        )
    )
    assert "coming back" in rendered.text.lower()
    assert _pending().summary in rendered.text


def test_only_supplied_service_facts_are_rendered() -> None:
    rendered = DeterministicResponseRenderer().render(
        _input(
            _plan(goal=ConversationMove.ANSWER_CURRENT_QUESTION, length=ResponseLength.MODERATE),
            context=TrustedRenderingContext(
                service_name="Approved service",
                service_facts=("includes the authorized directory review",),
            ),
        )
    )
    assert "authorized directory review" in rendered.text
    assert "guarantee" not in rendered.text.lower()
    assert "%" not in rendered.text


def test_missing_fact_uses_safe_uncertainty_instead_of_hallucinating() -> None:
    rendered = DeterministicResponseRenderer().render(
        _input(_plan(goal=ConversationMove.ANSWER_CURRENT_QUESTION))
    )
    assert "don't have enough confirmed detail" in rendered.text


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (ContactConfirmationStatus.UNCONFIRMED, "is it correct"),
        (ContactConfirmationStatus.CONFIRMED, "is confirmed"),
    ],
)
def test_contact_wording_reflects_trusted_confirmation_status(
    status: ContactConfirmationStatus, expected: str
) -> None:
    rendered = DeterministicResponseRenderer().render(
        _input(_plan(), context=TrustedRenderingContext(contact_status=status))
    )
    assert expected in rendered.text.lower()


def test_escalation_never_makes_commercial_promise() -> None:
    rendered = DeterministicResponseRenderer().render(
        _input(_plan(), result=AuthoritativeResultKind.ESCALATE)
    )
    assert "confirmed" in rendered.text
    assert "discount" not in rendered.text.lower()
    assert "%" not in rendered.text


@pytest.mark.parametrize(
    "candidate",
    [
        {"text": "Done", "next_state": "end_call"},
        {"execute_action": "persist_contact"},
        "",
        "hello\x00world",
        "hello\nworld",
    ],
)
def test_malicious_empty_and_invalid_outputs_are_rejected(candidate: object) -> None:
    with pytest.raises(RenderValidationError):
        validate_rendered_text(candidate, _input(_plan()))


def test_over_budget_output_is_rejected() -> None:
    with pytest.raises(RenderValidationError):
        validate_rendered_text("One. Two.", _input(_plan(), sentences=1))


def test_invalid_candidate_uses_one_deterministic_safe_fallback(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    monkeypatch.setattr(renderer_module, "_compose", lambda _: "")
    renderer = DeterministicResponseRenderer()
    render_input = _input(_plan(clarification=True))

    first = renderer.render(render_input)
    second = renderer.render(render_input)
    assert first == second
    assert first.text == "Could you clarify that for me?"


def test_rendered_response_has_no_authority_fields() -> None:
    forbidden = {
        "next_state",
        "execute_action",
        "authorized_service",
        "confirmed_contact",
        "persist_contact",
        "approved_discount",
        "human_approval",
        "policy_override",
    }
    assert forbidden.isdisjoint({field.name for field in fields(RenderedResponse)})
