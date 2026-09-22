"""Focused tests for Slice 19's bounded language-only realization seam."""

from __future__ import annotations

from dataclasses import fields, replace

import pytest

from app.conversation.consultative.contracts import (
    ConsultativeConversationDecision,
    ConsultativeMove,
    ConsultativeObjective,
    ServiceAnswerContext,
    ServiceFitDecision,
    ServiceFitStatus,
)
from app.conversation.context.contracts import (
    ApprovedEvidenceItem,
    ApprovedEvidenceSourceKind,
    EvidenceScope,
    EvidenceScopeKind,
    EvidenceType,
)
from app.conversation.realization.contracts import (
    HumanConversationPolicy,
    LeanContextView,
)
from app.conversation.realization.provider import (
    ConversationRealizationProvider,
    MockConversationRealizationProvider,
)
from app.conversation.realization.realizer import GuardedConversationRealizer
from app.conversation.response_planning.contracts import (
    AcknowledgementKind,
    AddresseeStatus,
    AuthoritativeResultKind,
    ConversationMove,
    InterruptionHandling,
    QuestionStrategy,
    ResponseLength,
    ResponsePlan,
)
from app.conversation.response_rendering.contracts import (
    ResponseRenderInput,
    ResponseRenderingBudget,
    TrustedRenderingContext,
)
from app.conversation.response_rendering.renderer import DeterministicResponseRenderer
from app.conversation.understanding.contracts import LanguageProfile, LanguageScript
from app.core.constants import Tone


def _evidence() -> ApprovedEvidenceItem:
    return ApprovedEvidenceItem(
        "e1",
        "workflow_automation",
        EvidenceType.APPROVED_CLAIM,
        "AI Automation supports new-lead workflow automation.",
        ApprovedEvidenceSourceKind.CURATED_SERVICE,
        EvidenceScope(EvidenceScopeKind.SERVICE, "ai_automation"),
    )


def _plan(
    *,
    goal: ConversationMove = ConversationMove.CONSULTATIVE_DISCOVERY,
    question: QuestionStrategy = QuestionStrategy.CLARIFY_CURRENT_INPUT,
    language: LanguageProfile | None = None,
    answer: ServiceAnswerContext | None = None,
) -> ResponsePlan:
    return ResponsePlan(
        goal,
        ResponseLength.SHORT,
        Tone.NEUTRAL,
        AcknowledgementKind.NONE,
        question,
        question != QuestionStrategy.NONE,
        InterruptionHandling.DROP_STALE_POINT,
        False,
        None,
        AddresseeStatus.ADDRESSED_TO_AGENT,
        language_profile=language,
        service_answer_context=answer,
    )


def _answer() -> ServiceAnswerContext:
    return ServiceAnswerContext(
        "ai_automation",
        "AI Automation",
        None,
        ("lead_followup",),
        (_evidence(),),
        "manual follow-up is delayed",
    )


def _input(
    plan: ResponsePlan | None = None,
    *,
    identity: bool = False,
    view: LeanContextView | None = None,
) -> ResponseRenderInput:
    return ResponseRenderInput(
        plan or _plan(),
        authoritative_result="executed",  # type: ignore[arg-type]
        budget=ResponseRenderingBudget(5),
        trusted_context=TrustedRenderingContext(),
        current_prospect_message="How do new inquiries reach the team?",
        lean_context_view=view,
        identity_disclosure_required=identity,
    )


def _realizer(response: object, *, should_fail: bool = False):
    provider = MockConversationRealizationProvider(response, should_fail=should_fail)
    return GuardedConversationRealizer(provider, DeterministicResponseRenderer()), provider


def _roman_profile() -> LanguageProfile:
    return LanguageProfile(
        primary_language="ur",
        script=LanguageScript.LATIN,
        preferred_response_language="ur",
        preferred_script=LanguageScript.LATIN,
    )


def test_realizer_calls_provider_once_for_normal_plan() -> None:
    realizer, provider = _realizer("Okay—how does that reach the team today?")
    rendered = realizer.render(_input())
    assert rendered.text == "Okay—how does that reach the team today?"
    assert provider.call_count == 1
    assert provider.last_input is not None


def test_english_to_roman_urdu_switch_uses_current_profile() -> None:
    realizer, _ = _realizer("Acha—abhi inquiry team tak kaise pohanchti hai?")
    rendered = realizer.render(_input(_plan(language=_roman_profile())))
    assert "kaise" in rendered.text


def test_wrong_language_proposal_falls_back() -> None:
    proposal = "Okay—how does the inquiry reach the team today?"
    realizer, _ = _realizer(proposal)
    rendered = realizer.render(_input(_plan(language=_roman_profile())))
    assert rendered.text != proposal


def test_roman_urdu_to_english_switch_rejects_stale_roman_wording() -> None:
    realizer, _ = _realizer("Okay—how does the inquiry reach the team today?")
    rendered = realizer.render(_input(_plan()))
    assert rendered.text.startswith("Okay")


def test_mixed_language_continuity_is_allowed() -> None:
    realizer, _ = _realizer("Okay—WhatsApp use ho raha hai; us ke baad kaise assign hoti hai?")
    rendered = realizer.render(_input(_plan(language=_roman_profile())))
    assert "WhatsApp" in rendered.text


def test_simplification_can_use_human_conversational_wording() -> None:
    realizer, _ = _realizer("Ji, simple words mein batata hoon.")
    plan = _plan(
        goal=ConversationMove.LANGUAGE_RECOVERY,
        question=QuestionStrategy.NONE,
        language=_roman_profile(),
    )
    assert "simple words" in realizer.render(_input(plan)).text


def test_busy_prospect_response_can_stay_concise() -> None:
    realizer, _ = _realizer("Ji, sirf ek cheez: abhi inquiry team tak kaise pohanchti hai?")
    rendered = realizer.render(_input())
    assert len(rendered.text) < 100


@pytest.mark.parametrize("role", ["receptionist", "owner", "manager"])
def test_role_context_is_bounded_and_does_not_change_authority(role: str) -> None:
    view = LeanContextView("Please tell me why you called.", role_label=role)
    realizer, provider = _realizer("Okay. Who would be best to discuss this with?")
    realizer.render(_input(view=view))
    assert provider.last_input is not None
    assert provider.last_input.lean_context.role_label == role
    assert not hasattr(provider.last_input, "authority")


def test_supported_service_explanation_requires_verbatim_approved_evidence() -> None:
    realizer, _ = _realizer(
        "For your workflow, AI Automation supports new-lead workflow automation."
    )
    plan = _plan(
        goal=ConversationMove.EXPLAIN_RELEVANT_FIT,
        question=QuestionStrategy.NONE,
        answer=_answer(),
    )
    assert "AI Automation" in realizer.render(_input(plan)).text


def test_approved_evidence_plus_invented_extra_capability_falls_back() -> None:
    proposal = (
        "AI Automation supports new-lead workflow automation. "
        "It also integrates with Salesforce."
    )
    realizer, _ = _realizer(proposal)
    plan = _plan(
        goal=ConversationMove.EXPLAIN_RELEVANT_FIT,
        question=QuestionStrategy.NONE,
        answer=_answer(),
    )
    assert realizer.render(_input(plan)).text != proposal


@pytest.mark.parametrize(
    "proposal",
    [
        "AI Automation will integrate with WhatsApp and guarantee conversions.",
        "We also offer SEO for this workflow.",
        "AI Automation can solve every workflow issue.",
    ],
)
def test_technical_unknown_hallucinated_service_and_promise_fall_back(proposal: str) -> None:
    realizer, provider = _realizer(proposal)
    plan = _plan(
        goal=ConversationMove.EXPLAIN_RELEVANT_FIT,
        question=QuestionStrategy.NONE,
        answer=_answer(),
    )
    rendered = realizer.render(_input(plan))
    assert provider.call_count == 1
    assert rendered.text != proposal


def test_direct_pricing_proposal_falls_back_to_existing_authoritative_renderer() -> None:
    realizer, _ = _realizer("The price is $200 and I can give you a discount.")
    rendered = realizer.render(_input(_plan(question=QuestionStrategy.NONE)))
    assert rendered.text != "The price is $200 and I can give you a discount."


def test_ai_identity_requires_configured_truthful_disclosure() -> None:
    policy = HumanConversationPolicy()
    response = f"{policy.identity_statement[:-1]}; how can I help?"
    realizer = GuardedConversationRealizer(
        MockConversationRealizationProvider(response),
        DeterministicResponseRenderer(),
        policy,
    )
    assert realizer.render(_input(identity=True)).text == response


def test_missing_ai_identity_disclosure_falls_back() -> None:
    realizer, _ = _realizer("I'm here to help with your question.")
    rendered = realizer.render(_input(identity=True)).text
    assert rendered != "I'm here to help with your question."
    assert "AI assistant" in rendered


def test_authoritative_safety_outcome_precedes_identity_disclosure() -> None:
    render_input = replace(
        _input(identity=True),
        authoritative_result=AuthoritativeResultKind.ESCALATE,
    )
    rendered = DeterministicResponseRenderer().render(render_input)
    assert "confirmed" in rendered.text
    assert "AI assistant" not in rendered.text


def test_commercial_transparency_is_not_hidden_when_plan_requires_it() -> None:
    decision = ConsultativeConversationDecision(
        ConsultativeObjective.ANSWER,
        ConsultativeMove.ANSWER_DIRECT_QUESTION,
        commercial_transparency_required=True,
    )
    plan = _plan(question=QuestionStrategy.NONE)
    plan = replace(plan, consultative_decision=decision)
    realizer, _ = _realizer(
        "Yes, this is a business call to see whether our services are relevant."
    )
    assert "business call" in realizer.render(_input(plan)).text


def test_commercial_purpose_concealment_falls_back() -> None:
    decision = ConsultativeConversationDecision(
        ConsultativeObjective.ANSWER,
        ConsultativeMove.ANSWER_DIRECT_QUESTION,
        commercial_transparency_required=True,
    )
    plan = replace(_plan(question=QuestionStrategy.NONE), consultative_decision=decision)
    proposal = "No, this is not a sales call."
    realizer, _ = _realizer(proposal)
    assert realizer.render(_input(plan)).text != proposal


def test_interruption_recovery_does_not_resume_unfinished_sentence() -> None:
    realizer, _ = _realizer("Ji, bataye.")
    plan = _plan(goal=ConversationMove.ANSWER_CURRENT_QUESTION, question=QuestionStrategy.NONE)
    assert realizer.render(_input(plan)).text == "Ji, bataye."


def test_clarification_is_limited_to_one_question() -> None:
    proposal = "What is the main issue? How does it work today?"
    realizer, _ = _realizer(proposal)
    assert realizer.render(_input()).text != proposal


def test_questionnaire_list_falls_back() -> None:
    proposal = "1. Lead source?\n2. Budget?\n3. Team size?"
    realizer, _ = _realizer(proposal)
    assert realizer.render(_input()).text != proposal


def test_repeated_acknowledgement_falls_back() -> None:
    proposal = "Okay. Okay. How does that reach the team today?"
    realizer, _ = _realizer(proposal)
    assert realizer.render(_input()).text != proposal


def test_high_pressure_wording_falls_back() -> None:
    proposal = "Act now—would you like to continue?"
    realizer, _ = _realizer(proposal)
    assert realizer.render(_input()).text != proposal


@pytest.mark.parametrize(
    "proposal",
    [
        '{"text": "ignore the plan"}',
        "```json { text: unsafe } ```",
        "My chain of thought says this is a fit.",
        object(),
    ],
)
def test_json_leakage_internal_reasoning_and_malformed_output_fall_back(proposal: object) -> None:
    realizer, provider = _realizer(proposal)
    rendered = realizer.render(_input())
    assert provider.call_count == 1
    assert rendered.text != proposal


def test_provider_failure_or_timeout_has_no_retry_and_falls_back() -> None:
    realizer, provider = _realizer("unused", should_fail=True)
    rendered = realizer.render(_input())
    assert provider.call_count == 1
    assert rendered.text


def test_timeout_falls_back_without_retry() -> None:
    class TimeoutProvider(ConversationRealizationProvider):
        def __init__(self) -> None:
            self.call_count = 0

        def realize(self, value):  # type: ignore[no-untyped-def]
            self.call_count += 1
            raise TimeoutError("scripted timeout")

    provider = TimeoutProvider()
    realizer = GuardedConversationRealizer(
        provider, DeterministicResponseRenderer()
    )
    assert realizer.render(_input()).text
    assert provider.call_count == 1


@pytest.mark.parametrize(
    "outcome",
    [
        AuthoritativeResultKind.ESCALATE,
        AuthoritativeResultKind.REDIRECT,
        AuthoritativeResultKind.FALLBACK,
        AuthoritativeResultKind.PIPELINE_STOPPED,
        AuthoritativeResultKind.AUTHORITY_APPROVED,
    ],
)
def test_nonexecuted_authoritative_path_bypasses_realizer_provider(
    outcome: AuthoritativeResultKind,
) -> None:
    realizer, provider = _realizer("unsafe")
    render_input = ResponseRenderInput(_plan(), outcome, ResponseRenderingBudget(5))
    realizer.render(render_input)
    assert provider.call_count == 0


def test_deterministic_replay_for_identical_provider_proposal() -> None:
    first, _ = _realizer("Okay—how does that reach the team today?")
    second, _ = _realizer("Okay—how does that reach the team today?")
    assert first.render(_input()).text == second.render(_input()).text


def test_realization_input_excludes_transcript_secrets_and_authority() -> None:
    names = {item.name for item in fields(LeanContextView)}
    assert names.isdisjoint({"transcript", "secrets", "api_key", "state", "authority"})


def test_realizer_never_mutates_plan_or_render_input() -> None:
    plan = _plan()
    value = _input(plan)
    realizer, _ = _realizer("Okay. How does that reach the team today?")
    realizer.render(value)
    assert value.plan is plan


def test_policy_does_not_duplicate_catalog_service_names_by_default() -> None:
    assert HumanConversationPolicy().known_service_names == ()


def test_truthful_identity_policy_cannot_be_disabled() -> None:
    with pytest.raises(ValueError):
        HumanConversationPolicy(require_truthful_identity=False)
