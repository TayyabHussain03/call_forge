"""Deterministic, grounded response renderer and final-text validation."""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

from app.conversation.response_planning.contracts import (
    AddresseeStatus,
    AcknowledgementKind,
    AuthoritativeResultKind,
    ConversationMove,
    InterruptionHandling,
    QuestionStrategy,
    ResponseLength,
)
from app.conversation.response_rendering.contracts import (
    ContactConfirmationStatus,
    RenderedResponse,
    ResponseRenderInput,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_LENGTH_GUIDANCE = {
    ResponseLength.SHORT: 1,
    ResponseLength.MODERATE: 3,
    ResponseLength.DETAILED: 5,
}
_ACKNOWLEDGEMENTS = {
    AcknowledgementKind.NONE: "",
    AcknowledgementKind.GOT_IT: "Got it.",
    AcknowledgementKind.MAKES_SENSE: "That makes sense.",
    AcknowledgementKind.OKAY: "Okay.",
    AcknowledgementKind.RIGHT: "Right.",
    AcknowledgementKind.UNDERSTOOD: "Understood.",
}


class RenderValidationError(ValueError):
    """Candidate renderer output failed the deterministic acceptance boundary."""


class ResponseRenderer(ABC):
    """Provider-independent renderer interface."""

    @abstractmethod
    def render(self, render_input: ResponseRenderInput) -> RenderedResponse:
        """Render one accepted response without changing upstream state."""
        raise NotImplementedError


class DeterministicResponseRenderer(ResponseRenderer):
    """Small grounded renderer used for safety invariants and simulations."""

    def render(self, render_input: ResponseRenderInput) -> RenderedResponse:
        """Render and validate once; invalid output becomes a safe fallback."""
        try:
            text = _compose(render_input)
            return validate_rendered_text(text, render_input)
        except RenderValidationError:
            return validate_rendered_text(_fallback_text(render_input), render_input)


def validate_rendered_text(
    candidate: object, render_input: ResponseRenderInput
) -> RenderedResponse:
    """Accept plain bounded text only; structured commands fail closed."""
    if not isinstance(candidate, str):
        raise RenderValidationError("renderer output must be plain text")
    if any(ord(character) < 32 for character in candidate):
        raise RenderValidationError("renderer output contains control characters")
    text = " ".join(candidate.split())
    if not text:
        raise RenderValidationError("renderer output is empty")
    if len(text) > render_input.budget.max_characters:
        raise RenderValidationError("renderer output exceeds character budget")
    if _sentence_count(text) > _effective_sentence_limit(render_input):
        raise RenderValidationError("renderer output exceeds sentence budget")
    return RenderedResponse(
        text=text,
        communicative_goal=render_input.plan.communicative_goal,
        length_class=render_input.plan.response_length,
        clarification_required=render_input.plan.clarification_required,
    )


def _compose(render_input: ResponseRenderInput) -> str:
    plan = render_input.plan
    context = render_input.trusted_context

    if plan.addressee_status == AddresseeStatus.ADDRESSEE_UNCERTAIN:
        choices = (
            "Sorry, was that for me?",
            "Just checking—were you saying that to me?",
            "Was that for me, or someone there with you?",
        )
        return choices[_stable_variant(render_input.variation_seed, len(choices))]
    if plan.addressee_status == AddresseeStatus.NOT_ADDRESSED_TO_AGENT:
        if plan.resume_previous_point and plan.pending_intent is not None:
            return f"No problem—coming back to where we left off: {plan.pending_intent.summary}"
        return "No problem—take your time."

    if render_input.authoritative_result == AuthoritativeResultKind.ESCALATE:
        return "I'd need to have that confirmed before giving you a definite answer."
    if render_input.authoritative_result == AuthoritativeResultKind.REDIRECT:
        return "I can't confirm that here, but I can help with the relevant next step."
    if render_input.authoritative_result == AuthoritativeResultKind.FALLBACK:
        return "I may have missed that. Could you clarify what you'd like help with?"

    if context.contact_status == ContactConfirmationStatus.UNCONFIRMED:
        return "Let me make sure I have that contact detail right—is it correct?"
    if context.contact_status == ContactConfirmationStatus.CONFIRMED:
        return "Thanks, that contact detail is confirmed."

    sentences: list[str] = []
    acknowledgement = _ACKNOWLEDGEMENTS[plan.acknowledgement]
    if acknowledgement:
        sentences.append(acknowledgement)

    fact_sentences = _grounded_fact_sentences(render_input)
    goal = plan.communicative_goal
    if goal == ConversationMove.ANSWER_CURRENT_QUESTION:
        sentences.extend(fact_sentences or [_missing_fact_text()])
        if (
            plan.resume_previous_point
            and plan.pending_intent is not None
            and plan.interruption_handling == InterruptionHandling.INTEGRATE_IF_USEFUL
        ):
            sentences.append(f"That also connects to {plan.pending_intent.summary}")
    elif goal == ConversationMove.EXPLORE_OBJECTION:
        sentences.extend(fact_sentences)
        sentences.append(
            "Before suggesting anything further, it would help to understand "
            "what did not work."
        )
    elif goal == ConversationMove.INCORPORATE_CORRECTION:
        sentences.extend(fact_sentences or ["I'll use that correction going forward."])
    elif goal == ConversationMove.FOLLOW_NEW_DIRECTION:
        sentences.extend(fact_sentences or ["Let's focus on that instead."])
    elif goal == ConversationMove.CLARIFY_MEANING:
        sentences.append("Could you clarify what you mean so I respond accurately?")
    else:
        sentences.extend(fact_sentences or ["How would you like to continue?"])

    if plan.question_strategy == QuestionStrategy.ANSWER_THEN_FOLLOW_UP:
        sentences.append("What part would be most useful to clarify next?")
    elif plan.question_strategy == QuestionStrategy.EXPLORE_WITH_ONE_QUESTION:
        sentences.append(
            "Was the main issue reaching people, or getting inquiries once they engaged?"
        )

    return _fit_to_budget(sentences, render_input)


def _grounded_fact_sentences(render_input: ResponseRenderInput) -> list[str]:
    context = render_input.trusted_context
    facts: list[str] = []
    if context.primary_fact:
        facts.append(_as_sentence(context.primary_fact))
    if context.service_name and context.service_facts:
        facts.append(_as_sentence(f"{context.service_name}: {context.service_facts[0]}"))
        facts.extend(_as_sentence(fact) for fact in context.service_facts[1:])
    return facts


def _fit_to_budget(sentences: list[str], render_input: ResponseRenderInput) -> str:
    limit = _effective_sentence_limit(render_input)
    if len(sentences) <= limit:
        return " ".join(sentences)
    if limit == 1:
        if render_input.plan.question_strategy != QuestionStrategy.NONE:
            lead_index = (
                -2
                if render_input.plan.communicative_goal == ConversationMove.EXPLORE_OBJECTION
                else 0
            )
            lead = sentences[lead_index].rstrip(".!?")
            question = sentences[-1]
            return f"{lead}; {question[0].lower()}{question[1:]}"
        return sentences[0]
    question = sentences[-1] if sentences[-1].endswith("?") else None
    selected = sentences[: limit - (1 if question else 0)]
    if (
        question
        and render_input.plan.communicative_goal == ConversationMove.EXPLORE_OBJECTION
    ):
        selected = sentences[-limit:-1]
    if question:
        selected.append(question)
    return " ".join(selected)


def _effective_sentence_limit(render_input: ResponseRenderInput) -> int:
    return min(
        render_input.budget.max_sentences,
        _LENGTH_GUIDANCE[render_input.plan.response_length],
    )


def _sentence_count(text: str) -> int:
    return len([part for part in _SENTENCE_SPLIT.split(text) if part.strip()])


def _as_sentence(value: str) -> str:
    cleaned = " ".join(value.split()).strip()
    return cleaned if cleaned.endswith((".", "!", "?")) else f"{cleaned}."


def _missing_fact_text() -> str:
    return "I don't have enough confirmed detail to answer that accurately."


def _stable_variant(seed: str, count: int) -> int:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:2], "big") % count


def _fallback_text(render_input: ResponseRenderInput) -> str:
    plan = render_input.plan
    if plan.addressee_status == AddresseeStatus.ADDRESSEE_UNCERTAIN:
        return "Sorry, was that for me?"
    if render_input.authoritative_result == AuthoritativeResultKind.ESCALATE:
        return "I need to have that confirmed first."
    if plan.clarification_required:
        return "Could you clarify that for me?"
    if render_input.authoritative_result == AuthoritativeResultKind.REDIRECT:
        return "I can't confirm that here."
    return "Could you clarify what you need?"
