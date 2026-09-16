"""Deterministic, grounded response renderer and final-text validation."""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

from app.conversation.consultative.contracts import (
    ConsultativeMove,
    ProblemField,
    QuestionPolicy,
)
from app.conversation.context.contracts import EvidenceType
from app.conversation.escalation.contracts import (
    EscalationCapability,
    EscalationReason,
    RecoveryMode,
)
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
_CONFIRMED_PROMISE_PATTERNS = (
    re.compile(r"\b(?:i|we)(?:'ll| will) (?:transfer|email|send|book|schedule)\b", re.I),
    re.compile(r"\b(?:engineering|our team) will (?:confirm|email|send)\b", re.I),
    re.compile(r"\btransfer you now\b", re.I),
    re.compile(r"\b(?:callback|follow-up) (?:is|has been) (?:scheduled|confirmed)\b", re.I),
)


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
    decision = render_input.plan.escalation_decision
    if decision is not None and any(
        pattern.search(text) for pattern in _CONFIRMED_PROMISE_PATTERNS
    ):
        raise RenderValidationError("escalation output contains a confirmed promise")
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

    escalation_text = _escalation_text(render_input)
    if escalation_text is not None:
        return escalation_text

    if render_input.authoritative_result == AuthoritativeResultKind.ESCALATE:
        return "I'd need to have that confirmed before giving you a definite answer."
    if render_input.authoritative_result == AuthoritativeResultKind.REDIRECT:
        return "I can't confirm that here, but I can help with the relevant next step."
    if render_input.authoritative_result == AuthoritativeResultKind.FALLBACK:
        return "I may have missed that. Could you clarify what you'd like help with?"

    consultative_text = _consultative_text(render_input)
    if consultative_text is not None:
        return consultative_text

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


def _consultative_text(render_input: ResponseRenderInput) -> str | None:
    plan = render_input.plan
    decision = plan.consultative_decision
    if decision is None:
        return None
    if decision.commercial_transparency_required:
        return (
            "Yes—this is a business call to understand whether our services are "
            "relevant; I won't assume there is a fit."
        )
    if decision.move == ConsultativeMove.SIMPLIFY_EXPLANATION:
        language = plan.language_profile
        if language is not None and language.preferred_response_language in {"ur", "hi"}:
            return "Ji, simple words mein batata hoon—main baat ko seedha aur asaan rakhunga."
        return "Let me put that more simply without changing the underlying point."
    if decision.move == ConsultativeMove.EXPLAIN_RELEVANT_FIT:
        return _grounded_service_fit(render_input)
    if decision.move == ConsultativeMove.ANSWER_DIRECT_QUESTION:
        if plan.service_answer_context is not None:
            return _grounded_service_fit(render_input)
        return _missing_fact_text()
    if decision.move == ConsultativeMove.LOW_PRESSURE_CALLBACK:
        return (
            "Since now is busy, we can keep this brief and only discuss a callback "
            "if useful."
        )
    if decision.move == ConsultativeMove.GRACEFUL_CLOSE:
        return "It sounds like there may be nothing useful to change right now, so I won't force it."
    if decision.move == ConsultativeMove.ASK_MICRO_COMMITMENT:
        return "Would it be useful to take one small next step, without treating it as confirmed?"
    if decision.question_policy != QuestionPolicy.NONE:
        return _one_consultative_question(
            decision.primary_information_gap,
            decision.question_policy,
        )
    return None


def _grounded_service_fit(render_input: ResponseRenderInput) -> str:
    context = render_input.plan.service_answer_context
    if context is None or not context.disclosure_allowed or not context.approved_evidence:
        return _missing_fact_text()
    evidence = context.approved_evidence[0]
    if evidence.evidence_type == EvidenceType.CASE_STUDY:
        return (
            f"The relevant approved example for {context.service_name} is: "
            f"{evidence.statement} This is context, not a guarantee of your outcome."
        )
    return (
        f"Based on {context.problem_summary}, the relevant part of "
        f"{context.service_name} is: {evidence.statement}"
    )


def _one_consultative_question(
    gap: ProblemField | None,
    policy: QuestionPolicy,
) -> str:
    if policy == QuestionPolicy.SHORT_CONTRAST:
        return "Is the issue mainly with the process itself, or with how the team handles it?"
    questions = {
        ProblemField.UNDERLYING_PROBLEM: "What are you mainly hoping to improve?",
        ProblemField.CURRENT_PROCESS: "How does that process work today?",
        ProblemField.FRICTION: "What's creating the most difficulty in that process?",
        ProblemField.IMPACT: "What usually happens when that issue comes up?",
        ProblemField.DESIRED_OUTCOME: "What would a useful improvement look like?",
        ProblemField.SOURCE_OR_CHANNEL: "Where does that work enter the process today?",
        ProblemField.PROVIDER_SATISFACTION: (
            "Is the current setup working well, or is there still something to improve?"
        ),
        ProblemField.OBJECTION_REASON: "What part concerns you most?",
        ProblemField.ROLE_ROUTING: "Who is the right person to discuss how this works today?",
        ProblemField.ALREADY_OFFERED: "Is there a different part of the problem to focus on?",
    }
    return questions.get(gap, "What would be most useful to understand first?")


def _escalation_text(render_input: ResponseRenderInput) -> str | None:
    decision = render_input.plan.escalation_decision
    if decision is None or decision.recovery_mode == RecoveryMode.PROCEED_NORMALLY:
        return None
    if decision.recovery_mode == RecoveryMode.ANSWER_WITH_EVIDENCE:
        allowed = [
            item.statement
            for item in render_input.trusted_context.approved_evidence
            if item.evidence_id in decision.evidence_ids
        ]
        if not allowed:
            return _missing_fact_text()
        return _with_resume(_as_sentence(allowed[0]), render_input)
    if decision.recovery_mode == RecoveryMode.ASK_CLARIFYING_QUESTION:
        if decision.capability_required in {
            EscalationCapability.EMAIL,
            EscalationCapability.MESSAGE,
        }:
            return (
                "A confirmed contact method is needed before an information "
                "follow-up can be requested. Which contact method should we verify?"
            )
        return "Could you clarify the specific detail you need so I don't guess?"
    if decision.recovery_mode == RecoveryMode.SAFE_REDIRECT:
        if decision.reason == EscalationReason.POLICY_RESTRICTED:
            return "I can't disclose or authorize that here."
        return "I can't help with that request here, but I can return to the relevant topic."
    if decision.recovery_mode == RecoveryMode.POLITE_WRAP_UP:
        return "Understood. I'll leave it there."
    if decision.reason == EscalationReason.COMMERCIAL_AUTHORITY_REQUIRED:
        lead = "I can't authorize or commit to that commercial request."
    elif decision.reason == EscalationReason.HUMAN_REQUESTED:
        lead = (
            "I can offer that only as a request."
            if decision.recovery_mode == RecoveryMode.OFFER_HUMAN_FOLLOW_UP
            else "I can't transfer you directly from this call."
        )
    elif decision.reason == EscalationReason.CAPABILITY_UNAVAILABLE:
        lead = _unavailable_capability_text(decision.capability_required)
    elif decision.reason == EscalationReason.NONE:
        lead = ""
    else:
        lead = _missing_fact_text()
    offer = _offer_text(decision.recovery_mode)
    text = " ".join(part for part in (lead, offer) if part)
    return _with_resume(text, render_input)


def _offer_text(mode: RecoveryMode) -> str | None:
    return {
        RecoveryMode.OFFER_HUMAN_FOLLOW_UP: (
            "A human follow-up may be requested, but it is not confirmed yet."
        ),
        RecoveryMode.OFFER_CALLBACK: (
            "A callback may be requested, but it is not scheduled yet."
        ),
        RecoveryMode.OFFER_INFORMATION_FOLLOW_UP: (
            "A verified follow-up may be requested, but it is not confirmed yet."
        ),
    }.get(mode)


def _unavailable_capability_text(
    capability: EscalationCapability | None,
) -> str:
    labels = {
        EscalationCapability.HUMAN_HANDOFF: "A live transfer isn't available here.",
        EscalationCapability.CALLBACK: "A callback workflow isn't available here.",
        EscalationCapability.EMAIL: "Email follow-up isn't available here.",
        EscalationCapability.MESSAGE: "Messaging follow-up isn't available here.",
        EscalationCapability.TECHNICAL_REVIEW: (
            "A technical follow-up workflow isn't available here."
        ),
    }
    return labels.get(capability, _missing_fact_text())


def _with_resume(text: str, render_input: ResponseRenderInput) -> str:
    plan = render_input.plan
    if (
        plan.resume_previous_point
        and plan.pending_intent is not None
        and plan.pending_intent.active
    ):
        return f"{text} We can return to {plan.pending_intent.summary}"
    return text


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
