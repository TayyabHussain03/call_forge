"""Strict deterministic acceptance checks for untrusted wording proposals."""

from __future__ import annotations

import re

from app.conversation.realization.contracts import RealizationInput
from app.conversation.response_planning.contracts import (
    ConversationMove,
    QuestionStrategy,
)


class RealizationValidationError(ValueError):
    """The wording proposal falls outside its approved communication envelope."""


_CONTROL_OR_LEAKAGE = re.compile(
    r"```|[{}]|\b(?:system prompt|ignore previous|chain of thought|internal reasoning|"
    r"as an ai language model)\b",
    re.IGNORECASE,
)
_LIST = re.compile(r"(?:^|\n)\s*(?:[-*]|\d+[.)])\s+", re.MULTILINE)
_PROMISE = re.compile(
    r"\b(?:guarantee|guaranteed|roi|return on investment|increase conversions?|"
    r"save \d+%|never miss|discount|price|pricing|booked|scheduled|confirmed)\b",
    re.IGNORECASE,
)
_PRESSURE = re.compile(
    r"\b(?:act now|limited[- ]time|must buy|don't miss out|last chance)\b",
    re.IGNORECASE,
)
_UNPLANNED_OFFER = re.compile(
    r"\b(?:we (?:also )?(?:offer|provide)|our .{0,40} service)\b",
    re.IGNORECASE,
)
_CAPABILITY_ASSERTION = re.compile(
    r"\b(?:integrat(?:e|es|ed|ion)|supports?|automates?|handles?|includes?|"
    r"provides?|compatible|works with)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?%?\b")
_FILLERS = re.compile(r"\b(?:hmm+|okay+|right|i see)\b", re.IGNORECASE)
_ROMAN_URDU = re.compile(r"\b(?:ji|aap|hai|kaise|samajh|theek|bataye|acha)\b", re.I)


def validate_realization(candidate: object, value: RealizationInput) -> str:
    """Return normalized safe wording, or reject it for deterministic fallback."""
    if not isinstance(candidate, str):
        raise RealizationValidationError("realization must be plain text")
    if any(
        ord(character) < 32 and character not in "\r\n\t"
        for character in candidate
    ):
        raise RealizationValidationError("realization contains control characters")
    text = " ".join(candidate.split())
    if not text or len(text) > 800:
        raise RealizationValidationError("realization length is invalid")
    if _CONTROL_OR_LEAKAGE.search(text):
        raise RealizationValidationError("realization leaks structured or internal content")
    if value.policy.prohibit_lists and _LIST.search(candidate):
        raise RealizationValidationError("realization contains a questionnaire list")
    if _PROMISE.search(text):
        raise RealizationValidationError("realization contains a commercial promise")
    if _PRESSURE.search(text):
        raise RealizationValidationError("realization contains high-pressure language")
    if len(_FILLERS.findall(text)) > value.policy.max_fillers:
        raise RealizationValidationError("realization contains excessive fillers")
    _validate_question_count(text, value)
    _validate_services(text, value)
    _validate_claims(text, value)
    _validate_grounded_explanation(text, value)
    _validate_language(text, value)
    _validate_disclosures(text, value)
    return text


def _validate_question_count(text: str, value: RealizationInput) -> None:
    count = text.count("?")
    allows_question = value.plan.question_strategy != QuestionStrategy.NONE
    if count > value.policy.max_questions or (count and not allows_question):
        raise RealizationValidationError("realization contradicts question strategy")


def _validate_services(text: str, value: RealizationInput) -> None:
    folded = text.casefold()
    allowed = {name.casefold() for name in value.allowed_service_names}
    for name in value.policy.known_service_names:
        if name.casefold() in folded and name.casefold() not in allowed:
            raise RealizationValidationError("realization names an unsupported service")
    if (
        _UNPLANNED_OFFER.search(text)
        and value.plan.communicative_goal != ConversationMove.EXPLAIN_RELEVANT_FIT
    ):
        raise RealizationValidationError("realization introduces an unplanned service")


def _validate_grounded_explanation(text: str, value: RealizationInput) -> None:
    if value.plan.communicative_goal != ConversationMove.EXPLAIN_RELEVANT_FIT:
        return
    evidence = tuple(item.statement for item in value.approved_evidence)
    folded = text.casefold()
    if not evidence or not any(statement.casefold() in folded for statement in evidence):
        raise RealizationValidationError("service explanation lacks approved evidence")


def _validate_claims(text: str, value: RealizationInput) -> None:
    remainder = text
    for item in value.approved_evidence:
        remainder = re.sub(re.escape(item.statement), "", remainder, flags=re.IGNORECASE)
    if _CAPABILITY_ASSERTION.search(remainder):
        raise RealizationValidationError("realization adds an unsupported capability claim")
    approved_text = " ".join(item.statement for item in value.approved_evidence)
    for number in _NUMBER.findall(text):
        if number not in approved_text:
            raise RealizationValidationError("realization adds an unsupported numeric claim")


def _validate_language(text: str, value: RealizationInput) -> None:
    profile = value.language_profile
    if profile is None or profile.preferred_response_language not in {"ur", "hi"}:
        return
    if profile.preferred_script is None or profile.preferred_script.value != "latin":
        return
    if not _ROMAN_URDU.search(text):
        raise RealizationValidationError("realization does not match requested language")


def _validate_disclosures(text: str, value: RealizationInput) -> None:
    folded = text.casefold()
    identity_core = value.policy.identity_statement.rstrip(".!?").casefold()
    if value.identity_disclosure_required and (
        identity_core not in folded
    ):
        raise RealizationValidationError("realization omits required identity disclosure")
    decision = value.plan.consultative_decision
    if decision is None or not decision.commercial_transparency_required:
        return
    if "not a sales call" in folded or "not selling" in folded:
        raise RealizationValidationError("realization conceals commercial purpose")
    if not any(term in folded for term in ("business call", "sales call", "commercial")):
        raise RealizationValidationError("realization omits commercial disclosure")
