"""Deterministic validation for advisory sales cognition guidance."""

from __future__ import annotations

from dataclasses import fields

from app.conversation.consultative.contracts import ServiceFitStatus
from app.conversation.sales_cognition.contracts import (
    BuyingReadinessGuidance,
    InterestStrength,
    SalesCognitionInput,
    SalesConversationGuidance,
)


class SalesGuidanceValidationError(ValueError):
    """Guidance contradicted known conversational evidence or ownership."""


_FORBIDDEN_FIELDS = {
    "action",
    "next_state",
    "service_id",
    "approved_evidence",
    "price",
    "discount",
    "callback_confirmed",
    "demo_confirmed",
    "booking_confirmed",
    "authority",
    "persistence",
}


def validate_sales_guidance(
    guidance: SalesConversationGuidance,
    value: SalesCognitionInput,
) -> SalesConversationGuidance:
    """Fail closed if advisory guidance claims unavailable conversational readiness."""
    names = {item.name for item in fields(SalesConversationGuidance)}
    if names.intersection(_FORBIDDEN_FIELDS):
        raise SalesGuidanceValidationError("guidance contains forbidden authority fields")
    if guidance.question_focus in value.recent_question_concepts:
        raise SalesGuidanceValidationError("guidance repeats a recent question concept")
    if (
        guidance.buying_guidance == BuyingReadinessGuidance.EXPLAIN_FIT
        and value.service_fit.status != ServiceFitStatus.SUPPORTED_FIT
    ):
        raise SalesGuidanceValidationError("fit explanation requires supported fit")
    if (
        guidance.buying_guidance
        == BuyingReadinessGuidance.SUGGEST_MICRO_COMMITMENT
        and guidance.interest != InterestStrength.BUYING_SIGNAL
    ):
        raise SalesGuidanceValidationError("micro commitment requires a buying signal")
    return guidance
