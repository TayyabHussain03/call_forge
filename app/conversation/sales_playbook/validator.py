"""Fail-closed validation for advisory sales playbook guidance."""

from __future__ import annotations

from dataclasses import fields

from app.conversation.sales_playbook.contracts import (
    OpportunityGuidance,
    PitchReadiness,
    PlaybookRestriction,
    SalesPlaybookInput,
)

_FORBIDDEN_FIELDS = frozenset(
    {
        "action",
        "next_state",
        "approved_evidence",
        "execution",
        "response_text",
        "price",
        "discount",
        "booking",
        "authority",
    }
)
_MANDATORY_RESTRICTIONS = frozenset(
    {
        PlaybookRestriction.NO_GUARANTEED_REVENUE,
        PlaybookRestriction.NO_GUARANTEED_ROI,
        PlaybookRestriction.NO_PRESSURE_LANGUAGE,
    }
)


class PlaybookGuidanceValidationError(ValueError):
    """Advisory guidance escaped its eligibility or safety envelope."""


def validate_playbook_guidance(
    guidance: OpportunityGuidance, value: SalesPlaybookInput
) -> OpportunityGuidance:
    """Return guidance only when it remains bounded and non-authoritative."""
    names = {item.name for item in fields(OpportunityGuidance)}
    if names & _FORBIDDEN_FIELDS:
        raise PlaybookGuidanceValidationError("guidance contains forbidden fields")
    if guidance.selected_service_id is not None and (
        guidance.selected_service_id not in value.eligible_service_ids
    ):
        raise PlaybookGuidanceValidationError("selected service is not eligible")
    if any(item not in value.eligible_service_ids for item in guidance.future_opportunity_ids):
        raise PlaybookGuidanceValidationError("future opportunity is not eligible")
    if guidance.secondary_question is not None and not value.context.true_ambiguity:
        raise PlaybookGuidanceValidationError(
            "a secondary question requires true ambiguity"
        )
    if not _MANDATORY_RESTRICTIONS.issubset(guidance.restrictions):
        raise PlaybookGuidanceValidationError("mandatory restrictions are missing")
    if guidance.existing_customer_optimization and (
        guidance.selected_service_id not in value.context.existing_service_ids
    ):
        raise PlaybookGuidanceValidationError("optimization requires an existing service")
    if guidance.pitch_readiness == PitchReadiness.NO_FIT and (
        guidance.selected_service_id is not None
        or guidance.primary_question is not None
        or guidance.future_opportunity_ids
    ):
        raise PlaybookGuidanceValidationError("no-fit guidance must not sell")
    return guidance
