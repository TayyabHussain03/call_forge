"""Deterministic communication recovery policy for unknown or unsafe requests."""

from __future__ import annotations

from app.brain.authority.models import AuthorityTier
from app.brain.scope.validator import ScopeCategory
from app.conversation.context.contracts import ContactContextStatus
from app.conversation.prospect_intelligence.contracts import PreferredNextStep
from app.conversation.escalation.contracts import (
    DisclosureStatus,
    EscalationCapability,
    EscalationDecision,
    EscalationReason,
    GracefulEscalationInput,
    KnowledgeRequestKind,
    RecoveryMode,
    RequestedFollowUp,
)
from app.conversation.escalation.evidence import (
    ApprovedEvidenceMatcher,
    EvidenceMatchStatus,
)


class GracefulEscalationPolicy:
    """Select safe communication behavior without granting or executing authority."""

    def __init__(self, matcher: ApprovedEvidenceMatcher | None = None) -> None:
        self._matcher = matcher or ApprovedEvidenceMatcher()

    def evaluate(self, policy_input: GracefulEscalationInput) -> EscalationDecision:
        """Fail-safe wrapper around pure deterministic evaluation."""
        try:
            return self._evaluate(policy_input)
        except Exception:
            return _decision(
                EscalationReason.MISSING_APPROVED_EVIDENCE,
                RecoveryMode.ACKNOWLEDGE_UNKNOWN,
                policy_input,
            )

    def _evaluate(self, value: GracefulEscalationInput) -> EscalationDecision:
        if value.scope_category == ScopeCategory.OUT_OF_SCOPE:
            return _decision(
                EscalationReason.OUT_OF_SCOPE,
                RecoveryMode.SAFE_REDIRECT,
                value,
            )
        if value.scope_category == ScopeCategory.UNKNOWN_SCOPE:
            return _decision(
                EscalationReason.AMBIGUOUS_REQUEST,
                RecoveryMode.ASK_CLARIFYING_QUESTION,
                value,
            )
        if value.authority_tier == AuthorityTier.HUMAN_APPROVAL_REQUIRED:
            return self._commercial_authority(value)
        if value.authority_tier == AuthorityTier.DENIED:
            return _decision(
                EscalationReason.POLICY_RESTRICTED,
                RecoveryMode.SAFE_REDIRECT,
                value,
            )

        if _explicit_wrap_up_requested(value):
            return _decision(
                EscalationReason.NONE,
                RecoveryMode.POLITE_WRAP_UP,
                value,
            )
        if _explicit_human_requested(value):
            return self._human_request(value)
        if value.request.kind == KnowledgeRequestKind.AMBIGUOUS:
            return _decision(
                EscalationReason.AMBIGUOUS_REQUEST,
                RecoveryMode.ASK_CLARIFYING_QUESTION,
                value,
            )
        follow_up = self._requested_follow_up(value)
        if follow_up is not None:
            return follow_up
        if value.request.kind == KnowledgeRequestKind.NONE:
            return _decision(
                EscalationReason.NONE,
                RecoveryMode.PROCEED_NORMALLY,
                value,
            )

        match = self._matcher.match(
            value.context.approved_evidence,
            fact_key=value.request.fact_key,
            evidence_type=value.request.evidence_type,
            campaign_id=value.context.campaign.campaign_id,
            service_id=value.request.service_id,
        )
        if match.status == EvidenceMatchStatus.CONFLICT:
            return _decision(
                EscalationReason.EVIDENCE_CONFLICT,
                RecoveryMode.ACKNOWLEDGE_UNKNOWN,
                value,
            )
        if value.request.disclosure_status == DisclosureStatus.RESTRICTED:
            return _decision(
                EscalationReason.POLICY_RESTRICTED,
                RecoveryMode.SAFE_REDIRECT,
                value,
            )
        if match.status == EvidenceMatchStatus.MATCHED:
            return EscalationDecision(
                EscalationReason.NONE,
                RecoveryMode.ANSWER_WITH_EVIDENCE,
                True,
                tuple(item.evidence_id for item in match.items),
                resume_previous_goal=_resume(value),
            )
        return self._unknown(value)

    def _unknown(self, value: GracefulEscalationInput) -> EscalationDecision:
        reason = {
            KnowledgeRequestKind.TECHNICAL_DETAIL: EscalationReason.UNKNOWN_TECHNICAL_DETAIL,
            KnowledgeRequestKind.PRODUCT_CAPABILITY: (
                EscalationReason.UNKNOWN_PRODUCT_CAPABILITY
            ),
        }.get(value.request.kind, EscalationReason.MISSING_APPROVED_EVIDENCE)
        if value.request.kind in {
            KnowledgeRequestKind.TECHNICAL_DETAIL,
            KnowledgeRequestKind.PRODUCT_CAPABILITY,
        } and value.capabilities.technical_followup_available:
            return _decision(
                reason,
                RecoveryMode.OFFER_INFORMATION_FOLLOW_UP,
                value,
                EscalationCapability.TECHNICAL_REVIEW,
                True,
            )
        return _decision(reason, RecoveryMode.ACKNOWLEDGE_UNKNOWN, value)

    def _human_request(self, value: GracefulEscalationInput) -> EscalationDecision:
        available = value.capabilities.human_handoff_available
        if not available and value.capabilities.callback_workflow_available:
            return _decision(
                EscalationReason.HUMAN_REQUESTED,
                RecoveryMode.OFFER_CALLBACK,
                value,
                EscalationCapability.CALLBACK,
                True,
            )
        if (
            not available
            and value.capabilities.email_followup_available
            and value.context.contact.status == ContactContextStatus.CONFIRMED
        ):
            return _decision(
                EscalationReason.HUMAN_REQUESTED,
                RecoveryMode.OFFER_INFORMATION_FOLLOW_UP,
                value,
                EscalationCapability.EMAIL,
                True,
            )
        return _decision(
            EscalationReason.HUMAN_REQUESTED,
            (
                RecoveryMode.OFFER_HUMAN_FOLLOW_UP
                if available
                else RecoveryMode.ACKNOWLEDGE_UNKNOWN
            ),
            value,
            EscalationCapability.HUMAN_HANDOFF,
            available,
        )

    def _commercial_authority(
        self, value: GracefulEscalationInput
    ) -> EscalationDecision:
        available = value.capabilities.human_handoff_available
        return _decision(
            EscalationReason.COMMERCIAL_AUTHORITY_REQUIRED,
            (
                RecoveryMode.OFFER_HUMAN_FOLLOW_UP
                if available
                else RecoveryMode.ACKNOWLEDGE_UNKNOWN
            ),
            value,
            EscalationCapability.HUMAN_HANDOFF,
            available,
        )

    def _requested_follow_up(
        self, value: GracefulEscalationInput
    ) -> EscalationDecision | None:
        requested = value.request.requested_follow_up
        if requested == RequestedFollowUp.NONE:
            return None
        capability = EscalationCapability(requested.value)
        available = value.capabilities.supports(capability)
        if not available:
            return _decision(
                EscalationReason.CAPABILITY_UNAVAILABLE,
                RecoveryMode.ACKNOWLEDGE_UNKNOWN,
                value,
                capability,
                False,
            )
        if requested in {RequestedFollowUp.EMAIL, RequestedFollowUp.MESSAGE} and (
            value.context.contact.status != ContactContextStatus.CONFIRMED
        ):
            return _decision(
                EscalationReason.CAPABILITY_UNAVAILABLE,
                RecoveryMode.ASK_CLARIFYING_QUESTION,
                value,
                capability,
                True,
            )
        recovery = {
            RequestedFollowUp.HUMAN_HANDOFF: RecoveryMode.OFFER_HUMAN_FOLLOW_UP,
            RequestedFollowUp.CALLBACK: RecoveryMode.OFFER_CALLBACK,
            RequestedFollowUp.EMAIL: RecoveryMode.OFFER_INFORMATION_FOLLOW_UP,
            RequestedFollowUp.MESSAGE: RecoveryMode.OFFER_INFORMATION_FOLLOW_UP,
            RequestedFollowUp.TECHNICAL_REVIEW: RecoveryMode.OFFER_INFORMATION_FOLLOW_UP,
        }[requested]
        return _decision(
            EscalationReason.NONE,
            recovery,
            value,
            capability,
            True,
        )


def _explicit_human_requested(value: GracefulEscalationInput) -> bool:
    prospect = value.context.prospect
    return bool(
        prospect is not None
        and prospect.human_requested is not None
        and prospect.human_requested.value
    )


def _explicit_wrap_up_requested(value: GracefulEscalationInput) -> bool:
    prospect = value.context.prospect
    return bool(
        prospect is not None
        and prospect.explicit_next_step is not None
        and prospect.explicit_next_step.value == PreferredNextStep.END_CONVERSATION
    )


def _resume(value: GracefulEscalationInput) -> bool:
    pending = value.context.pending_intent
    return bool(pending is not None and pending.active)


def _decision(
    reason: EscalationReason,
    recovery: RecoveryMode,
    value: GracefulEscalationInput,
    capability: EscalationCapability | None = None,
    available: bool | None = None,
) -> EscalationDecision:
    return EscalationDecision(
        reason,
        recovery,
        False,
        capability_required=capability,
        capability_available=available,
        resume_previous_goal=_resume(value),
    )
