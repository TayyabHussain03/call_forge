"""Typed communication-only contracts for graceful unknown handling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.brain.authority.models import AuthorityTier
from app.brain.scope.validator import ScopeCategory
from app.conversation.context.contracts import EvidenceType, LeanTurnContext


class EscalationReason(str, Enum):
    NONE = "none"
    MISSING_APPROVED_EVIDENCE = "missing_approved_evidence"
    UNKNOWN_TECHNICAL_DETAIL = "unknown_technical_detail"
    UNKNOWN_PRODUCT_CAPABILITY = "unknown_product_capability"
    COMMERCIAL_AUTHORITY_REQUIRED = "commercial_authority_required"
    HUMAN_REQUESTED = "human_requested"
    OUT_OF_SCOPE = "out_of_scope"
    AMBIGUOUS_REQUEST = "ambiguous_request"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    POLICY_RESTRICTED = "policy_restricted"
    EVIDENCE_CONFLICT = "evidence_conflict"


class RecoveryMode(str, Enum):
    PROCEED_NORMALLY = "proceed_normally"
    ANSWER_WITH_EVIDENCE = "answer_with_evidence"
    ACKNOWLEDGE_UNKNOWN = "acknowledge_unknown"
    ASK_CLARIFYING_QUESTION = "ask_clarifying_question"
    OFFER_HUMAN_FOLLOW_UP = "offer_human_follow_up"
    OFFER_CALLBACK = "offer_callback"
    OFFER_INFORMATION_FOLLOW_UP = "offer_information_follow_up"
    SAFE_REDIRECT = "safe_redirect"
    POLITE_WRAP_UP = "polite_wrap_up"


class EscalationCapability(str, Enum):
    HUMAN_HANDOFF = "human_handoff"
    CALLBACK = "callback"
    EMAIL = "email"
    MESSAGE = "message"
    TECHNICAL_REVIEW = "technical_review"


@dataclass(frozen=True)
class AvailableEscalationCapabilities:
    """Trusted operational availability; every capability defaults unavailable."""

    human_handoff_available: bool = False
    callback_workflow_available: bool = False
    email_followup_available: bool = False
    messaging_followup_available: bool = False
    technical_followup_available: bool = False

    def __post_init__(self) -> None:
        if any(not isinstance(value, bool) for value in self.__dict__.values()):
            raise TypeError("escalation capability flags must be booleans")

    def supports(self, capability: EscalationCapability) -> bool:
        return {
            EscalationCapability.HUMAN_HANDOFF: self.human_handoff_available,
            EscalationCapability.CALLBACK: self.callback_workflow_available,
            EscalationCapability.EMAIL: self.email_followup_available,
            EscalationCapability.MESSAGE: self.messaging_followup_available,
            EscalationCapability.TECHNICAL_REVIEW: self.technical_followup_available,
        }[capability]


class KnowledgeRequestKind(str, Enum):
    NONE = "none"
    FACT = "fact"
    TECHNICAL_DETAIL = "technical_detail"
    PRODUCT_CAPABILITY = "product_capability"
    COMMERCIAL = "commercial"
    AMBIGUOUS = "ambiguous"


class RequestedFollowUp(str, Enum):
    NONE = "none"
    HUMAN_HANDOFF = "human_handoff"
    CALLBACK = "callback"
    EMAIL = "email"
    MESSAGE = "message"
    TECHNICAL_REVIEW = "technical_review"


class DisclosureStatus(str, Enum):
    ALLOWED = "allowed"
    RESTRICTED = "restricted"


@dataclass(frozen=True)
class EscalationRequest:
    """Trusted structured description of what needs factual handling."""

    kind: KnowledgeRequestKind = KnowledgeRequestKind.NONE
    fact_key: str | None = None
    evidence_type: EvidenceType | None = None
    service_id: str | None = None
    requested_follow_up: RequestedFollowUp = RequestedFollowUp.NONE
    disclosure_status: DisclosureStatus = DisclosureStatus.ALLOWED

    def __post_init__(self) -> None:
        for value, name in (
            (self.fact_key, "fact key"),
            (self.service_id, "service id"),
        ):
            if value is not None and (not value.strip() or len(value) > 100):
                raise ValueError(f"{name} must contain 1-100 characters")


@dataclass(frozen=True)
class GracefulEscalationInput:
    """Current trusted context plus existing deterministic classifications."""

    context: LeanTurnContext
    request: EscalationRequest = EscalationRequest()
    capabilities: AvailableEscalationCapabilities = AvailableEscalationCapabilities()
    scope_category: ScopeCategory | None = None
    authority_tier: AuthorityTier | None = None


@dataclass(frozen=True)
class EscalationDecision:
    """Communication recovery only; it cannot execute or confirm anything."""

    reason: EscalationReason
    recovery_mode: RecoveryMode
    answerable: bool
    evidence_ids: tuple[str, ...] = ()
    capability_required: EscalationCapability | None = None
    capability_available: bool | None = None
    resume_previous_goal: bool = False

    def __post_init__(self) -> None:
        evidence_ids = tuple(self.evidence_ids)
        if len(evidence_ids) > 8 or len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("decision evidence ids must be unique and limited to eight")
        if evidence_ids != tuple(sorted(evidence_ids)):
            raise ValueError("decision evidence ids must be stably ordered")
        if self.answerable != bool(evidence_ids):
            raise ValueError("answerable decisions require matching evidence")
        if self.recovery_mode == RecoveryMode.ANSWER_WITH_EVIDENCE and not self.answerable:
            raise ValueError("evidence answer requires approved evidence")
        if self.answerable and self.recovery_mode != RecoveryMode.ANSWER_WITH_EVIDENCE:
            raise ValueError("only an evidence answer may be marked answerable")
        if (self.capability_required is None) != (self.capability_available is None):
            raise ValueError("capability requirement and availability must be paired")
        required_by_offer = {
            RecoveryMode.OFFER_HUMAN_FOLLOW_UP: EscalationCapability.HUMAN_HANDOFF,
            RecoveryMode.OFFER_CALLBACK: EscalationCapability.CALLBACK,
        }
        expected = required_by_offer.get(self.recovery_mode)
        if expected is not None and (
            self.capability_required != expected or self.capability_available is not True
        ):
            raise ValueError("offered recovery requires its available capability")
        if self.recovery_mode == RecoveryMode.OFFER_INFORMATION_FOLLOW_UP and (
            self.capability_required
            not in {
                EscalationCapability.EMAIL,
                EscalationCapability.MESSAGE,
                EscalationCapability.TECHNICAL_REVIEW,
            }
            or self.capability_available is not True
        ):
            raise ValueError("information follow-up requires an available channel")
        object.__setattr__(self, "evidence_ids", evidence_ids)
