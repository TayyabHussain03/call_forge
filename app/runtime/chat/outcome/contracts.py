"""Immutable contracts for Post-Conversation Outcome Intelligence.

LeadOutcome is descriptive/advisory only. It must not send emails,
schedule meetings, write CRM, modify Excel, or place calls.

TRUST BOUNDARY: outcome is derived from authoritative pipeline state,
not from raw transcript or LLM inference. DNC has highest priority
and is never collapsed into NOT_INTERESTED. Runtime/provider failure
is distinct from customer outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OutcomeStatus(str, Enum):
    """Primary customer-facing classification of the conversation result.

    This describes what the prospect wanted, NOT whether the pipeline
    completed cleanly. See ExecutionStatus for runtime fault reporting.
    """

    INTERESTED = "interested"
    NOT_INTERESTED = "not_interested"
    DNC = "dnc"
    FOLLOW_UP = "follow_up"
    NO_FIT = "no_fit"
    NEEDS_MORE_INFORMATION = "needs_more_information"
    COMPLETED_NO_COMMITMENT = "completed_no_commitment"
    UNKNOWN = "unknown"


class ExecutionStatus(str, Enum):
    """Runtime/execution status, orthogonal to the customer outcome."""

    COMPLETED = "completed"
    FAILED = "failed"


class InterestLevel(str, Enum):
    """Observed interest, never inferred from absence."""

    EXPLICIT_INTEREST = "explicit_interest"
    POSSIBLE_INTEREST = "possible_interest"
    NO_INTEREST = "no_interest"
    UNKNOWN = "unknown"


class DNCStatus(str, Enum):
    """DNC is a separate, highest-priority classification."""

    NOT_DNC = "not_dnc"
    DNC_CONFIRMED = "dnc_confirmed"


class DecisionMakerStatus(str, Enum):
    """Whether a decision-maker was identified."""

    CONFIRMED = "confirmed"
    NOT_DECISION_MAKER = "not_decision_maker"
    UNKNOWN = "unknown"


class FollowUpNeed(str, Enum):
    """Whether follow-up is indicated."""

    NEEDED = "needed"
    NOT_NEEDED = "not_needed"
    UNKNOWN = "unknown"


class TerminationReason(str, Enum):
    """Why the session ended."""

    NATURAL_COMPLETION = "natural_completion"
    DNC_REQUEST = "dnc_request"
    NOT_INTERESTED_EXIT = "not_interested_exit"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MANUAL_TERMINATION = "manual_termination"
    RUNTIME_ERROR = "runtime_error"
    UNKNOWN = "unknown"


class FieldConfidence(str, Enum):
    """Whether a field is observed, inferred, or unknown."""

    OBSERVED = "observed"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IdentifiedProblem:
    """One problem observed during the conversation."""

    description: str
    confidence: FieldConfidence

    def __post_init__(self) -> None:
        if not self.description or not self.description.strip():
            raise ValueError("problem description must not be empty")
        if len(self.description) > 200:
            raise ValueError("problem description exceeds 200 chars")
        if not isinstance(self.confidence, FieldConfidence):
            raise TypeError("confidence must be FieldConfidence")


@dataclass(frozen=True)
class RelevantService:
    """A service identified as relevant through existing authorization logic."""

    service_id: str
    match_reason: str
    confidence: FieldConfidence

    def __post_init__(self) -> None:
        if not self.service_id or not self.service_id.strip():
            raise ValueError("service_id must not be empty")
        if not self.match_reason or not self.match_reason.strip():
            raise ValueError("match_reason must not be empty")
        if len(self.match_reason) > 200:
            raise ValueError("match_reason exceeds 200 chars")


@dataclass(frozen=True)
class RecordedObjection:
    """An objection observed during the conversation."""

    objection_type: str
    confidence: FieldConfidence

    def __post_init__(self) -> None:
        if not self.objection_type or not self.objection_type.strip():
            raise ValueError("objection_type must not be empty")


@dataclass(frozen=True)
class OutcomeSummaryLine:
    """One structured line in the factual conversation summary."""

    label: str
    value: str

    def __post_init__(self) -> None:
        if not self.label or not self.label.strip():
            raise ValueError("label must not be empty")
        if not self.value or not self.value.strip():
            raise ValueError("value must not be empty")
        if len(self.label) > 60:
            raise ValueError("label exceeds 60 chars")
        if len(self.value) > 300:
            raise ValueError("value exceeds 300 chars")


# ---------------------------------------------------------------------------
# LeadOutcome
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeadOutcome:
    """Immutable, advisory-only post-conversation outcome.

    Descriptive only — must not trigger emails, CRM writes, meetings,
    calls, or any execution side effect.
    """

    lead_id: str
    campaign_id: str
    session_id: str
    tenant_id: str

    outcome_status: OutcomeStatus
    interest_level: InterestLevel
    dnc_status: DNCStatus
    execution_status: ExecutionStatus
    termination_reason: TerminationReason

    identified_problems: tuple[IdentifiedProblem, ...] = ()
    relevant_services: tuple[RelevantService, ...] = ()
    objections: tuple[RecordedObjection, ...] = ()

    decision_maker_status: DecisionMakerStatus = DecisionMakerStatus.UNKNOWN
    qualification_completeness: str | None = None
    next_step: str | None = None
    follow_up_needed: FollowUpNeed = FollowUpNeed.UNKNOWN
    outstanding_questions: tuple[str, ...] = ()

    summary_lines: tuple[OutcomeSummaryLine, ...] = ()

    def __post_init__(self) -> None:
        if not self.lead_id or not self.lead_id.strip():
            raise ValueError("lead_id must not be empty")
        if not self.campaign_id or not self.campaign_id.strip():
            raise ValueError("campaign_id must not be empty")
        if not self.session_id or not self.session_id.strip():
            raise ValueError("session_id must not be empty")
        if not self.tenant_id or not self.tenant_id.strip():
            raise ValueError("tenant_id must not be empty")
        if not isinstance(self.outcome_status, OutcomeStatus):
            raise TypeError("outcome_status must be OutcomeStatus")
        if not isinstance(self.dnc_status, DNCStatus):
            raise TypeError("dnc_status must be DNCStatus")
        if not isinstance(self.execution_status, ExecutionStatus):
            raise TypeError("execution_status must be ExecutionStatus")
        if self.dnc_status == DNCStatus.DNC_CONFIRMED and self.outcome_status != OutcomeStatus.DNC:
            raise ValueError("DNC_CONFIRMED must have DNC outcome_status")
        if len(self.identified_problems) > 10:
            raise ValueError("too many identified problems")
        if len(self.relevant_services) > 10:
            raise ValueError("too many relevant services")
        if len(self.objections) > 10:
            raise ValueError("too many objections")
        if len(self.outstanding_questions) > 10:
            raise ValueError("too many outstanding questions")
        if len(self.summary_lines) > 20:
            raise ValueError("too many summary lines")


# ---------------------------------------------------------------------------
# Outcome input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutcomeIntelligenceInput:
    """Trusted pipeline state collected after session termination."""

    lead_id: str
    campaign_id: str
    session_id: str
    tenant_id: str

    is_dnc: bool = False
    is_not_interested: bool = False
    session_error: bool = False
    termination_reason: TerminationReason = TerminationReason.UNKNOWN
    conversation_terminal: bool = False

    prospect_observed: object | None = None
    prospect_inferred: object | None = None
    business_conversation: object | None = None
    business_diagnostic: object | None = None
    qualification: object | None = None
    business_memory: object | None = None
    playbook_guidance: object | None = None
    offered_service_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.lead_id or not self.lead_id.strip():
            raise ValueError("lead_id must not be empty")
        if not self.session_id or not self.session_id.strip():
            raise ValueError("session_id must not be empty")
