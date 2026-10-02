"""Immutable contracts for Campaign & Lead Queue Runtime.

Campaigns own lead queues and control calling eligibility. This layer decides
WHICH lead is next and WHETHER a call is allowed — it never places calls,
generates responses, or modifies conversation state.

TRUST BOUNDARY: campaign configuration is trusted organizational input.
Queue state is authoritative — only deterministic transitions change it.
DNC has highest priority and is never overridden.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# Campaign enums
# ---------------------------------------------------------------------------


class CampaignStatus(str, Enum):
    """Lifecycle status of a campaign."""

    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class LeadQueueStatus(str, Enum):
    """Status of a lead within a campaign queue."""

    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    DNC = "dnc"
    NEEDS_REVIEW = "needs_review"
    SKIPPED = "skipped"
    RETRY_SCHEDULED = "retry_scheduled"
    EXHAUSTED = "exhausted"


class CallEligibility(str, Enum):
    """Whether a call attempt is currently allowed."""

    ELIGIBLE = "eligible"
    DAILY_LIMIT_REACHED = "daily_limit_reached"
    OUTSIDE_CALLING_WINDOW = "outside_calling_window"
    GAP_NOT_ELAPSED = "gap_not_elapsed"
    CAMPAIGN_PAUSED = "campaign_paused"
    CAMPAIGN_NOT_ACTIVE = "campaign_not_active"
    LEAD_DNC = "lead_dnc"
    LEAD_NOT_READY = "lead_not_ready"
    MAX_RETRIES_EXHAUSTED = "max_retries_exhausted"
    CONCURRENCY_LIMIT = "concurrency_limit"
    NO_LEADS_AVAILABLE = "no_leads_available"


class RetryEligibility(str, Enum):
    """Whether a lead may be retried after a failed attempt."""

    ELIGIBLE = "eligible"
    MAX_RETRIES_REACHED = "max_retries_reached"
    DNC = "dnc"
    COMPLETED = "completed"


class NoAnswerPolicy(str, Enum):
    """What to do when a call is not answered."""

    RETRY_LATER = "retry_later"
    SKIP = "skip"
    MARK_EXHAUSTED = "mark_exhausted"


# ---------------------------------------------------------------------------
# Calling window
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallingWindow:
    """Time window when calls are allowed (24h format, HH:MM)."""

    start_hour: int
    start_minute: int
    end_hour: int
    end_minute: int
    timezone: str

    def __post_init__(self) -> None:
        if not 0 <= self.start_hour <= 23:
            raise ValueError("start hour must be 0-23")
        if not 0 <= self.start_minute <= 59:
            raise ValueError("start minute must be 0-59")
        if not 0 <= self.end_hour <= 23:
            raise ValueError("end hour must be 0-23")
        if not 0 <= self.end_minute <= 59:
            raise ValueError("end minute must be 0-59")
        if not self.timezone or not self.timezone.strip():
            raise ValueError("timezone must not be empty")

    def start_minutes(self) -> int:
        return self.start_hour * 60 + self.start_minute

    def end_minutes(self) -> int:
        return self.end_hour * 60 + self.end_minute

    def contains_time(self, hour: int, minute: int) -> bool:
        current = hour * 60 + minute
        start = self.start_minutes()
        end = self.end_minutes()
        if start <= end:
            return start <= current < end
        return current >= start or current < end


# ---------------------------------------------------------------------------
# Campaign configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CampaignConfig:
    """Trusted organizational campaign configuration."""

    campaign_id: str
    tenant_id: str
    name: str

    status: CampaignStatus = CampaignStatus.DRAFT

    authorized_service_ids: tuple[str, ...] = ()
    target_market: str | None = None
    language_preference: str | None = None
    tone: str | None = None

    daily_call_limit: int = 50
    gap_between_calls_seconds: int = 30
    max_concurrent_calls: int = 1
    max_retries_per_lead: int = 3
    no_answer_policy: NoAnswerPolicy = NoAnswerPolicy.RETRY_LATER

    calling_window: CallingWindow | None = None

    def __post_init__(self) -> None:
        if not self.campaign_id or not self.campaign_id.strip():
            raise ValueError("campaign_id must not be empty")
        if not self.tenant_id or not self.tenant_id.strip():
            raise ValueError("tenant_id must not be empty")
        if not self.name or not self.name.strip():
            raise ValueError("campaign name must not be empty")
        if self.daily_call_limit < 1:
            raise ValueError("daily call limit must be at least 1")
        if self.gap_between_calls_seconds < 0:
            raise ValueError("gap between calls must be non-negative")
        if self.max_concurrent_calls < 1:
            raise ValueError("max concurrent calls must be at least 1")
        if self.max_retries_per_lead < 0:
            raise ValueError("max retries must be non-negative")


# ---------------------------------------------------------------------------
# Lead queue entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeadQueueEntry:
    """A lead's position and state in a campaign queue."""

    lead_id: str
    campaign_id: str
    tenant_id: str

    queue_position: int
    status: LeadQueueStatus = LeadQueueStatus.PENDING

    attempt_count: int = 0
    last_attempt_at: str | None = None

    is_dnc: bool = False

    def __post_init__(self) -> None:
        if not self.lead_id or not self.lead_id.strip():
            raise ValueError("lead_id must not be empty")
        if not self.campaign_id or not self.campaign_id.strip():
            raise ValueError("campaign_id must not be empty")
        if not self.tenant_id or not self.tenant_id.strip():
            raise ValueError("tenant_id must not be empty")
        if self.queue_position < 0:
            raise ValueError("queue position must be non-negative")
        if self.attempt_count < 0:
            raise ValueError("attempt count must be non-negative")
        if self.is_dnc and self.status != LeadQueueStatus.DNC:
            object.__setattr__(self, "status", LeadQueueStatus.DNC)


# ---------------------------------------------------------------------------
# Queue state snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueueStateSnapshot:
    """Immutable snapshot of campaign queue state for decision-making."""

    campaign_id: str
    tenant_id: str
    total_leads: int
    pending: int
    ready: int
    in_progress: int
    completed: int
    failed: int
    dnc: int
    exhausted: int
    calls_today: int
    active_calls: int
    last_call_completed_at: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("total_leads", "pending", "ready", "in_progress",
                           "completed", "failed", "dnc", "exhausted",
                           "calls_today", "active_calls"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")


# ---------------------------------------------------------------------------
# Eligibility decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallEligibilityDecision:
    """Deterministic decision on whether a call may proceed."""

    eligible: bool
    reason: CallEligibility
    lead_id: str | None = None
    campaign_id: str | None = None

    def __post_init__(self) -> None:
        if self.eligible and self.reason != CallEligibility.ELIGIBLE:
            raise ValueError("eligible decision must have ELIGIBLE reason")
        if not self.eligible and self.reason == CallEligibility.ELIGIBLE:
            raise ValueError("ineligible decision must not have ELIGIBLE reason")


@dataclass(frozen=True)
class NextLeadDecision:
    """Result of deterministic next-lead selection."""

    available: bool
    lead_entry: LeadQueueEntry | None = None
    ineligibility_reason: CallEligibility | None = None

    def __post_init__(self) -> None:
        if self.available and self.lead_entry is None:
            raise ValueError("available decision must include a lead entry")
        if not self.available and self.lead_entry is not None:
            raise ValueError("unavailable decision must not include a lead entry")
