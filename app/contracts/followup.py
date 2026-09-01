"""Contracts for follow-ups and scheduled callbacks.

TRUST BOUNDARY: internal/TRUSTED.
    Follow-up tasks system khud banata hai (call outcome ke basis par), external
    source se nahi. Isliye ye trusted-internal contracts hain — lekin phir bhi
    strict, taake galat scheduling data persist na ho.

DESIGN: Follow-up ek intent ka result hai, tone ka nahi. Ek busy/harsh client
    bhi high-interest follow-up ban sakta hai. Reason enum isi ko capture karta
    hai, lead quality se independent.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class FollowupReason(str, Enum):
    """Why a follow-up is required.

    Ye reason lead quality se independent hai — busy hone ka matlab bura lead
    nahi. Reason sirf ye batata hai ke follow-up kyun queue hua.
    """

    BUSY = "busy"                          # client abhi busy tha
    CALLBACK_REQUESTED = "callback_requested"  # client ne khud time maanga
    DISCONNECTED = "disconnected"          # call interested-state mein cut gayi
    NO_ANSWER = "no_answer"                # pick nahi hui
    CONTACT_NOT_OBTAINED = "contact_not_obtained"  # interested but email na mila


class FollowupStatus(str, Enum):
    """Lifecycle of a follow-up task."""

    PENDING = "pending"
    SCHEDULED = "scheduled"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class CallbackSchedule(BaseModel):
    """A specific time window the client asked to be called back.

    Attributes:
        earliest_at: Is waqt se pehle call nahi karni. None agar koi lower
            bound nahi.
        latest_at: Is waqt tak call karni chahiye. None agar koi upper bound
            nahi.
        timezone: IANA timezone name (e.g. "America/Chicago") agar pata ho.
        note: Client ka bola hua time-related note (e.g. "tomorrow afternoon").
    """

    model_config = ConfigDict(extra="forbid")

    earliest_at: datetime | None = None
    latest_at: datetime | None = None
    timezone: str | None = Field(default=None, max_length=60)
    note: str | None = Field(default=None, max_length=300)


class FollowupTask(BaseModel):
    """A follow-up action the system must take for a lead.

    Ye system dwara generated trusted task hai. Persistence/worker layer isko
    queue mein daalti hai. Reason aur interest ALAG rakhe gaye hain jaan-boojh
    kar.

    Attributes:
        lead_id: Kis lead ke liye.
        contact_id: Agar kisi specific person se follow-up karna hai.
        reason: Follow-up kyun banaya (enum, tone-independent).
        status: Task ka current lifecycle stage.
        schedule: Optional specific callback timing.
        attempts: Ab tak kitni follow-up koshishein hui.
        max_attempts: Kitni koshishon ke baad give up karna (config-driven
            default upar layer set karti hai).
        created_at: Task kab bana.
    """

    model_config = ConfigDict(extra="forbid")

    lead_id: str
    contact_id: str | None = None
    reason: FollowupReason
    status: FollowupStatus = FollowupStatus.PENDING
    schedule: CallbackSchedule | None = None
    attempts: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=3, ge=1, le=10)
    created_at: datetime | None = None

    @property
    def is_exhausted(self) -> bool:
        """Whether follow-up attempts have hit the ceiling.

        Returns:
            bool: True when attempts >= max_attempts.
        """
        return self.attempts >= self.max_attempts