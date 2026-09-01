"""Followup ORM model — scheduled callbacks and follow-up tasks.

Guide ke mutabiq: ek follow-up Business + Contact + previous Call ko reference
kar sakta hai, taake agent conversation ko naturally resume kar sake. Reason
tone-independent hai (busy/harsh client bhi valid high-interest follow-up).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import ModelBase


class Followup(ModelBase):
    """A follow-up/callback task for a lead.

    Attributes:
        business_id: Kis business ke liye (FK).
        contact_id: Kis person se follow-up (FK, nullable).
        previous_call_id: Kis call ke baad ye follow-up bana (FK, nullable) —
            resume ke liye context.
        reason: Follow-up kyun (e.g. "busy", "callback_requested").
        status: Task lifecycle (pending/scheduled/completed/cancelled).
        earliest_at: Is waqt se pehle call na karo (UTC).
        latest_at: Is waqt tak call karo (UTC).
        timezone: Client ka timezone (IANA), agar pata ho.
        note: Client ka time-related note (e.g. "tomorrow afternoon").
        attempts: Ab tak kitni follow-up koshishein hui.
        max_attempts: Max koshishein.
        business: Parent business relationship.
        contact: Associated person relationship (nullable).
    """

    __tablename__ = "followups"
    __table_args__ = (
        Index("ix_followups_business_id", "business_id"),
        Index("ix_followups_status", "status"),
        Index("ix_followups_earliest_at", "earliest_at"),
    )

    business_id: Mapped[str] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    contact_id: Mapped[str | None] = mapped_column(
        ForeignKey("contacts.id", ondelete="SET NULL"), nullable=True
    )
    previous_call_id: Mapped[str | None] = mapped_column(
        ForeignKey("calls.id", ondelete="SET NULL"), nullable=True
    )

    reason: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")

    earliest_at: Mapped[datetime | None] = mapped_column(nullable=True)
    latest_at: Mapped[datetime | None] = mapped_column(nullable=True)
    timezone: Mapped[str | None] = mapped_column(String(60), nullable=True)
    note: Mapped[str | None] = mapped_column(String(300), nullable=True)

    attempts: Mapped[int] = mapped_column(nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(nullable=False, default=3)

    business: Mapped["Business"] = relationship("Business")  # noqa: F821
    contact: Mapped["Contact | None"] = relationship("Contact")  # noqa: F821

    def __repr__(self) -> str:
        """Return a concise representation.

        Returns:
            str: Debug-safe representation with id, business, reason, status.
        """
        return (
            f"<Followup id={self.id!r} business_id={self.business_id!r} "
            f"reason={self.reason!r} status={self.status!r}>"
        )