"""Call ORM model — one record per outbound call attempt.

APPEND-ORIENTED (guide ke mutabiq): har call apna alag record hai. Purani calls
overwrite NAHI hoti — Business → Call #1, #2, #3 sab preserve rehte hain. Sirf
processing fields (status, ended_at, outcome) update hote hain jab tak call
zinda hai; complete hone ke baad woh historical record ban jaati hai.

RELATIONSHIPS: ek Call optionally ek Contact se judi hoti hai (kis person se
baat hui), aur ek Business se. Transcript turns Call se one-to-many hain.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.constants import CallStatus
from app.models.base import ModelBase

if TYPE_CHECKING:
    from app.models.transcript import TranscriptTurn


class Call(ModelBase):
    """A single outbound call attempt and its outcome.

    Attributes:
        business_id: Kis business ko call kiya (FK).
        contact_id: Kis person se baat hui, agar identify hua (FK, nullable).
        status: Call lifecycle status (queued → dialing → ... → completed).
        provider_call_id: Voice platform (Vapi) ka apna call id, agar mila.
        attempt_number: Is business ke liye kaunsi koshish (1-based).
        started_at: Call connect hone ka UTC waqt.
        ended_at: Call khatam hone ka UTC waqt.
        outcome_summary: Short post-call summary (analyzer bharega).
        business: Parent business relationship.
        contact: Associated person relationship (nullable).
        turns: Is call ke transcript turns (one-to-many).
    """

    __tablename__ = "calls"
    __table_args__ = (
        Index("ix_calls_business_id", "business_id"),
        Index("ix_calls_contact_id", "contact_id"),
        Index("ix_calls_status", "status"),
    )

    business_id: Mapped[str] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    # Contact delete par call ko mat udaao — call history preserve rehni chahiye.
    # Isliye SET NULL: person record chala jaye to bhi call ka record bacha rahe.
    contact_id: Mapped[str | None] = mapped_column(
        ForeignKey("contacts.id", ondelete="SET NULL"), nullable=True
    )

    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=CallStatus.QUEUED.value
    )
    provider_call_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    attempt_number: Mapped[int] = mapped_column(nullable=False, default=1)

    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(nullable=True)
    outcome_summary: Mapped[str | None] = mapped_column(String(2000), nullable=True)

    business: Mapped["Business"] = relationship("Business")  # noqa: F821
    contact: Mapped["Contact | None"] = relationship("Contact")  # noqa: F821
    turns: Mapped[list[TranscriptTurn]] = relationship(
        "TranscriptTurn",
        back_populates="call",
        cascade="all, delete-orphan",
        order_by="TranscriptTurn.turn_index",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        """Return a PII-light representation.

        Returns:
            str: Debug-safe representation with id, business, status.
        """
        return (
            f"<Call id={self.id!r} business_id={self.business_id!r} "
            f"status={self.status!r}>"
        )