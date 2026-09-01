"""LeadState ORM model — lifecycle status of a business as a lead.

Guide ka core requirement: lead lifecycle (NEW → QUEUED → CONTACTED → ... →
QUALIFIED) business IDENTITY se ALAG hai. Business kaun hai (naam, phone) stable
rehta hai; uska lead status badalta rehta hai. Isliye alag model.

Ye business-LEVEL status hai. Person-level (John interested, Mike unknown) alag
hai — woh Contact aur ConversationMemory par rehta hai. Dono ko mila dena galat
hoga.

One-to-one: har business ka ek current lead state.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.constants import LeadStatus
from app.models.base import ModelBase


class LeadState(ModelBase):
    """Current lifecycle status of a business-as-a-lead.

    Attributes:
        business_id: Kis business ka state (FK, unique — one-to-one).
        status: Current lead lifecycle status.
        score: 0–100 heuristic lead score (analyzer update karta hai).
        call_attempts: Ab tak kitni call koshishein hui.
        last_call_at: Aakhri call ka UTC waqt.
        last_status_change_at: Status aakhri baar kab badla.
        business: Parent business relationship.
    """

    __tablename__ = "lead_states"
    __table_args__ = (
        # One-to-one: ek business ka ek hi lead state.
        UniqueConstraint("business_id", name="uq_lead_states_business_id"),
        Index("ix_lead_states_status", "status"),
    )

    business_id: Mapped[str] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(30), nullable=False, default=LeadStatus.NEW.value
    )
    score: Mapped[int] = mapped_column(nullable=False, default=0)
    call_attempts: Mapped[int] = mapped_column(nullable=False, default=0)
    last_call_at: Mapped[datetime | None] = mapped_column(nullable=True)
    last_status_change_at: Mapped[datetime | None] = mapped_column(nullable=True)

    business: Mapped["Business"] = relationship("Business")  # noqa: F821

    def __repr__(self) -> str:
        """Return a concise representation.

        Returns:
            str: Debug-safe representation with business, status, score.
        """
        return (
            f"<LeadState business_id={self.business_id!r} "
            f"status={self.status!r} score={self.score}>"
        )