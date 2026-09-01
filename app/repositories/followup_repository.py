"""Follow-up task persistence repository.

Callback/follow-up tasks banata aur query karta hai. Due tasks (jinka waqt aa
gaya) fetch karne ka method deta hai, jo baad mein worker use karega.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import utcnow
from app.models.followup import Followup
from app.repositories.base_repository import BaseRepository


class FollowupRepository(BaseRepository[Followup]):
    """Persistence operations for follow-up tasks.

    Attributes:
        model: Followup ORM model.
    """

    model = Followup

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with an active async session.

        Args:
            session: The async session to run queries against.
        """
        super().__init__(session)

    async def create_followup(
        self,
        business_id: str,
        reason: str,
        *,
        contact_id: str | None = None,
        previous_call_id: str | None = None,
        earliest_at: datetime | None = None,
        latest_at: datetime | None = None,
        timezone: str | None = None,
        note: str | None = None,
    ) -> Followup:
        """Create and stage a follow-up task.

        Args:
            business_id: The business needing follow-up.
            reason: Why (e.g. "busy", "callback_requested").
            contact_id: The person, if known.
            previous_call_id: The call that triggered this follow-up.
            earliest_at: Don't call before this UTC time.
            latest_at: Call by this UTC time.
            timezone: Client timezone (IANA), if known.
            note: Client's time note (e.g. "tomorrow afternoon").

        Returns:
            Followup: The staged follow-up.
        """
        fu = Followup(
            business_id=business_id,
            reason=reason,
            contact_id=contact_id,
            previous_call_id=previous_call_id,
            earliest_at=earliest_at,
            latest_at=latest_at,
            timezone=timezone,
            note=note,
        )
        self.session.add(fu)
        return fu

    async def list_due(self, now: datetime | None = None, limit: int = 50) -> list[Followup]:
        """Return pending follow-ups whose time window has arrived.

        Due = status pending AND (koi earliest_at nahi, ya earliest_at guzar
        chuka). Worker isse queue banata hai.

        Args:
            now: Reference time (defaults to current UTC).
            limit: Max tasks to return.

        Returns:
            list[Followup]: Due follow-up tasks.
        """
        ref = now or utcnow()
        result = await self.session.execute(
            select(Followup)
            .where(
                Followup.status == "pending",
                or_(Followup.earliest_at.is_(None), Followup.earliest_at <= ref),
            )
            .order_by(Followup.earliest_at.is_(None).desc(), Followup.earliest_at)
            .limit(limit)
        )
        return list(result.scalars().all())