"""Call and transcript persistence repository.

Call records append-oriented hain (overwrite nahi). Transcript turns ordered
append hote hain. Ye repo call lifecycle aur transcript growth handle karta hai.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.call import Call
from app.models.transcript import TranscriptTurn
from app.repositories.base_repository import BaseRepository


class CallRepository(BaseRepository[Call]):
    """Persistence operations for calls and their transcript turns.

    Attributes:
        model: Call ORM model.
    """

    model = Call

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with an active async session.

        Args:
            session: The async session to run queries against.
        """
        super().__init__(session)

    async def create_call(
        self, business_id: str, contact_id: str | None, attempt_number: int
    ) -> Call:
        """Create and stage a new call record.

        Args:
            business_id: The business being called.
            contact_id: The person, if known (nullable).
            attempt_number: Which attempt this is for the business.

        Returns:
            Call: The staged call.
        """
        call = Call(
            business_id=business_id,
            contact_id=contact_id,
            attempt_number=attempt_number,
        )
        self.session.add(call)
        return call

    async def list_for_business(self, business_id: str) -> list[Call]:
        """Return all calls for a business, newest first.

        Args:
            business_id: The business id.

        Returns:
            list[Call]: Calls, ordered by creation descending.
        """
        result = await self.session.execute(
            select(Call)
            .where(Call.business_id == business_id)
            .order_by(Call.created_at.desc())
        )
        return list(result.scalars().all())

    async def next_attempt_number(self, business_id: str) -> int:
        """Compute the next attempt number for a business.

        Existing calls count karke +1 deta hai, taake attempt numbering
        append-oriented aur gap-free rahe.

        Args:
            business_id: The business id.

        Returns:
            int: The next attempt number (1 if no prior calls).
        """
        result = await self.session.execute(
            select(func.count()).select_from(Call).where(Call.business_id == business_id)
        )
        count = result.scalar_one()
        return int(count) + 1

    async def append_turn(
        self,
        call_id: str,
        turn_index: int,
        speaker: str,
        text: str,
        stt_confidence: float | None = None,
    ) -> TranscriptTurn:
        """Append one transcript turn to a call.

        Args:
            call_id: Parent call id.
            turn_index: 0-based order within the call.
            speaker: "agent" or "client".
            text: Spoken text (client text is untrusted).
            stt_confidence: STT confidence for client turns, if available.

        Returns:
            TranscriptTurn: The staged turn.
        """
        turn = TranscriptTurn(
            call_id=call_id,
            turn_index=turn_index,
            speaker=speaker,
            text=text,
            stt_confidence=stt_confidence,
        )
        self.session.add(turn)
        return turn