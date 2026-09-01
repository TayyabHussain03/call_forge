"""Conversation memory persistence repository.

Cross-call continuity ka data access. Memory PERSON-level hai (contact_id), aur
structured fields update hote hain — poora blob overwrite nahi. Ek person ka ek
memory row (get-or-create pattern).
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import utcnow
from app.models.conversation_memory import ConversationMemory
from app.repositories.base_repository import BaseRepository


class MemoryRepository(BaseRepository[ConversationMemory]):
    """Persistence operations for per-person conversation memory.

    Attributes:
        model: ConversationMemory ORM model.
    """

    model = ConversationMemory

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with an active async session.

        Args:
            session: The async session to run queries against.
        """
        super().__init__(session)

    async def get_for_contact(self, contact_id: str) -> ConversationMemory | None:
        """Fetch the memory row for a specific person.

        Args:
            contact_id: The person's contact id.

        Returns:
            ConversationMemory | None: The memory, or None if none yet.
        """
        result = await self.session.execute(
            select(ConversationMemory).where(
                ConversationMemory.contact_id == contact_id
            )
        )
        return result.scalar_one_or_none()

    async def get_or_create(
        self, business_id: str, contact_id: str
    ) -> ConversationMemory:
        """Return the person's memory, creating an empty one if absent.

        Get-or-create pattern — har person ka exactly ek memory row rehta hai
        (DB constraint bhi isi ko enforce karta hai).

        Args:
            business_id: The parent business id.
            contact_id: The person's contact id.

        Returns:
            ConversationMemory: Existing or newly-staged memory.
        """
        existing = await self.get_for_contact(contact_id)
        if existing is not None:
            return existing
        mem = ConversationMemory(business_id=business_id, contact_id=contact_id)
        self.session.add(mem)
        return mem

    async def record_interaction(
        self,
        memory: ConversationMemory,
        *,
        last_topic: str | None = None,
        interest: str | None = None,
        callback_reason: str | None = None,
        website_discussed: bool | None = None,
        pricing_discussed: bool | None = None,
        objection: str | None = None,
        last_promise: str | None = None,
        last_call_id: str | None = None,
    ) -> ConversationMemory:
        """Update structured memory fields after an interaction.

        Sirf provided fields update hote hain (None = "is field ko chhoro"),
        taake ek call sirf woh cheezein badle jo actually pata chali. `previous_
        conversation_exists` aur `last_interacted_at` yahan set ho jaate hain.

        Args:
            memory: The memory row to update.
            last_topic: New last topic, if changed.
            interest: New interest level string, if changed.
            callback_reason: Callback reason, if any.
            website_discussed: Set True if website was explained this call.
            pricing_discussed: Set True if pricing was explained this call.
            objection: Latest objection text, if any.
            last_promise: What the agent promised, if any.
            last_call_id: The call this update came from.

        Returns:
            ConversationMemory: The updated memory row.
        """
        if last_topic is not None:
            memory.last_topic = last_topic
        if interest is not None:
            memory.interest = interest
        if callback_reason is not None:
            memory.callback_reason = callback_reason
        if website_discussed is not None:
            memory.website_discussed = website_discussed
        if pricing_discussed is not None:
            memory.pricing_discussed = pricing_discussed
        if objection is not None:
            memory.objection = objection
        if last_promise is not None:
            memory.last_promise = last_promise
        if last_call_id is not None:
            memory.last_call_id = last_call_id
        memory.previous_conversation_exists = True
        memory.last_interacted_at = utcnow()
        return memory