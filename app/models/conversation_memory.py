"""ConversationMemory ORM model — structured cross-call continuity.

Ye system ka continuity ka dil hai. Guide ka strict requirement: memory ek giant
JSON blob NAHI hai — structured fields hain, taake query/update/reason kiya ja
sake bina poora transcript LLM ko dobara diye.

SCENARIO ye enable karta hai:
    Call 1 — John/Owner: "interested, but busy, call me tomorrow"
        → yahan persist: last_topic=website_preview, interest=high,
          callback_reason=busy, previous_conversation=true, person=John
    Call 2 — John: agent bina zero se shuru kiye continue karta hai
    Call 3 — Mike/Manager: alag person, John ki memory intact rehti hai

DESIGN: memory PERSON-level hai (contact_id se bandhi), business-level nahi.
    Isliye John ka interest Mike se alag rehta hai. Ek business ke multiple
    contacts ki apni-apni memory ho sakti hai.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.constants import Interest
from app.models.base import ModelBase


class ConversationMemory(ModelBase):
    """Structured memory of prior conversations with one person.

    Har field ek discrete, queryable fact hai — deliberately no free-form blob.
    Agli call se pehle context_builder inhi fields se ek short LLM context banata
    hai.

    Attributes:
        business_id: Kis business ke andar (FK).
        contact_id: Kis person ki memory (FK, unique — one memory per person).
        last_topic: Aakhri conversation ka main topic (e.g. "website_preview").
        interest: Is person ka last-known interest level.
        callback_reason: Agar callback chahiye tha to kyun (e.g. "busy").
        website_discussed: Kya website offer explain ho chuka.
        pricing_discussed: Kya pricing explain ho chuki.
        objection: Aakhri objection agar koi tha (short text).
        last_promise: Agent ne kya promise kiya (e.g. "send preview to email").
        previous_conversation_exists: Kya isse pehle koi baat hui.
        last_call_id: Aakhri call ka id (traceability).
        last_interacted_at: Is person se aakhri interaction ka UTC waqt.
        business: Parent business relationship.
        contact: The person this memory belongs to.
    """

    __tablename__ = "conversation_memories"
    __table_args__ = (
        # One memory row per person.
        UniqueConstraint("contact_id", name="uq_conversation_memories_contact_id"),
        Index("ix_conversation_memories_business_id", "business_id"),
    )

    business_id: Mapped[str] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    contact_id: Mapped[str] = mapped_column(
        ForeignKey("contacts.id", ondelete="CASCADE"), nullable=False
    )

    last_topic: Mapped[str | None] = mapped_column(String(120), nullable=True)
    interest: Mapped[str] = mapped_column(
        String(20), nullable=False, default=Interest.NONE.value
    )
    callback_reason: Mapped[str | None] = mapped_column(String(120), nullable=True)
    website_discussed: Mapped[bool] = mapped_column(nullable=False, default=False)
    pricing_discussed: Mapped[bool] = mapped_column(nullable=False, default=False)
    objection: Mapped[str | None] = mapped_column(String(500), nullable=True)
    last_promise: Mapped[str | None] = mapped_column(String(500), nullable=True)
    previous_conversation_exists: Mapped[bool] = mapped_column(
        nullable=False, default=False
    )
    last_call_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    last_interacted_at: Mapped[datetime | None] = mapped_column(nullable=True)

    business: Mapped["Business"] = relationship("Business")  # noqa: F821
    contact: Mapped["Contact"] = relationship("Contact")  # noqa: F821

    def __repr__(self) -> str:
        """Return a concise representation.

        Returns:
            str: Debug-safe representation with contact, interest, last topic.
        """
        return (
            f"<ConversationMemory contact_id={self.contact_id!r} "
            f"interest={self.interest!r} last_topic={self.last_topic!r}>"
        )