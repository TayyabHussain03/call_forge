"""Contact persistence repository.

Ek business ke multiple contacts (John, Mike) handle karta hai, aur contact info
updates ko PROVENANCE-AWARE rakhta hai — confirmed value ko baad ki untrusted
extraction se blindly overwrite nahi karta.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import utcnow
from app.models.contact import Contact, ContactInfoStatus
from app.repositories.base_repository import BaseRepository


class ContactRepository(BaseRepository[Contact]):
    """Persistence operations for contacts (people at a business).

    Attributes:
        model: Contact ORM model.
    """

    model = Contact

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with an active async session.

        Args:
            session: The async session to run queries against.
        """
        super().__init__(session)

    async def list_for_business(self, business_id: str) -> list[Contact]:
        """Return all contacts belonging to a business.

        Args:
            business_id: The parent business id.

        Returns:
            list[Contact]: Contacts for that business (may be empty).
        """
        result = await self.session.execute(
            select(Contact).where(Contact.business_id == business_id)
        )
        return list(result.scalars().all())

    async def find_by_name(self, business_id: str, name: str) -> Contact | None:
        """Find a contact at a business by name (case-insensitive).

        Cross-call continuity ke liye: agli call par "John" ko pehchan-ne mein
        madad karta hai.

        Args:
            business_id: The parent business id.
            name: The person's name to match.

        Returns:
            Contact | None: Matching contact, or None.
        """
        result = await self.session.execute(
            select(Contact).where(
                Contact.business_id == business_id,
                Contact.name.ilike(name.strip()),
            )
        )
        return result.scalars().first()

    async def create(
        self, business_id: str, name: str | None = None, role: str = "unknown"
    ) -> Contact:
        """Create and stage a new contact for a business.

        Args:
            business_id: Parent business id.
            name: Person's name, if known.
            role: Role string (default "unknown").

        Returns:
            Contact: The staged (unpersisted-until-commit) contact.
        """
        contact = Contact(business_id=business_id, name=name, role=role)
        self.session.add(contact)
        return contact

    async def update_email(
        self,
        contact: Contact,
        *,
        raw: str,
        normalized: str,
        status: ContactInfoStatus,
        provenance: str,
    ) -> Contact:
        """Update a contact's email, respecting confirmation precedence.

        RULE (guide ke mutabiq): ek pehle se CONFIRMED email ko sirf tab
        overwrite karo jab naya value BHI confirmed ho. Warna confirmed data ko
        untrusted later extraction se protect karte hain.

        Args:
            contact: The contact to update.
            raw: Raw email value.
            normalized: Lowercased canonical email.
            status: New confirmation status for this value.
            provenance: Where this value came from.

        Returns:
            Contact: The (possibly unchanged) contact.
        """
        already_confirmed = contact.email_status == ContactInfoStatus.CONFIRMED.value
        incoming_confirmed = status == ContactInfoStatus.CONFIRMED
        if already_confirmed and not incoming_confirmed:
            return contact  # protect confirmed value

        contact.email_raw = raw
        contact.email_normalized = normalized
        contact.email_status = status.value
        contact.email_provenance = provenance
        contact.last_seen_at = utcnow()
        return contact