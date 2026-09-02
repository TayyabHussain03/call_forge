"""ContactMethod persistence repository.

Canonical communication methods ka data access. Responsibilities SIRF
persistence/constraints:
    - upsert with identity dedup (contact_id, channel, value_normalized)
    - explicit provenance precedence (centralized policy consume karta hai)
    - preferred-per-channel (flag-only, history preserve)
    - status monotonicity (confirmed downgrade nahi)
    - null/empty normalized value reject (identity boundary)

NON-responsibilities (guide ke mutabiq): reference resolution NAHI, contact
parsing/normalization NAHI, LLM/network NAHI. Normalized value resolver/
normalization layer deta hai; repo use persist karta hai.

Business.phone_e164 (immutable source) ko ye repo KABHI write nahi karta.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ValidationError
from app.models.base import utcnow
from app.models.contact import ContactInfoStatus
from app.models.contact_method import (
    ContactMethod,
    can_transition_status,
    provenance_rank,
)
from app.repositories.base_repository import BaseRepository


class ContactMethodRepository(BaseRepository[ContactMethod]):
    """Persistence operations for canonical contact methods.

    Attributes:
        model: ContactMethod ORM model.
    """

    model = ContactMethod

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with an active async session.

        Args:
            session: The async session to run queries against.
        """
        super().__init__(session)

    @staticmethod
    def _require_value(value_normalized: str | None) -> str:
        """Validate that a normalized value can form an identity, or reject.

        Identity boundary (guide safeguard 4): null/empty/whitespace-only value
        se koi ContactMethod identity nahi ban sakti. Unresolved references ko
        yahan tak aana hi nahi chahiye — resolver pehle resolve karta hai.

        Args:
            value_normalized: The canonical value provided by the resolver.

        Returns:
            str: The validated (stripped) value.

        Raises:
            ValidationError: Agar value None/empty/whitespace ho.
        """
        if value_normalized is None:
            raise ValidationError(
                "ContactMethod identity requires a resolved value (got None)."
            )
        stripped = value_normalized.strip()
        if not stripped:
            raise ValidationError(
                "ContactMethod identity requires a non-empty normalized value."
            )
        return stripped

    async def find_identity(
        self, contact_id: str, channel: str, value_normalized: str
    ) -> ContactMethod | None:
        """Find a contact method by its identity key.

        Args:
            contact_id: The person id.
            channel: The channel string.
            value_normalized: The canonical value.

        Returns:
            ContactMethod | None: Matching method, or None.
        """
        result = await self.session.execute(
            select(ContactMethod).where(
                ContactMethod.contact_id == contact_id,
                ContactMethod.channel == channel,
                ContactMethod.value_normalized == value_normalized,
            )
        )
        return result.scalar_one_or_none()

    async def list_methods(self, contact_id: str) -> list[ContactMethod]:
        """Return all contact methods for a person.

        Args:
            contact_id: The person id.

        Returns:
            list[ContactMethod]: All methods (may be empty).
        """
        result = await self.session.execute(
            select(ContactMethod).where(ContactMethod.contact_id == contact_id)
        )
        return list(result.scalars().all())

    async def list_for_channel(
        self, contact_id: str, channel: str
    ) -> list[ContactMethod]:
        """Return a person's methods for a specific channel.

        Args:
            contact_id: The person id.
            channel: The channel string.

        Returns:
            list[ContactMethod]: Methods for that channel.
        """
        result = await self.session.execute(
            select(ContactMethod).where(
                ContactMethod.contact_id == contact_id,
                ContactMethod.channel == channel,
            )
        )
        return list(result.scalars().all())

    async def upsert_method(
        self,
        contact_id: str,
        channel: str,
        value_normalized: str,
        provenance: str,
        *,
        value_raw: str | None = None,
        note: str | None = None,
    ) -> tuple[ContactMethod, bool]:
        """Create a new method or reuse an existing one by identity.

        Reuse par (identity match): last_observed_at update, aur provenance sirf
        tab upgrade jab naya provenance CENTRALIZED precedence mein higher ho —
        koi arbitrary "stronger" logic nahi. Status yahan touch nahi hota
        (update_status alag). Naya row DETECTED status se banta hai.

        Args:
            contact_id: The person id.
            channel: The channel string.
            value_normalized: Canonical resolved value (resolver provides).
            provenance: Where this observation came from.
            value_raw: Original raw value, if available.
            note: Optional structured note.

        Returns:
            tuple[ContactMethod, bool]: (method, created).

        Raises:
            ValidationError: Agar value null/empty ho.
        """
        value = self._require_value(value_normalized)
        existing = await self.find_identity(contact_id, channel, value)

        if existing is not None:
            existing.last_observed_at = utcnow()
            # Explicit precedence upgrade only.
            if provenance_rank(provenance) > provenance_rank(existing.provenance):
                existing.provenance = provenance
            if note is not None:
                existing.note = note
            return existing, False

        method = ContactMethod(
            contact_id=contact_id,
            channel=channel,
            value_normalized=value,
            value_raw=value_raw,
            provenance=provenance,
            status=ContactInfoStatus.DETECTED.value,
            note=note,
        )
        self.session.add(method)
        return method, True

    async def update_status(
        self, method: ContactMethod, new_status: ContactInfoStatus
    ) -> ContactMethod:
        """Update a method's status under the monotonicity policy.

        Confirmed method ko weaker status se downgrade nahi hone deta (centralized
        `can_transition_status`). Downgrade attempt silently ignore hota hai
        (method unchanged) — confirmed data protect rehti hai.

        Args:
            method: The method to update.
            new_status: Proposed new status.

        Returns:
            ContactMethod: The (possibly unchanged) method.
        """
        if can_transition_status(method.status, new_status.value):
            method.status = new_status.value
        return method

    async def set_preferred(
        self, contact_id: str, channel: str, method_id: str
    ) -> None:
        """Mark one method as preferred for a channel; unset others.

        Invariant: per (contact_id, channel) at most ek is_preferred=True. Ye
        flag-only operation hai — koi row delete/overwrite NAHI, history intact.
        SQLite par application-level (transaction); Postgres par partial unique
        index isko concurrency-safe banayega (model note dekho).

        Args:
            contact_id: The person id.
            channel: The channel string.
            method_id: The method to mark preferred.
        """
        methods = await self.list_for_channel(contact_id, channel)
        for m in methods:
            m.is_preferred = m.id == method_id

    async def get_preferred(
        self, contact_id: str, channel: str
    ) -> ContactMethod | None:
        """Return the preferred method for a channel, if any.

        Args:
            contact_id: The person id.
            channel: The channel string.

        Returns:
            ContactMethod | None: The preferred method, or None.
        """
        result = await self.session.execute(
            select(ContactMethod).where(
                ContactMethod.contact_id == contact_id,
                ContactMethod.channel == channel,
                ContactMethod.is_preferred.is_(True),
            )
        )
        return result.scalars().first()