"""Business/lead persistence repository.

Business ingestion ko idempotent banata hai: same lead dobara aaye to nayi row
nahi banti. Protection ka final layer DATABASE ka unique constraint hai — ye
repo uska sahi istemaal karta hai (check-then-insert race se bachte hue).

CONVERSION: Ye repo Pydantic contracts (BusinessIdentity/LeadIngestion) aur ORM
    Business ke beech explicit conversion karta hai — dono ek dusre ki copy
    nahi.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.contracts.lead import LeadIngestion
from app.models.business import Business
from app.repositories.base_repository import BaseRepository


class LeadRepository(BaseRepository[Business]):
    """Persistence operations for businesses/leads.

    Attributes:
        model: Business ORM model.
    """

    model = Business

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with an active async session.

        Args:
            session: The async session to run queries against.
        """
        super().__init__(session)

    async def get_by_dedup_key(self, dedup_key: str) -> Business | None:
        """Fetch a business by its deduplication key.

        Args:
            dedup_key: The deterministic dedup key.

        Returns:
            Business | None: Matching business, or None.
        """
        result = await self.session.execute(
            select(Business).where(Business.dedup_key == dedup_key)
        )
        return result.scalar_one_or_none()

    async def get_by_phone_e164(self, phone_e164: str) -> Business | None:
        """Fetch a business by canonical phone number.

        Args:
            phone_e164: E.164 phone string.

        Returns:
            Business | None: Matching business, or None.
        """
        result = await self.session.execute(
            select(Business).where(Business.phone_e164 == phone_e164)
        )
        return result.scalar_one_or_none()

    def _to_model(self, ingestion: LeadIngestion) -> Business:
        """Convert a validated LeadIngestion contract into a Business ORM row.

        Ye explicit conversion hai — contract ke fields ko model columns par map
        karta hai. Yahin computed `phone_e164`/`dedup_key` contract se liye jaate
        hain (contract ne unhe deterministically compute kiya).

        Args:
            ingestion: The validated ingestion contract.

        Returns:
            Business: A new (unpersisted) Business instance.
        """
        b = ingestion.business
        return Business(
            business_name=b.name,
            phone_raw=b.phone,
            phone_e164=b.phone_e164,
            default_region=b.default_region,
            address=b.address,
            city=b.city,
            state=b.state,
            industry=b.industry,
            source=ingestion.source.value,
            source_external_id=ingestion.external_id,
            verification=ingestion.verification.value,
            dedup_key=ingestion.dedup_key,
        )

    async def upsert_from_ingestion(self, ingestion: LeadIngestion) -> tuple[Business, bool]:
        """Idempotently persist a lead: insert if new, else return existing.

        Strategy (guide ke mutabiq):
            1. Pehle dedup_key se existing business dhoondo.
            2. Mil gaya → wahi return karo (koi duplicate nahi).
            3. Nahi mila → insert karke flush karo. Agar is beech koi aur same
               key insert kar chuka ho, DB ka UNIQUE constraint IntegrityError
               dega — us case mein rollback karke existing fetch kar lo.

        Ye check-then-insert ki race window ko DB constraint se cover karta hai;
        sirf Python check par depend nahi karta.

        Args:
            ingestion: The validated ingestion contract.

        Returns:
            tuple[Business, bool]: (business, created) — created True jab nayi row
            bani, False jab existing mili.
        """
        from sqlalchemy.exc import IntegrityError

        existing = await self.get_by_dedup_key(ingestion.dedup_key)
        if existing is not None:
            return existing, False

        entity = self._to_model(ingestion)
        self.session.add(entity)
        try:
            await self.session.flush()
        except IntegrityError:
            # Race: kisi aur ne same key insert kar diya — rollback + fetch.
            await self.session.rollback()
            existing = await self.get_by_dedup_key(ingestion.dedup_key)
            if existing is None:
                raise  # asli unexpected integrity error — chhupao mat
            return existing, False
        return entity, True