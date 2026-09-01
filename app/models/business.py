"""Business ORM model — the company being contacted.

Ye persistence-level representation hai us business ka jise call kiya jaayega.
Data mostly scraped (Google Maps) hai — isliye UNVERIFIED by default, aur source
metadata explicitly track hoti hai.

KEY GUARANTEES (guide ke mutabiq):
    - Deduplication DB level par: `dedup_key` par UNIQUE constraint. Sirf
      Python-level `if exists()` nahi — database final protection deta hai
      (race-condition safe).
    - Business identity aur lead lifecycle ALAG. Ye model identity + source +
      dedup rakhta hai; lifecycle status alag `LeadState` model mein jaayega.
    - Ek Business ke MULTIPLE Contacts ho sakte hain (John, Mike) — relationship
      contact model define karta hai; business unhe reference karta hai.

DESIGN: Ye ORM model Pydantic `BusinessIdentity`/`LeadIngestion` contracts ki
    exact copy NAHI hai. Conversion repository layer mein explicit hota hai.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.constants import LeadStatus
from app.models.base import ModelBase

if TYPE_CHECKING:
    from app.models.contact import Contact


class Business(ModelBase):
    """A business/company record targeted for outbound calling.

    Attributes:
        business_name: Display name (scraped, untrusted).
        phone_raw: Original phone string as scraped. UNTRUSTED, audit ke liye.
        phone_e164: Canonical E.164 form (international). Dedup/lookup ka basis.
            None agar number parse na ho.
        default_region: ISO country hint (US/PK/GB/AE) national-format numbers
            ke liye.
        address: Full address, agar available.
        city: Sheher.
        state: State/region.
        country: Country name/code, agar available.
        industry: Business category (e.g. "roofing").
        website_url: Website agar mila.
        website_status: Website ki state (e.g. "none", "active") — offer targeting
            ke liye useful.
        source: Lead kahan se aaya (e.g. "google_maps").
        source_external_id: Source ka apna id (e.g. Maps place id).
        verification: Verification status string (default "unverified").
        dedup_key: Deterministic dedup key. UNIQUE at DB level.
        contacts: Is business se jude saare log (John, Mike, ...).
    """

    __tablename__ = "businesses"
    __table_args__ = (
        # DB-level duplicate protection — final authority, Python check nahi.
        UniqueConstraint("dedup_key", name="uq_businesses_dedup_key"),
        # Real query patterns ke liye indexes.
        Index("ix_businesses_phone_e164", "phone_e164"),
        Index("ix_businesses_source_external_id", "source", "source_external_id"),
    )

    business_name: Mapped[str] = mapped_column(String(300), nullable=False)
    phone_raw: Mapped[str] = mapped_column(String(40), nullable=False)
    phone_e164: Mapped[str | None] = mapped_column(String(20), nullable=True)
    default_region: Mapped[str | None] = mapped_column(String(2), nullable=True)

    address: Mapped[str | None] = mapped_column(String(500), nullable=True)
    city: Mapped[str | None] = mapped_column(String(120), nullable=True)
    state: Mapped[str | None] = mapped_column(String(120), nullable=True)
    country: Mapped[str | None] = mapped_column(String(120), nullable=True)
    industry: Mapped[str | None] = mapped_column(String(120), nullable=True)

    website_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    website_status: Mapped[str | None] = mapped_column(String(40), nullable=True)

    source: Mapped[str] = mapped_column(String(40), nullable=False, default="google_maps")
    source_external_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    verification: Mapped[str] = mapped_column(
        String(30), nullable=False, default="unverified"
    )

    dedup_key: Mapped[str] = mapped_column(String(340), nullable=False)

    # One business → many contacts. Delete par contacts bhi jaayein (business hi
    # nahi raha to uske log ka koi matlab nahi) — lekin business delete khud
    # rare/careful operation hai (call history preserve karni hai).
    contacts: Mapped[list[Contact]] = relationship(
        "Contact",
        back_populates="business",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        """Return a concise, PII-light representation.

        Poora phone/address deliberately nahi dikhate — logs mein PII leak na
        ho. Sirf id aur naam.

        Returns:
            str: Debug-safe representation.
        """
        return f"<Business id={self.id!r} name={self.business_name!r}>"

    @property
    def default_lead_status(self) -> LeadStatus:
        """Convenience default lifecycle status for a fresh business.

        Actual lifecycle alag LeadState model mein track hoga; ye sirf ingestion
        ke waqt ka sane default deta hai.

        Returns:
            LeadStatus: NEW.
        """
        return LeadStatus.NEW