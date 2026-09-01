"""Contact ORM model — a person associated with a business.

Ek business ke MULTIPLE contacts ho sakte hain (John/Owner, Mike/Manager). Ye
model person-level identity aur unke contact info ko business se ALAG rakhta
hai — jo cross-call continuity ka foundation hai.

KEY GUARANTEES (guide ke mutabiq):
    - Person identity ≠ Business identity. John ka interest Mike se independent.
    - Contact info ka provenance track hota hai (lead_import vs client_spoken)
      aur confirmation status — taake confirmed value ko baad ki untrusted
      extraction se blindly overwrite na kiya jaaye.
    - detected → validated → confirmed lifecycle contact info par bhi lagti hai.

PII NOTE: name/email/phone sensitive hain — __repr__ inhe expose nahi karta.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import ModelBase, utcnow

if TYPE_CHECKING:
    from app.models.business import Business


class ContactRole(str, Enum):
    """A contact's role relative to the business decision."""

    UNKNOWN = "unknown"
    OWNER = "owner"
    MANAGER = "manager"
    EMPLOYEE = "employee"
    DECISION_MAKER = "decision_maker"
    NON_DECISION_MAKER = "non_decision_maker"


class ContactProvenance(str, Enum):
    """Where a piece of contact info originated.

    Confirmed client-spoken data ko lead-import ya inferred data se distinguish
    karna zaroori hai — taake update logic sahi precedence rakhe.
    """

    LEAD_IMPORT = "lead_import"      # scraped/imported ke saath aaya
    CLIENT_SPOKEN = "client_spoken"  # call mein client ne khud bataya
    MANUAL = "manual"               # human ne daala
    INFERRED = "inferred"           # system ne guess kiya


class ContactInfoStatus(str, Enum):
    """Confirmation lifecycle of a contact's email/phone."""

    NONE = "none"
    DETECTED = "detected"
    VALIDATED = "validated"
    CONFIRMED = "confirmed"


class Contact(ModelBase):
    """A person associated with a business.

    Attributes:
        business_id: Parent business (FK).
        name: Person ka naam, agar pata ho.
        role: Business mein role (owner/manager/...).
        email_raw: Raw email jaise mila. UNTRUSTED.
        email_normalized: Lowercased canonical email.
        email_status: Email ka confirmation lifecycle stage.
        email_provenance: Email kahan se aaya.
        phone_raw: Raw phone jaise mila.
        phone_e164: Canonical E.164 phone (international).
        phone_status: Phone ka confirmation lifecycle stage.
        phone_provenance: Phone kahan se aaya.
        is_decision_maker: Kya ye person decide kar sakta hai. None = unknown.
        first_seen_at: Pehli baar kab is person se contact hua.
        last_seen_at: Aakhri baar kab.
        business: Parent business relationship.
    """

    __tablename__ = "contacts"
    __table_args__ = (
        Index("ix_contacts_business_id", "business_id"),
        Index("ix_contacts_email_normalized", "email_normalized"),
        Index("ix_contacts_phone_e164", "phone_e164"),
    )

    business_id: Mapped[str] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )

    name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    role: Mapped[str] = mapped_column(
        String(30), nullable=False, default=ContactRole.UNKNOWN.value
    )

    email_raw: Mapped[str | None] = mapped_column(String(320), nullable=True)
    email_normalized: Mapped[str | None] = mapped_column(String(320), nullable=True)
    email_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=ContactInfoStatus.NONE.value
    )
    email_provenance: Mapped[str | None] = mapped_column(String(20), nullable=True)

    phone_raw: Mapped[str | None] = mapped_column(String(40), nullable=True)
    phone_e164: Mapped[str | None] = mapped_column(String(20), nullable=True)
    phone_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=ContactInfoStatus.NONE.value
    )
    phone_provenance: Mapped[str | None] = mapped_column(String(20), nullable=True)

    is_decision_maker: Mapped[bool | None] = mapped_column(nullable=True)

    first_seen_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)

    business: Mapped[Business] = relationship("Business", back_populates="contacts")

    def __repr__(self) -> str:
        """Return a PII-light representation (no name/email/phone).

        Returns:
            str: Debug-safe representation with id, business, role only.
        """
        return (
            f"<Contact id={self.id!r} business_id={self.business_id!r} "
            f"role={self.role!r}>"
        )

    @property
    def has_confirmed_email(self) -> bool:
        """Whether this contact has a client-confirmed email.

        Returns:
            bool: True when email_status is CONFIRMED.
        """
        return self.email_status == ContactInfoStatus.CONFIRMED.value

    @property
    def has_confirmed_phone(self) -> bool:
        """Whether this contact has a client-confirmed phone.

        Returns:
            bool: True when phone_status is CONFIRMED.
        """
        return self.phone_status == ContactInfoStatus.CONFIRMED.value