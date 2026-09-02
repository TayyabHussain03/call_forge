"""Canonical communication methods discovered during the product lifecycle.

Ye model source data (Business.phone_e164, immutable) aur legacy Contact fields
(backward-compat) se ALAG hai. Har client-provided/referenced contact method
yahan aata hai — original lead phone kabhi overwrite nahi hota.

CORE RULES (locked design):
    - Identity = UNIQUE(contact_id, channel, value_normalized). Sirf RESOLVED
      values — unresolved reference (NULL/empty value) kabhi identity row nahi.
    - Provenance aur status ALAG dimensions. Provenance = kahan se aaya; status =
      value+channel confirmation lifecycle.
    - status=CONFIRMED = confirmed value+channel association (client ne kaha ke
      ye value is channel ke liye sahi hai).
    - Provenance precedence CENTRALIZED aur explicit (neeche) — repositories ise
      consume karti hain, apni "stronger" logic invent nahi karti.

Centralized policies (provenance precedence, status monotonicity) yahin rehti
hain taake har layer same rule follow kare.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import ModelBase, utcnow
from app.models.contact import ContactInfoStatus, ContactProvenance

if TYPE_CHECKING:
    from app.models.contact import Contact


# ── Centralized provenance precedence policy ──
# Explicit ordered list, lowest → highest trust for client-intent. Repository
# isko consume karti hai; "MANUAL = highest" koi permanent undocumented truth
# NAHI — ye evolvable policy hai. Index jitna bada, utna higher precedence.
PROVENANCE_PRECEDENCE: tuple[ContactProvenance, ...] = (
    ContactProvenance.INFERRED,
    ContactProvenance.LEAD_IMPORT,
    ContactProvenance.CLIENT_REFERENCED_CURRENT_CALL_NUMBER,
    ContactProvenance.CLIENT_SPOKEN,
    ContactProvenance.CLIENT_PROVIDED,
    ContactProvenance.MANUAL,
)


def provenance_rank(provenance: str) -> int:
    """Return the precedence rank of a provenance value (higher = more trusted).

    Args:
        provenance: The provenance string.

    Returns:
        int: Index in PROVENANCE_PRECEDENCE, or -1 if unknown (treated lowest).
    """
    try:
        return PROVENANCE_PRECEDENCE.index(ContactProvenance(provenance))
    except (ValueError, KeyError):
        return -1


# ── Centralized status monotonicity policy ──
# CONFIRMED ko unconfirmed/weaker se downgrade nahi karna (jab tak koi explicit
# re-verification flow future mein na aaye). Ye rank-based hai.
_STATUS_RANK: dict[str, int] = {
    ContactInfoStatus.NONE.value: 0,
    ContactInfoStatus.DETECTED.value: 1,
    ContactInfoStatus.VALIDATED.value: 2,
    ContactInfoStatus.CONFIRMED.value: 3,
}


def can_transition_status(current: str, new: str) -> bool:
    """Whether a status change is allowed under the monotonicity policy.

    Confirmed se neeche jaana block hai. Barabar ya upar theek. Ye policy
    centralized hai taake har layer same rule use kare.

    Args:
        current: Current status value.
        new: Proposed new status value.

    Returns:
        bool: True agar transition allowed (new_rank >= current_rank).
    """
    return _STATUS_RANK.get(new, 0) >= _STATUS_RANK.get(current, 0)


class ContactMethod(ModelBase):
    """A canonical communication method for a contact (person).

    Identity: (contact_id, channel, value_normalized). Same value+channel = ek
    row; same value alag channel = alag rows.

    Attributes:
        contact_id: Parent contact/person (FK).
        channel: Communication channel (phone/email/whatsapp/sms).
        value_normalized: Canonical resolved value — identity ka hissa. NEVER
            null/empty (resolver deta hai; unresolved reference row nahi banti).
        value_raw: Original raw value as observed.
        provenance: Where this method came from (explicit precedence policy).
        status: Confirmation lifecycle (value+channel association).
        is_preferred: Kya ye is channel ka preferred method (per-channel ≤1 True).
        note: Optional short structured note (e.g. client channel-assignment).
        first_observed_at: Pehli baar kab observe hua.
        last_observed_at: Aakhri baar kab.
        contact: Parent contact relationship.
    """

    __tablename__ = "contact_methods"
    __table_args__ = (
        # Canonical identity — DB-level dedup (race-safe).
        UniqueConstraint(
            "contact_id",
            "channel",
            "value_normalized",
            name="uq_contact_methods_identity",
        ),
        Index("ix_contact_methods_contact_id", "contact_id"),
        Index("ix_contact_methods_contact_channel", "contact_id", "channel"),
        # NOTE (Postgres migration): per (contact_id, channel) at most one
        # is_preferred=True ko DB level par enforce karne ke liye ek PARTIAL
        # UNIQUE INDEX chahiye:
        #   CREATE UNIQUE INDEX ... ON contact_methods (contact_id, channel)
        #   WHERE is_preferred = true;
        # SQLite partial-index support limited hai, isliye abhi application-level
        # enforcement (set_preferred transaction). Ye final concurrency guarantee
        # NAHI — Postgres par partial index isko pukhta karega.
    )

    contact_id: Mapped[str] = mapped_column(
        ForeignKey("contacts.id", ondelete="CASCADE"), nullable=False
    )
    channel: Mapped[str] = mapped_column(String(20), nullable=False)
    value_normalized: Mapped[str] = mapped_column(String(320), nullable=False)
    value_raw: Mapped[str | None] = mapped_column(String(320), nullable=True)
    provenance: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=ContactInfoStatus.DETECTED.value
    )
    is_preferred: Mapped[bool] = mapped_column(nullable=False, default=False)
    note: Mapped[str | None] = mapped_column(String(300), nullable=True)
    first_observed_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)
    last_observed_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)

    contact: Mapped["Contact"] = relationship("Contact")  # noqa: F821

    def __repr__(self) -> str:
        """Return a PII-light representation (no contact value).

        Returns:
            str: Debug-safe representation with id, channel, status.
        """
        return (
            f"<ContactMethod id={self.id!r} contact_id={self.contact_id!r} "
            f"channel={self.channel!r} status={self.status!r}>"
        )