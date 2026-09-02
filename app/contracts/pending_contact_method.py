"""Persistence-intent contract for a confirmed contact method.

Ye contract sirf ye represent karta hai: "conversation successfully us point tak
pohanch gayi jahan ye contact method ab persist hone ke layak hai." Iska koi
behaviour NAHI — na DB, na repository, na confirmation logic, na resolution.

ENGINE isko banata hai (successful confirmation par). SERVICE isko consume karke
ContactMethodRepository ko persist karta hai (status CONFIRMED). Engine khud DB
nahi karta.

`status` yahan NAHI — service hamesha CONFIRMED persist karta hai (locked rule:
successful confirmation → CONFIRMED). Engine arbitrary status inject nahi karta.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PendingContactMethod:
    """A contact method eligible for persistence after successful confirmation.

    Sirf persistence ke liye zaroori trusted data. Value+channel+provenance
    synchronized hote hain (engine context se leta hai). contact_id engine ke
    trusted current_contact_id se aata hai — resolver/LLM se kabhi nahi.

    Attributes:
        contact_id: The person this method belongs to (engine's trusted current
            contact). Never guessed.
        channel: Communication channel (email/phone/whatsapp/sms).
        value_normalized: Canonical resolved value.
        value_raw: Original raw value where available.
        provenance: Provenance carried from the resolver (never remapped).
    """

    contact_id: str
    channel: str
    value_normalized: str
    provenance: str
    value_raw: str | None = None