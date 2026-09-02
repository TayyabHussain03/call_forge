"""Contracts for contact information extracted during a call.

TRUST BOUNDARY: UNTRUSTED at entry.
    Contact data yahan STT/LLM se aata hai — dono galat ho sakte hain. Isliye
    ye contracts ek EXPLICIT LIFECYCLE encode karte hain: detected → normalized
    → validated → confirmed_by_person. Koi bhi contact tab tak "trusted" nahi
    jab tak client ne khud confirm na kiya ho AND deterministic validation pass
    na ho.

CORE RULE: LLM/STT confidence != correctness.
    High confidence ka matlab "shayad theek hai", "pakka theek hai" nahi. Isliye
    persistence layer (baad mein) sirf `confirmed_by_person == True` waale
    contacts ko hi confirmed maanega. Ye contracts sirf shape aur deterministic
    format validation guarantee karte hain — hallucination nahi rokte.

Note: In models mein koi telephony/DB/business logic NAHI hai — sirf data shape
    aur format-level validation. Business decisions (kab clarify karna hai, kab
    save karna hai) upar ke layers karte hain.
"""

from __future__ import annotations

import re
from enum import Enum

import phonenumbers
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ContactStatus(str, Enum):
    """Lifecycle stage of an extracted contact value.

    Ye stages sequential hain. Ek value tab tak aage nahi badhti jab tak
    current stage ka requirement poora na ho. `CONFIRMED_BY_PERSON` sirf tab set
    hota hai jab client ne verbally confirm kiya ho.
    """

    DETECTED = "detected"                # STT/LLM ne kuch nikala, abhi raw
    NORMALIZED = "normalized"            # canonical form mein convert hua
    VALIDATED = "validated"              # deterministic format check pass
    CONFIRMED_BY_PERSON = "confirmed_by_person"  # client ne verbally confirm kiya
    REJECTED = "rejected"               # format invalid ya client ne mana kiya


class ContactChannel(str, Enum):
    """Which contact medium this value represents."""

    EMAIL = "email"
    PHONE = "phone"
    WHATSAPP = "whatsapp"


# Deterministic email format pattern. "Sahi format" check karta hai, "asli
# maujood hai" nahi — woh alag concern hai (email verification service, etc.).
_EMAIL_RE = re.compile(r"^[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@"
                       r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")


class ExtractedContact(BaseModel):
    """A single contact value moving through the extraction lifecycle.

    Ye untrusted-origin data ka structured representation hai. Har field ka
    trust level alag hai — `raw_value` bilkul untrusted, `status` deterministic
    checks ka result. Model sirf format validate karta hai; kab confirm/persist
    karna hai woh business layer decide karti hai.

    Attributes:
        channel: Contact ka type (email/phone/whatsapp).
        raw_value: STT/LLM se aaya bilkul raw string. UNTRUSTED. Kabhi seedha
            save/use nahi karna — sirf audit/debug ke liye.
        normalized_value: Canonical form (lowercased email, digits-only phone).
            None jab tak normalization na hui ho.
        status: Lifecycle stage. Default DETECTED.
        extraction_confidence: Extractor ka apna confidence (0.0–1.0). Sirf
            extraction ke liye — isko intent ya STT confidence ke saath mat
            milao. Correctness ka proof NAHI.
        source_turn_id: Kis conversation turn se aaya (traceability).
        format_valid: Deterministic format check pass hua ya nahi. None jab tak
            validate na hua ho.
    """

    model_config = ConfigDict(extra="forbid", frozen=False)

    channel: ContactChannel
    raw_value: str = Field(min_length=1, max_length=320)
    normalized_value: str | None = Field(default=None, max_length=320)
    status: ContactStatus = ContactStatus.DETECTED
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    source_turn_id: str | None = None
    format_valid: bool | None = None

    @field_validator("raw_value")
    @classmethod
    def _strip_raw(cls, v: str) -> str:
        """Trim surrounding whitespace from the raw value.

        Args:
            v: The incoming raw string.

        Returns:
            str: Whitespace-trimmed value.
        """
        return v.strip()

    @property
    def is_confirmed(self) -> bool:
        """Whether this contact has been explicitly confirmed by the client.

        Returns:
            bool: True only when status is CONFIRMED_BY_PERSON.
        """
        return self.status == ContactStatus.CONFIRMED_BY_PERSON

    @property
    def is_clear_for_progression(self) -> bool:
        """Whether the extraction is clear enough to move toward confirmation.

        Ye clarification ke liye canonical "clear/usable" signal hai — confirmation
        se ALAG (woh baad mein client karta hai). "Clear" ka matlab: format valid
        HO aur extractor ne value ko kam-se-kam VALIDATED/CONFIRMED status tak
        process kiya ho (yani raw DETECTED se aage). Ye existing status + format
        signals par bana hai — engine mein koi naya arbitrary confidence threshold
        hardcode NAHI hota; "clear" ka faisla extractor ke set kiye status se aata
        hai.

        Returns:
            bool: True agar contact clarification-progression ke liye clear hai.
        """
        return bool(self.format_valid) and self.status in (
            ContactStatus.VALIDATED,
            ContactStatus.CONFIRMED_BY_PERSON,
        )

    @property
    def is_safe_to_persist(self) -> bool:
        """Whether this contact may be stored as a confirmed contact.

        Sirf tab True jab format valid HO aur client ne confirm kiya HO. Ye
        do-taraffa gate hallucinated/misheard contacts ko silently save hone se
        rokta hai.

        Returns:
            bool: True when both format-valid and person-confirmed.
        """
        return bool(self.format_valid) and self.is_confirmed


class ContactValidationResult(BaseModel):
    """Deterministic validation outcome for a contact value.

    Ye extraction se ALAG step hai — pure format/pattern check, koi AI nahi.
    Isi ka result `ExtractedContact.format_valid` set karta hai.

    Attributes:
        channel: Kis channel ko validate kiya.
        candidate: Jo value check hui (normalized form expected).
        is_valid: Format pattern pass hua ya nahi.
        reason: Agar invalid, to short machine-readable reason.
    """

    model_config = ConfigDict(extra="forbid")

    channel: ContactChannel
    candidate: str
    is_valid: bool
    reason: str | None = None

    @classmethod
    def check_email(cls, candidate: str) -> ContactValidationResult:
        """Deterministically validate an email's format.

        Args:
            candidate: Email string (lowercased/normalized recommended).

        Returns:
            ContactValidationResult: Validation outcome for the email.
        """
        norm = candidate.strip().lower()
        ok = bool(_EMAIL_RE.match(norm))
        return cls(
            channel=ContactChannel.EMAIL,
            candidate=norm,
            is_valid=ok,
            reason=None if ok else "email_format_invalid",
        )

    @classmethod
    def check_phone(cls, candidate: str, region: str | None = None) -> ContactValidationResult:
        """Deterministically validate a phone number's format (international).

        Google's libphonenumber use karti hai, isliye kisi bhi country (US, PK,
        UK, UAE, EU) aur landline/mobile ko sahi validate karti hai. Result ka
        `candidate` canonical E.164 hota hai jab number valid ho — taake
        downstream storage/dedup consistent rahe.

        Args:
            candidate: Phone string, spoken or formatted.
            region: ISO country hint (e.g. "PK", "GB") for national-format
                numbers that don't start with "+". "+"-prefixed numbers isko
                ignore karte hain.

        Returns:
            ContactValidationResult: Valid hone par candidate E.164 form mein;
            warna is_valid=False with a reason.
        """
        try:
            parsed = phonenumbers.parse(candidate, region.upper() if region else None)
        except phonenumbers.NumberParseException:
            return cls(
                channel=ContactChannel.PHONE,
                candidate=candidate.strip(),
                is_valid=False,
                reason="phone_unparseable",
            )
        if not phonenumbers.is_valid_number(parsed):
            return cls(
                channel=ContactChannel.PHONE,
                candidate=candidate.strip(),
                is_valid=False,
                reason="phone_format_invalid",
            )
        e164 = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
        return cls(
            channel=ContactChannel.PHONE,
            candidate=e164,
            is_valid=True,
            reason=None,
        )