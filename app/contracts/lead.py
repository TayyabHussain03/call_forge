"""Contracts for leads ingested from Google Maps and their lifecycle.

TRUST BOUNDARY: scraped input is UNTRUSTED and UNVERIFIED.
    Google Maps se aaya data automatically "sahi" ya "verified" nahi hai —
    phone galat ho sakta hai, business band ho sakta hai. Isliye ingestion ka
    `source` aur `verification` status EXPLICIT hai. Ek scraped lead default
    UNVERIFIED hota hai; verification alag step hai.

DESIGN (guide ke mutabiq):
    Business identity (naam, phone, address) aur lead lifecycle (new → queued →
    ... → qualified) ALAG concepts hain, alag models mein. Ek business ki
    identity stable hai; uska lifecycle badalta rehta hai.

IDEMPOTENCY: duplicate scraped leads protect karne ke liye har lead ka ek
    deterministic `dedup_key` hota hai (phone-based). Same business dobara
    scrape ho to nayi row nahi banni chahiye — ye key wahi decision enable
    karti hai.
"""

from __future__ import annotations

from enum import Enum

import phonenumbers
from pydantic import BaseModel, ConfigDict, Field, computed_field

from app.core.constants import LeadStatus


class LeadSource(str, Enum):
    """Where a lead originally came from."""

    GOOGLE_MAPS = "google_maps"
    MANUAL = "manual"
    CSV_IMPORT = "csv_import"
    OTHER = "other"


class VerificationStatus(str, Enum):
    """Whether the scraped business data has been verified.

    Scraped data default UNVERIFIED hota hai. Verification (phone reachable,
    business active) ek alag downstream step hai — ingestion nahi.
    """

    UNVERIFIED = "unverified"
    PHONE_VALID = "phone_valid"      # format-level phone check pass
    VERIFIED = "verified"            # confirmed reachable/active
    INVALID = "invalid"             # data unusable


class BusinessIdentity(BaseModel):
    """Stable identifying information about a business.

    Ye lead ke lifecycle se ALAG hai — identity change nahi hoti chahe lead
    status kuch bhi ho. Fields mostly scraped hain, isliye untrusted; sirf
    shape/format validate hota hai.

    Attributes:
        name: Business ka naam. Required.
        phone: Contact number, raw scraped form. UNTRUSTED. Kisi bhi country ka
            ho sakta hai (US, PK, UK, UAE, EU), mobile ya landline.
        default_region: ISO country code hint (e.g. "US", "PK", "GB", "AE") jo
            tab use hota hai jab number "+" se shuru na ho. National-format
            numbers (jaise "0300 1234567") ko sahi country mein resolve karne ke
            liye zaroori. "+" wale numbers ke liye ignore hota hai.
        city: Sheher, agar available.
        state: State/region, agar available.
        address: Full address string, agar available.
        industry: Business category (e.g. "roofing"), agar available.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=300)
    phone: str = Field(min_length=1, max_length=40)
    default_region: str | None = Field(default=None, max_length=2, min_length=2)
    city: str | None = Field(default=None, max_length=120)
    state: str | None = Field(default=None, max_length=120)
    address: str | None = Field(default=None, max_length=500)
    industry: str | None = Field(default=None, max_length=120)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def phone_e164(self) -> str | None:
        """Phone in canonical E.164 form, or None if unparseable.

        International-safe: Google's libphonenumber (phonenumbers) use karti
        hai, jo har country (US/PK/UK/UAE/EU) aur landline/mobile handle karti
        hai. Same number alag formats mein same E.164 deta hai — yehi
        deduplication ko reliable banata hai across regions.

        "+" wale numbers bina region hint ke parse hote hain. National-format
        numbers ke liye `default_region` chahiye; agar na ho aur number "+" se
        shuru na ho, to parse fail hoke None milega (galat guess se behtar).

        Returns:
            str | None: E.164 string (e.g. "+923001234567") ya None agar
            number parse/validate nahi hua.
        """
        region = self.default_region.upper() if self.default_region else None
        try:
            parsed = phonenumbers.parse(self.phone, region)
        except phonenumbers.NumberParseException:
            return None
        if not phonenumbers.is_valid_number(parsed):
            return None
        return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def phone_digits(self) -> str:
        """Deduplication key form of the phone.

        E.164 ko prefer karta hai (canonical, cross-region consistent). Agar
        number parse na ho (invalid/unknown region), to raw ko digits/plus tak
        strip karke fallback deta hai — taake dedup phir bhi best-effort chale.

        Returns:
            str: Canonical E.164 if parseable, else a stripped raw fallback.
        """
        e164 = self.phone_e164
        if e164:
            return e164
        import re

        stripped = re.sub(r"[^\d+]", "", self.phone.strip())
        return stripped


class LeadIngestion(BaseModel):
    """A lead as it enters the system, before any calling happens.

    Ye ingestion-boundary contract hai: scraped/imported data ka pehla trusted-
    shape representation. Isme lifecycle abhi NEW hota hai aur verification
    abhi UNVERIFIED — dono ko yahan claim nahi kiya jaata.

    Attributes:
        business: Business identity (untrusted-origin, shape-validated).
        source: Lead kahan se aaya.
        verification: Verification status (default UNVERIFIED).
        status: Lifecycle status (default NEW).
        external_id: Source ka apna id agar ho (e.g. Maps place id).
            Idempotency mein help karta hai.
    """

    model_config = ConfigDict(extra="forbid")

    business: BusinessIdentity
    source: LeadSource = LeadSource.GOOGLE_MAPS
    verification: VerificationStatus = VerificationStatus.UNVERIFIED
    status: LeadStatus = LeadStatus.NEW
    external_id: str | None = Field(default=None, max_length=200)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dedup_key(self) -> str:
        """Deterministic key for duplicate detection.

        External id ko prefer karta hai (sabse reliable); warna phone digits par
        girta hai. Same business dobara ingest ho to yehi key match hogi, jisse
        persistence layer duplicate row banane se bach sakti hai.

        Returns:
            str: A stable dedup key for this lead.
        """
        if self.external_id:
            return f"ext:{self.external_id.strip().lower()}"
        return f"phone:{self.business.phone_digits}"


class LeadSummary(BaseModel):
    """A lead's current state for dashboards and API responses.

    Ye OUTBOUND (response) contract hai — jo system bahar dikhata hai. Ingestion
    se alag rakha gaya taake read-model aur write-model independently evolve ho
    sakein.

    Attributes:
        lead_id: Internal persisted id.
        business_name: Convenience field for display.
        status: Current lifecycle status.
        verification: Current verification status.
        call_attempts: Ab tak kitni baar call try hui.
        has_confirmed_contact: Koi confirmed email/phone mila ya nahi.
    """

    model_config = ConfigDict(extra="forbid")

    lead_id: str
    business_name: str
    status: LeadStatus
    verification: VerificationStatus
    call_attempts: int = Field(default=0, ge=0)
    has_confirmed_contact: bool = False