"""Phone normalization using existing phonenumbers infrastructure.

Reuses the same libphonenumber approach as app.contracts.lead.BusinessIdentity
but returns a structured NormalizedPhone with provenance.
"""

from __future__ import annotations

import phonenumbers

from app.leads.contracts import NormalizedPhone, PhoneNormalizationStatus


def normalize_phone(
    raw: str,
    country_context: str | None = None,
) -> NormalizedPhone:
    """Normalize a raw phone string to E.164 where possible."""
    raw = raw.strip()
    if not raw:
        return NormalizedPhone(
            raw=raw or "(empty)",
            normalized=None,
            status=PhoneNormalizationStatus.INVALID,
            country_context=country_context,
        )

    region = country_context.upper() if country_context else None
    try:
        parsed = phonenumbers.parse(raw, region)
    except phonenumbers.NumberParseException:
        if region is None and not raw.startswith("+"):
            return NormalizedPhone(
                raw=raw,
                normalized=None,
                status=PhoneNormalizationStatus.NEEDS_REVIEW,
                country_context=country_context,
            )
        return NormalizedPhone(
            raw=raw,
            normalized=None,
            status=PhoneNormalizationStatus.INVALID,
            country_context=country_context,
        )

    if not phonenumbers.is_valid_number(parsed):
        return NormalizedPhone(
            raw=raw,
            normalized=None,
            status=PhoneNormalizationStatus.INVALID,
            country_context=country_context,
        )

    e164 = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    return NormalizedPhone(
        raw=raw,
        normalized=e164,
        status=PhoneNormalizationStatus.VALID_E164,
        country_context=country_context,
    )
