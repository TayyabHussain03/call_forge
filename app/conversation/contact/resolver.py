"""Deterministic contact resolver.

ContactUnderstanding (untrusted) + ResolutionContext (trusted, injected) →
ResolutionResult. Ye pure/deterministic hai: same input → same output.

STRICT PROHIBITIONS (guide ke mutabiq): resolver DB query/write nahi, network
nahi, LLM nahi, ClarificationEngine invoke nahi, contact_id/ownership infer
nahi, Business.phone_e164 modify nahi, preferred decide nahi, confirmation
decide nahi, malformed value guess/repair nahi.

RESOLVED ≠ CONFIRMED — resolver sirf usable candidate deta hai; confirmation alag
conversation step hai.
"""

from __future__ import annotations

from app.contracts.contact_info import ContactChannel, ContactValidationResult
from app.contracts.contact_understanding import (
    ContactReference,
    ContactUnderstanding,
    ResolutionContext,
    ResolutionOutcome,
    ResolutionResult,
)
from app.models.contact import ContactProvenance

# Kaunse channels phone-based hain (phone-style validation ke liye).
_PHONE_CHANNELS = frozenset(
    {ContactChannel.PHONE, ContactChannel.WHATSAPP, ContactChannel.SMS}
)


class ContactResolver:
    """Resolves a ContactUnderstanding into a safe contact candidate.

    Stateless aur deterministic — koi injected dependency nahi (trusted data har
    call mein ResolutionContext se aata hai).
    """

    def resolve(
        self, understanding: ContactUnderstanding, context: ResolutionContext
    ) -> ResolutionResult:
        """Resolve an understanding into a deterministic result.

        Flow: input-consistency check → explicit value OR reference resolution →
        normalize + deterministic validate → ResolutionResult.

        Args:
            understanding: The (untrusted) LLM interpretation.
            context: Trusted, application-injected resolution values.

        Returns:
            ResolutionResult: RESOLVED candidate, ya ek non-resolved outcome. Koi
            fake ContactMethod value nahi.
        """
        value = understanding.value
        reference = understanding.reference

        # 1. Input consistency: value + reference dono → INVALID (resolver choose
        #    nahi karega kis ko priority di jaaye).
        if value is not None and value.strip() != "" and reference is not None:
            return ResolutionResult(
                outcome=ResolutionOutcome.INVALID,
                reason="contradictory: both explicit value and reference present",
            )

        # 2. Neither value nor reference → ambiguous, clarification chahiye.
        has_value = value is not None and value.strip() != ""
        if not has_value and reference is None:
            return ResolutionResult(
                outcome=ResolutionOutcome.NEEDS_CLARIFICATION,
                reason="no explicit value and no reference",
            )

        # 3. Reference resolution (trusted context se).
        if reference is not None:
            return self._resolve_reference(understanding, context)

        # 4. Explicit value resolution.
        return self._resolve_explicit(understanding, context)

    def _resolve_explicit(
        self, understanding: ContactUnderstanding, context: ResolutionContext
    ) -> ResolutionResult:
        """Resolve an explicit client-provided value (untrusted → validated).

        Channel UNKNOWN par resolver channel guess NAHI karega — NEEDS_
        CLARIFICATION. Malformed value → INVALID (koi repair nahi).

        Args:
            understanding: The understanding carrying the explicit value.

        Returns:
            ResolutionResult: RESOLVED (validated), INVALID, ya
            NEEDS_CLARIFICATION.
        """
        channel = understanding.channel
        raw = (understanding.value or "").strip()

        if channel == ContactChannel.UNKNOWN:
            return ResolutionResult(
                outcome=ResolutionOutcome.NEEDS_CLARIFICATION,
                reason="explicit value but channel unknown",
            )

        return self._validate_and_build(
            channel, raw, ContactProvenance.CLIENT_PROVIDED, region=context.region
        )

    def _resolve_reference(
        self, understanding: ContactUnderstanding, context: ResolutionContext
    ) -> ResolutionResult:
        """Resolve a reference against trusted context values only.

        Args:
            understanding: The understanding carrying the reference.
            context: Trusted injected values.

        Returns:
            ResolutionResult: RESOLVED, UNAVAILABLE_REFERENCE, or
            UNSUPPORTED_REFERENCE.
        """
        reference = understanding.reference
        channel = understanding.channel

        # Reference → (trusted value, provenance, implied-channel-if-any).
        if reference == ContactReference.CURRENT_CALL_NUMBER:
            trusted = context.call_number
            provenance = ContactProvenance.CLIENT_REFERENCED_CURRENT_CALL_NUMBER
            # channel client ne diya (WhatsApp/phone/sms). UNKNOWN → clarify.
            resolve_channel = channel
        elif reference == ContactReference.BUSINESS_PHONE:
            trusted = context.business_phone
            provenance = ContactProvenance.LEAD_IMPORT
            resolve_channel = (
                channel if channel != ContactChannel.UNKNOWN else ContactChannel.PHONE
            )
        elif reference == ContactReference.BUSINESS_EMAIL:
            trusted = context.business_email
            provenance = ContactProvenance.LEAD_IMPORT
            resolve_channel = (
                channel if channel != ContactChannel.UNKNOWN else ContactChannel.EMAIL
            )
        elif reference == ContactReference.PREVIOUSLY_PROVIDED:
            method = context.previously_provided_method
            if method is None:
                return ResolutionResult(
                    outcome=ResolutionOutcome.UNAVAILABLE_REFERENCE,
                    reason="no previously-provided method supplied",
                )
            trusted = method.value_normalized
            provenance = ContactProvenance.CLIENT_PROVIDED
            resolve_channel = (
                channel if channel != ContactChannel.UNKNOWN else method.channel
            )
        else:
            return ResolutionResult(
                outcome=ResolutionOutcome.UNSUPPORTED_REFERENCE,
                reason=f"unsupported reference: {reference}",
            )

        if trusted is None or trusted.strip() == "":
            return ResolutionResult(
                outcome=ResolutionOutcome.UNAVAILABLE_REFERENCE,
                reason=f"trusted source unavailable for {reference.value}",
            )

        if resolve_channel == ContactChannel.UNKNOWN:
            return ResolutionResult(
                outcome=ResolutionOutcome.NEEDS_CLARIFICATION,
                reason="reference resolvable but channel unknown",
            )

        return self._validate_and_build(
            resolve_channel, trusted.strip(), provenance, region=context.region
        )

    def _validate_and_build(
        self,
        channel: ContactChannel,
        raw: str,
        provenance: ContactProvenance,
        region: str | None,
    ) -> ResolutionResult:
        """Normalize + deterministically validate, then build a result.

        Format validation `ContactValidationResult` se reuse hoti hai. Valid
        format ≠ confirmed — RESOLVED sirf candidate hai. Malformed → INVALID.

        Args:
            channel: The resolved channel.
            raw: The raw value to validate.
            provenance: Provenance for persistence.
            region: Optional region hint for phone normalization.

        Returns:
            ResolutionResult: RESOLVED (with normalized value) or INVALID.
        """
        if channel == ContactChannel.EMAIL:
            check = ContactValidationResult.check_email(raw)
        elif channel in _PHONE_CHANNELS:
            check = ContactValidationResult.check_phone(raw, region)
        else:
            return ResolutionResult(
                outcome=ResolutionOutcome.NEEDS_CLARIFICATION,
                reason=f"unhandled channel: {channel}",
            )

        if not check.is_valid:
            return ResolutionResult(
                outcome=ResolutionOutcome.INVALID,
                channel=channel,
                reason=check.reason,
            )

        return ResolutionResult(
            outcome=ResolutionOutcome.RESOLVED,
            channel=channel,
            value_normalized=check.candidate,  # canonical form from validator
            value_raw=raw,
            provenance=provenance,
        )