"""Contracts for contact understanding and deterministic resolution.

TRUST BOUNDARY:
    - ContactUnderstanding = LLM/NLP output → UNTRUSTED. Iska `value` (client ka
      bola number/email) untrusted hai.
    - ResolutionContext = application dwara inject ki gayi TRUSTED values
      (call_number, business_phone, ...). Client transcript se kabhi nahi.

CORE RULES:
    - RESOLVED ≠ CONFIRMED. Resolver ek usable candidate deta hai; confirmation
      ek alag conversation step hai.
    - value + reference DONO present → INVALID (resolver choose nahi karega).
    - value + reference DONO absent → NEEDS_CLARIFICATION (ambiguous).
    - Unresolved kabhi fake ContactMethod nahi banata.

Resolver ko `contact_id` ki zaroorat NAHI — woh persistence ka concern hai.
Resolver: understanding + trusted context → result.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.contracts.contact_info import ContactChannel
from app.models.contact import ContactProvenance


class ContactIntent(str, Enum):
    """What the client is trying to do regarding contact info.

    Conversation-level Intent (interested/busy/dnc) se ALAG — ye contact
    sub-intent hai.
    """

    PROVIDE_CONTACT = "provide_contact"   # value ya reference de raha
    CONFIRM_CONTACT = "confirm_contact"   # "yes that's right"
    CORRECT_CONTACT = "correct_contact"   # "no, use 214-..."
    DECLINE_CONTACT = "decline_contact"   # "I won't share"
    UNCLEAR = "unclear"                   # ambiguous


class ContactReference(str, Enum):
    """A reference to a contact value resolved from trusted application data.

    Reference = "kis cheez ki taraf ishaara" — actual value application ke
    trusted source se aata hai, LLM se kabhi nahi.
    """

    CURRENT_CALL_NUMBER = "current_call_number"    # "is number par"
    BUSINESS_PHONE = "business_phone"              # "office number"
    BUSINESS_EMAIL = "business_email"              # "business email"
    PREVIOUSLY_PROVIDED = "previously_provided"    # "jo maine pehle diya"


class ResolutionOutcome(str, Enum):
    """The outcome category of a contact resolution attempt."""

    RESOLVED = "resolved"                            # usable candidate mila
    NEEDS_CLARIFICATION = "needs_clarification"      # ambiguous, aur poocho
    INVALID = "invalid"                              # explicit value malformed / contradictory input
    UNSUPPORTED_REFERENCE = "unsupported_reference"  # reference type handle nahi hota
    UNAVAILABLE_REFERENCE = "unavailable_reference"  # reference supported, source missing


@dataclass(frozen=True)
class ContactUnderstanding:
    """Structured interpretation of a client's contact-related utterance.

    LLM/NLP produce karta hai — UNTRUSTED. Resolver ise consume karta hai lekin
    semantic understanding khud nahi karta.

    Attributes:
        intent: Contact sub-intent.
        channel: Intended channel; UNKNOWN allowed (resolver guess nahi karega).
        value: Explicit spoken value (untrusted), ya None.
        reference: Trusted-source reference, ya None.
        interpretation_confidence: LLM ka interpretation confidence (0.0–1.0).
    """

    intent: ContactIntent
    channel: ContactChannel
    value: str | None = None
    reference: ContactReference | None = None
    interpretation_confidence: float = 0.0


@dataclass(frozen=True)
class ResolvedMethod:
    """A trusted, previously-resolved contact method for PREVIOUSLY_PROVIDED.

    Ye arbitrary string nahi — application explicitly ek trusted resolved object
    deta hai, taake PREVIOUSLY_PROVIDED galat method select na kare.

    Attributes:
        channel: The method's channel.
        value_normalized: The canonical value.
    """

    channel: ContactChannel
    value_normalized: str


@dataclass(frozen=True)
class ResolutionContext:
    """Trusted values injected by the application for reference resolution.

    Ye sab application ke trusted sources hain — client transcript se kabhi nahi.
    Resolver in par DB/network query NAHI karta; values yahan pehle se inject
    hoti hain.

    Attributes:
        call_number: Trusted telephony session number (E.164), ya None.
        business_phone: Business.phone_e164 (read-only source), ya None.
        business_email: Trusted business email, ya None.
        previously_provided_method: A trusted resolved method for
            PREVIOUSLY_PROVIDED, ya None.
        region: Optional ISO region hint for phone normalization.
    """

    call_number: str | None = None
    business_phone: str | None = None
    business_email: str | None = None
    previously_provided_method: ResolvedMethod | None = None
    region: str | None = None


@dataclass(frozen=True)
class ResolutionResult:
    """The deterministic result of a contact resolution attempt.

    Unresolved conditions kabhi fake ContactMethod value nahi bante. RESOLVED par
    hi value_normalized present hota hai — aur woh CONFIRMED nahi (alag step).

    Attributes:
        outcome: The resolution outcome category.
        channel: Resolved channel (RESOLVED par), ya None.
        value_normalized: Canonical value (RESOLVED par), ya None.
        value_raw: Original raw value where meaningful, ya None.
        provenance: Provenance for persistence (RESOLVED par), ya None.
        reason: Machine-readable detail for non-resolved outcomes (debug).
    """

    outcome: ResolutionOutcome
    channel: ContactChannel | None = None
    value_normalized: str | None = None
    value_raw: str | None = None
    provenance: ContactProvenance | None = None
    reason: str | None = None

    @property
    def is_resolved(self) -> bool:
        """Whether a usable candidate was produced.

        Returns:
            bool: True only for RESOLVED.
        """
        return self.outcome == ResolutionOutcome.RESOLVED