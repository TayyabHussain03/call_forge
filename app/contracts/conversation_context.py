"""Session-scoped conversation context.

Ye orchestrator ka structured input/output context hai. IMMUTABLE from the
engine's perspective: engine ise padhta hai aur `with_updates()` se ek naya copy
banata hai — original kabhi mutate nahi hota. Caller/session isko own karta hai
aur agle turn ke liye returned copy rakhta hai.

DESIGN:
    - Frozen dataclass — top-level fields reassign nahi ho sakte.
    - Nested `clarification` bhi frozen (ClarificationState 2C-B mein frozen hai),
      toa caller iske through nested state mutate nahi kar sakta.
    - Sirf woh fields jo existing components (validator, priority, machine)
      genuinely chahiye — catch-all object nahi.
    - Memory yahan carry hoti hai, StateMachine mein NAHI.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from app.conversation.guardrails.clarification import ClarificationState


@dataclass(frozen=True)
class ConversationContext:
    """Immutable per-session conversation context.

    Attributes:
        call_id: Is call/session ka unique id.
        business_id: Kis business ki call (optional).
        current_contact_id: Abhi jis person se baat ho rahi (John vs Mike distinct).
        previous_contact_id: Pichli baar jis person se baat hui thi (follow-up).
        previous_conversation_exists: Kya is person se pehle baat hui — follow_up
            state ka precondition.
        contact_candidate: Abhi tak mila contact candidate (unconfirmed, channel-
            agnostic). Validator `contact_candidate_exists` isse dekhta hai.
            RESOLVED resolution ka value yahan aata hai — DB mein NAHI (E1).
        contact_candidate_channel: Us candidate ka channel (email/phone/...).
        contact_candidate_provenance: Us candidate ka provenance (resolver se).
            Value + channel + provenance synchronized rehte hain — kabhi mismatched
            nahi.
        contact_confirmed: Kya candidate client ne confirm kiya.
        callback: Callback context/time string agar mila.
        dnc_pending: Kya client ne DNC maanga (validator safety-net isse dekhta).
        interest_preserved: True jab busy→callback ne interest barqarar rakha.
        call_number: TRUSTED telephony session number (E.164). Caller/application
            populate karta hai — LLM/transcript ise mutate NAHI kar sakta.
            Resolver CURRENT_CALL_NUMBER isse resolve karta hai.
        business_phone: TRUSTED Business.phone_e164 (read-only source). Caller
            populate karta hai. Resolver BUSINESS_PHONE isse resolve karta hai.
        business_email: TRUSTED business email. Caller populate karta hai.
        clarification: Active contact-clarification session state (frozen), ya None.
    """

    call_id: str
    business_id: str | None = None
    current_contact_id: str | None = None
    previous_contact_id: str | None = None
    previous_conversation_exists: bool = False
    contact_candidate: str | None = None
    contact_candidate_channel: str | None = None
    contact_candidate_provenance: str | None = None
    contact_confirmed: bool = False
    callback: str | None = None
    dnc_pending: bool = False
    interest_preserved: bool = False
    call_number: str | None = None
    business_phone: str | None = None
    business_email: str | None = None
    clarification: ClarificationState | None = None

    def with_updates(self, **changes: Any) -> ConversationContext:
        """Return a new context with the given fields changed.

        Original object NEVER mutate hota — `dataclasses.replace` ek naya frozen
        instance deta hai. Caller returned copy ko agle turn ke liye rakhta hai.

        Args:
            **changes: Field names → new values.

        Returns:
            ConversationContext: A new context with updates applied.
        """
        return replace(self, **changes)

    def to_validator_context(self) -> dict[str, Any]:
        """Produce the flat dict the ActionValidator/machine expect.

        Validator prerequisite keys (contact_candidate, contact_confirmed,
        callback, previous_conversation_exists) aur DNC safety-net key
        (dnc_pending) is dict se aate hain. Ye ek naya dict banata hai — context
        mutate nahi hota.

        Returns:
            dict[str, Any]: Flat read-only context for validation.
        """
        return {
            "contact_candidate": self.contact_candidate,
            "contact_confirmed": self.contact_confirmed,
            "callback": self.callback,
            "previous_conversation_exists": self.previous_conversation_exists,
            "dnc_pending": self.dnc_pending,
        }