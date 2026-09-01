"""Contracts for classified conversation intent.

TRUST BOUNDARY: UNTRUSTED at entry.
    Intent LLM output se aata hai. LLM galat intent, invalid enum, ya galat
    confidence de sakta hai. Ye contract sirf shape aur range validate karta
    hai — iska matlab ye nahi ke classification "sahi" hai.

CORE RULE: Intent, Interest, aur Tone ALAG dimensions hain.
    Ek harsh-tone client bhi high-interest ho sakta hai ("I'm busy, just email
    it"). In teeno ko kabhi ek field mein collapse mat karo, warna busy-but-
    interested leads lose ho jayenge. Guide ne isi ko strongly emphasize kiya.

CORE RULE: `reasoning` runtime authority NAHI hai.
    LLM ka free-form reasoning sirf human debugging/audit ke liye hai. Koi bhi
    business decision reasoning text par depend nahi karega — decisions enums
    aur deterministic rules par bante hain.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.core.constants import Intent, Interest, Tone


class IntentSignal(BaseModel):
    """A single classified reading of the client's intent for one turn/call.

    Ye LLM ke intent classification ka structured, validated container hai. Har
    dimension (intent/tone/interest) apne enum se bandha hai — arbitrary string
    allowed nahi. Confidence sirf INTENT classification ka hai, aur kisi aur
    confidence (STT, extraction) ke saath mix nahi hota.

    Attributes:
        intent: Client kya convey kar raha hai (finite enum).
        tone: Client ka emotional tone (finite enum). Lead quality se
            independent.
        interest: Qualification level (finite enum). Tone se independent.
        intent_confidence: Sirf intent classification ka confidence (0.0–1.0).
            STT/extraction confidence ke saath combine mat karo.
        is_decision_maker: LLM ka reading ke ye owner/authorized person hai ya
            nahi. None jab abhi pata na ho. Trusted-fact nahi — proposal hai.
        reasoning: Free-form explanation. SIRF audit/debug. Kisi runtime ya
            authorization decision mein use NAHI hoga. Optional.
    """

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    tone: Tone = Tone.NEUTRAL
    interest: Interest = Interest.NONE
    intent_confidence: float = Field(ge=0.0, le=1.0)
    is_decision_maker: bool | None = None
    reasoning: str | None = Field(default=None, max_length=2000)

    @property
    def is_terminal_intent(self) -> bool:
        """Whether this intent should end the call deterministically.

        NOT_INTERESTED aur DO_NOT_CALL par call band honi chahiye — chahe LLM
        kuch aur propose kare. Ye property business layer ko woh signal deti hai.

        Returns:
            bool: True for NOT_INTERESTED or DO_NOT_CALL.
        """
        return self.intent in {Intent.NOT_INTERESTED, Intent.DO_NOT_CALL}

    @property
    def requires_followup(self) -> bool:
        """Whether this intent implies a follow-up is needed.

        BUSY aur CALLBACK_REQUESTED ka matlab lead lost nahi — follow-up queue
        mein jaana chahiye. Interest se independent (busy client high-interest
        ho sakta hai).

        Returns:
            bool: True for BUSY or CALLBACK_REQUESTED.
        """
        return self.intent in {Intent.BUSY, Intent.CALLBACK_REQUESTED}