"""Contracts for call sessions, transcripts, and post-call results.

TRUST BOUNDARY: mixed.
    - CallSession lifecycle metadata system-controlled hai (TRUSTED).
    - TranscriptTurn ka client-side text UNTRUSTED (STT output).
    - CallResult post-call analyzer se banta hai — LLM-derived, isliye shape
      validated but non-authoritative for irreversible actions.

DESIGN: CallResult analyzer ka output hai jo call KHATAM hone ke baad banta hai
    (live path se bahar). Isliye ismein deep fields ho sakte hain bina live
    latency ki fikr ke — guide ka "post-call analyzer live path mein nahi" wala
    principle.

PII NOTE: TranscriptTurn poora client speech rakhta hai — ye sensitive hai.
    Logging layer (baad mein) isko default INFO par log NAHI karegi; redaction
    utilities alag se aayengi.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.contracts.intent import IntentSignal
from app.core.constants import CallStatus, Interest, LeadStatus, Tone


class CallSession(BaseModel):
    """Metadata about a single outbound call attempt.

    System-controlled lifecycle record. Voice platform events isko update karte
    hain (queued → dialing → answered → completed).

    Attributes:
        call_id: Internal unique id for this attempt.
        lead_id: Kis lead ko call kiya.
        status: Current call status.
        provider_call_id: Voice platform (Vapi) ka apna call id, agar mila.
        started_at: Call connect hone ka waqt.
        ended_at: Call khatam hone ka waqt.
        attempt_number: Is lead ke liye kaunsi koshish (1-based).
    """

    model_config = ConfigDict(extra="forbid")

    call_id: str
    lead_id: str
    status: CallStatus = CallStatus.QUEUED
    provider_call_id: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    attempt_number: int = Field(default=1, ge=1)


class TranscriptTurn(BaseModel):
    """A single turn in the call transcript.

    Speaker ya to agent hai ya client. Client text UNTRUSTED hai — kabhi
    instruction ki tarah treat nahi hota. STT confidence sirf is turn ke speech
    recognition ka hai (intent/extraction confidence se alag).

    Attributes:
        turn_index: Call ke andar 0-based turn number.
        speaker: "agent" ya "client".
        text: Bola gaya text. Client ke liye UNTRUSTED (STT output).
        stt_confidence: Speech-recognition confidence (0.0–1.0), sirf client
            turns ke liye meaningful. None agar available nahi.
        timestamp: Turn ka waqt.
    """

    model_config = ConfigDict(extra="forbid")

    turn_index: int = Field(ge=0)
    speaker: str = Field(pattern="^(agent|client)$")
    text: str = Field(max_length=8000)
    stt_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    timestamp: datetime | None = None


class CallResult(BaseModel):
    """Post-call analysis outcome for one call.

    Analyzer (offline) transcript + metadata se ye banata hai. Ye LLM-derived
    hai — shape validated hai, lekin irreversible actions (jaise DNC) ke liye
    business layer deterministic confirmation maangti hai, sirf isi par bharosa
    nahi karti.

    Attributes:
        call_id: Kis call ka result.
        final_intent: Poori call ka final intent reading.
        interest: Final qualification level.
        tone: Overall observed tone.
        email_obtained: Koi CONFIRMED email mila ya nahi. Ye confirmed-contact
            gate se aata hai, LLM ke daawe se nahi.
        callback_required: Follow-up chahiye ya nahi.
        suggested_lead_status: Analyzer ka SUGGESTED status. Non-authoritative
            — lead service final decide karti hai.
        lead_score: 0–100 heuristic score.
        summary: Short human-readable summary.
    """

    model_config = ConfigDict(extra="forbid")

    call_id: str
    final_intent: IntentSignal
    interest: Interest = Interest.NONE
    tone: Tone = Tone.NEUTRAL
    email_obtained: bool = False
    callback_required: bool = False
    suggested_lead_status: LeadStatus = LeadStatus.NEW
    lead_score: int = Field(default=0, ge=0, le=100)
    summary: str = Field(default="", max_length=2000)