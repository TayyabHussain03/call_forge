"""Contracts for structured conversation turns.

TRUST BOUNDARY: LLM output is UNTRUSTED and NON-AUTHORITATIVE.
    Ye is layer ka sabse important principle hai. LLM ek probabilistic
    proposer hai — system of record nahi, final authority nahi. Isliye har
    LLM-generated decision field `proposed_` se prefix hai. Naming khud ek
    reminder hai: ye proposal hai, order nahi.

FLOW (guide ke mutabiq):
    Client turn (untrusted text)
        → LLM → ProposedConversationDecision  (yeh file)
        → State Machine + business rules (Step 4) → ApprovedAction
        → Execution
    State Machine authoritative rehti hai. Agar LLM invalid action/state
    propose kare, business layer deterministic fallback deti hai — LLM ko
    dobara call NAHI karti (live latency bachane ke liye).

Ye contracts sirf shape guarantee karte hain. `proposed_action` valid enum
hai iska matlab ye NAHI ke woh current state mein allowed hai — woh check
guardrails layer karti hai.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.contracts.contact_info import ExtractedContact
from app.contracts.intent import IntentSignal
from app.core.constants import AgentAction, ConversationState


class TurnInput(BaseModel):
    """Everything the engine feeds the LLM for one conversation turn.

    Ye jaan-boojh kar client transcript ko baaki context se ALAG field mein
    rakhta hai. Prompt-injection defense ka structural hissa: system policy,
    state, allowed actions, aur client speech kabhi ek ambiguous blob mein
    concatenate nahi hote — har cheez apni typed jagah par.

    Attributes:
        call_id: Kis call ka turn (traceability).
        current_state: State machine ki authoritative current state. TRUSTED
            (system ne set kiya, LLM ne nahi).
        allowed_actions: Is state mein legally allowed actions. TRUSTED. LLM ko
            inhi mein se propose karna chahiye.
        client_utterance: Client ka bola hua (STT text). UNTRUSTED. Ise kabhi
            instruction ki tarah treat nahi karna — ye pure data hai.
        turn_index: Call ke andar is turn ka number (0-based).
    """

    model_config = ConfigDict(extra="forbid")

    call_id: str
    current_state: ConversationState
    allowed_actions: list[AgentAction] = Field(min_length=1)
    client_utterance: str = Field(max_length=8000)
    turn_index: int = Field(ge=0)


class ProposedConversationDecision(BaseModel):
    """The LLM's PROPOSED decision for a turn — not an authoritative command.

    Har field jo LLM generate karta hai woh yahan proposal ke roop mein aata
    hai. Business/state layer isko validate karke approve ya reject karti hai.
    `proposed_` prefix intentional hai — koi consumer galti se ise final na
    samjhe.

    Attributes:
        detected_intent: LLM ka intent reading (validated shape, non-
            authoritative).
        response_text: Jo agent bolega. Approve hone ke baad hi TTS/Vapi ko
            jaayega. Length-capped taake voice mein monologue na ho.
        proposed_action: LLM jo action lena chahta hai (valid enum, lekin state
            mein allowed hai ya nahi woh guardrails decide karti hai).
        proposed_next_state: LLM jahan jaana chahta hai. NON-AUTHORITATIVE —
            state machine actual transition decide karti hai.
        extracted_contact: Agar client ne contact diya to uska extracted form.
            None agar is turn mein koi contact nahi. Untrusted lifecycle mein
            hi rehta hai (detected/normalized), confirmed nahi.
        turn_confidence: LLM ka overall confidence is decision par (0.0–1.0).
            Intent/extraction confidence se ALAG.
    """

    model_config = ConfigDict(extra="forbid")

    detected_intent: IntentSignal
    response_text: str = Field(max_length=1000)
    proposed_action: AgentAction
    proposed_next_state: ConversationState
    extracted_contact: ExtractedContact | None = None
    turn_confidence: float = Field(ge=0.0, le=1.0)


class ApprovedAction(BaseModel):
    """A decision AFTER the state machine has validated the LLM's proposal.

    Ye TRUSTED internal representation hai — execution layer sirf isi ko chalata
    hai, kabhi raw `ProposedConversationDecision` ko nahi. Is object ka wujood
    hi iska proof hai ke deterministic validation pass ho chuki.

    Attributes:
        call_id: Kis call ke liye.
        action: Approved action (allowed-in-state guarantee ho chuki).
        response_text: Jo agent actually bolega.
        next_state: State machine dwara decided actual next state (LLM ka
            proposal nahi — final).
        was_fallback: True agar ye LLM proposal reject hone par deterministic
            fallback se bana. Analytics/debug ke liye useful.
    """

    model_config = ConfigDict(extra="forbid")

    call_id: str
    action: AgentAction
    response_text: str = Field(max_length=1000)
    next_state: ConversationState
    was_fallback: bool = False