"""Brain input/output contracts.

Ye future ReasoningProvider ke boundaries hain — ABHI koi provider/orchestrator
nahi (sirf contracts). Trust tagging explicit:
    - BrainInput  = TRUSTED container (orchestrator-assembled), lekin uske andar
      `current_utterance` UNTRUSTED DATA hai (instruction nahi).
    - BrainProposal = poori tarah UNTRUSTED model output.

BrainProposal mein KOI execution authority nahi: no authority level, no next_state,
no pricing/discount/quote, no selected service_id, no parsed contact value, no
persistence command, no catalog authorization, no provider-specific field, no
universal confidence. Sirf flags aur proposals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.brain.business_intelligence import (
    InferredSignal,
    ObservedSignal,
    UnknownSlot,
)
from app.contracts.contact_understanding import ContactUnderstanding
from app.core.constants import AgentAction, ConversationState, Intent, TopicCategory, Tone


@dataclass(frozen=True)
class BudgetState:
    """Remaining conversation budgets (deterministic-derived, read-only for Brain).

    Four SEPARATE budgets — collapse nahi hote. Ye orchestrator populate karega;
    Brain sirf padhega (early-exit reasoning ke liye), enforce deterministic layer
    karegi.

    Attributes:
        turns_remaining: Turns left before turn-budget exhaustion.
        seconds_remaining: Call-duration seconds left.
        reasoning_calls_remaining: Reasoning/model calls left this call.
        max_response_sentences: Response length ceiling (per turn).
    """

    turns_remaining: int | None = None
    seconds_remaining: int | None = None
    reasoning_calls_remaining: int | None = None
    max_response_sentences: int | None = None


@dataclass(frozen=True)
class BrainInput:
    """Trusted, bounded context assembled by the (future) orchestrator.

    Container TRUSTED hai (deterministic code ne banaya), lekin uske andar sab
    fields automatically truthful business-fact NAHI — especially `current_utterance`
    UNTRUSTED DATA hai. Bounded: koi full transcript, koi DB dump, koi poora catalog.

    Attributes:
        current_utterance: Prospect ka bola hua. UNTRUSTED DATA — kabhi instruction
            ki tarah treat nahi.
        current_state: Trusted current conversational state.
        current_goal: Current conversation goal (short label).
        recent_turns: A few bounded recent turns (NOT full transcript).
        business_intelligence: Snapshot of structured BI (read-only).
        known_signals: Detected eligibility/context signals.
        eligible_service_ids: Fresh eligible service ids (ids only, not catalog).
        offered_service_ids: Already-offered ids.
        campaign_policy_summary: Short policy summary (not full config).
        budget: Remaining budgets.
        resolved_contact_context: Read-only resolved contact context from the
            contact pipeline, ya None (Brain consume karta hai, re-derive nahi).
    """

    current_utterance: str
    current_state: ConversationState
    current_goal: str | None = None
    recent_turns: tuple[str, ...] = ()
    business_intelligence: "BusinessIntelligenceSnapshot | None" = None
    known_signals: frozenset[str] = field(default_factory=frozenset)
    eligible_service_ids: tuple[str, ...] = ()
    offered_service_ids: tuple[str, ...] = ()
    campaign_policy_summary: str | None = None
    budget: BudgetState | None = None
    resolved_contact_context: str | None = None


@dataclass(frozen=True)
class BusinessIntelligenceSnapshot:
    """A read-only snapshot of BI passed into BrainInput.

    Historical signals ka bounded view — Brain ise padhta hai, own nahi karta
    (single source of truth persistence/BI layer mein). Ye jaan-boojh kar
    lightweight hai (full history nahi, relevant signals).

    Attributes:
        observed: Relevant observed signals (bounded).
        inferred: Relevant inferred signals (bounded).
        unknown: Known-unknown slots (bounded).
    """

    observed: tuple[ObservedSignal, ...] = ()
    inferred: tuple[InferredSignal, ...] = ()
    unknown: tuple[UnknownSlot, ...] = ()


@dataclass(frozen=True)
class BrainProposal:
    """Untrusted proposal produced by the (future) reasoning provider.

    UNTRUSTED — deterministic layers (priority/policy/authority/validator/state
    machine) authority rakhte hain. Ye sirf SUGGEST karta hai. KOI authority field,
    no next_state, no pricing, no selected service_id, no parsed contact value.

    Attributes:
        detected_intent: Interpreted client intent (proposal — priority authority
            rakhti hai actual DNC decision).
        tone: Interpreted tone.
        topic_category: Structured conversational topic (UNTRUSTED — Brain proposes).
            ScopePolicyValidator isse deterministically scope check karta hai, raw
            text/keyword ke bina. Default UNKNOWN → fail-closed.
        proposed_action: Suggested next action (non-authoritative — ActionValidator
            + state machine decide karte hain).
        needs_service_decision: FLAG — "service decision chahiye". Chosen service
            NAHI (ServiceOfferingService/Selector handle karte hain).
        involves_contact: FLAG — "is turn mein contact hai". Parsed value NAHI
            (ContactResolver handle karta hai).
        contact_understanding: Optional interpreted contact (untrusted) jo contact
            pipeline validate karega. Parsed/normalized value NAHI.
        intelligence_updates: Proposed BI updates (observed/inferred/unknown, tagged).
            Deterministic layer inhe BI mein record karega.
        proposed_goal_update: Suggested new goal, ya None (deterministic layer
            accept/gate karega).
        reasoning: Free-form audit/debug text — NON-authoritative.
        action_confidence: Confidence in THIS proposal only (0.0–1.0). Distinct —
            interpretation/selection/inference confidence se mix NAHI.
    """

    detected_intent: Intent
    tone: Tone = Tone.NEUTRAL
    topic_category: TopicCategory = TopicCategory.UNKNOWN
    proposed_action: AgentAction | None = None
    needs_service_decision: bool = False
    involves_contact: bool = False
    contact_understanding: ContactUnderstanding | None = None
    intelligence_updates: tuple[ObservedSignal | InferredSignal | UnknownSlot, ...] = ()
    proposed_goal_update: str | None = None
    reasoning: str | None = None
    action_confidence: float = 0.0