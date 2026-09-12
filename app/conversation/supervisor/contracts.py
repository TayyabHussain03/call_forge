"""Bounded immutable input and output contracts for advisory supervision."""

from __future__ import annotations

from dataclasses import dataclass

from app.conversation.prospect_intelligence.contracts import (
    ProspectEvidence,
    ProspectIntelligenceSnapshot,
)
from app.conversation.response_planning.contracts import (
    AddresseeStatus,
    InterruptionCategory,
    InterruptionContext,
)
from app.conversation.strategy.contracts import (
    ConversationStrategy,
    ConversationStrategyHint,
)
from app.core.constants import ConversationState


@dataclass(frozen=True)
class SupervisorInput:
    """One completed turn's bounded advisory-analysis input."""

    call_id: str
    turn_id: str
    source_turn_sequence: int
    current_turn_excerpt: str
    current_state: ConversationState
    prospect_intelligence: ProspectIntelligenceSnapshot
    conversation_strategy: ConversationStrategy
    recent_context: tuple[str, ...] = ()
    campaign_context_summary: str | None = None
    interruption: InterruptionContext = InterruptionContext()
    conversation_category: InterruptionCategory = InterruptionCategory.OTHER
    addressee_status: AddresseeStatus = AddresseeStatus.ADDRESSED_TO_AGENT

    def __post_init__(self) -> None:
        _validate_identity(self.call_id, "call id")
        _validate_identity(self.turn_id, "turn id")
        if self.source_turn_sequence < 0:
            raise ValueError("source turn sequence must be non-negative")
        _validate_text(self.current_turn_excerpt, "current turn excerpt", 500)
        if not isinstance(self.current_state, ConversationState):
            raise TypeError("current_state must be ConversationState")
        if not isinstance(self.prospect_intelligence, ProspectIntelligenceSnapshot):
            raise TypeError("prospect_intelligence has an invalid type")
        if not isinstance(self.conversation_strategy, ConversationStrategy):
            raise TypeError("conversation_strategy has an invalid type")
        if not isinstance(self.interruption, InterruptionContext):
            raise TypeError("interruption has an invalid type")
        if not isinstance(self.conversation_category, InterruptionCategory):
            raise TypeError("conversation_category has an invalid type")
        if not isinstance(self.addressee_status, AddresseeStatus):
            raise TypeError("addressee_status has an invalid type")
        recent = tuple(self.recent_context)
        if len(recent) > 4:
            raise ValueError("recent supervisor context is limited to four items")
        for item in recent:
            _validate_text(item, "recent context item", 300)
        object.__setattr__(self, "recent_context", recent)
        if self.campaign_context_summary is not None:
            _validate_text(
                self.campaign_context_summary,
                "campaign context summary",
                500,
            )


@dataclass(frozen=True)
class SupervisorInsight:
    """Versioned advisory result containing neither reasoning nor authority."""

    call_id: str
    source_turn_id: str
    source_turn_sequence: int
    prospect_evidence: ProspectEvidence | None = None
    recommended_strategy_hint: ConversationStrategyHint | None = None

    def __post_init__(self) -> None:
        _validate_identity(self.call_id, "call id")
        _validate_identity(self.source_turn_id, "source turn id")
        if self.source_turn_sequence < 0:
            raise ValueError("source turn sequence must be non-negative")
        if self.prospect_evidence is None and self.recommended_strategy_hint is None:
            raise ValueError("supervisor insight requires evidence or a strategy hint")
        if self.prospect_evidence is not None:
            if not isinstance(self.prospect_evidence, ProspectEvidence):
                raise TypeError("prospect_evidence has an invalid type")
            if self.prospect_evidence.observed is not None:
                raise ValueError("supervisor cannot emit observed prospect facts")
            inferred = self.prospect_evidence.inferred
            if inferred is None or inferred.source_turn_id != self.source_turn_id:
                raise ValueError("supervisor inference must match its source turn")
        if self.recommended_strategy_hint is not None and not isinstance(
            self.recommended_strategy_hint, ConversationStrategyHint
        ):
            raise TypeError("recommended_strategy_hint has an invalid type")


def _validate_identity(value: str, name: str) -> None:
    if not value or len(value) > 100:
        raise ValueError(f"{name} must contain 1-100 characters")


def _validate_text(value: str, name: str, limit: int) -> None:
    if not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must contain 1-{limit} characters")
