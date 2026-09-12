"""Bounded immutable input and output contracts for advisory supervision."""

from __future__ import annotations

from dataclasses import dataclass

from app.conversation.context.contracts import SupervisorContextView
from app.conversation.prospect_intelligence.contracts import ProspectEvidence
from app.conversation.strategy.contracts import ConversationStrategyHint


@dataclass(frozen=True)
class SupervisorInput:
    """One completed turn's bounded advisory-analysis input."""

    call_id: str
    turn_id: str
    source_turn_sequence: int
    context: SupervisorContextView

    def __post_init__(self) -> None:
        _validate_identity(self.call_id, "call id")
        _validate_identity(self.turn_id, "turn id")
        if self.source_turn_sequence < 0:
            raise ValueError("source turn sequence must be non-negative")
        if not isinstance(self.context, SupervisorContextView):
            raise TypeError("context must be a SupervisorContextView")


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
