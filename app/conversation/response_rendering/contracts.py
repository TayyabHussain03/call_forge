"""Minimal immutable contracts for grounded response rendering."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.conversation.response_planning.contracts import (
    AuthoritativeResultKind,
    ConversationMove,
    ResponseLength,
    ResponsePlan,
)


class ContactConfirmationStatus(str, Enum):
    """Trusted contact status used only to choose accurate wording."""

    NONE = "none"
    UNCONFIRMED = "unconfirmed"
    CONFIRMED = "confirmed"


@dataclass(frozen=True)
class ResponseRenderingBudget:
    """Authoritative per-response resource ceiling, never mutable by rendering."""

    max_sentences: int
    max_characters: int = 800

    def __post_init__(self) -> None:
        if self.max_sentences < 1:
            raise ValueError("max_sentences must be positive")
        if self.max_characters < 40:
            raise ValueError("max_characters must be at least 40")


@dataclass(frozen=True)
class TrustedRenderingContext:
    """Bounded facts approved by upstream application/catalog components."""

    primary_fact: str | None = None
    service_name: str | None = None
    service_facts: tuple[str, ...] = ()
    contact_status: ContactConfirmationStatus = ContactConfirmationStatus.NONE
    role_label: str | None = None

    def __post_init__(self) -> None:
        values = (self.primary_fact, self.service_name, self.role_label)
        if any(value is not None and len(value) > 160 for value in values):
            raise ValueError("trusted rendering value exceeds bounded length")
        if len(self.service_facts) > 4 or any(
            not fact.strip() or len(fact) > 200 for fact in self.service_facts
        ):
            raise ValueError("service facts must contain at most four bounded facts")


@dataclass(frozen=True)
class ResponseRenderInput:
    """Provider-independent input downstream of authoritative decisions."""

    plan: ResponsePlan
    authoritative_result: AuthoritativeResultKind
    budget: ResponseRenderingBudget
    trusted_context: TrustedRenderingContext = TrustedRenderingContext()
    variation_seed: str = "default"

    def __post_init__(self) -> None:
        if len(self.variation_seed) > 80:
            raise ValueError("variation seed exceeds bounded length")


@dataclass(frozen=True)
class RenderedResponse:
    """Accepted conversational text plus communication-only provenance."""

    text: str
    communicative_goal: ConversationMove
    length_class: ResponseLength
    clarification_required: bool
