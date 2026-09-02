"""Contracts for service selection.

Boundary (locked): LLM/selector SUGGESTS, catalog AUTHORIZES. Selector ko sirf
fresh eligible ids milte hain (poora catalog nahi), aur uska output UNTRUSTED
proposal hai jo SelectorValidator deterministically authorize karta hai.

confidence semantics: `selection_confidence` sirf selector ki RELEVANCE confidence
hai — contact/intent/STT confidence ya lead quality se ALAG. Kabhi mix nahi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True)
class ServiceSelectionInput:
    """Trusted input the engine hands to a selector.

    Selector ko SIRF eligible ids milte hain — poora catalog nahi. Isse LLM ka
    scope structurally bounded hai (catalog se bahar ja hi nahi sakta).

    Attributes:
        eligible_service_ids: Fresh, catalog-computed, non-repeated eligible ids.
        campaign_id: Active campaign (for context).
        known_signals: Detected eligibility signals.
        offered_service_ids: Already-offered ids (for context; also excluded from
            eligible upstream).
    """

    eligible_service_ids: tuple[str, ...]
    campaign_id: str
    known_signals: frozenset[str] = field(default_factory=frozenset)
    offered_service_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ServiceSelectionProposal:
    """A selector's UNTRUSTED proposal.

    Sirf teen fields — kaunsa id relevant, kitna confident, kyun. Koi claims/
    pricing/wording NAHI. Final authorization SelectorValidator karta hai.

    Attributes:
        proposed_service_id: The suggested service id (untrusted — may be invalid/
            invented; validator catches it).
        selection_confidence: Relevance confidence (0.0–1.0). Relevance only.
        reason: Free-form rationale — audit/debug only, NON-authoritative.
    """

    proposed_service_id: str
    selection_confidence: float = 0.0
    reason: str | None = None


class SelectionOutcome(str, Enum):
    """Outcome of validating a service selection."""

    APPROVED = "approved"                    # id authorized for OFFER_SERVICE
    NOT_ELIGIBLE = "not_eligible"            # id not in fresh eligible set (invent-guard)
    ALREADY_OFFERED = "already_offered"      # id already offered this call
    LOW_CONFIDENCE = "low_confidence"        # below threshold
    UNAUTHORIZED = "unauthorized"            # campaign authorization sanity-check failed
    NO_ELIGIBLE = "no_eligible"              # nothing eligible → selector shouldn't run
    SELECTOR_FAILED = "selector_failed"      # selector error/timeout


@dataclass(frozen=True)
class SelectionResult:
    """The deterministic result of validating a selection.

    Attributes:
        outcome: The validation outcome.
        approved_service_id: The authorized id (APPROVED only), ya None.
        reason: Machine-readable detail (debug).
    """

    outcome: SelectionOutcome
    approved_service_id: str | None = None
    reason: str | None = None

    @property
    def is_approved(self) -> bool:
        """Whether the selection was approved.

        Returns:
            bool: True only for APPROVED.
        """
        return self.outcome == SelectionOutcome.APPROVED