"""Service offering capability for the conversation engine.

Engine ko catalog/eligibility/selector internals se SHIELD karta hai. Engine sirf
`offer(...)` call karta hai; ye service fresh eligibility compute karke selection+
authorization run karta hai aur ek clean `OfferResult` deta hai.

BOUNDARY (locked): engine ko ScopedCatalog, compute_eligible_alternatives,
selector implementation, ya Gemini/mock ka pata NAHI. Future real-Gemini swap
sirf yahan hota hai.

AUTHORITY SOURCE (locked): non-repetition `offered_service_ids` par based hai
(actual approved services). `service_offers_made` sirf observability — eligibility
usse affect NAHI hoti.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.catalog.eligibility import compute_eligible_alternatives
from app.catalog.scoped_catalog import ScopedCatalog
from app.catalog.selection.contracts import (
    SelectionOutcome,
    SelectionResult,
    ServiceSelectionInput,
)
from app.catalog.selection.selection_service import ServiceSelectionService


@dataclass(frozen=True)
class OfferResult:
    """The result of an offering attempt (engine-facing).

    Attributes:
        approved_service_id: The authorized service to offer (offer possible), ya
            None (no offer — close/fallback).
        selection: The underlying SelectionResult (traceability).
        eligible_service_ids: The FRESH eligible ids computed this attempt (engine
            context update ke liye).
    """

    approved_service_id: str | None
    selection: SelectionResult
    eligible_service_ids: tuple[str, ...]

    @property
    def can_offer(self) -> bool:
        """Whether an offer was authorized.

        Returns:
            bool: True if an approved service is available.
        """
        return self.approved_service_id is not None


class ServiceOfferingService:
    """Encapsulates fresh eligibility + selection + authorization.

    Attributes:
        catalog: The campaign-scoped catalog.
        selection: The selection+authorization service.
        max_offers: Configured max offers (bounded offering).
    """

    def __init__(
        self,
        scoped_catalog: ScopedCatalog,
        selection_service: ServiceSelectionService,
        max_offers: int = 3,
    ) -> None:
        """Initialize with a scoped catalog, selection service, and max offers.

        Args:
            scoped_catalog: Campaign-scoped catalog.
            selection_service: Selection + authorization.
            max_offers: Max offers allowed this call.
        """
        self._catalog = scoped_catalog
        self._selection = selection_service
        self._max_offers = max_offers

    def offer(
        self,
        campaign_id: str,
        known_signals: frozenset[str],
        offered_service_ids: tuple[str, ...],
        offers_made: int,
    ) -> OfferResult:
        """Compute fresh eligibility, select, and authorize an offer.

        Non-repetition authority = `offered_service_ids` (approved services). The
        `offers_made` count is used only for the bounded-offering ceiling, not for
        eligibility identity.

        Args:
            campaign_id: Active campaign.
            known_signals: Detected eligibility signals.
            offered_service_ids: Already-approved-offered services (authority).
            offers_made: Offers made so far (observability/ceiling).

        Returns:
            OfferResult: approved id + selection + fresh eligible ids.
        """
        # FRESH eligibility — every attempt recomputed (no stale).
        eligible = compute_eligible_alternatives(
            self._catalog, known_signals, offered_service_ids,
            offers_made, self._max_offers,
        )

        selection_input = ServiceSelectionInput(
            eligible_service_ids=eligible,
            campaign_id=campaign_id,
            known_signals=known_signals,
            offered_service_ids=offered_service_ids,
        )
        result = self._selection.select_and_authorize(selection_input)

        approved = (
            result.approved_service_id
            if result.outcome == SelectionOutcome.APPROVED
            else None
        )
        return OfferResult(
            approved_service_id=approved,
            selection=result,
            eligible_service_ids=eligible,
        )