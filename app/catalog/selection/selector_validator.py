"""Deterministic validation of service-selection proposals.

Ye woh authority hai jo selector (LLM) ke UNTRUSTED proposal ko authorize karti
hai. Locked rule: "LLM service SUGGEST kar sakta hai, AUTHORIZE nahi."

PRIMARY AUTHORITY = eligibility. Agar proposed id fresh eligible_service_ids mein
nahi, wahin reject — isse LLM kabhi catalog se bahar service inject nahi kar
sakta.

Order (guide ke mutabiq):
    proposed_id ∈ fresh eligible_service_ids   (invent-guard, PRIMARY)
      → not already offered
      → confidence valid
      → campaign authorization sanity-check (ClaimValidator)
      → APPROVED

Selector ko claims/pricing/guarantees ka koi concern nahi — woh pitch stage.
"""

from __future__ import annotations

from app.catalog.claim_validator import ClaimValidator
from app.catalog.selection.contracts import (
    SelectionOutcome,
    SelectionResult,
    ServiceSelectionInput,
    ServiceSelectionProposal,
)


class SelectorValidator:
    """Deterministically authorizes a selector's proposal.

    Attributes:
        claim_validator: For campaign authorization sanity-check.
        min_confidence: Minimum relevance confidence to accept.
    """

    def __init__(
        self, claim_validator: ClaimValidator, min_confidence: float = 0.5
    ) -> None:
        """Initialize with a claim validator and confidence threshold.

        Args:
            claim_validator: ClaimValidator for the active campaign.
            min_confidence: Minimum selection_confidence to approve.
        """
        self._claims = claim_validator
        self._min_confidence = min_confidence

    def validate(
        self,
        proposal: ServiceSelectionProposal,
        selection_input: ServiceSelectionInput,
    ) -> SelectionResult:
        """Authorize a proposal against fresh eligibility (primary) + checks.

        Args:
            proposal: The selector's untrusted proposal.
            selection_input: The trusted input (fresh eligible ids etc.).

        Returns:
            SelectionResult: APPROVED with id, ya a rejection outcome.
        """
        eligible = set(selection_input.eligible_service_ids)

        # 0. Nothing eligible → selector should not have run.
        if not eligible:
            return SelectionResult(
                SelectionOutcome.NO_ELIGIBLE,
                reason="no eligible services",
            )

        sid = proposal.proposed_service_id

        # 1. PRIMARY: id must be in FRESH eligible set (invent-guard). LLM catalog
        #    se bahar / invented / campaign-mismatch id yahin girta hai.
        if sid not in eligible:
            return SelectionResult(
                SelectionOutcome.NOT_ELIGIBLE,
                reason=f"proposed id not in fresh eligible set: {sid}",
            )

        # 2. Not already offered (defensive — eligible already excludes offered,
        #    lekin double-guard).
        if sid in set(selection_input.offered_service_ids):
            return SelectionResult(
                SelectionOutcome.ALREADY_OFFERED,
                reason=f"service already offered: {sid}",
            )

        # 3. Confidence valid.
        if proposal.selection_confidence < self._min_confidence:
            return SelectionResult(
                SelectionOutcome.LOW_CONFIDENCE,
                reason=f"confidence {proposal.selection_confidence} < {self._min_confidence}",
            )

        # 4. Campaign authorization sanity-check (ClaimValidator).
        service_check = self._claims.validate_service(sid)
        if not service_check.authorized:
            return SelectionResult(
                SelectionOutcome.UNAUTHORIZED,
                reason=f"campaign authorization failed: {service_check.category.value}",
            )

        return SelectionResult(
            SelectionOutcome.APPROVED, approved_service_id=sid
        )