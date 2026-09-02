"""Service selection orchestration.

Poora flow tie karta hai: fresh eligible ids → (0/1/2+ branching) → select →
validate → offered-service update. Ye deterministic orchestration hai (mock
selector ke saath testable); engine baad mein isse wire karega.

BRANCHING (locked decision B):
    0 eligible → NO_ELIGIBLE (safe close; selector call NAHI)
    1 eligible → deterministic pick → validate
    2+ eligible → ServiceSelector → validate

Selector failure → SELECTOR_FAILED (caller deterministic fallback le).
"""

from __future__ import annotations

from app.catalog.selection.contracts import (
    SelectionOutcome,
    SelectionResult,
    ServiceSelectionInput,
)
from app.catalog.selection.selector import (
    SelectorError,
    ServiceSelector,
    deterministic_single_pick,
)
from app.catalog.selection.selector_validator import SelectorValidator


class ServiceSelectionService:
    """Orchestrates deterministic-or-LLM selection with authorization.

    Attributes:
        selector: The (mock/LLM) service selector for 2+ eligible.
        validator: The deterministic SelectorValidator (authority).
    """

    def __init__(
        self, selector: ServiceSelector, validator: SelectorValidator
    ) -> None:
        """Initialize with a selector and validator.

        Args:
            selector: Used only when 2+ services are eligible.
            validator: Deterministic authorization.
        """
        self._selector = selector
        self._validator = validator

    def select_and_authorize(
        self, selection_input: ServiceSelectionInput
    ) -> SelectionResult:
        """Run the full selection pipeline and return an authorized result.

        Args:
            selection_input: Trusted input with FRESH eligible ids.

        Returns:
            SelectionResult: APPROVED with id, ya a rejection/failure outcome.
        """
        eligible = selection_input.eligible_service_ids

        # 0 eligible → no selector, safe close.
        if not eligible:
            return SelectionResult(
                SelectionOutcome.NO_ELIGIBLE, reason="no eligible services"
            )

        # 1 eligible → deterministic pick (no LLM).
        single = deterministic_single_pick(selection_input)
        if single is not None:
            return self._validator.validate(single, selection_input)

        # 2+ eligible → selector (mock/LLM). Failure → deterministic fallback.
        try:
            proposal = self._selector.select(selection_input)
        except SelectorError as exc:
            return SelectionResult(
                SelectionOutcome.SELECTOR_FAILED, reason=str(exc)
            )

        return self._validator.validate(proposal, selection_input)