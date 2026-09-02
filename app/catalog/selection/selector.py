"""Service selector interface and deterministic/mock implementations.

Locked design: selector interface + mock provider — real Gemini/LLM baad mein
replace/add hoga bina engine rewrite ke (jaise LLMProvider pattern).

SELECTION STRATEGY (locked decision B):
    0 eligible → koi selector nahi (safe close upstream)
    1 eligible → DETERMINISTIC pick (LLM unnecessary — no cost/latency)
    2+ eligible → ServiceSelector (LLM/mock) ranks relevance

Selector RELEVANCE deta hai — authorization nahi (woh SelectorValidator).
Selector failure/timeout → SelectorError raise; caller deterministic fallback le.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.catalog.selection.contracts import (
    ServiceSelectionInput,
    ServiceSelectionProposal,
)


class SelectorError(Exception):
    """Raised when a selector fails/times out — caller falls back deterministically."""


class ServiceSelector(ABC):
    """Interface for selecting a relevant service from eligible ones.

    Real Gemini provider isi interface ko implement karega. Selector sirf
    relevance propose karta hai; authorization SelectorValidator karta hai.
    """

    @abstractmethod
    def select(
        self, selection_input: ServiceSelectionInput
    ) -> ServiceSelectionProposal:
        """Propose the most relevant eligible service.

        Args:
            selection_input: Trusted input with fresh eligible ids.

        Returns:
            ServiceSelectionProposal: Untrusted proposal (validated downstream).

        Raises:
            SelectorError: On failure/timeout.
        """
        raise NotImplementedError


def deterministic_single_pick(
    selection_input: ServiceSelectionInput,
) -> ServiceSelectionProposal | None:
    """Deterministically pick when exactly one service is eligible.

    Locked decision B: exactly-one-eligible → deterministic pick (no LLM). Ye
    single-option case ko bulletproof deterministic banata hai.

    Args:
        selection_input: The trusted input.

    Returns:
        ServiceSelectionProposal | None: A full-confidence proposal for the single
        eligible id, ya None agar count != 1 (caller selector use kare).
    """
    if len(selection_input.eligible_service_ids) == 1:
        return ServiceSelectionProposal(
            proposed_service_id=selection_input.eligible_service_ids[0],
            selection_confidence=1.0,
            reason="single eligible service (deterministic pick)",
        )
    return None


class MockServiceSelector(ServiceSelector):
    """Deterministic mock selector for testing (no LLM).

    Configurable behaviour taake saare cases test ho sakein: default pehla eligible
    id high-confidence, ya explicit override (invalid/low-confidence/fail).

    Attributes:
        forced_id: Agar set, ye id propose karta hai (invalid-id test ke liye).
        forced_confidence: Confidence override.
        should_fail: True → SelectorError raise (failure/timeout test).
    """

    def __init__(
        self,
        forced_id: str | None = None,
        forced_confidence: float = 0.9,
        should_fail: bool = False,
    ) -> None:
        """Initialize the mock selector.

        Args:
            forced_id: Force this proposed id (else first eligible).
            forced_confidence: Confidence to report.
            should_fail: If True, raise SelectorError.
        """
        self._forced_id = forced_id
        self._forced_confidence = forced_confidence
        self._should_fail = should_fail

    def select(
        self, selection_input: ServiceSelectionInput
    ) -> ServiceSelectionProposal:
        """Return a deterministic proposal (or fail).

        Args:
            selection_input: Trusted input.

        Returns:
            ServiceSelectionProposal: The mock proposal.

        Raises:
            SelectorError: If configured to fail.
        """
        if self._should_fail:
            raise SelectorError("mock selector configured to fail")
        if self._forced_id is not None:
            chosen = self._forced_id
        elif selection_input.eligible_service_ids:
            chosen = selection_input.eligible_service_ids[0]
        else:
            chosen = ""  # empty → validator will NOT_ELIGIBLE
        return ServiceSelectionProposal(
            proposed_service_id=chosen,
            selection_confidence=self._forced_confidence,
            reason="mock selection",
        )