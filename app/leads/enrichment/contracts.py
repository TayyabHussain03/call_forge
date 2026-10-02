"""Provider-neutral enrichment boundary contracts.

Enrichment information is UNTRUSTED until reconciled. No provider SDK is
required in this slice.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from app.leads.contracts import EnrichmentProvenance


# ---------------------------------------------------------------------------
# Enrichment input — bounded lead identity only
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeadEnrichmentInput:
    """Only bounded lead identity for enrichment. Never conversation or Brain state."""

    business_name: str
    normalized_phone: str | None
    city: str | None = None
    state: str | None = None
    country: str | None = None
    website: str | None = None

    def __post_init__(self) -> None:
        if not self.business_name or not self.business_name.strip():
            raise ValueError("business name must not be empty")


# ---------------------------------------------------------------------------
# Enrichment result — candidate fields with provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnrichmentCandidateField:
    """A single candidate field from an enrichment provider."""

    value: str
    provenance: EnrichmentProvenance

    def __post_init__(self) -> None:
        if not self.value or not self.value.strip():
            raise ValueError("enrichment candidate value must not be empty")
        if not isinstance(self.provenance, EnrichmentProvenance):
            raise TypeError("provenance must be typed")


@dataclass(frozen=True)
class LeadEnrichmentResult:
    """Enrichment candidate data — all fields optional and untrusted."""

    provider_name: str
    success: bool

    website: EnrichmentCandidateField | None = None
    category: EnrichmentCandidateField | None = None
    address: EnrichmentCandidateField | None = None
    city: EnrichmentCandidateField | None = None
    state: EnrichmentCandidateField | None = None
    country: EnrichmentCandidateField | None = None
    email: EnrichmentCandidateField | None = None
    business_profile_url: EnrichmentCandidateField | None = None
    business_status: EnrichmentCandidateField | None = None

    error_message: str | None = None

    def __post_init__(self) -> None:
        if not self.provider_name or not self.provider_name.strip():
            raise ValueError("provider name must not be empty")


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------


class LeadEnrichmentProvider(abc.ABC):
    """Provider-neutral enrichment boundary."""

    @abc.abstractmethod
    def enrich(self, input: LeadEnrichmentInput) -> LeadEnrichmentResult:
        """Return candidate public-business information."""


# ---------------------------------------------------------------------------
# Mock provider for testing
# ---------------------------------------------------------------------------


class MockLeadEnrichmentProvider(LeadEnrichmentProvider):
    """Deterministic, scripted mock provider for offline tests."""

    def __init__(
        self,
        results: dict[str, LeadEnrichmentResult] | None = None,
        default_result: LeadEnrichmentResult | None = None,
    ) -> None:
        self._results = results or {}
        self._default = default_result or LeadEnrichmentResult(
            provider_name="mock",
            success=False,
            error_message="no enrichment configured",
        )

    def enrich(self, input: LeadEnrichmentInput) -> LeadEnrichmentResult:
        key = input.normalized_phone or input.business_name
        return self._results.get(key, self._default)
