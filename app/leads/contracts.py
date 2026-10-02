"""Immutable contracts for Lead Intake, Enrichment & Pre-Call Context.

TRUST BOUNDARY: imported lead data is UNTRUSTED user input. Enrichment data
is UNTRUSTED external data. Both must be normalized, validated, and reconciled
before entering the sales intelligence pipeline.

Missing ≠ False. Blank website ≠ no website. Lead data provides context;
conversation discovers the problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class WebsiteStatus(str, Enum):
    """Explicit representation of website knowledge state."""

    UNKNOWN = "unknown"
    VERIFIED_PRESENT = "verified_present"
    VERIFIED_ABSENT = "verified_absent"
    UNVERIFIED_CANDIDATE = "unverified_candidate"
    INVALID = "invalid"


class PhoneNormalizationStatus(str, Enum):
    """Result of phone normalization attempt."""

    VALID_E164 = "valid_e164"
    NEEDS_REVIEW = "needs_review"
    INVALID = "invalid"


class LeadImportReadiness(str, Enum):
    """Import-level readiness only, not sales qualification."""

    READY_FOR_CONTACT = "ready_for_contact"
    NEEDS_REVIEW = "needs_review"
    INVALID = "invalid"


class DuplicateClassification(str, Enum):
    """How a lead relates to existing records."""

    UNIQUE = "unique"
    DUPLICATE_EXISTING = "duplicate_existing"
    DUPLICATE_IN_BATCH = "duplicate_in_batch"


class FieldKnowledgeState(str, Enum):
    """Whether a field value is known, unknown, conflicting, or unverified."""

    KNOWN = "known"
    UNKNOWN = "unknown"
    CONFLICTING = "conflicting"
    UNVERIFIED = "unverified"


class EnrichmentFieldSource(str, Enum):
    """Provenance priority for reconciliation."""

    USER_PROVIDED = "user_provided"
    TRUSTED_EXISTING_RECORD = "trusted_existing_record"
    ENRICHMENT_CANDIDATE = "enrichment_candidate"
    UNKNOWN = "unknown"


class ImportRowStatus(str, Enum):
    """Status of an individual row during import."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"


# ---------------------------------------------------------------------------
# Phone normalization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizedPhone:
    """Phone normalization result preserving raw input."""

    raw: str
    normalized: str | None
    status: PhoneNormalizationStatus
    country_context: str | None = None

    def __post_init__(self) -> None:
        if not self.raw or not self.raw.strip():
            raise ValueError("raw phone must not be empty")
        if self.status == PhoneNormalizationStatus.VALID_E164 and not self.normalized:
            raise ValueError("valid E.164 status requires a normalized value")


# ---------------------------------------------------------------------------
# Lead record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeadRecord:
    """Immutable lead record after import and normalization."""

    lead_id: str
    tenant_id: str
    campaign_id: str

    business_name: str
    phone: NormalizedPhone

    website: str | None = None
    website_status: WebsiteStatus = WebsiteStatus.UNKNOWN

    category: str | None = None
    city: str | None = None
    state: str | None = None
    country: str | None = None
    address: str | None = None

    email: str | None = None
    contact_name: str | None = None

    source: str | None = None
    notes: str | None = None

    import_batch_id: str | None = None
    row_number: int | None = None

    readiness: LeadImportReadiness = LeadImportReadiness.READY_FOR_CONTACT
    duplicate_status: DuplicateClassification = DuplicateClassification.UNIQUE

    def __post_init__(self) -> None:
        if not self.lead_id or not self.lead_id.strip():
            raise ValueError("lead_id must not be empty")
        if not self.tenant_id or not self.tenant_id.strip():
            raise ValueError("tenant_id must not be empty")
        if not self.campaign_id or not self.campaign_id.strip():
            raise ValueError("campaign_id must not be empty")
        if not self.business_name or not self.business_name.strip():
            raise ValueError("business_name must not be empty")
        if not isinstance(self.phone, NormalizedPhone):
            raise TypeError("phone must be a NormalizedPhone")
        if not isinstance(self.website_status, WebsiteStatus):
            raise TypeError("website_status must be typed")
        if self.phone.status == PhoneNormalizationStatus.NEEDS_REVIEW:
            if self.readiness == LeadImportReadiness.READY_FOR_CONTACT:
                object.__setattr__(self, "readiness", LeadImportReadiness.NEEDS_REVIEW)
        if self.phone.status == PhoneNormalizationStatus.INVALID:
            object.__setattr__(self, "readiness", LeadImportReadiness.INVALID)


# ---------------------------------------------------------------------------
# Import result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RejectedRow:
    """A row that failed import validation."""

    row_number: int
    reason: str
    raw_data: dict[str, str]

    def __post_init__(self) -> None:
        if self.row_number < 1:
            raise ValueError("row number must be positive")
        if not self.reason or not self.reason.strip():
            raise ValueError("rejection reason must not be empty")


@dataclass(frozen=True)
class ImportWarning:
    """A non-fatal warning during import."""

    row_number: int
    field: str
    message: str


@dataclass(frozen=True)
class ImportStatistics:
    """Summary statistics for an import batch."""

    total_rows: int
    accepted: int
    rejected: int
    duplicates: int

    def __post_init__(self) -> None:
        if self.total_rows < 0 or self.accepted < 0 or self.rejected < 0 or self.duplicates < 0:
            raise ValueError("import statistics must be non-negative")
        if self.accepted + self.rejected + self.duplicates != self.total_rows:
            raise ValueError("statistics must sum to total rows")


@dataclass(frozen=True)
class LeadImportResult:
    """Complete result of a lead file import."""

    batch_id: str
    accepted: tuple[LeadRecord, ...]
    rejected: tuple[RejectedRow, ...]
    duplicates: tuple[LeadRecord, ...]
    warnings: tuple[ImportWarning, ...]
    statistics: ImportStatistics

    def __post_init__(self) -> None:
        if not self.batch_id or not self.batch_id.strip():
            raise ValueError("batch_id must not be empty")


# ---------------------------------------------------------------------------
# Enrichment provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnrichmentProvenance:
    """Provenance for a single enrichment field."""

    source_provider: str
    source_reference: str | None = None
    retrieved_at: str | None = None
    confidence_category: str | None = None

    def __post_init__(self) -> None:
        if not self.source_provider or not self.source_provider.strip():
            raise ValueError("source provider must not be empty")


@dataclass(frozen=True)
class EnrichedField:
    """A single enrichment field with its provenance and knowledge state."""

    value: str | None
    state: FieldKnowledgeState
    source: EnrichmentFieldSource
    provenance: EnrichmentProvenance | None = None

    def __post_init__(self) -> None:
        if self.value is not None and self.state == FieldKnowledgeState.UNKNOWN:
            raise ValueError("a field with a value cannot be unknown")
        if self.state == FieldKnowledgeState.KNOWN and self.value is None:
            raise ValueError("a known field must have a value")


# ---------------------------------------------------------------------------
# Pre-call business context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreCallBusinessContext:
    """Immutable pre-call context for the sales intelligence pipeline.

    This is what the pipeline receives before a conversation begins. It carries
    no service selection, sales diagnosis, language inference, or execution
    authority.
    """

    lead_id: str
    tenant_id: str
    campaign_id: str

    business_name: str
    phone: NormalizedPhone

    website_status: WebsiteStatus
    website: EnrichedField

    category: EnrichedField
    city: EnrichedField
    state: EnrichedField
    country: EnrichedField
    address: EnrichedField

    contact_name: EnrichedField
    email: EnrichedField

    lead_source: str | None
    campaign_context: str | None = None
    import_batch_id: str | None = None

    def __post_init__(self) -> None:
        if not self.lead_id or not self.lead_id.strip():
            raise ValueError("lead_id must not be empty")
        if not self.tenant_id or not self.tenant_id.strip():
            raise ValueError("tenant_id must not be empty")
        if not self.campaign_id or not self.campaign_id.strip():
            raise ValueError("campaign_id must not be empty")
        if not self.business_name or not self.business_name.strip():
            raise ValueError("business_name must not be empty")
        if not isinstance(self.phone, NormalizedPhone):
            raise TypeError("phone must be a NormalizedPhone")
        if not isinstance(self.website_status, WebsiteStatus):
            raise TypeError("website_status must be typed")
        if not isinstance(self.website, EnrichedField):
            raise TypeError("website must be an enriched field")
