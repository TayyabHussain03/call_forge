"""Immutable contracts for the Lead Outcome Export Writer.

Export facts, do not infer facts. Stable identity beats fuzzy matching.
Customer outcome and execution status stay separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExportStatus(str, Enum):
    """High-level outcome of an export operation."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    SOURCE_FILE_NOT_FOUND = "source_file_not_found"
    UNSUPPORTED_FORMAT = "unsupported_format"
    INVALID_WORKBOOK = "invalid_workbook"
    WRITE_FAILED = "write_failed"


class RowMatchStatus(str, Enum):
    """Per-row match outcome."""

    MATCHED = "matched"
    UNMATCHED = "unmatched"
    AMBIGUOUS = "ambiguous"
    DUPLICATE_OUTCOME = "duplicate_outcome"


class MultipleOutcomePolicy(str, Enum):
    """How to resolve multiple outcomes for one lead."""

    LATEST_TRUSTED_OUTCOME = "latest_trusted_outcome"
    FAIL_ON_AMBIGUITY = "fail_on_ambiguity"


class OutputFormat(str, Enum):
    """Supported export formats."""

    XLSX = "xlsx"
    CSV = "csv"


# Deterministic result column names — reused across XLSX and CSV
RESULT_COLUMNS: tuple[str, ...] = (
    "Outcome Status",
    "Interest Level",
    "DNC Status",
    "Execution Status",
    "Termination Reason",
    "Relevant Services",
    "Identified Problems",
    "Objections",
    "Decision Maker Status",
    "Qualification Completeness",
    "Next Step",
    "Follow Up Needed",
    "Outstanding Questions",
    "Conversation Summary",
    "Session ID",
    "Campaign ID",
)

# Deterministic delimiter for serialized collections
COLLECTION_DELIMITER = " | "


@dataclass(frozen=True)
class ExportWarning:
    """One warning encountered during export."""

    row_number: int | None
    lead_id: str | None
    code: str
    detail: str

    def __post_init__(self) -> None:
        if not self.code or not self.code.strip():
            raise ValueError("warning code must not be empty")
        if not self.detail or not self.detail.strip():
            raise ValueError("warning detail must not be empty")


@dataclass(frozen=True)
class LeadOutcomeExportResult:
    """Immutable result of one export run."""

    export_status: ExportStatus
    output_path: str | None = None
    output_format: OutputFormat | None = None
    rows_total: int = 0
    rows_updated: int = 0
    rows_unmatched: int = 0
    ambiguous_matches: int = 0
    duplicate_outcomes: int = 0
    warnings: tuple[ExportWarning, ...] = ()
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.rows_total < 0 or self.rows_updated < 0:
            raise ValueError("row counts must be non-negative")
        if self.export_status in (
            ExportStatus.SUCCESS, ExportStatus.PARTIAL_SUCCESS
        ) and not self.output_path:
            raise ValueError("successful export must include output_path")
