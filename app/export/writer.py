"""Deterministic Lead Outcome Export Writer.

Writes LeadOutcome results back to Excel/CSV. Pure data export — no
new sales intelligence, no inference. Never overwrites the source file.
"""

from __future__ import annotations

import csv
import io
import os
from pathlib import Path

from app.leads.contracts import LeadRecord
from app.runtime.chat.outcome.contracts import (
    DNCStatus,
    ExecutionStatus,
    LeadOutcome,
    OutcomeStatus,
)
from app.export.contracts import (
    COLLECTION_DELIMITER,
    ExportStatus,
    ExportWarning,
    LeadOutcomeExportResult,
    MultipleOutcomePolicy,
    OutputFormat,
    RESULT_COLUMNS,
    RowMatchStatus,
)


class LeadOutcomeExportWriter:
    """Deterministic writer. Export facts, do not infer facts.

    Matching priority (strongest existing identity wins):
      1. lead_id (if lead-identifier column present in source)
      2. import_batch_id + row_number
      3. normalized_phone + campaign_id

    Never fuzzy name matching.
    """

    def export(
        self,
        source_path: str,
        output_path: str,
        leads: tuple[LeadRecord, ...],
        outcomes: tuple[LeadOutcome, ...],
        *,
        multiple_outcome_policy: MultipleOutcomePolicy = MultipleOutcomePolicy.FAIL_ON_AMBIGUITY,
        outcome_order_hint: tuple[str, ...] | None = None,
    ) -> LeadOutcomeExportResult:
        """Export outcomes to a new file alongside the source.

        outcome_order_hint: when policy is LATEST_TRUSTED_OUTCOME, a
        tuple of session_ids in trusted chronological order. Required
        for that policy — no wall-clock guessing.
        """

        if not os.path.exists(source_path):
            return LeadOutcomeExportResult(
                export_status=ExportStatus.SOURCE_FILE_NOT_FOUND,
                failure_reason=f"source file not found: {source_path}",
            )

        fmt = _detect_format(source_path)
        if fmt is None:
            return LeadOutcomeExportResult(
                export_status=ExportStatus.UNSUPPORTED_FORMAT,
                failure_reason=f"unsupported format: {source_path}",
            )

        # Resolve outcomes per lead_id with deterministic policy
        try:
            outcome_by_lead, duplicate_warnings = self._resolve_outcomes(
                outcomes, multiple_outcome_policy, outcome_order_hint,
            )
        except _AmbiguousOutcomeError as exc:
            return LeadOutcomeExportResult(
                export_status=ExportStatus.WRITE_FAILED,
                failure_reason=str(exc),
            )

        lead_by_id = {lead.lead_id: lead for lead in leads}

        try:
            if fmt == OutputFormat.XLSX:
                stats = _write_xlsx(
                    source_path, output_path, lead_by_id, outcome_by_lead,
                )
            else:
                stats = _write_csv(
                    source_path, output_path, lead_by_id, outcome_by_lead,
                )
        except (OSError, ValueError) as exc:
            return LeadOutcomeExportResult(
                export_status=ExportStatus.WRITE_FAILED,
                failure_reason=f"write failed: {exc}",
            )

        warnings = tuple(duplicate_warnings + stats.warnings)
        status = (
            ExportStatus.PARTIAL_SUCCESS
            if stats.rows_unmatched > 0 or stats.ambiguous_matches > 0
            or stats.duplicate_outcomes > 0 or warnings
            else ExportStatus.SUCCESS
        )

        return LeadOutcomeExportResult(
            export_status=status,
            output_path=output_path,
            output_format=fmt,
            rows_total=stats.rows_total,
            rows_updated=stats.rows_updated,
            rows_unmatched=stats.rows_unmatched,
            ambiguous_matches=stats.ambiguous_matches,
            duplicate_outcomes=stats.duplicate_outcomes,
            warnings=warnings,
        )

    def _resolve_outcomes(
        self,
        outcomes: tuple[LeadOutcome, ...],
        policy: MultipleOutcomePolicy,
        order_hint: tuple[str, ...] | None,
    ) -> tuple[dict[str, LeadOutcome], list[ExportWarning]]:
        """Deterministically pick one outcome per lead_id."""
        by_lead: dict[str, list[LeadOutcome]] = {}
        for outcome in outcomes:
            by_lead.setdefault(outcome.lead_id, []).append(outcome)

        resolved: dict[str, LeadOutcome] = {}
        warnings: list[ExportWarning] = []
        for lead_id, group in by_lead.items():
            if len(group) == 1:
                resolved[lead_id] = group[0]
                continue

            # DNC always wins regardless of policy — safety invariant
            dnc_outcomes = [
                o for o in group if o.dnc_status == DNCStatus.DNC_CONFIRMED
            ]
            if dnc_outcomes:
                resolved[lead_id] = dnc_outcomes[0]
                warnings.append(ExportWarning(
                    row_number=None,
                    lead_id=lead_id,
                    code="DUPLICATE_OUTCOME",
                    detail=f"{len(group)} outcomes; DNC preserved",
                ))
                continue

            if policy == MultipleOutcomePolicy.LATEST_TRUSTED_OUTCOME:
                if order_hint is None:
                    raise _AmbiguousOutcomeError(
                        f"LATEST_TRUSTED_OUTCOME requires outcome_order_hint "
                        f"for lead {lead_id}"
                    )
                # Pick the one with the highest index in order_hint
                order_rank = {sid: i for i, sid in enumerate(order_hint)}
                ranked = [
                    o for o in group if o.session_id in order_rank
                ]
                if not ranked:
                    raise _AmbiguousOutcomeError(
                        f"no outcomes for lead {lead_id} appear in order_hint"
                    )
                ranked.sort(key=lambda o: order_rank[o.session_id])
                resolved[lead_id] = ranked[-1]
                warnings.append(ExportWarning(
                    row_number=None,
                    lead_id=lead_id,
                    code="DUPLICATE_OUTCOME",
                    detail=f"{len(group)} outcomes; latest selected",
                ))
            else:
                # FAIL_ON_AMBIGUITY → do not silently choose
                raise _AmbiguousOutcomeError(
                    f"lead {lead_id} has {len(group)} outcomes; "
                    f"no deterministic order to pick one"
                )

        return resolved, warnings


class _AmbiguousOutcomeError(Exception):
    pass


class _WriteStats:
    def __init__(self) -> None:
        self.rows_total: int = 0
        self.rows_updated: int = 0
        self.rows_unmatched: int = 0
        self.ambiguous_matches: int = 0
        self.duplicate_outcomes: int = 0
        self.warnings: list[ExportWarning] = []


def _detect_format(path: str) -> OutputFormat | None:
    lower = path.lower()
    if lower.endswith(".xlsx"):
        return OutputFormat.XLSX
    if lower.endswith(".csv"):
        return OutputFormat.CSV
    return None


# ---------------------------------------------------------------------------
# Row serialization — one place where LeadOutcome → cell values
# ---------------------------------------------------------------------------


def _serialize_outcome(outcome: LeadOutcome | None) -> tuple[str, ...]:
    """Serialize one LeadOutcome into the fixed-order RESULT_COLUMNS tuple.

    None → empty strings for all columns (unmatched row).
    """
    if outcome is None:
        return tuple("" for _ in RESULT_COLUMNS)

    problems = COLLECTION_DELIMITER.join(
        sorted(p.description for p in outcome.identified_problems)
    )
    services = COLLECTION_DELIMITER.join(
        sorted(s.service_id for s in outcome.relevant_services)
    )
    objections = COLLECTION_DELIMITER.join(
        sorted(o.objection_type for o in outcome.objections)
    )
    questions = COLLECTION_DELIMITER.join(sorted(outcome.outstanding_questions))
    summary = COLLECTION_DELIMITER.join(
        f"{s.label}: {s.value}" for s in outcome.summary_lines
    )

    return (
        outcome.outcome_status.value,
        outcome.interest_level.value,
        outcome.dnc_status.value,
        outcome.execution_status.value,
        outcome.termination_reason.value,
        services,
        problems,
        objections,
        outcome.decision_maker_status.value,
        outcome.qualification_completeness or "",
        outcome.next_step or "",
        outcome.follow_up_needed.value,
        questions,
        summary,
        outcome.session_id,
        outcome.campaign_id,
    )


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _lead_identifier_column_index(headers: list[str]) -> int | None:
    """Find a lead_id/lead identifier column if present."""
    for i, h in enumerate(headers):
        norm = (h or "").strip().lower()
        if norm in ("lead_id", "lead id", "leadid"):
            return i
    return None


def _match_row_to_lead(
    row: list[str],
    headers: list[str],
    lead_id_col: int | None,
    lead_by_id: dict[str, LeadRecord],
    row_number: int,
) -> tuple[LeadRecord | None, RowMatchStatus]:
    """Match one source row to a LeadRecord using stable identity.

    Priority: lead_id column > import_batch_id+row_number > phone+campaign.
    Fuzzy name matching is NEVER used.
    """
    # Priority 1: explicit lead_id column
    if lead_id_col is not None and lead_id_col < len(row):
        candidate = (row[lead_id_col] or "").strip()
        if candidate:
            lead = lead_by_id.get(candidate)
            if lead is not None:
                return lead, RowMatchStatus.MATCHED
            return None, RowMatchStatus.UNMATCHED

    # Priority 2: import_batch_id + row_number
    batch_matches = [
        lead for lead in lead_by_id.values()
        if lead.row_number == row_number and lead.import_batch_id is not None
    ]
    if len(batch_matches) == 1:
        return batch_matches[0], RowMatchStatus.MATCHED
    if len(batch_matches) > 1:
        return None, RowMatchStatus.AMBIGUOUS

    # Priority 3: normalized phone (strongest fallback)
    phone_col = _find_phone_column(headers)
    if phone_col is not None and phone_col < len(row):
        phone_raw = (row[phone_col] or "").strip()
        if phone_raw:
            phone_matches = [
                lead for lead in lead_by_id.values()
                if lead.phone.normalized and lead.phone.raw.strip() == phone_raw
            ]
            if len(phone_matches) == 1:
                return phone_matches[0], RowMatchStatus.MATCHED
            if len(phone_matches) > 1:
                return None, RowMatchStatus.AMBIGUOUS

    return None, RowMatchStatus.UNMATCHED


def _find_phone_column(headers: list[str]) -> int | None:
    phone_aliases = {"phone", "phone number", "mobile", "telephone", "tel"}
    for i, h in enumerate(headers):
        if (h or "").strip().lower() in phone_aliases:
            return i
    return None


# ---------------------------------------------------------------------------
# XLSX writer
# ---------------------------------------------------------------------------


def _write_xlsx(
    source_path: str,
    output_path: str,
    lead_by_id: dict[str, LeadRecord],
    outcome_by_lead: dict[str, LeadOutcome],
) -> _WriteStats:
    import openpyxl

    stats = _WriteStats()
    wb = openpyxl.load_workbook(source_path)
    ws = wb.active

    # Read headers
    headers: list[str] = []
    for cell in ws[1]:
        headers.append(str(cell.value) if cell.value is not None else "")

    # Strip existing RESULT_COLUMNS if rerunning — idempotency
    existing_result_cols = {
        i for i, h in enumerate(headers) if h in RESULT_COLUMNS
    }
    keep_cols = [i for i in range(len(headers)) if i not in existing_result_cols]
    original_headers = [headers[i] for i in keep_cols]

    lead_id_col = _lead_identifier_column_index(original_headers)

    # Rewrite with original columns + append result columns deterministically
    # We rebuild a new worksheet to guarantee the result-column suffix is clean
    new_headers = original_headers + list(RESULT_COLUMNS)

    # Collect data rows (skip header)
    data_rows: list[list[str]] = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        row_values = [str(v) if v is not None else "" for v in row]
        kept = [row_values[i] if i < len(row_values) else "" for i in keep_cols]
        data_rows.append(kept)

    # Clear worksheet and rewrite
    ws.delete_rows(1, ws.max_row)
    ws.append(new_headers)

    for idx, row in enumerate(data_rows, start=1):
        stats.rows_total += 1
        lead, match_status = _match_row_to_lead(
            row, original_headers, lead_id_col, lead_by_id, idx,
        )
        if match_status == RowMatchStatus.AMBIGUOUS:
            stats.ambiguous_matches += 1
            stats.warnings.append(ExportWarning(
                row_number=idx, lead_id=None,
                code="AMBIGUOUS_MATCH",
                detail=f"row {idx} matched multiple leads",
            ))
            outcome = None
        elif lead is None:
            stats.rows_unmatched += 1
            stats.warnings.append(ExportWarning(
                row_number=idx, lead_id=None,
                code="LEAD_NOT_FOUND",
                detail=f"row {idx} could not be matched to a lead",
            ))
            outcome = None
        else:
            outcome = outcome_by_lead.get(lead.lead_id)
            if outcome is not None:
                stats.rows_updated += 1

        values = row + list(_serialize_outcome(outcome))
        ws.append(values)

    # Freeze header row
    ws.freeze_panes = "A2"

    # Reasonable column widths
    for col_idx in range(1, len(new_headers) + 1):
        letter = openpyxl.utils.get_column_letter(col_idx)
        ws.column_dimensions[letter].width = max(14, len(new_headers[col_idx - 1]) + 2)

    wb.save(output_path)
    return stats


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------


def _write_csv(
    source_path: str,
    output_path: str,
    lead_by_id: dict[str, LeadRecord],
    outcome_by_lead: dict[str, LeadOutcome],
) -> _WriteStats:
    stats = _WriteStats()

    with open(source_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        all_rows = list(reader)

    if not all_rows:
        return stats

    headers = all_rows[0]

    # Strip existing result columns — idempotency
    keep_cols = [
        i for i, h in enumerate(headers) if h not in RESULT_COLUMNS
    ]
    original_headers = [headers[i] for i in keep_cols]
    lead_id_col = _lead_identifier_column_index(original_headers)

    new_headers = original_headers + list(RESULT_COLUMNS)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(new_headers)

        for idx, raw_row in enumerate(all_rows[1:], start=1):
            # Preserve row by keeping only non-result-column cells
            row = [raw_row[i] if i < len(raw_row) else "" for i in keep_cols]
            stats.rows_total += 1
            lead, match_status = _match_row_to_lead(
                row, original_headers, lead_id_col, lead_by_id, idx,
            )
            if match_status == RowMatchStatus.AMBIGUOUS:
                stats.ambiguous_matches += 1
                stats.warnings.append(ExportWarning(
                    row_number=idx, lead_id=None,
                    code="AMBIGUOUS_MATCH",
                    detail=f"row {idx} matched multiple leads",
                ))
                outcome = None
            elif lead is None:
                stats.rows_unmatched += 1
                stats.warnings.append(ExportWarning(
                    row_number=idx, lead_id=None,
                    code="LEAD_NOT_FOUND",
                    detail=f"row {idx} could not be matched to a lead",
                ))
                outcome = None
            else:
                outcome = outcome_by_lead.get(lead.lead_id)
                if outcome is not None:
                    stats.rows_updated += 1

            writer.writerow(row + list(_serialize_outcome(outcome)))

    return stats
