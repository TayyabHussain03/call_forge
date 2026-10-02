"""Lead file importer for XLSX and CSV with column alias support.

Validates required columns, normalizes phones, handles rejected rows
independently, and returns a complete import result with provenance.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import BinaryIO

from app.leads.contracts import (
    DuplicateClassification,
    ImportStatistics,
    ImportWarning,
    LeadImportReadiness,
    LeadImportResult,
    LeadRecord,
    PhoneNormalizationStatus,
    RejectedRow,
    WebsiteStatus,
)
from app.leads.normalizer import normalize_phone


# ---------------------------------------------------------------------------
# Column aliases
# ---------------------------------------------------------------------------

DEFAULT_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "business_name": ("business name", "company", "company name", "business", "name"),
    "phone": ("phone", "phone number", "mobile", "telephone", "tel"),
    "website": ("website", "web", "url", "site"),
    "category": ("category", "business category", "industry", "type"),
    "city": ("city", "town"),
    "state": ("state", "province", "region"),
    "country": ("country", "nation"),
    "address": ("address", "street", "location"),
    "email": ("email", "e-mail", "email address"),
    "contact_name": ("contact name", "contact", "person", "contact person"),
    "notes": ("notes", "note", "comments", "comment"),
    "source": ("source", "lead source", "origin"),
}

REQUIRED_FIELDS = ("business_name", "phone")


def _resolve_columns(
    headers: list[str],
    aliases: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, int]:
    """Map canonical field names to column indices using aliases."""
    alias_map = aliases or DEFAULT_COLUMN_ALIASES
    normalized_headers = [h.strip().lower() for h in headers]

    mapping: dict[str, int] = {}
    for field, field_aliases in alias_map.items():
        for alias in field_aliases:
            if alias in normalized_headers:
                mapping[field] = normalized_headers.index(alias)
                break
    return mapping


def _extract_row(
    row: list[str],
    column_map: dict[str, int],
) -> dict[str, str]:
    """Extract field values from a row using the column mapping."""
    result: dict[str, str] = {}
    for field, idx in column_map.items():
        if idx < len(row):
            val = row[idx].strip() if row[idx] else ""
            if val:
                result[field] = val
    return result


def _determine_website_status(website: str | None) -> WebsiteStatus:
    """Blank website = UNKNOWN, never VERIFIED_ABSENT."""
    if not website:
        return WebsiteStatus.UNKNOWN
    website = website.strip()
    if not website:
        return WebsiteStatus.UNKNOWN
    if "." not in website and "://" not in website:
        return WebsiteStatus.INVALID
    return WebsiteStatus.UNVERIFIED_CANDIDATE


def _parse_csv_rows(content: str) -> tuple[list[str], list[list[str]]]:
    """Parse CSV content into headers and data rows."""
    reader = csv.reader(io.StringIO(content))
    rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _parse_xlsx_rows(file_content: bytes) -> tuple[list[str], list[list[str]]]:
    """Parse XLSX content into headers and data rows."""
    import openpyxl

    wb = openpyxl.load_workbook(io.BytesIO(file_content), read_only=True, data_only=True)
    ws = wb.active
    rows: list[list[str]] = []
    for row in ws.iter_rows(values_only=True):
        rows.append([str(cell) if cell is not None else "" for cell in row])
    wb.close()
    if not rows:
        return [], []
    return rows[0], rows[1:]


def import_leads(
    file_path: str | Path | None = None,
    file_content: bytes | None = None,
    file_type: str | None = None,
    tenant_id: str = "default",
    campaign_id: str = "default",
    batch_id: str = "batch_1",
    country_context: str | None = None,
    column_aliases: dict[str, tuple[str, ...]] | None = None,
    existing_phones: frozenset[str] = frozenset(),
) -> LeadImportResult:
    """Import leads from an XLSX or CSV file."""
    if file_path is not None:
        path = Path(file_path)
        suffix = path.suffix.lower()
        content_bytes = path.read_bytes()
    elif file_content is not None:
        suffix = f".{file_type}" if file_type else ".csv"
        content_bytes = file_content
    else:
        raise ValueError("either file_path or file_content must be provided")

    if suffix == ".xlsx":
        headers, data_rows = _parse_xlsx_rows(content_bytes)
    elif suffix == ".csv":
        headers, data_rows = _parse_csv_rows(content_bytes.decode("utf-8-sig"))
    else:
        raise ValueError(f"unsupported file type: {suffix}")

    column_map = _resolve_columns(headers, column_aliases)

    for req in REQUIRED_FIELDS:
        if req not in column_map:
            raise ValueError(f"required column not found: {req}")

    accepted: list[LeadRecord] = []
    rejected: list[RejectedRow] = []
    duplicates: list[LeadRecord] = []
    warnings: list[ImportWarning] = []
    seen_phones: dict[str, int] = {}

    for row_idx, row in enumerate(data_rows):
        row_number = row_idx + 2  # 1-indexed, header is row 1
        fields = _extract_row(row, column_map)
        raw_data = {h: row[i] if i < len(row) else "" for h, i in column_map.items()}

        business_name = fields.get("business_name", "").strip()
        if not business_name:
            rejected.append(RejectedRow(
                row_number=row_number,
                reason="missing required field: business_name",
                raw_data=raw_data,
            ))
            continue

        raw_phone = fields.get("phone", "").strip()
        if not raw_phone:
            rejected.append(RejectedRow(
                row_number=row_number,
                reason="missing required field: phone",
                raw_data=raw_data,
            ))
            continue

        row_country = fields.get("country") or country_context
        phone = normalize_phone(raw_phone, row_country)

        website_raw = fields.get("website")
        website_status = _determine_website_status(website_raw)

        lead_id = f"{batch_id}_row_{row_number}"

        lead = LeadRecord(
            lead_id=lead_id,
            tenant_id=tenant_id,
            campaign_id=campaign_id,
            business_name=business_name,
            phone=phone,
            website=website_raw,
            website_status=website_status,
            category=fields.get("category"),
            city=fields.get("city"),
            state=fields.get("state"),
            country=fields.get("country"),
            address=fields.get("address"),
            email=fields.get("email"),
            contact_name=fields.get("contact_name"),
            source=fields.get("source"),
            notes=fields.get("notes"),
            import_batch_id=batch_id,
            row_number=row_number,
        )

        dedup_key = phone.normalized or raw_phone
        if phone.normalized and phone.normalized in existing_phones:
            lead = LeadRecord(
                lead_id=lead.lead_id,
                tenant_id=lead.tenant_id,
                campaign_id=lead.campaign_id,
                business_name=lead.business_name,
                phone=lead.phone,
                website=lead.website,
                website_status=lead.website_status,
                category=lead.category,
                city=lead.city,
                state=lead.state,
                country=lead.country,
                address=lead.address,
                email=lead.email,
                contact_name=lead.contact_name,
                source=lead.source,
                notes=lead.notes,
                import_batch_id=lead.import_batch_id,
                row_number=lead.row_number,
                duplicate_status=DuplicateClassification.DUPLICATE_EXISTING,
            )
            duplicates.append(lead)
            continue

        if dedup_key in seen_phones:
            lead = LeadRecord(
                lead_id=lead.lead_id,
                tenant_id=lead.tenant_id,
                campaign_id=lead.campaign_id,
                business_name=lead.business_name,
                phone=lead.phone,
                website=lead.website,
                website_status=lead.website_status,
                category=lead.category,
                city=lead.city,
                state=lead.state,
                country=lead.country,
                address=lead.address,
                email=lead.email,
                contact_name=lead.contact_name,
                source=lead.source,
                notes=lead.notes,
                import_batch_id=lead.import_batch_id,
                row_number=lead.row_number,
                duplicate_status=DuplicateClassification.DUPLICATE_IN_BATCH,
            )
            duplicates.append(lead)
            continue

        seen_phones[dedup_key] = row_number
        accepted.append(lead)

    stats = ImportStatistics(
        total_rows=len(data_rows),
        accepted=len(accepted),
        rejected=len(rejected),
        duplicates=len(duplicates),
    )

    return LeadImportResult(
        batch_id=batch_id,
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        duplicates=tuple(duplicates),
        warnings=tuple(warnings),
        statistics=stats,
    )
