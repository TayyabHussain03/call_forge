"""Comprehensive tests for Lead Intake, Enrichment & Pre-Call Context Engine."""

from __future__ import annotations

import io
import tempfile
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from app.leads.contracts import (
    DuplicateClassification,
    EnrichedField,
    EnrichmentFieldSource,
    FieldKnowledgeState,
    ImportStatistics,
    LeadImportReadiness,
    LeadRecord,
    NormalizedPhone,
    PhoneNormalizationStatus,
    PreCallBusinessContext,
    RejectedRow,
    WebsiteStatus,
)
from app.leads.enrichment.contracts import (
    EnrichmentCandidateField,
    EnrichmentProvenance,
    LeadEnrichmentInput,
    LeadEnrichmentResult,
    MockLeadEnrichmentProvider,
)
from app.leads.enrichment.reconciler import reconcile
from app.leads.importer import import_leads
from app.leads.normalizer import normalize_phone


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _csv_bytes(rows: list[list[str]]) -> bytes:
    """Build CSV bytes from list of rows."""
    import csv

    buf = io.StringIO()
    writer = csv.writer(buf)
    for row in rows:
        writer.writerow(row)
    return buf.getvalue().encode("utf-8")


def _xlsx_bytes(rows: list[list[str]]) -> bytes:
    """Build XLSX bytes from list of rows."""
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    for row in rows:
        ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _lead(
    lead_id: str = "lead_1",
    business_name: str = "ABC Restaurant",
    raw_phone: str = "+12125551234",
    normalized: str | None = "+12125551234",
    phone_status: PhoneNormalizationStatus = PhoneNormalizationStatus.VALID_E164,
    website: str | None = None,
    website_status: WebsiteStatus = WebsiteStatus.UNKNOWN,
    **kwargs,
) -> LeadRecord:
    return LeadRecord(
        lead_id=lead_id,
        tenant_id=kwargs.get("tenant_id", "tenant_1"),
        campaign_id=kwargs.get("campaign_id", "campaign_1"),
        business_name=business_name,
        phone=NormalizedPhone(
            raw=raw_phone,
            normalized=normalized,
            status=phone_status,
        ),
        website=website,
        website_status=website_status,
        **{k: v for k, v in kwargs.items() if k not in ("tenant_id", "campaign_id")},
    )


def _enrichment_result(
    provider: str = "mock",
    success: bool = True,
    website: str | None = None,
    category: str | None = None,
    city: str | None = None,
) -> LeadEnrichmentResult:
    def _field(val):
        if val is None:
            return None
        return EnrichmentCandidateField(
            value=val,
            provenance=EnrichmentProvenance(source_provider=provider),
        )
    return LeadEnrichmentResult(
        provider_name=provider,
        success=success,
        website=_field(website),
        category=_field(category),
        city=_field(city),
    )


# ===========================================================================
# Import tests
# ===========================================================================


class TestCSVImport:
    def test_basic_csv_import(self):
        data = _csv_bytes([
            ["Business Name", "Phone"],
            ["ABC Restaurant", "+12125551234"],
            ["XYZ HVAC", "+12025551234"],
        ])
        result = import_leads(file_content=data, file_type="csv", country_context="US")
        assert result.statistics.accepted == 2
        assert result.statistics.rejected == 0
        assert len(result.accepted) == 2

    def test_csv_with_optional_fields(self):
        data = _csv_bytes([
            ["Business Name", "Phone", "Website", "City", "Email"],
            ["ABC Restaurant", "+12125551234", "abc.com", "Dallas", "abc@test.com"],
        ])
        result = import_leads(file_content=data, file_type="csv", country_context="US")
        assert result.accepted[0].city == "Dallas"
        assert result.accepted[0].email == "abc@test.com"
        assert result.accepted[0].website == "abc.com"


class TestXLSXImport:
    def test_basic_xlsx_import(self):
        data = _xlsx_bytes([
            ["Business Name", "Phone"],
            ["ABC Restaurant", "+12125551234"],
        ])
        result = import_leads(file_content=data, file_type="xlsx")
        assert result.statistics.accepted == 1
        assert result.accepted[0].business_name == "ABC Restaurant"


class TestMinimumFields:
    def test_business_name_and_phone_sufficient(self):
        data = _csv_bytes([
            ["Business Name", "Phone"],
            ["ABC Restaurant", "+12125551234"],
        ])
        result = import_leads(file_content=data, file_type="csv")
        assert result.statistics.accepted == 1

    def test_optional_fields_absent(self):
        data = _csv_bytes([
            ["Business Name", "Phone"],
            ["ABC", "+12125551234"],
        ])
        result = import_leads(file_content=data, file_type="csv")
        lead = result.accepted[0]
        assert lead.website is None
        assert lead.city is None
        assert lead.email is None
        assert lead.contact_name is None


class TestColumnAliases:
    def test_company_alias(self):
        data = _csv_bytes([
            ["Company", "Telephone"],
            ["ABC Restaurant", "+12125551234"],
        ])
        result = import_leads(file_content=data, file_type="csv")
        assert result.statistics.accepted == 1
        assert result.accepted[0].business_name == "ABC Restaurant"

    def test_custom_aliases(self):
        data = _csv_bytes([
            ["Firma", "Nummer"],
            ["ABC", "+12125551234"],
        ])
        custom = {
            "business_name": ("firma",),
            "phone": ("nummer",),
        }
        result = import_leads(file_content=data, file_type="csv", column_aliases=custom)
        assert result.statistics.accepted == 1


class TestMissingRequiredColumn:
    def test_missing_phone_column_raises(self):
        data = _csv_bytes([
            ["Business Name"],
            ["ABC"],
        ])
        with pytest.raises(ValueError, match="required column not found: phone"):
            import_leads(file_content=data, file_type="csv")

    def test_missing_business_name_column_raises(self):
        data = _csv_bytes([
            ["Phone"],
            ["+12125551234"],
        ])
        with pytest.raises(ValueError, match="required column not found: business_name"):
            import_leads(file_content=data, file_type="csv")


class TestEmptyBusinessName:
    def test_empty_name_rejected(self):
        data = _csv_bytes([
            ["Business Name", "Phone"],
            ["", "+12125551234"],
        ])
        result = import_leads(file_content=data, file_type="csv")
        assert result.statistics.rejected == 1
        assert "business_name" in result.rejected[0].reason


class TestMalformedPhone:
    def test_empty_phone_rejected(self):
        data = _csv_bytes([
            ["Business Name", "Phone"],
            ["ABC", ""],
        ])
        result = import_leads(file_content=data, file_type="csv")
        assert result.statistics.rejected == 1
        assert "phone" in result.rejected[0].reason


class TestRowProvenance:
    def test_row_number_preserved(self):
        data = _csv_bytes([
            ["Business Name", "Phone"],
            ["ABC", "+12125551234"],
            ["XYZ", "+12025551234"],
        ])
        result = import_leads(file_content=data, file_type="csv", batch_id="b1")
        assert result.accepted[0].row_number == 2
        assert result.accepted[1].row_number == 3
        assert result.accepted[0].import_batch_id == "b1"

    def test_rejected_row_provenance(self):
        data = _csv_bytes([
            ["Business Name", "Phone"],
            ["", "+12125551234"],
        ])
        result = import_leads(file_content=data, file_type="csv")
        assert result.rejected[0].row_number == 2


class TestRejectedRowReporting:
    def test_rejected_rows_never_silently_discarded(self):
        data = _csv_bytes([
            ["Business Name", "Phone"],
            ["Good", "+12125551234"],
            ["", "+12025551234"],
            ["Also Good", "+14155551234"],
        ])
        result = import_leads(file_content=data, file_type="csv")
        assert result.statistics.accepted == 2
        assert result.statistics.rejected == 1
        assert result.statistics.total_rows == 3


# ===========================================================================
# Normalization tests
# ===========================================================================


class TestPhoneNormalization:
    def test_valid_e164(self):
        result = normalize_phone("+12125551234")
        assert result.status == PhoneNormalizationStatus.VALID_E164
        assert result.normalized == "+12125551234"

    def test_explicit_country_normalization(self):
        result = normalize_phone("2125551234", country_context="US")
        assert result.status == PhoneNormalizationStatus.VALID_E164
        assert result.normalized is not None
        assert result.normalized.startswith("+1")

    def test_missing_country_needs_review(self):
        result = normalize_phone("2125551234")
        assert result.status == PhoneNormalizationStatus.NEEDS_REVIEW

    def test_raw_phone_retained(self):
        result = normalize_phone("(212) 555-1234", country_context="US")
        assert result.raw == "(212) 555-1234"

    def test_no_inferred_country_from_name(self):
        result = normalize_phone("2125551234")
        assert result.status == PhoneNormalizationStatus.NEEDS_REVIEW
        assert result.country_context is None

    def test_stable_normalized_identity(self):
        r1 = normalize_phone("+12125551234")
        r2 = normalize_phone("+1 212 555 1234")
        assert r1.normalized == r2.normalized


# ===========================================================================
# Deduplication tests
# ===========================================================================


class TestDeduplication:
    def test_same_phone_in_batch(self):
        data = _csv_bytes([
            ["Business Name", "Phone"],
            ["ABC Restaurant", "+12125551234"],
            ["ABC Diner", "+12125551234"],
        ])
        result = import_leads(file_content=data, file_type="csv")
        assert result.statistics.accepted == 1
        assert result.statistics.duplicates == 1
        assert result.duplicates[0].duplicate_status == DuplicateClassification.DUPLICATE_IN_BATCH

    def test_same_phone_different_campaign_is_unique(self):
        data = _csv_bytes([
            ["Business Name", "Phone"],
            ["ABC Restaurant", "+12125551234"],
        ])
        r1 = import_leads(file_content=data, file_type="csv", campaign_id="c1")
        r2 = import_leads(file_content=data, file_type="csv", campaign_id="c2")
        assert r1.statistics.accepted == 1
        assert r2.statistics.accepted == 1

    def test_similar_names_not_merged(self):
        data = _csv_bytes([
            ["Business Name", "Phone"],
            ["ABC Restaurant", "+12125551234"],
            ["ABC Restaurants", "+12025551234"],
        ])
        result = import_leads(file_content=data, file_type="csv")
        assert result.statistics.accepted == 2

    def test_existing_phone_is_duplicate_existing(self):
        data = _csv_bytes([
            ["Business Name", "Phone"],
            ["ABC Restaurant", "+12125551234"],
        ])
        result = import_leads(
            file_content=data, file_type="csv",
            existing_phones=frozenset({"+12125551234"}),
        )
        assert result.statistics.duplicates == 1
        assert result.duplicates[0].duplicate_status == DuplicateClassification.DUPLICATE_EXISTING


# ===========================================================================
# Website state tests
# ===========================================================================


class TestWebsiteState:
    def test_blank_website_is_unknown(self):
        lead = _lead(website=None)
        assert lead.website_status == WebsiteStatus.UNKNOWN

    def test_explicit_website_is_unverified_candidate(self):
        data = _csv_bytes([
            ["Business Name", "Phone", "Website"],
            ["ABC", "+12125551234", "abc.com"],
        ])
        result = import_leads(file_content=data, file_type="csv")
        assert result.accepted[0].website_status == WebsiteStatus.UNVERIFIED_CANDIDATE

    def test_trusted_verified_absence(self):
        lead = _lead(website_status=WebsiteStatus.VERIFIED_ABSENT)
        assert lead.website_status == WebsiteStatus.VERIFIED_ABSENT

    def test_enrichment_candidate_is_not_verified(self):
        lead = _lead()
        enrichment = _enrichment_result(website="abc.com")
        ctx = reconcile(lead, enrichment)
        assert ctx.website_status == WebsiteStatus.UNVERIFIED_CANDIDATE

    def test_invalid_url(self):
        data = _csv_bytes([
            ["Business Name", "Phone", "Website"],
            ["ABC", "+12125551234", "notaurl"],
        ])
        result = import_leads(file_content=data, file_type="csv")
        assert result.accepted[0].website_status == WebsiteStatus.INVALID

    def test_no_false_no_website_inference(self):
        lead = _lead(website=None)
        ctx = reconcile(lead)
        assert ctx.website_status == WebsiteStatus.UNKNOWN
        assert ctx.website.state == FieldKnowledgeState.UNKNOWN


# ===========================================================================
# Enrichment tests
# ===========================================================================


class TestEnrichmentProvider:
    def test_provider_neutral_contract(self):
        provider = MockLeadEnrichmentProvider()
        inp = LeadEnrichmentInput(
            business_name="ABC Restaurant",
            normalized_phone="+12125551234",
        )
        result = provider.enrich(inp)
        assert isinstance(result, LeadEnrichmentResult)

    def test_deterministic_mock_provider(self):
        expected = _enrichment_result(website="abc.com")
        provider = MockLeadEnrichmentProvider(
            results={"+12125551234": expected},
        )
        inp = LeadEnrichmentInput(
            business_name="ABC",
            normalized_phone="+12125551234",
        )
        result = provider.enrich(inp)
        assert result.website is not None
        assert result.website.value == "abc.com"


class TestEnrichmentReconciliation:
    def test_imported_data_beats_enrichment(self):
        lead = _lead(city="Dallas")
        enrichment = _enrichment_result(city="Fort Worth")
        ctx = reconcile(lead, enrichment)
        assert ctx.city.value == "Dallas"
        assert ctx.city.state == FieldKnowledgeState.CONFLICTING

    def test_enrichment_fills_unknown(self):
        lead = _lead()
        enrichment = _enrichment_result(category="Restaurant")
        ctx = reconcile(lead, enrichment)
        assert ctx.category.value == "Restaurant"
        assert ctx.category.state == FieldKnowledgeState.UNVERIFIED

    def test_conflict_preserved(self):
        lead = _lead(website="abc.com", website_status=WebsiteStatus.UNVERIFIED_CANDIDATE)
        enrichment = _enrichment_result(website="abcrestaurant.com")
        ctx = reconcile(lead, enrichment)
        assert ctx.website.state == FieldKnowledgeState.CONFLICTING
        assert ctx.website.provenance is not None

    def test_no_silent_overwrite(self):
        lead = _lead(city="Dallas")
        enrichment = _enrichment_result(city="Dallas")
        ctx = reconcile(lead, enrichment)
        assert ctx.city.source == EnrichmentFieldSource.USER_PROVIDED

    def test_provenance_retained(self):
        lead = _lead()
        enrichment = _enrichment_result(category="Restaurant")
        ctx = reconcile(lead, enrichment)
        assert ctx.category.provenance is not None
        assert ctx.category.provenance.source_provider == "mock"

    def test_provider_failure_leaves_lead_usable(self):
        lead = _lead()
        failed = LeadEnrichmentResult(
            provider_name="mock",
            success=False,
            error_message="timeout",
        )
        ctx = reconcile(lead, failed)
        assert ctx.business_name == lead.business_name
        assert ctx.category.state == FieldKnowledgeState.UNKNOWN

    def test_timeout_does_not_corrupt_lead(self):
        lead = _lead(city="Dallas")
        failed = LeadEnrichmentResult(
            provider_name="mock",
            success=False,
            error_message="timeout",
        )
        ctx = reconcile(lead, failed)
        assert ctx.city.value == "Dallas"
        assert ctx.city.state == FieldKnowledgeState.KNOWN


# ===========================================================================
# Pre-call context tests
# ===========================================================================


class TestPreCallBusinessContext:
    def test_immutable(self):
        lead = _lead()
        ctx = reconcile(lead)
        with pytest.raises(FrozenInstanceError):
            ctx.business_name = "Changed"

    def test_known_vs_unknown_separation(self):
        lead = _lead(city="Dallas")
        ctx = reconcile(lead)
        assert ctx.city.state == FieldKnowledgeState.KNOWN
        assert ctx.category.state == FieldKnowledgeState.UNKNOWN

    def test_conflict_separation(self):
        lead = _lead(city="Dallas")
        enrichment = _enrichment_result(city="Fort Worth")
        ctx = reconcile(lead, enrichment)
        assert ctx.city.state == FieldKnowledgeState.CONFLICTING

    def test_bounded_fields(self):
        lead = _lead()
        ctx = reconcile(lead)
        assert ctx.lead_id == lead.lead_id
        assert ctx.tenant_id == lead.tenant_id
        assert ctx.campaign_id == lead.campaign_id

    def test_tenant_isolation(self):
        l1 = _lead(tenant_id="t1")
        l2 = _lead(tenant_id="t2", lead_id="lead_2")
        c1 = reconcile(l1)
        c2 = reconcile(l2)
        assert c1.tenant_id != c2.tenant_id

    def test_campaign_isolation(self):
        l1 = _lead(campaign_id="c1")
        l2 = _lead(campaign_id="c2", lead_id="lead_2")
        c1 = reconcile(l1)
        c2 = reconcile(l2)
        assert c1.campaign_id != c2.campaign_id


class TestNoLanguageInference:
    def test_no_language_inference_from_metadata(self):
        lead = _lead(country="PK")
        ctx = reconcile(lead)
        assert not hasattr(ctx, "language")
        assert not hasattr(ctx, "preferred_language")


class TestNoServiceSelection:
    def test_no_service_fields(self):
        lead = _lead()
        ctx = reconcile(lead)
        assert not hasattr(ctx, "recommended_service")
        assert not hasattr(ctx, "sell_website")
        assert not hasattr(ctx, "service_fit")


class TestNoSalesDiagnosis:
    def test_no_pain_inference(self):
        lead = _lead()
        ctx = reconcile(lead)
        assert not hasattr(ctx, "pain")
        assert not hasattr(ctx, "problem")
        assert not hasattr(ctx, "opportunity")


class TestNoAuthorityMutation:
    def test_no_authority_fields(self):
        lead = _lead()
        ctx = reconcile(lead)
        assert not hasattr(ctx, "authority")
        assert not hasattr(ctx, "transition")
        assert not hasattr(ctx, "fsm_state")


# ===========================================================================
# Replay determinism
# ===========================================================================


class TestReplayDeterminism:
    def test_same_input_same_output(self):
        data = _csv_bytes([
            ["Business Name", "Phone"],
            ["ABC Restaurant", "+12125551234"],
        ])
        r1 = import_leads(file_content=data, file_type="csv", batch_id="b1")
        r2 = import_leads(file_content=data, file_type="csv", batch_id="b1")
        assert r1.statistics == r2.statistics
        assert r1.accepted[0].business_name == r2.accepted[0].business_name
        assert r1.accepted[0].phone == r2.accepted[0].phone


# ===========================================================================
# Safety tests
# ===========================================================================


class TestNoNetworkInTests:
    def test_no_telephony_import(self):
        import app.leads.contracts as c
        source = Path(c.__file__).read_text()
        assert "twilio" not in source.lower()
        assert "telnyx" not in source.lower()
        assert "vapi" not in source.lower()

    def test_no_llm_import_in_contracts(self):
        import app.leads.contracts as c
        source = Path(c.__file__).read_text()
        assert "gemini" not in source.lower()
        assert "openai" not in source.lower()
        assert "anthropic" not in source.lower()

    def test_no_web_scraper_in_domain(self):
        import app.leads.importer as m
        source = Path(m.__file__).read_text()
        assert "requests.get" not in source
        assert "httpx" not in source
        assert "urllib.request" not in source

    def test_no_approved_evidence_generation(self):
        import app.leads.contracts as c
        source = Path(c.__file__).read_text()
        assert "ApprovedEvidence" not in source

    def test_no_customer_response_generation(self):
        import app.leads.contracts as c
        source = Path(c.__file__).read_text()
        assert "generate_response" not in source
        assert "HumanConversation" not in source
