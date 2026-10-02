"""Deterministic lead enrichment reconciliation.

Priority: USER_PROVIDED > TRUSTED_EXISTING > ENRICHMENT_CANDIDATE > UNKNOWN.
Enrichment cannot silently overwrite user-supplied data. Conflicts are preserved.
"""

from __future__ import annotations

from app.leads.contracts import (
    EnrichedField,
    EnrichmentFieldSource,
    FieldKnowledgeState,
    LeadRecord,
    NormalizedPhone,
    PreCallBusinessContext,
    WebsiteStatus,
)
from app.leads.enrichment.contracts import (
    EnrichmentCandidateField,
    LeadEnrichmentResult,
)


def _reconcile_field(
    user_value: str | None,
    enrichment: EnrichmentCandidateField | None,
) -> EnrichedField:
    """Reconcile a single field with explicit priority."""
    if user_value and enrichment:
        if user_value.strip().lower() == enrichment.value.strip().lower():
            return EnrichedField(
                value=user_value,
                state=FieldKnowledgeState.KNOWN,
                source=EnrichmentFieldSource.USER_PROVIDED,
                provenance=enrichment.provenance,
            )
        return EnrichedField(
            value=user_value,
            state=FieldKnowledgeState.CONFLICTING,
            source=EnrichmentFieldSource.USER_PROVIDED,
            provenance=enrichment.provenance,
        )

    if user_value:
        return EnrichedField(
            value=user_value,
            state=FieldKnowledgeState.KNOWN,
            source=EnrichmentFieldSource.USER_PROVIDED,
        )

    if enrichment:
        return EnrichedField(
            value=enrichment.value,
            state=FieldKnowledgeState.UNVERIFIED,
            source=EnrichmentFieldSource.ENRICHMENT_CANDIDATE,
            provenance=enrichment.provenance,
        )

    return EnrichedField(
        value=None,
        state=FieldKnowledgeState.UNKNOWN,
        source=EnrichmentFieldSource.UNKNOWN,
    )


def _reconcile_website_status(
    lead: LeadRecord,
    enrichment: LeadEnrichmentResult | None,
) -> WebsiteStatus:
    """Reconcile website status preserving UNKNOWN for blanks."""
    if lead.website_status != WebsiteStatus.UNKNOWN:
        return lead.website_status
    if enrichment and enrichment.website:
        return WebsiteStatus.UNVERIFIED_CANDIDATE
    return WebsiteStatus.UNKNOWN


def reconcile(
    lead: LeadRecord,
    enrichment: LeadEnrichmentResult | None = None,
) -> PreCallBusinessContext:
    """Build pre-call context by reconciling lead data with enrichment."""
    enrich_website = enrichment.website if enrichment and enrichment.success else None
    enrich_category = enrichment.category if enrichment and enrichment.success else None
    enrich_city = enrichment.city if enrichment and enrichment.success else None
    enrich_state = enrichment.state if enrichment and enrichment.success else None
    enrich_country = enrichment.country if enrichment and enrichment.success else None
    enrich_address = enrichment.address if enrichment and enrichment.success else None
    enrich_email = enrichment.email if enrichment and enrichment.success else None

    return PreCallBusinessContext(
        lead_id=lead.lead_id,
        tenant_id=lead.tenant_id,
        campaign_id=lead.campaign_id,
        business_name=lead.business_name,
        phone=lead.phone,
        website_status=_reconcile_website_status(lead, enrichment),
        website=_reconcile_field(lead.website, enrich_website),
        category=_reconcile_field(lead.category, enrich_category),
        city=_reconcile_field(lead.city, enrich_city),
        state=_reconcile_field(lead.state, enrich_state),
        country=_reconcile_field(lead.country, enrich_country),
        address=_reconcile_field(lead.address, enrich_address),
        contact_name=_reconcile_field(lead.contact_name, None),
        email=_reconcile_field(lead.email, enrich_email),
        lead_source=lead.source,
        import_batch_id=lead.import_batch_id,
    )
