"""Deterministic Pre-Call Intelligence Engine (PCIE).

Translates PreCallBusinessContext + CampaignConfig + PreCallDiscoveryPolicy
into a PreCallConversationPlan. No model calls, no retrieval, no state
mutation. Known facts → preparation; unknown facts → verification targets.
"""

from __future__ import annotations

from app.leads.campaign.contracts import CampaignConfig
from app.leads.contracts import (
    EnrichedField,
    FieldKnowledgeState,
    PreCallBusinessContext,
    WebsiteStatus,
)
from app.precall.contracts import (
    AssumptionGuardrail,
    InitialPosture,
    OpeningObjective,
    PreCallConversationPlan,
    PreCallDiscoveryPolicy,
    PreCallFact,
    PreCallPlanStatus,
    RiskFlag,
    ServicePrerequisiteGap,
    VerificationPriority,
    VerificationTarget,
)


# ---------------------------------------------------------------------------
# Stable fact keys — generic semantic namespace
# ---------------------------------------------------------------------------

# Keys are intentionally free strings so new campaigns/services can introduce
# new keys via config without adding Python branches here. The constants below
# are provided for internal engine use; callers may use their own keys.

FACT_WEBSITE_PRESENCE = "website.presence"
FACT_BUSINESS_CATEGORY = "business.category"
FACT_CONTACT_NAME = "contact.name"
FACT_CONTACT_EMAIL = "contact.email"
FACT_LOCATION_CITY = "location.city"
FACT_LOCATION_STATE = "location.state"
FACT_LOCATION_COUNTRY = "location.country"
FACT_LOCATION_ADDRESS = "location.address"


# ---------------------------------------------------------------------------
# Priority ordering — deterministic
# ---------------------------------------------------------------------------

_PRIORITY_RANK = {
    VerificationPriority.CRITICAL: 0,
    VerificationPriority.HIGH: 1,
    VerificationPriority.NORMAL: 2,
    VerificationPriority.LOW: 3,
}


class PreCallIntelligenceEngine:
    """Prepare, do not predict. Unknown → verify. Unknown ≠ negative."""

    def plan(
        self,
        context: PreCallBusinessContext,
        campaign: CampaignConfig,
        policy: PreCallDiscoveryPolicy,
    ) -> PreCallConversationPlan:
        if context.tenant_id != campaign.tenant_id:
            raise ValueError("tenant mismatch between context and campaign")
        if context.campaign_id != campaign.campaign_id:
            raise ValueError("campaign mismatch between context and campaign")

        facts = _build_facts(context)
        buckets = _bucket(facts)

        authorized = tuple(campaign.authorized_service_ids)
        gaps = _service_prerequisite_gaps(authorized, policy, facts)

        targets = _verification_targets(facts, authorized, policy, gaps)
        primary = targets[0] if targets else None

        guardrails = _guardrails(facts)
        knowledge_hints = _knowledge_hints(authorized, policy)
        risks = _risk_flags(context, authorized, buckets)
        posture = _initial_posture(buckets, context)
        status = _plan_status(authorized, buckets)
        opening = _opening_objective(primary, context, buckets)
        discovery_priorities = tuple(t.fact_key for t in targets)

        return PreCallConversationPlan(
            lead_id=context.lead_id,
            campaign_id=context.campaign_id,
            tenant_id=context.tenant_id,
            policy_id=policy.policy_id,
            policy_version=policy.version,
            known_facts=buckets[FieldKnowledgeState.KNOWN],
            unknown_facts=buckets[FieldKnowledgeState.UNKNOWN],
            unverified_facts=buckets[FieldKnowledgeState.UNVERIFIED],
            conflicting_facts=buckets[FieldKnowledgeState.CONFLICTING],
            primary_initial_discovery_target=primary,
            verification_targets=targets,
            opening_objective=opening,
            discovery_priorities=discovery_priorities,
            allowed_service_context=authorized,
            service_prerequisite_gaps=gaps,
            assumption_guardrails=guardrails,
            knowledge_topics_likely_needed=knowledge_hints,
            risk_flags=risks,
            initial_posture=posture,
            plan_status=status,
        )


# ---------------------------------------------------------------------------
# Fact extraction
# ---------------------------------------------------------------------------


def _build_facts(context: PreCallBusinessContext) -> tuple[PreCallFact, ...]:
    """Project PreCallBusinessContext into generic PCIE facts."""
    facts: list[PreCallFact] = []

    facts.append(_website_fact(context.website_status))

    facts.append(_fact_from_enriched(FACT_BUSINESS_CATEGORY, context.category))
    facts.append(_fact_from_enriched(FACT_CONTACT_NAME, context.contact_name))
    facts.append(_fact_from_enriched(FACT_CONTACT_EMAIL, context.email))
    facts.append(_fact_from_enriched(FACT_LOCATION_CITY, context.city))
    facts.append(_fact_from_enriched(FACT_LOCATION_STATE, context.state))
    facts.append(_fact_from_enriched(FACT_LOCATION_COUNTRY, context.country))
    facts.append(_fact_from_enriched(FACT_LOCATION_ADDRESS, context.address))

    return tuple(facts)


def _website_fact(status: WebsiteStatus) -> PreCallFact:
    """Map WebsiteStatus to a PCIE fact. UNKNOWN never becomes 'no website'."""
    if status == WebsiteStatus.VERIFIED_PRESENT:
        return PreCallFact(
            key=FACT_WEBSITE_PRESENCE,
            state=FieldKnowledgeState.KNOWN,
            value="present",
            sources=("verified",),
        )
    if status == WebsiteStatus.VERIFIED_ABSENT:
        return PreCallFact(
            key=FACT_WEBSITE_PRESENCE,
            state=FieldKnowledgeState.KNOWN,
            value="absent",
            sources=("verified",),
        )
    if status == WebsiteStatus.UNVERIFIED_CANDIDATE:
        return PreCallFact(
            key=FACT_WEBSITE_PRESENCE,
            state=FieldKnowledgeState.UNVERIFIED,
            value="candidate",
            sources=("unverified_candidate",),
        )
    if status == WebsiteStatus.INVALID:
        return PreCallFact(
            key=FACT_WEBSITE_PRESENCE,
            state=FieldKnowledgeState.UNVERIFIED,
            value="invalid_input",
            sources=("invalid",),
        )
    return PreCallFact(
        key=FACT_WEBSITE_PRESENCE,
        state=FieldKnowledgeState.UNKNOWN,
    )


def _fact_from_enriched(key: str, field: EnrichedField) -> PreCallFact:
    state = field.state
    sources = (field.source.value,) if field.source is not None else ()

    if state == FieldKnowledgeState.CONFLICTING:
        return PreCallFact(
            key=key,
            state=state,
            value=field.value,
            sources=sources,
            conflict_detail=f"source {field.source.value} vs enrichment",
        )
    if state == FieldKnowledgeState.KNOWN and field.value is not None:
        return PreCallFact(key=key, state=state, value=field.value, sources=sources)
    if state == FieldKnowledgeState.UNVERIFIED and field.value is not None:
        return PreCallFact(key=key, state=state, value=field.value, sources=sources)
    return PreCallFact(key=key, state=FieldKnowledgeState.UNKNOWN)


def _bucket(
    facts: tuple[PreCallFact, ...],
) -> dict[FieldKnowledgeState, tuple[PreCallFact, ...]]:
    buckets: dict[FieldKnowledgeState, list[PreCallFact]] = {
        FieldKnowledgeState.KNOWN: [],
        FieldKnowledgeState.UNKNOWN: [],
        FieldKnowledgeState.UNVERIFIED: [],
        FieldKnowledgeState.CONFLICTING: [],
    }
    for fact in facts:
        buckets[fact.state].append(fact)
    return {k: tuple(v) for k, v in buckets.items()}


# ---------------------------------------------------------------------------
# Service prerequisites
# ---------------------------------------------------------------------------


def _service_prerequisite_gaps(
    authorized: tuple[str, ...],
    policy: PreCallDiscoveryPolicy,
    facts: tuple[PreCallFact, ...],
) -> tuple[ServicePrerequisiteGap, ...]:
    fact_by_key = {f.key: f for f in facts}
    resolved_states = {FieldKnowledgeState.KNOWN}

    gaps: list[ServicePrerequisiteGap] = []
    for service_id in authorized:
        rule = policy.rule_for(service_id)
        if rule is None:
            continue
        missing = tuple(
            k for k in rule.required_fact_keys
            if (k not in fact_by_key or fact_by_key[k].state not in resolved_states)
        )
        if missing:
            gaps.append(ServicePrerequisiteGap(
                service_id=service_id,
                missing_fact_keys=missing,
            ))
    return tuple(gaps)


# ---------------------------------------------------------------------------
# Verification targets (deterministic ordering)
# ---------------------------------------------------------------------------


def _verification_targets(
    facts: tuple[PreCallFact, ...],
    authorized: tuple[str, ...],
    policy: PreCallDiscoveryPolicy,
    gaps: tuple[ServicePrerequisiteGap, ...],
) -> tuple[VerificationTarget, ...]:
    required_for_services: dict[str, list[str]] = {}
    for gap in gaps:
        for key in gap.missing_fact_keys:
            required_for_services.setdefault(key, []).append(gap.service_id)

    optional_for_services: dict[str, list[str]] = {}
    for service_id in authorized:
        rule = policy.rule_for(service_id)
        if rule is None:
            continue
        for key in rule.optional_fact_keys:
            optional_for_services.setdefault(key, []).append(service_id)

    global_required = frozenset(policy.global_required_facts)
    global_optional = frozenset(policy.global_optional_facts)

    seen: set[str] = set()
    candidates: list[tuple[VerificationPriority, int, str, str]] = []

    for insertion_order, fact in enumerate(facts):
        if fact.state == FieldKnowledgeState.KNOWN:
            continue
        if fact.key in seen:
            continue
        seen.add(fact.key)

        priority = _priority_for(
            fact, policy,
            required_for_services, global_required,
            optional_for_services, global_optional,
        )
        if priority is None:
            continue
        reason = _reason_for(fact, required_for_services, global_required)
        candidates.append((priority, insertion_order, fact.key, reason))

    for insertion_order, key in enumerate(
        [*policy.global_required_facts, *policy.global_optional_facts]
    ):
        if key in seen:
            continue
        if any(f.key == key for f in facts):
            continue
        seen.add(key)
        priority = (
            policy.priority_override_for(key)
            or (VerificationPriority.HIGH if key in global_required else VerificationPriority.LOW)
        )
        candidates.append((
            priority, 1000 + insertion_order, key,
            "campaign-wide requirement" if key in global_required else "campaign-wide context",
        ))

    candidates.sort(key=lambda x: (_PRIORITY_RANK[x[0]], x[1], x[2]))

    return tuple(
        VerificationTarget(
            fact_key=key,
            priority=priority,
            reason=reason,
            tie_breaker_index=tie,
        )
        for priority, tie, key, reason in candidates
    )


def _priority_for(
    fact: PreCallFact,
    policy: PreCallDiscoveryPolicy,
    required_for_services: dict[str, list[str]],
    global_required: frozenset[str],
    optional_for_services: dict[str, list[str]],
    global_optional: frozenset[str],
) -> VerificationPriority | None:
    override = policy.priority_override_for(fact.key)
    if override is not None:
        return override

    if fact.state == FieldKnowledgeState.CONFLICTING:
        return VerificationPriority.CRITICAL

    is_required = fact.key in required_for_services or fact.key in global_required
    is_optional = fact.key in optional_for_services or fact.key in global_optional

    if is_required:
        return VerificationPriority.HIGH
    if fact.state == FieldKnowledgeState.UNVERIFIED:
        return VerificationPriority.NORMAL
    if is_optional:
        return VerificationPriority.LOW
    return None


def _reason_for(
    fact: PreCallFact,
    required_for_services: dict[str, list[str]],
    global_required: frozenset[str],
) -> str:
    if fact.state == FieldKnowledgeState.CONFLICTING:
        return "conflicting information requires resolution"
    services = required_for_services.get(fact.key, [])
    if services:
        return f"required for authorized service(s): {','.join(sorted(services))}"
    if fact.key in global_required:
        return "campaign-wide required fact"
    if fact.state == FieldKnowledgeState.UNVERIFIED:
        return "unverified candidate requires confirmation"
    return "optional discovery context"


# ---------------------------------------------------------------------------
# Guardrails / risk / posture
# ---------------------------------------------------------------------------


# Generic mapping — one row per fact key the engine knows how to guardrail
# against. New fact keys get entries here; the dispatch stays data-driven.
_UNKNOWN_GUARDRAILS: dict[str, AssumptionGuardrail] = {
    FACT_WEBSITE_PRESENCE: AssumptionGuardrail.DO_NOT_ASSUME_NO_WEBSITE,
    "workflow.current_process": AssumptionGuardrail.DO_NOT_ASSUME_MANUAL_WORKFLOW,
    "software.current_tools": AssumptionGuardrail.DO_NOT_ASSUME_MANUAL_WORKFLOW,
    "customer_acquisition.primary_channel": AssumptionGuardrail.DO_NOT_ASSUME_CUSTOMER_ACQUISITION,
    "seo.quality": AssumptionGuardrail.DO_NOT_ASSUME_BAD_SEO,
    "decision_maker.identity": AssumptionGuardrail.DO_NOT_ASSUME_DECISION_MAKER,
    "budget.availability": AssumptionGuardrail.DO_NOT_ASSUME_BUDGET,
    "language.preferred": AssumptionGuardrail.DO_NOT_ASSUME_LANGUAGE,
}

# Facts whose UNKNOWN state ALWAYS triggers the baseline guardrails, even
# when the fact key is not present in the context. These guard against the
# most common silent inferences (budget / decision maker / language).
_ALWAYS_UNKNOWN_GUARDRAILS: frozenset[AssumptionGuardrail] = frozenset({
    AssumptionGuardrail.DO_NOT_ASSUME_DECISION_MAKER,
    AssumptionGuardrail.DO_NOT_ASSUME_BUDGET,
    AssumptionGuardrail.DO_NOT_ASSUME_LANGUAGE,
})


def _guardrails(facts: tuple[PreCallFact, ...]) -> frozenset[AssumptionGuardrail]:
    result: set[AssumptionGuardrail] = set(_ALWAYS_UNKNOWN_GUARDRAILS)
    for fact in facts:
        if fact.state in (FieldKnowledgeState.UNKNOWN, FieldKnowledgeState.UNVERIFIED):
            guard = _UNKNOWN_GUARDRAILS.get(fact.key)
            if guard is not None:
                result.add(guard)
    return frozenset(result)


def _knowledge_hints(
    authorized: tuple[str, ...],
    policy: PreCallDiscoveryPolicy,
) -> tuple[str, ...]:
    seen: set[str] = set()
    hints: list[str] = []
    for service_id in authorized:
        rule = policy.rule_for(service_id)
        if rule is None:
            continue
        for hint in rule.knowledge_topic_hints:
            if hint not in seen:
                seen.add(hint)
                hints.append(hint)
    return tuple(hints)


def _risk_flags(
    context: PreCallBusinessContext,
    authorized: tuple[str, ...],
    buckets: dict[FieldKnowledgeState, tuple[PreCallFact, ...]],
) -> frozenset[RiskFlag]:
    flags: set[RiskFlag] = set()
    if not authorized:
        flags.add(RiskFlag.NO_CAMPAIGN_SERVICE_AUTHORIZED)
    if len(buckets[FieldKnowledgeState.KNOWN]) <= 1:
        flags.add(RiskFlag.THIN_PROFILE)
    if buckets[FieldKnowledgeState.CONFLICTING]:
        flags.add(RiskFlag.CONFLICTING_ENRICHMENT)
    if not context.email.value and not context.contact_name.value:
        flags.add(RiskFlag.UNVERIFIED_CONTACT)
    return frozenset(flags)


def _initial_posture(
    buckets: dict[FieldKnowledgeState, tuple[PreCallFact, ...]],
    context: PreCallBusinessContext,
) -> InitialPosture:
    if buckets[FieldKnowledgeState.CONFLICTING]:
        return InitialPosture.CONFLICTED_CONTEXT
    known_count = len(buckets[FieldKnowledgeState.KNOWN])
    if known_count >= 5:
        return InitialPosture.INFORMED_CONTEXT
    if known_count >= 2:
        return InitialPosture.NORMAL_CONTEXT
    return InitialPosture.LOW_CONTEXT


def _plan_status(
    authorized: tuple[str, ...],
    buckets: dict[FieldKnowledgeState, tuple[PreCallFact, ...]],
) -> PreCallPlanStatus:
    if not authorized:
        return PreCallPlanStatus.DEGRADED
    return PreCallPlanStatus.READY


def _opening_objective(
    primary: VerificationTarget | None,
    context: PreCallBusinessContext,
    buckets: dict[FieldKnowledgeState, tuple[PreCallFact, ...]],
) -> OpeningObjective:
    if primary is not None and primary.priority == VerificationPriority.CRITICAL:
        return OpeningObjective.VERIFY_CRITICAL_FACT
    if context.campaign_context is not None and context.campaign_context.strip():
        return OpeningObjective.RESUME_EXISTING_CONTEXT
    if len(buckets[FieldKnowledgeState.KNOWN]) >= 2:
        return OpeningObjective.CONFIRM_BUSINESS_CONTEXT
    if not context.contact_name.value and not context.email.value:
        return OpeningObjective.ESTABLISH_CONTACT
    return OpeningObjective.EARN_PERMISSION
