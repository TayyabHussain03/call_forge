"""Immutable contracts for the Pre-Call Intelligence Engine (PCIE).

PCIE prepares the salesperson. It does NOT decide what the prospect
needs. Known facts become preparation; unknown facts become verification
targets. Unknown is never collapsed into negative.

TRUST BOUNDARY: PCIE output is advisory. It does not authorize services,
mutate state, select pitches, or trigger retrieval. Conversation truth
always supersedes pre-call preparation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.leads.contracts import FieldKnowledgeState


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class VerificationPriority(str, Enum):
    """Categorical priority. Deterministic ordering, no scoring."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class OpeningObjective(str, Enum):
    """Bounded conversation-opening objective. No wording implied."""

    ESTABLISH_CONTACT = "establish_contact"
    EARN_PERMISSION = "earn_permission"
    CONFIRM_BUSINESS_CONTEXT = "confirm_business_context"
    VERIFY_CRITICAL_FACT = "verify_critical_fact"
    RESUME_EXISTING_CONTEXT = "resume_existing_context"


class InitialPosture(str, Enum):
    """Advisory posture snapshot for how much context exists pre-call."""

    LOW_CONTEXT = "low_context"
    NORMAL_CONTEXT = "normal_context"
    INFORMED_CONTEXT = "informed_context"
    CONFLICTED_CONTEXT = "conflicted_context"


class PreCallPlanStatus(str, Enum):
    """Whether the plan is usable."""

    READY = "ready"
    DEGRADED = "degraded"


class AssumptionGuardrail(str, Enum):
    """Generic do-not-assume guardrails derived from unresolved facts."""

    DO_NOT_ASSUME_NO_WEBSITE = "do_not_assume_no_website"
    DO_NOT_ASSUME_BAD_SEO = "do_not_assume_bad_seo"
    DO_NOT_ASSUME_MANUAL_WORKFLOW = "do_not_assume_manual_workflow"
    DO_NOT_ASSUME_DECISION_MAKER = "do_not_assume_decision_maker"
    DO_NOT_ASSUME_BUDGET = "do_not_assume_budget"
    DO_NOT_ASSUME_LANGUAGE = "do_not_assume_language"
    DO_NOT_ASSUME_CUSTOMER_ACQUISITION = "do_not_assume_customer_acquisition"


class RiskFlag(str, Enum):
    """Bounded advisory risk markers; no execution authority."""

    THIN_PROFILE = "thin_profile"
    CONFLICTING_ENRICHMENT = "conflicting_enrichment"
    NO_CAMPAIGN_SERVICE_AUTHORIZED = "no_campaign_service_authorized"
    UNVERIFIED_CONTACT = "unverified_contact"


# ---------------------------------------------------------------------------
# Fact model — reuses FieldKnowledgeState from leads
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreCallFact:
    """One pre-call semantic fact keyed by a stable string.

    The state uses app.leads.contracts.FieldKnowledgeState — the exact same
    four-state model lead intake uses, so KNOWN/UNKNOWN/UNVERIFIED/CONFLICTING
    semantics are consistent across the pipeline.
    """

    key: str
    state: FieldKnowledgeState
    value: str | None = None
    sources: tuple[str, ...] = ()
    conflict_detail: str | None = None

    def __post_init__(self) -> None:
        if not self.key or not self.key.strip():
            raise ValueError("fact key must not be empty")
        if len(self.key) > 100:
            raise ValueError("fact key exceeds 100 chars")
        if not isinstance(self.state, FieldKnowledgeState):
            raise TypeError("state must be FieldKnowledgeState")
        if self.value is not None and len(self.value) > 300:
            raise ValueError("fact value exceeds 300 chars")
        if self.state == FieldKnowledgeState.KNOWN and self.value is None:
            raise ValueError("KNOWN fact requires a value")
        if self.state == FieldKnowledgeState.UNKNOWN and self.value is not None:
            raise ValueError("UNKNOWN fact must not carry a value")
        if (
            self.state == FieldKnowledgeState.CONFLICTING
            and (self.conflict_detail is None or not self.conflict_detail.strip())
        ):
            raise ValueError("CONFLICTING fact requires conflict_detail")


# ---------------------------------------------------------------------------
# Verification target
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationTarget:
    """One fact PCIE recommends verifying during the conversation."""

    fact_key: str
    priority: VerificationPriority
    reason: str
    tie_breaker_index: int

    def __post_init__(self) -> None:
        if not self.fact_key or not self.fact_key.strip():
            raise ValueError("fact_key must not be empty")
        if len(self.fact_key) > 100:
            raise ValueError("fact_key exceeds 100 chars")
        if not isinstance(self.priority, VerificationPriority):
            raise TypeError("priority must be VerificationPriority")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must not be empty")
        if len(self.reason) > 200:
            raise ValueError("reason exceeds 200 chars")
        if self.tie_breaker_index < 0:
            raise ValueError("tie_breaker_index must be non-negative")


# ---------------------------------------------------------------------------
# Service prerequisite reference
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServicePrerequisiteGap:
    """A service and the fact keys that still need to be resolved for it.

    PCIE identifies MISSING prerequisites only. It does not authorize or
    recommend the service — that remains with existing conversation
    intelligence (ServiceRelevanceResolver, catalog eligibility).
    """

    service_id: str
    missing_fact_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.service_id or not self.service_id.strip():
            raise ValueError("service_id must not be empty")
        if not self.missing_fact_keys:
            raise ValueError("gap must have at least one missing fact")
        if len(self.missing_fact_keys) > 20:
            raise ValueError("too many missing facts")


# ---------------------------------------------------------------------------
# Policy — dynamic, config-driven, versioned
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServiceDiscoveryRule:
    """One service's discovery requirements, declared as fact keys."""

    service_id: str
    required_fact_keys: tuple[str, ...] = ()
    optional_fact_keys: tuple[str, ...] = ()
    knowledge_topic_hints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.service_id or not self.service_id.strip():
            raise ValueError("service_id must not be empty")
        if len(self.required_fact_keys) > 20:
            raise ValueError("too many required facts")
        if len(self.optional_fact_keys) > 20:
            raise ValueError("too many optional facts")
        if len(self.knowledge_topic_hints) > 20:
            raise ValueError("too many knowledge topic hints")
        duplicates = set(self.required_fact_keys) & set(self.optional_fact_keys)
        if duplicates:
            raise ValueError(f"fact keys cannot be both required and optional: {duplicates}")


@dataclass(frozen=True)
class PreCallDiscoveryPolicy:
    """Deterministic, versioned configuration governing PCIE behavior.

    Adding a new service requires updating this configuration (or its
    knowledge source), NOT writing new Python branches. Policy identity
    is (policy_id, version) — simulations remain reproducible when rules
    change by pinning to a version.
    """

    policy_id: str
    version: str
    service_rules: tuple[ServiceDiscoveryRule, ...] = ()
    global_required_facts: tuple[str, ...] = ()
    global_optional_facts: tuple[str, ...] = ()
    priority_overrides: tuple[tuple[str, VerificationPriority], ...] = ()

    def __post_init__(self) -> None:
        if not self.policy_id or not self.policy_id.strip():
            raise ValueError("policy_id must not be empty")
        if not self.version or not self.version.strip():
            raise ValueError("version must not be empty")
        if len(self.service_rules) > 50:
            raise ValueError("too many service rules")
        service_ids = [r.service_id for r in self.service_rules]
        if len(service_ids) != len(set(service_ids)):
            raise ValueError("duplicate service_id in rules")
        for key, prio in self.priority_overrides:
            if not isinstance(prio, VerificationPriority):
                raise TypeError("priority override must be VerificationPriority")
            if not key or len(key) > 100:
                raise ValueError("priority override key invalid")

    def rule_for(self, service_id: str) -> ServiceDiscoveryRule | None:
        for rule in self.service_rules:
            if rule.service_id == service_id:
                return rule
        return None

    def priority_override_for(self, fact_key: str) -> VerificationPriority | None:
        for key, prio in self.priority_overrides:
            if key == fact_key:
                return prio
        return None


# ---------------------------------------------------------------------------
# PreCallConversationPlan — PCIE output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreCallConversationPlan:
    """Immutable, read-only advisory plan feeding the chat runtime.

    This plan does NOT authorize services, mutate state, or generate
    wording. Downstream components may consult it as context; conversation
    truth from the live session always supersedes it.
    """

    lead_id: str
    campaign_id: str
    tenant_id: str
    policy_id: str
    policy_version: str

    known_facts: tuple[PreCallFact, ...] = ()
    unknown_facts: tuple[PreCallFact, ...] = ()
    unverified_facts: tuple[PreCallFact, ...] = ()
    conflicting_facts: tuple[PreCallFact, ...] = ()

    primary_initial_discovery_target: VerificationTarget | None = None
    verification_targets: tuple[VerificationTarget, ...] = ()

    opening_objective: OpeningObjective = OpeningObjective.ESTABLISH_CONTACT
    discovery_priorities: tuple[str, ...] = ()

    allowed_service_context: tuple[str, ...] = ()
    service_prerequisite_gaps: tuple[ServicePrerequisiteGap, ...] = ()

    assumption_guardrails: frozenset[AssumptionGuardrail] = frozenset()
    knowledge_topics_likely_needed: tuple[str, ...] = ()

    risk_flags: frozenset[RiskFlag] = frozenset()
    initial_posture: InitialPosture = InitialPosture.LOW_CONTEXT
    plan_status: PreCallPlanStatus = PreCallPlanStatus.READY

    def __post_init__(self) -> None:
        if not self.lead_id or not self.lead_id.strip():
            raise ValueError("lead_id must not be empty")
        if not self.campaign_id or not self.campaign_id.strip():
            raise ValueError("campaign_id must not be empty")
        if not self.tenant_id or not self.tenant_id.strip():
            raise ValueError("tenant_id must not be empty")
        if not self.policy_id or not self.policy_version:
            raise ValueError("policy identity required")

        for bucket, expected_state in (
            (self.known_facts, FieldKnowledgeState.KNOWN),
            (self.unknown_facts, FieldKnowledgeState.UNKNOWN),
            (self.unverified_facts, FieldKnowledgeState.UNVERIFIED),
            (self.conflicting_facts, FieldKnowledgeState.CONFLICTING),
        ):
            for fact in bucket:
                if fact.state != expected_state:
                    raise ValueError(
                        f"fact {fact.key} in {expected_state.value} bucket has wrong state"
                    )

        if self.primary_initial_discovery_target is not None:
            if self.primary_initial_discovery_target not in self.verification_targets:
                raise ValueError("primary target must appear in verification_targets")

        all_fact_keys = {f.key for f in (
            *self.known_facts, *self.unknown_facts,
            *self.unverified_facts, *self.conflicting_facts,
        )}
        if len(all_fact_keys) != sum(len(b) for b in (
            self.known_facts, self.unknown_facts,
            self.unverified_facts, self.conflicting_facts,
        )):
            raise ValueError("duplicate fact keys across buckets")

        if len(self.verification_targets) > 50:
            raise ValueError("too many verification targets")
        if len(self.service_prerequisite_gaps) > 20:
            raise ValueError("too many service prerequisite gaps")
