"""Pure deterministic construction of bounded model-facing context."""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.brain.authority.models import AuthorityPolicy
from app.brain.contracts import BusinessIntelligenceSnapshot
from app.catalog.claim_validator import ClaimValidator
from app.contracts.conversation_context import ConversationContext
from app.conversation.context.contracts import (
    DEFAULT_CONTEXT_BOUNDS,
    ApprovedEvidenceItem,
    ApprovedEvidenceSourceKind,
    BrainContextView,
    BusinessContext,
    CampaignContext,
    ContactContext,
    ContactContextStatus,
    ContextBounds,
    EvidenceScope,
    EvidenceScopeKind,
    EvidenceType,
    InferredBusinessSignal,
    LeanTurnContext,
    ObservedBusinessFact,
    PolicyContext,
    RecentTurn,
    SupervisorAdvisoryContext,
    SupervisorContextView,
    TurnSpeaker,
    UntrustedUserInput,
)
from app.conversation.prospect_intelligence.contracts import (
    InferredProspectState,
    ProspectIntelligenceSummary,
    ProspectIntelligenceSnapshot,
)
from app.conversation.prospect_intelligence.updater import ProspectIntelligenceUpdater
from app.conversation.response_planning.contracts import (
    AddresseeStatus,
    InterruptionCategory,
    InterruptionContext,
    PendingConversationIntent,
)
from app.conversation.strategy.contracts import ConversationStrategy
from app.conversation.supervisor.contracts import SupervisorInsight
from app.core.constants import ConversationState


class EvidenceConflictError(ValueError):
    """Two approved statements disagree for the same scoped fact key."""


@dataclass(frozen=True)
class LeanContextBuildInput:
    """Trusted source objects; only bounded projections leave the builder."""

    call_id: str
    current_turn_id: str
    current_turn_sequence: int
    current_user_message: str
    current_state: ConversationState
    conversation_context: ConversationContext
    strategy: ConversationStrategy | None = None
    recent_turns: tuple[RecentTurn, ...] = ()
    prospect_intelligence: ProspectIntelligenceSnapshot | None = None
    prospect_summary: ProspectIntelligenceSummary | None = None
    pending_intent: PendingConversationIntent | None = None
    approved_evidence: tuple[ApprovedEvidenceItem, ...] = ()
    authority_policy: AuthorityPolicy | None = None
    business_intelligence: BusinessIntelligenceSnapshot | None = None
    relevant_business_fields: tuple[str, ...] = ()
    campaign_goal: str | None = None
    discovery_priorities: tuple[str, ...] = ()
    contact: ContactContext | None = None
    interruption: InterruptionContext = InterruptionContext()
    conversation_category: InterruptionCategory = InterruptionCategory.OTHER
    addressee_status: AddresseeStatus = AddresseeStatus.ADDRESSED_TO_AGENT
    supervisor_insight: SupervisorInsight | None = None

    def __post_init__(self) -> None:
        if not self.call_id or not self.current_turn_id:
            raise ValueError("call and turn identity are required")
        if self.current_turn_sequence < 0:
            raise ValueError("current turn sequence must be non-negative")
        if self.conversation_context.call_id != self.call_id:
            raise ValueError("conversation context belongs to a different call")
        if self.prospect_intelligence is not None and self.prospect_summary is not None:
            raise ValueError("supply a prospect snapshot or projection, not both")


class LeanContextBuilder:
    """Build fresh context using bounded, stable, in-memory projections only."""

    def __init__(self, bounds: ContextBounds = DEFAULT_CONTEXT_BOUNDS) -> None:
        self._bounds = bounds

    def build(self, source: LeanContextBuildInput) -> LeanTurnContext:
        """Return the same immutable output for the same trusted input."""
        message = source.current_user_message[: self._bounds.max_current_message_chars]
        if not message.strip():
            raise ValueError("current user message cannot be empty")
        recent_candidates = tuple(
            item for item in source.recent_turns if item.turn_id != source.current_turn_id
        )
        recent = tuple(
            RecentTurn(
                item.turn_id,
                item.speaker,
                item.content[: self._bounds.max_recent_turn_chars],
            )
            for item in recent_candidates[-self._bounds.max_recent_turns :]
        )
        eligible = _stable_ids(
            source.conversation_context.eligible_alternative_service_ids,
            self._bounds.max_service_ids,
        )
        offered = _stable_ids(
            source.conversation_context.offered_service_ids,
            self._bounds.max_service_ids,
        )
        evidence = self._evidence(
            source.approved_evidence,
            source.conversation_context.campaign_id,
            frozenset((*eligible, *offered)),
        )
        prospect = (
            source.prospect_intelligence.to_brain_summary()
            if source.prospect_intelligence is not None
            else source.prospect_summary
        )
        return LeanTurnContext(
            untrusted_user_input=UntrustedUserInput(message),
            current_state=source.current_state,
            strategy=source.strategy,
            recent_turns=recent,
            prospect=prospect,
            pending_intent=source.pending_intent,
            eligible_service_ids=eligible,
            offered_service_ids=offered,
            approved_evidence=evidence,
            policy=_policy(source.authority_policy),
            business=self._business(
                source.business_intelligence, source.relevant_business_fields
            ),
            campaign=CampaignContext(
                campaign_id=source.conversation_context.campaign_id,
                campaign_goal=_optional(source.campaign_goal, 200),
                discovery_priorities=tuple(
                    item[:100]
                    for item in source.discovery_priorities[
                        : self._bounds.max_discovery_priorities
                    ]
                    if item.strip()
                ),
            ),
            contact=source.contact or _contact(source.conversation_context),
            interruption=source.interruption,
            conversation_category=source.conversation_category,
            addressee_status=source.addressee_status,
            supervisor_advisory=_supervisor(source, prospect),
        )

    @staticmethod
    def for_brain(context: LeanTurnContext) -> BrainContextView:
        return BrainContextView(context)

    @staticmethod
    def for_supervisor(context: LeanTurnContext) -> SupervisorContextView:
        return SupervisorContextView(context)

    def _evidence(
        self,
        items: tuple[ApprovedEvidenceItem, ...],
        campaign_id: str | None,
        service_ids: frozenset[str],
    ) -> tuple[ApprovedEvidenceItem, ...]:
        applicable = [
            item
            for item in items
            if _scope_applies(item.scope, campaign_id, service_ids)
        ]
        by_fact: dict[tuple[EvidenceScope, str], ApprovedEvidenceItem] = {}
        by_id: dict[str, ApprovedEvidenceItem] = {}
        for item in applicable:
            existing_id = by_id.get(item.evidence_id)
            if existing_id is not None and existing_id != item:
                raise EvidenceConflictError("conflicting evidence id")
            key = (item.scope, item.fact_key)
            existing_fact = by_fact.get(key)
            if existing_fact is not None and existing_fact.statement != item.statement:
                raise EvidenceConflictError("conflicting scoped evidence fact")
            by_id[item.evidence_id] = item
            by_fact[key] = item
        return tuple(
            sorted(by_id.values(), key=lambda item: item.evidence_id)[
                : self._bounds.max_evidence_items
            ]
        )

    def _business(
        self,
        snapshot: BusinessIntelligenceSnapshot | None,
        relevant_fields: tuple[str, ...],
    ) -> BusinessContext:
        if snapshot is None:
            return BusinessContext()
        fields = tuple(sorted(set(relevant_fields)))[: self._bounds.max_business_fields]
        observed = []
        inferred = []
        unknown = []
        for field in fields:
            observed_matches = [item for item in snapshot.observed if item.field == field]
            if observed_matches:
                item = observed_matches[-1]
                observed.append(
                    ObservedBusinessFact(
                        item.field[:80], item.value[:160], item.provenance.source_kind
                    )
                )
                continue
            inferred_matches = [item for item in snapshot.inferred if item.field == field]
            if inferred_matches:
                item = inferred_matches[-1]
                inferred.append(
                    InferredBusinessSignal(
                        item.field[:80],
                        item.value[:160],
                        item.inference_confidence,
                        tuple(value[:80] for value in item.basis[:3]),
                    )
                )
            elif any(item.field == field for item in snapshot.unknown):
                unknown.append(field[:80])
        return BusinessContext(tuple(observed), tuple(inferred), tuple(unknown))


def approve_catalog_claim(
    validator: ClaimValidator,
    *,
    evidence_id: str,
    fact_key: str,
    service_id: str,
    claim_id: str,
    subservice_id: str | None = None,
) -> ApprovedEvidenceItem | None:
    """Create evidence only after the existing ClaimValidator authorizes it."""
    result = validator.validate_claim(service_id, claim_id, subservice_id)
    if not result.authorized:
        return None
    return ApprovedEvidenceItem(
        evidence_id=evidence_id,
        fact_key=fact_key,
        evidence_type=EvidenceType.APPROVED_CLAIM,
        statement=claim_id,
        source_kind=ApprovedEvidenceSourceKind.CATALOG_CLAIM,
        source_reference=claim_id,
        scope=EvidenceScope(EvidenceScopeKind.SERVICE, service_id),
    )


def _stable_ids(values: tuple[str, ...], limit: int) -> tuple[str, ...]:
    return tuple(sorted({value for value in values if value.strip()}))[:limit]


def _optional(value: str | None, limit: int) -> str | None:
    return value[:limit] if value is not None and value.strip() else None


def _scope_applies(
    scope: EvidenceScope,
    campaign_id: str | None,
    service_ids: frozenset[str],
) -> bool:
    if scope.kind == EvidenceScopeKind.GLOBAL:
        return True
    if scope.kind == EvidenceScopeKind.CAMPAIGN:
        return scope.scope_id == campaign_id
    return scope.scope_id in service_ids


def _policy(policy: AuthorityPolicy | None) -> PolicyContext | None:
    if policy is None:
        return None
    disclosure = policy.pricing_disclosure
    return PolicyContext(
        policy.policy_id,
        disclosure is not None and disclosure.allowed,
        policy.default_tier,
    )


def _contact(context: ConversationContext) -> ContactContext:
    if context.contact_candidate is None:
        return ContactContext()
    status = (
        ContactContextStatus.CONFIRMED
        if context.contact_confirmed
        else ContactContextStatus.CANDIDATE
    )
    return ContactContext(status, context.contact_candidate_channel)


def _supervisor(
    source: LeanContextBuildInput,
    prospect,
) -> SupervisorAdvisoryContext | None:
    insight = source.supervisor_insight
    if (
        insight is None
        or insight.call_id != source.call_id
        or insight.source_turn_sequence >= source.current_turn_sequence
    ):
        return None
    inferred_evidence = (
        insight.prospect_evidence.inferred if insight.prospect_evidence else None
    )
    inferred = None
    if inferred_evidence is not None:
        inferred = ProspectIntelligenceUpdater().update(
            ProspectIntelligenceSnapshot(),
            insight.prospect_evidence,
        ).inferred
    if inferred is not None and prospect is not None:
        inferred = replace(
            inferred,
            likely_role=None if prospect.explicit_role else inferred.likely_role,
            decision_authority=(
                None
                if prospect.explicit_decision_authority
                else inferred.decision_authority
            ),
            objection_type=(
                None if prospect.explicit_objection else inferred.objection_type
            ),
            pain_summary=(
                None if prospect.observed_pain_points else inferred.pain_summary
            ),
            preferred_next_step=(
                None if prospect.explicit_next_step else inferred.preferred_next_step
            ),
        )
    return SupervisorAdvisoryContext(
        insight.source_turn_id,
        inferred,
        insight.recommended_strategy_hint,
    )
