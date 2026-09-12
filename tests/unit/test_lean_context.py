"""Focused behavioral tests for the canonical lean-context boundary."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace

import pytest

from app.brain.authority.models import AuthorityPolicy, AuthorityTier, PricingDisclosureRule
from app.brain.business_intelligence import (
    InferredSignal,
    ObservedSignal,
    Provenance,
    SourceKind,
    UnknownSlot,
)
from app.brain.contracts import BusinessIntelligenceSnapshot
from app.catalog.claim_validator import ClaimValidator
from app.catalog.loader import load_catalog
from app.catalog.scoped_catalog import ScopedCatalog
from app.contracts.conversation_context import ConversationContext
from app.conversation.context.builder import (
    EvidenceConflictError,
    LeanContextBuildInput,
    LeanContextBuilder,
    approve_catalog_claim,
)
from app.conversation.context.contracts import (
    ApprovedEvidenceItem,
    ApprovedEvidenceSourceKind,
    ContactContextStatus,
    EvidenceScope,
    EvidenceScopeKind,
    EvidenceType,
    LeanTurnContext,
    PolicyContext,
    RecentTurn,
    SupervisorAdvisoryContext,
    TurnSpeaker,
)
from app.conversation.prospect_intelligence.contracts import (
    InferenceCandidate,
    InferredProspectEvidence,
    ObservedProspectEvidence,
    PainCategory,
    PainEvidence,
    ProspectEvidence,
    ProspectIntelligenceSnapshot,
    ProspectRole,
)
from app.conversation.prospect_intelligence.updater import ProspectIntelligenceUpdater
from app.conversation.response_planning.contracts import (
    AddresseeStatus,
    InterruptionCategory,
    InterruptionContext,
    PendingConversationIntent,
    VoiceActivityMetadata,
)
from app.conversation.strategy.contracts import (
    ConversationMode,
    ConversationStrategy,
    ConversationStrategyHint,
    InformationGap,
    MicroCommitment,
    SalesStage,
    StrategyType,
)
from app.conversation.supervisor.contracts import SupervisorInput, SupervisorInsight
from app.core.constants import ConversationState


def _strategy(mode: ConversationMode = ConversationMode.NORMAL) -> ConversationStrategy:
    return ConversationStrategy(
        SalesStage.DISCOVERY,
        mode,
        "understand the current workflow",
        StrategyType.DISCOVER_NEED,
        (InformationGap.PAIN_POINT,),
        MicroCommitment.PERMISSION_TO_CONTINUE,
    )


def _evidence(
    evidence_id: str = "e-1",
    *,
    fact_key: str = "capability",
    statement: str = "The service supports workflow automation.",
    scope: EvidenceScope | None = None,
) -> ApprovedEvidenceItem:
    return ApprovedEvidenceItem(
        evidence_id,
        fact_key,
        EvidenceType.APPROVED_CLAIM,
        statement,
        ApprovedEvidenceSourceKind.CURATED_SERVICE,
        scope or EvidenceScope(EvidenceScopeKind.GLOBAL),
        "curated-1",
    )


def _source(**changes) -> LeanContextBuildInput:  # type: ignore[no-untyped-def]
    base = LeanContextBuildInput(
        call_id="call",
        current_turn_id="turn-5",
        current_turn_sequence=5,
        current_user_message="How would this help us?",
        current_state=ConversationState.LISTEN,
        conversation_context=ConversationContext(
            "call",
            campaign_id="campaign_a",
            eligible_alternative_service_ids=("seo", "website_development"),
            offered_service_ids=("website_development",),
        ),
        strategy=_strategy(),
    )
    return replace(base, **changes)


def _build(**changes) -> LeanTurnContext:  # type: ignore[no-untyped-def]
    return LeanContextBuilder().build(_source(**changes))


def _prospect() -> ProspectIntelligenceSnapshot:
    updater = ProspectIntelligenceUpdater()
    snapshot = updater.update(
        ProspectIntelligenceSnapshot(),
        ProspectEvidence(
            inferred=InferredProspectEvidence(
                "turn-1",
                likely_role=InferenceCandidate(ProspectRole.MANAGER, 0.8),
            )
        ),
    )
    return updater.update(
        snapshot,
        ProspectEvidence(
            observed=ObservedProspectEvidence(
                "turn-2",
                explicit_role=ProspectRole.OWNER,
                explicit_pain=PainEvidence(PainCategory.TIME, "manual follow-up"),
            )
        ),
    )


def test_lean_turn_context_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        _build().current_state = ConversationState.NEW_CALL  # type: ignore[misc]


def test_same_inputs_produce_same_context() -> None:
    source = _source()
    assert LeanContextBuilder().build(source) == LeanContextBuilder().build(source)


def test_full_transcript_is_not_a_context_field() -> None:
    assert "full_transcript" not in {item.name for item in fields(LeanTurnContext)}


def test_recent_turns_are_bounded() -> None:
    recent = tuple(RecentTurn(f"t-{i}", TurnSpeaker.USER, str(i)) for i in range(8))
    assert len(_build(recent_turns=recent).recent_turns) == 4


def test_recent_turns_keep_deterministic_recent_order() -> None:
    recent = tuple(RecentTurn(f"t-{i}", TurnSpeaker.USER, str(i)) for i in range(6))
    assert [item.content for item in _build(recent_turns=recent).recent_turns] == [
        "2", "3", "4", "5"
    ]


def test_prospect_observed_and_inferred_remain_distinct() -> None:
    prospect = _build(prospect_intelligence=_prospect()).prospect
    assert prospect is not None and prospect.explicit_role is not None
    assert prospect.likely_role is None


def test_strategy_history_is_excluded() -> None:
    assert "strategy_history" not in {item.name for item in fields(LeanTurnContext)}


def test_only_current_strategy_is_included() -> None:
    strategy = _strategy(ConversationMode.QUESTION_DETOUR)
    assert _build(strategy=strategy).strategy is strategy


def test_pending_intent_uses_existing_bounded_contract() -> None:
    pending = PendingConversationIntent("explain value", "unfinished point")
    assert _build(pending_intent=pending).pending_intent is pending


def test_stale_or_same_turn_supervisor_insight_is_excluded() -> None:
    insight = SupervisorInsight(
        "call",
        "turn-5",
        5,
        ProspectEvidence(
            inferred=InferredProspectEvidence(
                "turn-5", likely_role=InferenceCandidate(ProspectRole.MANAGER, 0.8)
            )
        ),
    )
    assert _build(supervisor_insight=insight).supervisor_advisory is None


def test_fresh_supervisor_insight_contributes_advisory_fields() -> None:
    hint = ConversationStrategyHint(StrategyType.DISCOVER_BUSINESS_IMPACT, 0.8)
    insight = SupervisorInsight("call", "turn-4", 4, recommended_strategy_hint=hint)
    advisory = _build(supervisor_insight=insight).supervisor_advisory
    assert advisory is not None and advisory.strategy_hint is hint


def test_raw_supervisor_reasoning_is_excluded() -> None:
    assert "reasoning" not in {
        item.name for item in fields(SupervisorAdvisoryContext)
    }


@pytest.mark.parametrize(
    "forbidden",
    ("provider_pricing", "token_pricing", "telephony_cost", "stt_pricing"),
)
def test_provider_pricing_is_excluded(forbidden: str) -> None:
    assert forbidden not in {item.name for item in fields(LeanTurnContext)}


def test_provider_credentials_are_excluded() -> None:
    assert "provider_credentials" not in {item.name for item in fields(LeanTurnContext)}


def test_environment_secrets_are_excluded() -> None:
    assert "environment" not in {item.name for item in fields(LeanTurnContext)}


def test_full_crm_history_is_excluded() -> None:
    assert "crm_history" not in {item.name for item in fields(LeanTurnContext)}


def test_operational_logs_are_excluded() -> None:
    assert "logs" not in {item.name for item in fields(LeanTurnContext)}


def test_eligible_services_are_bounded_and_sorted() -> None:
    values = tuple(f"service-{i:02}" for i in range(20, -1, -1))
    context = replace(_source().conversation_context, eligible_alternative_service_ids=values)
    result = _build(conversation_context=context).eligible_service_ids
    assert len(result) == 12 and result == tuple(sorted(result))


def test_offered_services_are_bounded_and_sorted() -> None:
    values = tuple(f"service-{i:02}" for i in range(20, -1, -1))
    context = replace(_source().conversation_context, offered_service_ids=values)
    result = _build(conversation_context=context).offered_service_ids
    assert len(result) == 12 and result == tuple(sorted(result))


def test_approved_evidence_is_bounded() -> None:
    items = tuple(_evidence(f"e-{i:02}", fact_key=f"fact-{i}") for i in range(12))
    assert len(_build(approved_evidence=items).approved_evidence) == 8


def test_evidence_retains_provenance() -> None:
    item = _build(approved_evidence=(_evidence(),)).approved_evidence[0]
    assert item.source_kind == ApprovedEvidenceSourceKind.CURATED_SERVICE
    assert item.source_reference == "curated-1"


def test_evidence_scope_is_preserved() -> None:
    scope = EvidenceScope(EvidenceScopeKind.CAMPAIGN, "campaign_a")
    assert _build(approved_evidence=(_evidence(scope=scope),)).approved_evidence[0].scope == scope


def test_conflicting_evidence_is_rejected_deterministically() -> None:
    first = _evidence("a", fact_key="same", statement="One statement")
    second = _evidence("b", fact_key="same", statement="Different statement")
    with pytest.raises(EvidenceConflictError, match="scoped evidence"):
        _build(approved_evidence=(second, first))


def test_missing_evidence_creates_no_claim() -> None:
    assert _build().approved_evidence == ()


def test_current_message_is_structurally_untrusted() -> None:
    assert _build().untrusted_user_input.message == "How would this help us?"


def test_policy_projection_has_no_action_authority() -> None:
    names = {item.name for item in fields(PolicyContext)}
    assert names.isdisjoint({"approved_action", "next_state", "execute"})


def test_contact_projection_omits_raw_value() -> None:
    context = replace(_source().conversation_context, contact_candidate="secret@example.com")
    contact = _build(conversation_context=context).contact
    assert "value" not in {item.name for item in fields(contact)}


def test_unconfirmed_contact_stays_candidate() -> None:
    context = replace(
        _source().conversation_context,
        contact_candidate="secret@example.com",
        contact_candidate_channel="email",
    )
    assert _build(conversation_context=context).contact.status == ContactContextStatus.CANDIDATE


def test_interruption_metadata_is_current_and_bounded() -> None:
    interruption = InterruptionContext(
        True,
        InterruptionCategory.QUESTION,
        PendingConversationIntent("finish", "one unfinished point"),
        VoiceActivityMetadata(True, True, False),
    )
    assert _build(interruption=interruption).interruption is interruption


def test_addressee_metadata_is_typed_current_signal() -> None:
    result = _build(addressee_status=AddresseeStatus.ADDRESSEE_UNCERTAIN)
    assert result.addressee_status == AddresseeStatus.ADDRESSEE_UNCERTAIN


def test_business_intelligence_is_projected_not_dumped() -> None:
    snapshot = BusinessIntelligenceSnapshot(
        observed=(
            ObservedSignal("website_status", "active", Provenance(SourceKind.CLIENT_STATED)),
            ObservedSignal("unrelated", "history", Provenance(SourceKind.LEAD_IMPORT)),
        )
    )
    business = _build(
        business_intelligence=snapshot,
        relevant_business_fields=("website_status",),
    ).business
    assert [item.field for item in business.observed] == ["website_status"]


def test_business_projection_preserves_unknown_and_inference() -> None:
    snapshot = BusinessIntelligenceSnapshot(
        inferred=(InferredSignal("workflow", "manual", 0.7),),
        unknown=(UnknownSlot("business_type"),),
    )
    business = _build(
        business_intelligence=snapshot,
        relevant_business_fields=("workflow", "business_type"),
    ).business
    assert business.inferred[0].confidence == 0.7
    assert business.unknown_fields == ("business_type",)


def test_campaign_context_is_bounded() -> None:
    result = _build(
        campaign_goal="g" * 300,
        discovery_priorities=tuple(f"priority-{i}" for i in range(8)),
    ).campaign
    assert len(result.campaign_goal or "") == 200
    assert len(result.discovery_priorities) == 4


def test_overflow_truncation_replays_identically() -> None:
    items = tuple(_evidence(f"e-{i:02}", fact_key=f"f-{i}") for i in range(12, -1, -1))
    source = _source(approved_evidence=items)
    assert LeanContextBuilder().build(source) == LeanContextBuilder().build(source)


def test_builder_does_not_mutate_input_objects() -> None:
    source = _source(recent_turns=(RecentTurn("old", TurnSpeaker.USER, "hello"),))
    before = source.recent_turns
    LeanContextBuilder().build(source)
    assert source.recent_turns == before


def test_brain_view_uses_canonical_context() -> None:
    context = _build()
    assert LeanContextBuilder.for_brain(context).turn is context


def test_supervisor_input_uses_canonical_bounded_view() -> None:
    context = _build()
    value = SupervisorInput("call", "turn-5", 5, LeanContextBuilder.for_supervisor(context))
    assert value.context.turn is context


def test_supervisor_view_has_no_full_transcript() -> None:
    view = LeanContextBuilder.for_supervisor(_build())
    assert "full_transcript" not in {item.name for item in fields(view.turn)}


def test_brain_view_has_no_provider_pricing() -> None:
    view = LeanContextBuilder.for_brain(_build())
    assert "pricing" not in {item.name for item in fields(view.turn)}


def test_service_scoped_evidence_does_not_apply_to_other_service() -> None:
    item = _evidence(scope=EvidenceScope(EvidenceScopeKind.SERVICE, "ai_automation"))
    assert _build(approved_evidence=(item,)).approved_evidence == ()


def test_catalog_claim_factory_reuses_claim_validator() -> None:
    catalog = load_catalog("app/config/service_config.yaml")
    validator = ClaimValidator(ScopedCatalog(catalog, "campaign_a"))
    allowed = approve_catalog_claim(
        validator,
        evidence_id="claim-1",
        fact_key="seo-capability",
        service_id="seo",
        claim_id="improve_visibility",
    )
    denied = approve_catalog_claim(
        validator,
        evidence_id="claim-2",
        fact_key="guarantee",
        service_id="seo",
        claim_id="guaranteed_first_page",
    )
    assert allowed is not None and allowed.statement == "improve_visibility"
    assert denied is None


def test_policy_projection_comes_from_existing_policy_metadata() -> None:
    policy = AuthorityPolicy(
        "policy",
        AuthorityTier.HUMAN_APPROVAL_REQUIRED,
        pricing_disclosure=PricingDisclosureRule(True, AuthorityTier.POLICY_BOUNDED),
    )
    result = _build(authority_policy=policy).policy
    assert result is not None and result.pricing_disclosure_allowed is True


def test_current_turn_is_not_duplicated_in_recent_window() -> None:
    recent = (RecentTurn("turn-5", TurnSpeaker.USER, "duplicate"),)
    assert _build(recent_turns=recent).recent_turns == ()


def test_long_current_message_is_deterministically_truncated() -> None:
    result = _build(current_user_message="x" * 2_500)
    assert len(result.untrusted_user_input.message) == 2_000


def test_observed_fact_hides_conflicting_supervisor_inference() -> None:
    insight = SupervisorInsight(
        "call",
        "turn-4",
        4,
        ProspectEvidence(
            inferred=InferredProspectEvidence(
                "turn-4", likely_role=InferenceCandidate(ProspectRole.MANAGER, 1.0)
            )
        ),
    )
    advisory = _build(
        prospect_intelligence=_prospect(), supervisor_insight=insight
    ).supervisor_advisory
    assert advisory is not None and advisory.inferred_prospect is not None
    assert advisory.inferred_prospect.likely_role is None


def test_context_contains_no_execution_or_transition_field() -> None:
    names = {item.name for item in fields(LeanTurnContext)}
    assert names.isdisjoint({"action", "approved_action", "next_state", "transition"})
