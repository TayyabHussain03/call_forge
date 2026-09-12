"""Slice 14 lean-context integration scenarios across existing typed domains."""

from __future__ import annotations

from dataclasses import replace

from app.brain.authority.models import AuthorityPolicy, AuthorityTier, PricingDisclosureRule
from app.contracts.conversation_context import ConversationContext
from app.conversation.context.builder import LeanContextBuildInput, LeanContextBuilder
from app.conversation.context.contracts import (
    ApprovedEvidenceItem,
    ApprovedEvidenceSourceKind,
    EvidenceScope,
    EvidenceScopeKind,
    EvidenceType,
    RecentTurn,
    TurnSpeaker,
)
from app.conversation.prospect_intelligence.contracts import (
    CurrentSolutionEvidence,
    InferenceCandidate,
    InferredProspectEvidence,
    ObjectionType,
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
)
from app.conversation.strategy.contracts import (
    ConversationMode,
    ConversationStrategy,
    SalesStage,
    StrategyType,
)
from app.conversation.supervisor.buffer import StrategyBuffer
from app.conversation.supervisor.contracts import SupervisorInsight
from app.core.constants import ConversationState


def _strategy(mode: ConversationMode = ConversationMode.NORMAL) -> ConversationStrategy:
    return ConversationStrategy(
        SalesStage.DISCOVERY,
        mode,
        "understand the prospect's current operational need",
        StrategyType.DISCOVER_NEED,
    )


def _source(**changes) -> LeanContextBuildInput:  # type: ignore[no-untyped-def]
    source = LeanContextBuildInput(
        call_id="call",
        current_turn_id="turn-5",
        current_turn_sequence=5,
        current_user_message="Current question",
        current_state=ConversationState.LISTEN,
        conversation_context=ConversationContext(
            "call",
            campaign_id="campaign_a",
            eligible_alternative_service_ids=("seo",),
        ),
        strategy=_strategy(),
    )
    return replace(source, **changes)


def test_scenario_a_normal_discovery_is_current_and_minimal() -> None:
    prospect = ProspectIntelligenceUpdater().update(
        ProspectIntelligenceSnapshot(),
        ProspectEvidence(
            observed=ObservedProspectEvidence(
                "turn-4",
                explicit_role=ProspectRole.OWNER,
                explicit_pain=PainEvidence(PainCategory.OPERATIONS, "manual intake"),
            )
        ),
    )
    recent = tuple(RecentTurn(f"t-{i}", TurnSpeaker.USER, str(i)) for i in range(8))
    context = LeanContextBuilder().build(
        _source(prospect_intelligence=prospect, recent_turns=recent)
    )
    assert context.current_state == ConversationState.LISTEN
    assert context.strategy == _strategy()
    assert len(context.recent_turns) == 4
    assert context.prospect is not None
    assert context.prospect.explicit_role is not None
    assert context.prospect.observed_pain_points[0].summary == "manual intake"
    assert not hasattr(context, "prospect_history")


def test_scenario_b_existing_provider_has_only_supported_service_evidence() -> None:
    prospect = ProspectIntelligenceUpdater().update(
        ProspectIntelligenceSnapshot(),
        ProspectEvidence(
            observed=ObservedProspectEvidence(
                "turn-4",
                current_solution=CurrentSolutionEvidence("Existing agency"),
                objection=ObjectionType.ALREADY_HAVE_PROVIDER,
            )
        ),
    )
    evidence = ApprovedEvidenceItem(
        "seo-1",
        "seo-visibility",
        EvidenceType.APPROVED_CLAIM,
        "SEO may improve organic visibility.",
        ApprovedEvidenceSourceKind.CURATED_SERVICE,
        EvidenceScope(EvidenceScopeKind.SERVICE, "seo"),
    )
    context = LeanContextBuilder().build(
        _source(prospect_intelligence=prospect, approved_evidence=(evidence,))
    )
    assert context.prospect is not None and context.prospect.current_solution is not None
    assert context.prospect.explicit_objection is not None
    assert context.approved_evidence == (evidence,)
    assert all("competitor" not in item.statement for item in context.approved_evidence)


def test_scenario_c_price_question_excludes_operational_provider_costs() -> None:
    policy = AuthorityPolicy(
        "standard",
        AuthorityTier.HUMAN_APPROVAL_REQUIRED,
        pricing_disclosure=PricingDisclosureRule(False, AuthorityTier.HUMAN_APPROVAL_REQUIRED),
    )
    context = LeanContextBuilder().build(
        _source(
            strategy=_strategy(ConversationMode.QUESTION_DETOUR),
            authority_policy=policy,
        )
    )
    assert context.strategy is not None
    assert context.strategy.conversation_mode == ConversationMode.QUESTION_DETOUR
    assert context.policy is not None
    assert context.policy.pricing_disclosure_allowed is False
    assert "provider_pricing" not in context.__dataclass_fields__
    assert context.approved_evidence == ()


def test_scenario_d_interruption_contains_one_pending_point_not_history() -> None:
    pending = PendingConversationIntent("answer", "finish the relevant explanation")
    interruption = InterruptionContext(
        True,
        InterruptionCategory.QUESTION,
        pending,
    )
    context = LeanContextBuilder().build(
        _source(
            current_user_message="But what does that cost?",
            pending_intent=pending,
            interruption=interruption,
            addressee_status=AddresseeStatus.ADDRESSED_TO_AGENT,
        )
    )
    assert context.untrusted_user_input.message == "But what does that cost?"
    assert context.pending_intent is pending
    assert context.interruption is interruption
    assert not hasattr(context, "interruption_history")


def test_scenario_e_only_latest_applicable_supervisor_advice_is_projected() -> None:
    buffer = StrategyBuffer("call")
    latest = SupervisorInsight(
        "call",
        "turn-4",
        4,
        ProspectEvidence(
            inferred=InferredProspectEvidence(
                "turn-4", likely_role=InferenceCandidate(ProspectRole.MANAGER, 0.9)
            )
        ),
    )
    stale = SupervisorInsight(
        "call",
        "turn-3",
        3,
        ProspectEvidence(
            inferred=InferredProspectEvidence(
                "turn-3", likely_role=InferenceCandidate(ProspectRole.RECEPTIONIST, 0.9)
            )
        ),
    )
    buffer.record(latest)
    buffer.record(stale)
    insight = buffer.consume_for(5)
    assert insight is latest
    context = LeanContextBuilder().build(_source(supervisor_insight=insight))
    assert context.supervisor_advisory is not None
    assert context.supervisor_advisory.source_turn_id == "turn-4"
    assert buffer.consume_for(6) is None
