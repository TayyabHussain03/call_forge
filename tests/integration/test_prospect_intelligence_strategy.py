"""Prospect-intelligence integration with deterministic conversation strategy."""

from __future__ import annotations

from dataclasses import fields

from app.conversation.prospect_intelligence.contracts import (
    CurrentSolutionEvidence,
    ObjectionType,
    ObservedProspectEvidence,
    PreferredNextStep,
    ProspectEvidence,
    ProspectIntelligenceSnapshot,
    ProspectRole,
)
from app.conversation.prospect_intelligence.updater import ProspectIntelligenceUpdater
from app.conversation.strategy.contracts import (
    ConversationMode,
    ConversationStrategyInput,
    MicroCommitment,
    SalesStage,
    StrategyType,
)
from app.conversation.strategy.engine import ConversationStrategyEngine
from app.core.constants import ConversationState


def _snapshot(**evidence_fields: object) -> ProspectIntelligenceSnapshot:
    return ProspectIntelligenceUpdater().update(
        ProspectIntelligenceSnapshot(),
        ProspectEvidence(
            observed=ObservedProspectEvidence("turn-1", **evidence_fields)
        ),
    )


def _recommend(snapshot: ProspectIntelligenceSnapshot):  # type: ignore[no-untyped-def]
    return ConversationStrategyEngine().recommend(
        ConversationStrategyInput(
            current_stage=SalesStage.DISCOVERY,
            current_state=ConversationState.LISTEN,
            prospect_intelligence=snapshot,
        )
    )


def test_receptionist_scenario_routes_respectfully_without_forced_pitch() -> None:
    result = _recommend(_snapshot(explicit_role=ProspectRole.RECEPTIONIST))

    assert result.conversation_mode == ConversationMode.ROLE_ROUTING
    assert result.strategy_type == StrategyType.ROUTE_TO_DECISION_MAKER
    assert "respectfully" in result.communication_goal


def test_owner_scenario_pursues_impact_without_pricing_authority() -> None:
    result = _recommend(_snapshot(explicit_role=ProspectRole.OWNER))

    assert result.strategy_type == StrategyType.DISCOVER_BUSINESS_IMPACT
    assert "impact" in result.communication_goal
    assert "pricing" not in {field.name for field in fields(result)}


def test_existing_provider_scenario_understands_gap_without_competitor_claims() -> None:
    result = _recommend(
        _snapshot(
            current_solution=CurrentSolutionEvidence("Existing Provider"),
            objection=ObjectionType.ALREADY_HAVE_PROVIDER,
        )
    )

    assert result.strategy_type == StrategyType.UNDERSTAND_CURRENT_SOLUTION
    assert result.conversation_mode == ConversationMode.OBJECTION
    assert result.communication_goal == (
        "understand satisfaction and any gap in the current solution"
    )


def test_busy_but_interested_scenario_preserves_both_and_reduces_pressure() -> None:
    snapshot = _snapshot(busy=True, interest=True)
    result = _recommend(snapshot)

    assert snapshot.observed.explicit_busy_signal is not None
    assert snapshot.observed.explicit_interest_signal is not None
    assert result.conversation_mode == ConversationMode.BUSY
    assert result.micro_commitment == MicroCommitment.CALLBACK
    assert "not_interested" not in {field.name for field in fields(snapshot.observed)}


def test_explicit_demo_scenario_accelerates_without_confirming_booking() -> None:
    result = _recommend(
        _snapshot(explicit_next_step=PreferredNextStep.DEMO, interest=True)
    )

    assert result.sales_stage == SalesStage.NEXT_STEP
    assert result.micro_commitment == MicroCommitment.DEMO
    assert "confirmed" not in {field.name for field in fields(result)}
