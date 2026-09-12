"""Focused behavioral tests for Slice 12 prospect intelligence."""

from __future__ import annotations

from dataclasses import fields

from app.brain.contracts import BrainInput
from app.config.settings import get_settings
from app.conversation.prospect_intelligence.contracts import (
    BuyingStage,
    CurrentSolutionEvidence,
    DecisionAuthority,
    InferenceCandidate,
    InferredProspectEvidence,
    InformationLevel,
    ObjectionType,
    ObservedProspectEvidence,
    PainCategory,
    PainEvidence,
    PreferredNextStep,
    ProspectEvidence,
    ProspectIntelligenceSnapshot,
    ProspectRole,
    SolutionSatisfaction,
)
from app.conversation.prospect_intelligence.updater import ProspectIntelligenceUpdater
from app.conversation.state_machine.machine import ConversationStateMachine
from app.conversation.state_machine.states import load_config
from app.conversation.strategy.contracts import (
    ConversationMode,
    ConversationStrategyInput,
    MicroCommitment,
    SalesStage,
    StrategyType,
)
from app.conversation.strategy.engine import ConversationStrategyEngine
from app.core.constants import ConversationState


def _update(
    previous: ProspectIntelligenceSnapshot | None = None,
    *,
    observed: ObservedProspectEvidence | None = None,
    inferred: InferredProspectEvidence | None = None,
) -> ProspectIntelligenceSnapshot:
    return ProspectIntelligenceUpdater().update(
        previous or ProspectIntelligenceSnapshot(),
        ProspectEvidence(observed=observed, inferred=inferred),
    )


def _strategy(snapshot: ProspectIntelligenceSnapshot):  # type: ignore[no-untyped-def]
    return ConversationStrategyEngine().recommend(
        ConversationStrategyInput(
            current_stage=SalesStage.DISCOVERY,
            current_state=ConversationState.LISTEN,
            prospect_intelligence=snapshot,
        )
    )


def test_explicit_role_is_stored_as_observed_fact() -> None:
    snapshot = _update(
        observed=ObservedProspectEvidence("turn-1", explicit_role=ProspectRole.OWNER)
    )

    assert snapshot.observed.explicit_role is not None
    assert snapshot.observed.explicit_role.value == ProspectRole.OWNER


def test_inferred_role_remains_in_inferred_state() -> None:
    snapshot = _update(
        inferred=InferredProspectEvidence(
            "turn-1", likely_role=InferenceCandidate(ProspectRole.OWNER, 0.91)
        )
    )

    assert snapshot.inferred.likely_role is not None
    assert snapshot.observed.explicit_role is None


def test_inferred_role_never_becomes_confirmed_automatically() -> None:
    snapshot = _update(
        inferred=InferredProspectEvidence(
            "turn-1", likely_role=InferenceCandidate(ProspectRole.MANAGER, 1.0)
        )
    )

    assert snapshot.observed.explicit_role is None


def test_explicit_role_overrides_contradictory_inferred_role() -> None:
    inferred = _update(
        inferred=InferredProspectEvidence(
            "turn-1", likely_role=InferenceCandidate(ProspectRole.OWNER, 0.9)
        )
    )
    revised = _update(
        inferred,
        observed=ObservedProspectEvidence(
            "turn-2", explicit_role=ProspectRole.RECEPTIONIST
        ),
    )

    assert revised.observed.explicit_role is not None
    assert revised.observed.explicit_role.value == ProspectRole.RECEPTIONIST
    assert revised.inferred.likely_role is None


def test_role_and_decision_authority_are_separate_fields() -> None:
    snapshot = _update(
        observed=ObservedProspectEvidence(
            "turn-1",
            explicit_role=ProspectRole.MANAGER,
            decision_authority_statement=DecisionAuthority.HIGH,
        )
    )

    assert snapshot.observed.explicit_role is not None
    assert snapshot.observed.explicit_decision_authority_statement is not None
    assert snapshot.observed.explicit_role.value == ProspectRole.MANAGER
    assert (
        snapshot.observed.explicit_decision_authority_statement.value
        == DecisionAuthority.HIGH
    )


def test_manager_role_does_not_imply_final_authority() -> None:
    snapshot = _update(
        observed=ObservedProspectEvidence(
            "turn-1", explicit_role=ProspectRole.MANAGER
        )
    )

    assert snapshot.observed.explicit_decision_authority_statement is None
    assert snapshot.inferred.decision_authority is None


def test_receptionist_strategy_is_routing_oriented() -> None:
    snapshot = _update(
        observed=ObservedProspectEvidence(
            "turn-1", explicit_role=ProspectRole.RECEPTIONIST
        )
    )

    result = _strategy(snapshot)
    assert result.conversation_mode == ConversationMode.ROLE_ROUTING
    assert result.strategy_type == StrategyType.ROUTE_TO_DECISION_MAKER


def test_owner_strategy_can_focus_on_business_impact() -> None:
    snapshot = _update(
        observed=ObservedProspectEvidence("turn-1", explicit_role=ProspectRole.OWNER)
    )

    assert _strategy(snapshot).strategy_type == StrategyType.DISCOVER_BUSINESS_IMPACT


def test_unknown_role_stays_unknown() -> None:
    snapshot = ProspectIntelligenceSnapshot()

    assert snapshot.observed.explicit_role is None
    assert snapshot.inferred.likely_role is None
    assert _strategy(snapshot).sales_stage == SalesStage.QUALIFICATION


def test_explicit_current_solution_is_stored_correctly() -> None:
    snapshot = _update(
        observed=ObservedProspectEvidence(
            "turn-1", current_solution=CurrentSolutionEvidence("HubSpot", "CRM")
        )
    )

    solution = snapshot.observed.explicit_current_solution
    assert solution is not None
    assert (solution.name, solution.category) == ("HubSpot", "CRM")


def test_current_solution_does_not_imply_dissatisfaction() -> None:
    snapshot = _update(
        observed=ObservedProspectEvidence(
            "turn-1", current_solution=CurrentSolutionEvidence("HubSpot")
        )
    )

    solution = snapshot.observed.explicit_current_solution
    assert solution is not None
    assert solution.satisfaction == SolutionSatisfaction.UNKNOWN


def test_new_current_solution_clears_stale_satisfaction_inference() -> None:
    inferred = _update(
        inferred=InferredProspectEvidence(
            "turn-1",
            current_solution_satisfaction=InferenceCandidate(
                SolutionSatisfaction.LOW, 0.6
            ),
        )
    )
    revised = _update(
        inferred,
        observed=ObservedProspectEvidence(
            "turn-2", current_solution=CurrentSolutionEvidence("New system")
        ),
    )

    assert revised.inferred.current_solution_satisfaction is None


def test_advisory_satisfaction_can_follow_known_solution_without_becoming_fact() -> None:
    observed = _update(
        observed=ObservedProspectEvidence(
            "turn-1", current_solution=CurrentSolutionEvidence("Current system")
        )
    )
    revised = _update(
        observed,
        inferred=InferredProspectEvidence(
            "turn-2",
            current_solution_satisfaction=InferenceCandidate(
                SolutionSatisfaction.LOW, 0.6
            ),
        ),
    )

    assert revised.inferred.current_solution_satisfaction is not None
    assert revised.observed.explicit_current_solution is not None
    assert (
        revised.observed.explicit_current_solution.satisfaction
        == SolutionSatisfaction.UNKNOWN
    )
    assert _strategy(revised).strategy_type == StrategyType.UNDERSTAND_CURRENT_SOLUTION


def test_explicit_pain_is_stored_as_observed() -> None:
    snapshot = _update(
        observed=ObservedProspectEvidence(
            "turn-1", explicit_pain=PainEvidence(PainCategory.TIME, "manual follow-up")
        )
    )

    assert snapshot.observed.explicit_pain_points[0].summary == "manual follow-up"
    assert snapshot.inferred.pain_summary is None


def test_inferred_pain_remains_inferred() -> None:
    snapshot = _update(
        inferred=InferredProspectEvidence(
            "turn-1", pain_summary=InferenceCandidate("possible delays", 0.6)
        )
    )

    assert snapshot.inferred.pain_summary is not None
    assert snapshot.observed.explicit_pain_points == ()


def test_objection_classification_is_advisory_only() -> None:
    snapshot = _update(
        inferred=InferredProspectEvidence(
            "turn-1",
            objection_type=InferenceCandidate(ObjectionType.PRICE, 0.7),
        )
    )

    assert snapshot.inferred.objection_type is not None
    assert "action" not in {field.name for field in fields(snapshot.inferred)}


def test_busy_does_not_equal_not_interested() -> None:
    snapshot = _update(
        observed=ObservedProspectEvidence("turn-1", busy=True)
    )

    assert snapshot.observed.explicit_busy_signal is not None
    assert snapshot.observed.explicit_busy_signal.value
    assert "not_interested" not in {field.name for field in fields(snapshot.observed)}


def test_interest_does_not_equal_next_step_confirmation() -> None:
    snapshot = _update(
        observed=ObservedProspectEvidence("turn-1", interest=True)
    )

    assert snapshot.observed.explicit_interest_signal is not None
    assert snapshot.observed.explicit_next_step_request is None


def test_explicit_demo_request_is_separate_from_interest() -> None:
    snapshot = _update(
        observed=ObservedProspectEvidence(
            "turn-1", interest=True, explicit_next_step=PreferredNextStep.DEMO
        )
    )

    assert snapshot.observed.explicit_interest_signal is not None
    assert snapshot.observed.explicit_next_step_request is not None
    assert snapshot.observed.explicit_next_step_request.value == PreferredNextStep.DEMO


def test_buying_stage_remains_advisory() -> None:
    snapshot = _update(
        inferred=InferredProspectEvidence(
            "turn-1",
            buying_stage=InferenceCandidate(BuyingStage.READY_FOR_NEXT_STEP, 0.8),
        )
    )

    assert snapshot.inferred.buying_stage is not None
    assert "next_state" not in {field.name for field in fields(snapshot.inferred)}


def test_openness_remains_advisory() -> None:
    snapshot = _update(
        inferred=InferredProspectEvidence(
            "turn-1", openness=InferenceCandidate(InformationLevel.HIGH, 0.8)
        )
    )

    assert snapshot.inferred.openness is not None
    assert snapshot.observed.explicit_interest_signal is None


def test_urgency_stays_unknown_without_evidence() -> None:
    snapshot = ProspectIntelligenceSnapshot()

    assert snapshot.inferred.urgency is None
    assert snapshot.observed.explicit_timing_preference is None


def test_provenance_includes_source_turn_id() -> None:
    snapshot = _update(
        observed=ObservedProspectEvidence("turn-42", explicit_role=ProspectRole.STAFF)
    )

    assert snapshot.observed.explicit_role is not None
    assert snapshot.observed.explicit_role.provenance.source_turn_id == "turn-42"


def test_confidence_does_not_create_authority() -> None:
    snapshot = _update(
        inferred=InferredProspectEvidence(
            "turn-1",
            decision_authority=InferenceCandidate(DecisionAuthority.FINAL, 1.0),
        )
    )

    assert snapshot.inferred.decision_authority is not None
    assert snapshot.observed.explicit_decision_authority_statement is None


def test_explicit_human_request_is_observed_but_does_not_execute_handoff() -> None:
    snapshot = _update(
        observed=ObservedProspectEvidence("turn-1", human_request=True)
    )
    result = _strategy(snapshot)

    assert snapshot.observed.explicit_human_request is not None
    assert result.micro_commitment == MicroCommitment.HUMAN_FOLLOW_UP
    assert "human_approval" not in {field.name for field in fields(result)}


def test_stale_authority_inference_is_cleared_by_observed_evidence() -> None:
    inferred = _update(
        inferred=InferredProspectEvidence(
            "turn-1",
            decision_authority=InferenceCandidate(DecisionAuthority.FINAL, 0.9),
        )
    )
    revised = _update(
        inferred,
        observed=ObservedProspectEvidence(
            "turn-2", decision_authority_statement=DecisionAuthority.LOW
        ),
    )

    assert revised.inferred.decision_authority is None
    assert revised.observed.explicit_decision_authority_statement is not None


def test_later_inference_cannot_shadow_existing_observed_role() -> None:
    observed = _update(
        observed=ObservedProspectEvidence(
            "turn-1", explicit_role=ProspectRole.RECEPTIONIST
        )
    )
    revised = _update(
        observed,
        inferred=InferredProspectEvidence(
            "turn-2", likely_role=InferenceCandidate(ProspectRole.OWNER, 1.0)
        ),
    )

    assert revised.observed.explicit_role is not None
    assert revised.observed.explicit_role.value == ProspectRole.RECEPTIONIST
    assert revised.inferred.likely_role is None


def test_update_produces_a_new_immutable_snapshot() -> None:
    previous = ProspectIntelligenceSnapshot()
    updated = _update(
        previous,
        observed=ObservedProspectEvidence("turn-1", interest=True),
    )

    assert updated is not previous
    assert updated.observed is not previous.observed


def test_previous_snapshot_remains_unchanged() -> None:
    previous = ProspectIntelligenceSnapshot()
    _update(
        previous,
        observed=ObservedProspectEvidence("turn-1", interest=True),
    )

    assert previous == ProspectIntelligenceSnapshot()


def test_same_snapshot_and_evidence_produce_same_result() -> None:
    previous = ProspectIntelligenceSnapshot()
    evidence = ObservedProspectEvidence("turn-1", busy=True)

    assert _update(previous, observed=evidence) == _update(previous, observed=evidence)


def test_prospect_intelligence_cannot_transition_fsm() -> None:
    machine = ConversationStateMachine(
        load_config(get_settings().conversation_config_path)
    )
    before = machine.current_state

    _update(
        observed=ObservedProspectEvidence(
            "turn-1", explicit_next_step=PreferredNextStep.DEMO
        )
    )

    assert machine.current_state == before


def test_prospect_intelligence_has_no_pricing_or_discount_authority() -> None:
    names = {
        field.name
        for contract in (ProspectIntelligenceSnapshot,)
        for field in fields(contract)
    }

    assert names.isdisjoint({"price", "pricing", "discount", "authority_policy"})


def test_prospect_intelligence_cannot_create_trusted_priority() -> None:
    names = {field.name for field in fields(ProspectIntelligenceSnapshot)}

    assert names.isdisjoint({"dnc", "trusted_priority", "not_interested"})


def test_bounded_brain_input_receives_safe_current_summary_only() -> None:
    snapshot = _update(
        observed=ObservedProspectEvidence("turn-1", explicit_role=ProspectRole.OWNER)
    )
    summary = snapshot.to_brain_summary()
    brain_input = BrainInput(
        current_utterance="hello",
        current_state=ConversationState.NEW_CALL,
        prospect_intelligence=summary,
    )

    assert brain_input.prospect_intelligence == summary
    assert "history" not in {field.name for field in fields(summary)}
    assert "transcript" not in {field.name for field in fields(summary)}


def test_demo_preference_accelerates_strategy_without_confirming_booking() -> None:
    snapshot = _update(
        observed=ObservedProspectEvidence(
            "turn-1", explicit_next_step=PreferredNextStep.DEMO
        )
    )
    result = _strategy(snapshot)

    assert result.sales_stage == SalesStage.NEXT_STEP
    assert result.micro_commitment == MicroCommitment.DEMO
    assert "confirmed" not in {field.name for field in fields(result)}
