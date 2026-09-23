"""Focused invariants for the deterministic business understanding layer."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from app.conversation.business_conversation.contracts import (
    BusinessConstraintKind,
    BusinessConversationEvidence,
    BusinessConversationSnapshot,
    BusinessFactKind,
    BusinessGoalKind,
    BusinessProblemKind,
    ConversationTopic,
    ObservedBusinessFact,
    ObservedBusinessProblem,
    RootCauseKind,
    UnknownArea,
)
from app.conversation.business_conversation.engine import BusinessConversationIntelligenceEngine


def _fact(kind: BusinessFactKind, value: str) -> ObservedBusinessFact:
    return ObservedBusinessFact(kind, value, "turn-1")


def _problem(kind: BusinessProblemKind, detail: str, *, primary: bool = False) -> ObservedBusinessProblem:
    return ObservedBusinessProblem(kind, detail, "turn-1", primary)


def test_long_narrative_decomposes_multiple_observed_pains_and_hypotheses() -> None:
    snapshot = BusinessConversationIntelligenceEngine().update(
        BusinessConversationSnapshot(),
        BusinessConversationEvidence(
            facts=(
                _fact(BusinessFactKind.BRANCHES, "4"),
                _fact(BusinessFactKind.CURRENT_WORKFLOW, "Excel order register"),
                _fact(BusinessFactKind.MANUAL_STEP, "staff manually updates orders"),
                _fact(BusinessFactKind.COMMUNICATION, "WhatsApp orders"),
            ),
            problems=(
                _problem(BusinessProblemKind.ERRORS, "duplicate orders", primary=True),
                _problem(BusinessProblemKind.FOLLOW_UPS, "missed follow-ups"),
            ),
            goals=frozenset({BusinessGoalKind.SCALE_OPERATIONS}),
            unknown_areas=frozenset({UnknownArea.CRM, UnknownArea.REPORTING_PROCESS}),
            current_focus=ConversationTopic.OPERATIONS,
        ),
    )
    assert len(snapshot.facts) == 4
    assert [item.kind for item in snapshot.problems] == [BusinessProblemKind.ERRORS, BusinessProblemKind.FOLLOW_UPS]
    assert RootCauseKind.MANUAL_DEPENDENCY in {item.kind for item in snapshot.hypotheses}
    assert RootCauseKind.NO_CENTRAL_SYSTEM in {item.kind for item in snapshot.hypotheses}
    assert all(item.supporting_observations and 0 <= item.confidence <= 1 for item in snapshot.hypotheses)


def test_correction_archives_old_fact_and_replaces_primary_pain() -> None:
    engine = BusinessConversationIntelligenceEngine()
    first = engine.update(BusinessConversationSnapshot(), BusinessConversationEvidence(
        facts=(_fact(BusinessFactKind.WEBSITE, "no website"),),
        problems=(_problem(BusinessProblemKind.ERRORS, "duplicate orders", primary=True),),
    ))
    corrected = engine.update(first, BusinessConversationEvidence(
        facts=(_fact(BusinessFactKind.WEBSITE, "website exists"),),
        problems=(_problem(BusinessProblemKind.FOLLOW_UPS, "missed follow-ups", primary=True),),
        corrections=frozenset({BusinessFactKind.WEBSITE, BusinessProblemKind.ERRORS}),
    ))
    assert corrected.facts[0].value == "website exists"
    assert corrected.problems[0].kind == BusinessProblemKind.FOLLOW_UPS
    assert corrected.archived_facts[0].value == "no website"
    assert corrected.archived_problems[0].kind == BusinessProblemKind.ERRORS


def test_topic_switch_preserves_prior_context_and_pending_topic() -> None:
    engine = BusinessConversationIntelligenceEngine()
    first = engine.update(BusinessConversationSnapshot(), BusinessConversationEvidence(
        facts=(_fact(BusinessFactKind.WEBSITE, "website exists"),), current_focus=ConversationTopic.WEBSITE,
    ))
    shifted = engine.update(first, BusinessConversationEvidence(
        facts=(_fact(BusinessFactKind.CRM, "CRM exists"),), current_focus=ConversationTopic.CRM,
        pending_topic=ConversationTopic.WEBSITE,
    ))
    assert {item.kind for item in shifted.facts} == {BusinessFactKind.WEBSITE, BusinessFactKind.CRM}
    assert shifted.current_focus == ConversationTopic.CRM
    assert shifted.pending_topic == ConversationTopic.WEBSITE


def test_unknowns_remain_unknown_until_observed() -> None:
    engine = BusinessConversationIntelligenceEngine()
    snapshot = engine.update(BusinessConversationSnapshot(), BusinessConversationEvidence(
        unknown_areas=frozenset({UnknownArea.CRM, UnknownArea.ORDER_VOLUME}),
    ))
    assert snapshot.unknown_areas == frozenset({UnknownArea.CRM, UnknownArea.ORDER_VOLUME})
    updated = engine.update(snapshot, BusinessConversationEvidence(facts=(_fact(BusinessFactKind.CRM, "HubSpot"),)))
    assert UnknownArea.CRM not in updated.unknown_areas
    assert UnknownArea.ORDER_VOLUME in updated.unknown_areas


def test_existing_software_manual_workflow_and_constraints_are_separate() -> None:
    snapshot = BusinessConversationIntelligenceEngine().update(BusinessConversationSnapshot(), BusinessConversationEvidence(
        facts=(_fact(BusinessFactKind.SOFTWARE, "Excel"), _fact(BusinessFactKind.MANUAL_STEP, "manual invoice entry")),
        constraints=frozenset({BusinessConstraintKind.BUDGET, BusinessConstraintKind.STAFF}),
        goals=frozenset({BusinessGoalKind.REDUCE_TIME}),
    ))
    assert {item.kind for item in snapshot.facts} == {BusinessFactKind.SOFTWARE, BusinessFactKind.MANUAL_STEP}
    assert snapshot.constraints == frozenset({BusinessConstraintKind.BUDGET, BusinessConstraintKind.STAFF})
    assert snapshot.goals == frozenset({BusinessGoalKind.REDUCE_TIME})


def test_replay_is_deterministic_and_contract_has_no_service_or_authority_fields() -> None:
    evidence = BusinessConversationEvidence(
        facts=(_fact(BusinessFactKind.MANUAL_STEP, "manual updates"),),
        problems=(_problem(BusinessProblemKind.TIME, "slow workflow"),),
    )
    engine = BusinessConversationIntelligenceEngine()
    assert engine.update(BusinessConversationSnapshot(), evidence) == engine.update(BusinessConversationSnapshot(), evidence)
    forbidden = {"service", "action", "authority", "state", "price", "execution", "recommendation"}
    assert not any(any(word in field.name for word in forbidden) for field in fields(BusinessConversationSnapshot))


def test_snapshot_is_immutable() -> None:
    snapshot = BusinessConversationIntelligenceEngine().update(BusinessConversationSnapshot(), BusinessConversationEvidence())
    with pytest.raises(FrozenInstanceError):
        snapshot.current_focus = ConversationTopic.SALES  # type: ignore[misc]
