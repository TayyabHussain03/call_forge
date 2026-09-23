"""Focused invariants for deterministic, non-commercial BCI diagnosis."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from app.conversation.business_conversation.contracts import (
    BusinessConversationSnapshot, BusinessFactKind, BusinessProblemKind,
    ObservedBusinessFact, ObservedBusinessProblem, UnknownArea,
)
from app.conversation.business_diagnostic.contracts import (
    BusinessDiagnosticInput, BusinessMaturity, CapabilityArea, DiagnosticConfidence,
    DiagnosticFocus,
)
from app.conversation.business_diagnostic.engine import BusinessDiagnosticEngine


def _fact(kind: BusinessFactKind, value: str) -> ObservedBusinessFact:
    return ObservedBusinessFact(kind, value, "turn-1")


def _problem(kind: BusinessProblemKind, detail: str) -> ObservedBusinessProblem:
    return ObservedBusinessProblem(kind, detail, "turn-1", True)


def _diagnose(snapshot: BusinessConversationSnapshot):
    return BusinessDiagnosticEngine().diagnose(BusinessDiagnosticInput(snapshot))


def test_manual_multi_branch_workflow_has_evidence_backed_diagnosis() -> None:
    diagnosis = _diagnose(BusinessConversationSnapshot(
        facts=(_fact(BusinessFactKind.BRANCHES, "four"), _fact(BusinessFactKind.CURRENT_WORKFLOW, "Excel"), _fact(BusinessFactKind.MANUAL_STEP, "manual entry")),
        problems=(_problem(BusinessProblemKind.ERRORS, "duplicate orders"),),
    ))
    assert diagnosis.diagnostic_focus == DiagnosticFocus.WORKFLOW
    assert {item.area for item in diagnosis.findings} == {CapabilityArea.WORKFLOW}
    assert all(item.confidence == DiagnosticConfidence.OBSERVED and item.evidence_ids for item in diagnosis.findings)
    assert {item.kind.value for item in diagnosis.hypotheses} >= {"manual_dependency", "centralization_gap"}


def test_existing_website_and_crm_produce_observed_capabilities_and_maturity() -> None:
    diagnosis = _diagnose(BusinessConversationSnapshot(
        facts=(_fact(BusinessFactKind.WEBSITE, "exists"), _fact(BusinessFactKind.CRM, "existing CRM"), _fact(BusinessFactKind.REPORTING, "reports exist")),
    ))
    assert diagnosis.maturity == BusinessMaturity.OPTIMIZING
    assert diagnosis.maturity_confidence == DiagnosticConfidence.OBSERVED
    assert {item.area for item in diagnosis.findings} >= {CapabilityArea.ONLINE_PRESENCE, CapabilityArea.CRM, CapabilityArea.REPORTING}


def test_unknowns_are_preserved_as_unknown_capability_gaps() -> None:
    diagnosis = _diagnose(BusinessConversationSnapshot(unknown_areas=frozenset({UnknownArea.CRM, UnknownArea.REPORTING_PROCESS})))
    assert {item.area for item in diagnosis.capability_gaps} == {CapabilityArea.CRM, CapabilityArea.REPORTING}
    assert all(item.confidence == DiagnosticConfidence.UNKNOWN for item in diagnosis.capability_gaps)


def test_follow_up_problem_selects_one_highest_value_focus() -> None:
    diagnosis = _diagnose(BusinessConversationSnapshot(
        facts=(_fact(BusinessFactKind.MANUAL_STEP, "manual entry"),),
        problems=(_problem(BusinessProblemKind.FOLLOW_UPS, "missed follow-ups"), _problem(BusinessProblemKind.ERRORS, "duplicates")),
    ))
    assert diagnosis.diagnostic_focus == DiagnosticFocus.LEAD_MANAGEMENT
    assert diagnosis.question_concept == diagnosis.diagnostic_focus
    assert diagnosis.summary.primary_problem == CapabilityArea.LEAD_TRACKING


def test_diagnosis_recalculates_from_corrected_bci_snapshot_and_replays() -> None:
    before = BusinessConversationSnapshot(facts=(_fact(BusinessFactKind.WEBSITE, "no website"),))
    after = BusinessConversationSnapshot(facts=(_fact(BusinessFactKind.WEBSITE, "website exists"),), archived_facts=before.facts)
    assert _diagnose(before) != _diagnose(after)
    assert _diagnose(after) == _diagnose(after)


def test_contract_is_immutable_and_contains_no_sales_or_authority_leakage() -> None:
    diagnosis = _diagnose(BusinessConversationSnapshot())
    with pytest.raises(FrozenInstanceError):
        diagnosis.diagnostic_focus = DiagnosticFocus.WORKFLOW  # type: ignore[misc]
    forbidden = {"service", "action", "authority", "state", "price", "execution", "recommend"}
    assert not any(any(word in field.name for word in forbidden) for field in fields(type(diagnosis)))
    assert not any(any(word in field.name for word in forbidden) for field in fields(type(diagnosis.summary)))
