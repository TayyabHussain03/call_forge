"""Pure deterministic BCI-to-diagnosis transformation."""

from __future__ import annotations

from app.conversation.business_conversation.contracts import BusinessFactKind, BusinessProblemKind
from app.conversation.business_diagnostic.contracts import *


class BusinessDiagnosticEngine:
    def diagnose(self, value: BusinessDiagnosticInput) -> BusinessDiagnosticSnapshot:
        facts = {item.kind: item for item in value.conversation.facts}
        problems = value.conversation.problems
        findings = _findings(facts)
        focus = _focus(facts, problems)
        gaps = _gaps(facts, value.conversation.unknown_areas)
        hypotheses = _hypotheses(facts, problems)
        maturity, confidence = _maturity(facts)
        areas = tuple(item.area for item in findings)
        primary = _area_for_problem(problems[0].kind) if problems else (areas[0] if areas else None)
        return BusinessDiagnosticSnapshot(
            maturity, confidence, findings, gaps, hypotheses, focus, focus,
            ConsultantSummary(areas, primary, tuple(area for area in areas if area != primary),
                             value.conversation.goals, BusinessFactKind.CURRENT_WORKFLOW in facts,
                             value.conversation.constraints,
                             tuple(item.area for item in gaps)),
        )


def _finding(area, confidence, fact):
    return DiagnosticFinding(area, confidence, (fact.value,), (fact.source_turn_id,))


def _findings(facts):
    mapping = {BusinessFactKind.WEBSITE: CapabilityArea.ONLINE_PRESENCE, BusinessFactKind.CRM: CapabilityArea.CRM,
               BusinessFactKind.REPORTING: CapabilityArea.REPORTING, BusinessFactKind.CURRENT_WORKFLOW: CapabilityArea.WORKFLOW,
               BusinessFactKind.MANUAL_STEP: CapabilityArea.WORKFLOW}
    return tuple(_finding(area, DiagnosticConfidence.OBSERVED, fact) for kind, area in mapping.items() if (fact := facts.get(kind)) is not None)


def _gaps(facts, unknowns):
    from app.conversation.business_conversation.contracts import UnknownArea
    mapping = {UnknownArea.CRM: CapabilityArea.CRM, UnknownArea.REPORTING_PROCESS: CapabilityArea.REPORTING,
               UnknownArea.ORDER_VOLUME: CapabilityArea.LEAD_TRACKING}
    return tuple(DiagnosticFinding(area, DiagnosticConfidence.UNKNOWN, (area.value,), ()) for unknown, area in mapping.items() if unknown in unknowns)


def _hypotheses(facts, problems):
    support = tuple(item.value for item in facts.values()) + tuple(item.detail for item in problems)
    ids = tuple(item.source_turn_id for item in facts.values()) + tuple(item.source_turn_id for item in problems)
    result = []
    if BusinessFactKind.MANUAL_STEP in facts:
        result.append(DiagnosticHypothesis(DiagnosticHypothesisKind.MANUAL_DEPENDENCY, DiagnosticConfidence.LIKELY, support[:6], ids[:6]))
    if BusinessProblemKind.ERRORS in {item.kind for item in problems} or BusinessProblemKind.FOLLOW_UPS in {item.kind for item in problems}:
        result.append(DiagnosticHypothesis(DiagnosticHypothesisKind.PROCESS_GAP, DiagnosticConfidence.LIKELY, support[:6], ids[:6]))
    if BusinessFactKind.BRANCHES in facts and BusinessFactKind.CURRENT_WORKFLOW in facts:
        result.append(DiagnosticHypothesis(DiagnosticHypothesisKind.CENTRALIZATION_GAP, DiagnosticConfidence.POSSIBLE, support[:6], ids[:6]))
    return tuple(result)


def _focus(facts, problems):
    kinds = {item.kind for item in problems}
    if BusinessProblemKind.FOLLOW_UPS in kinds: return DiagnosticFocus.LEAD_MANAGEMENT
    if (BusinessFactKind.MANUAL_STEP in facts or BusinessFactKind.CURRENT_WORKFLOW in facts
            or BusinessProblemKind.ERRORS in kinds): return DiagnosticFocus.WORKFLOW
    if BusinessFactKind.WEBSITE in facts: return DiagnosticFocus.WEBSITE_PERFORMANCE
    if BusinessFactKind.REPORTING in facts: return DiagnosticFocus.REPORTING
    return DiagnosticFocus.BUSINESS_CONTEXT


def _maturity(facts):
    digital = sum(kind in facts for kind in (BusinessFactKind.WEBSITE, BusinessFactKind.CRM, BusinessFactKind.REPORTING))
    if digital >= 3: return BusinessMaturity.OPTIMIZING, DiagnosticConfidence.OBSERVED
    if digital >= 2: return BusinessMaturity.GROWING, DiagnosticConfidence.OBSERVED
    if digital == 1: return BusinessMaturity.EARLY_DIGITAL, DiagnosticConfidence.OBSERVED
    return BusinessMaturity.UNKNOWN, DiagnosticConfidence.UNKNOWN


def _area_for_problem(kind):
    return {BusinessProblemKind.FOLLOW_UPS: CapabilityArea.LEAD_TRACKING, BusinessProblemKind.ERRORS: CapabilityArea.WORKFLOW,
            BusinessProblemKind.SALES: CapabilityArea.LEAD_TRACKING, BusinessProblemKind.MARKETING: CapabilityArea.ONLINE_PRESENCE}.get(kind, CapabilityArea.WORKFLOW)
