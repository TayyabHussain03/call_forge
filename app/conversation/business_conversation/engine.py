"""Pure reconciliation of typed business observations into consultant context."""

from __future__ import annotations

from app.conversation.business_conversation.contracts import (
    BusinessConversationEvidence, BusinessConversationSnapshot, BusinessFactKind,
    BusinessProblemKind, ObservedBusinessFact, ObservedBusinessProblem,
    RootCauseHypothesis, RootCauseKind,
)


class BusinessConversationIntelligenceEngine:
    """Maintains facts, unknowns, and evidence-based hypotheses without solutions."""

    def update(self, prior: BusinessConversationSnapshot, evidence: BusinessConversationEvidence) -> BusinessConversationSnapshot:
        facts, archived_facts = _reconcile(prior.facts, prior.archived_facts, evidence.facts, evidence.corrections, lambda item: item.kind)
        problems, archived_problems = _reconcile(prior.problems, prior.archived_problems, evidence.problems, evidence.corrections, lambda item: item.kind)
        hypotheses = _hypotheses(facts, problems)
        return BusinessConversationSnapshot(
            facts=facts, problems=problems,
            goals=prior.goals | evidence.goals,
            constraints=prior.constraints | evidence.constraints,
            unknown_areas=(prior.unknown_areas | evidence.unknown_areas) - _known_unknowns(facts),
            hypotheses=hypotheses,
            current_focus=evidence.current_focus or prior.current_focus,
            pending_topic=evidence.pending_topic if evidence.pending_topic is not None else prior.pending_topic,
            archived_facts=archived_facts, archived_problems=archived_problems,
        )


def _reconcile(current, archived, incoming, corrections, kind):
    current = tuple(current)
    archived = tuple(archived)
    affected = {kind(item) for item in incoming} | set(corrections)
    removed = tuple(item for item in current if kind(item) in affected)
    retained = tuple(item for item in current if kind(item) not in affected)
    # One current observation per kind makes explicit corrections authoritative over stale assumptions.
    latest = {kind(item): item for item in incoming}
    return (*retained, *latest.values())[-24:], (*archived, *removed)[-24:]


def _known_unknowns(facts: tuple[ObservedBusinessFact, ...]):
    from app.conversation.business_conversation.contracts import UnknownArea
    mapping = {BusinessFactKind.CRM: UnknownArea.CRM, BusinessFactKind.SOFTWARE: UnknownArea.CURRENT_SOFTWARE,
               BusinessFactKind.REPORTING: UnknownArea.REPORTING_PROCESS, BusinessFactKind.INTEGRATION: UnknownArea.INTEGRATIONS}
    return {mapping[item.kind] for item in facts if item.kind in mapping}


def _hypotheses(facts, problems) -> tuple[RootCauseHypothesis, ...]:
    fact_kinds = {item.kind for item in facts}
    problem_kinds = {item.kind for item in problems}
    support = tuple(item.detail for item in problems) + tuple(item.value for item in facts)
    candidates = []
    if BusinessFactKind.MANUAL_STEP in fact_kinds or BusinessFactKind.CURRENT_WORKFLOW in fact_kinds:
        candidates.append((RootCauseKind.MANUAL_DEPENDENCY, .75, "manual workflow is observed"))
    if BusinessProblemKind.ERRORS in problem_kinds or BusinessProblemKind.FOLLOW_UPS in problem_kinds:
        candidates.append((RootCauseKind.PROCESS_GAP, .65, "errors or missed follow-ups are observed"))
    if BusinessFactKind.BRANCHES in fact_kinds and BusinessFactKind.CURRENT_WORKFLOW in fact_kinds:
        candidates.append((RootCauseKind.NO_CENTRAL_SYSTEM, .55, "multiple branches and a workflow are observed"))
    if BusinessFactKind.REPORTING not in fact_kinds and BusinessProblemKind.OPERATIONAL in problem_kinds:
        candidates.append((RootCauseKind.REPORTING_GAP, .4, "operational issue is observed while reporting remains unknown"))
    return tuple(RootCauseHypothesis(kind, confidence, reason, support[:6]) for kind, confidence, reason in candidates[:6] if support)
