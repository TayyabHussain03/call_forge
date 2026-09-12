"""Pure deterministic updates for prospect-intelligence snapshots."""

from __future__ import annotations

from dataclasses import replace
from typing import TypeVar

from app.conversation.prospect_intelligence.contracts import (
    CurrentSolutionContext,
    EvidenceProvenance,
    EvidenceSourceKind,
    InferenceCandidate,
    InferredProspectEvidence,
    InferredProspectState,
    InferredValue,
    ObservedProspectEvidence,
    ObservedProspectFacts,
    ObservedValue,
    ProspectEvidence,
    ProspectIntelligenceSnapshot,
    ProspectPain,
    SolutionSatisfaction,
)

ValueT = TypeVar("ValueT")


class ProspectIntelligenceUpdater:
    """Merge typed evidence without mutation, inference promotion, or authority."""

    def update(
        self,
        previous: ProspectIntelligenceSnapshot,
        evidence: ProspectEvidence,
    ) -> ProspectIntelligenceSnapshot:
        """Return a new snapshot where observed evidence outranks inference."""
        if not isinstance(previous, ProspectIntelligenceSnapshot):
            raise TypeError("previous must be a ProspectIntelligenceSnapshot")
        if not isinstance(evidence, ProspectEvidence):
            raise TypeError("evidence must be ProspectEvidence")
        inferred = _merge_inferred(previous.inferred, evidence.inferred)
        observed, inferred = _merge_observed(
            previous.observed, inferred, evidence.observed
        )
        return ProspectIntelligenceSnapshot(observed=observed, inferred=inferred)


def _merge_inferred(
    previous: InferredProspectState,
    evidence: InferredProspectEvidence | None,
) -> InferredProspectState:
    if evidence is None:
        return previous
    changes = {
        name: _inferred(candidate, evidence.source_turn_id)
        for name, candidate in (
            ("likely_role", evidence.likely_role),
            ("decision_authority", evidence.decision_authority),
            ("influence_level", evidence.influence_level),
            ("buying_stage", evidence.buying_stage),
            ("openness", evidence.openness),
            ("urgency", evidence.urgency),
            ("objection_type", evidence.objection_type),
            ("pain_summary", evidence.pain_summary),
            (
                "current_solution_satisfaction",
                evidence.current_solution_satisfaction,
            ),
            ("preferred_next_step", evidence.preferred_next_step),
        )
        if candidate is not None
    }
    return replace(previous, **changes)


def _merge_observed(
    previous: ObservedProspectFacts,
    inferred: InferredProspectState,
    evidence: ObservedProspectEvidence | None,
) -> tuple[ObservedProspectFacts, InferredProspectState]:
    inferred = _clear_shadowed_inferences(previous, inferred)
    if evidence is None:
        return previous, inferred
    provenance = EvidenceProvenance(evidence.source_turn_id, evidence.source_kind)
    changes: dict[str, object] = {}
    inference_changes: dict[str, object] = {}

    if evidence.explicit_role is not None:
        changes["explicit_role"] = ObservedValue(evidence.explicit_role, provenance)
        inference_changes["likely_role"] = None
    if evidence.current_solution is not None:
        solution = evidence.current_solution
        changes["explicit_current_solution"] = CurrentSolutionContext(
            solution.name, solution.category, solution.satisfaction, provenance
        )
        inference_changes["current_solution_satisfaction"] = None
    if evidence.explicit_pain is not None:
        pain = ProspectPain(
            evidence.explicit_pain.category,
            evidence.explicit_pain.summary,
            provenance,
        )
        changes["explicit_pain_points"] = (
            *previous.explicit_pain_points,
            pain,
        )[-3:]
        inference_changes["pain_summary"] = None

    for target, value in (
        ("explicit_interest_signal", evidence.interest),
        ("explicit_busy_signal", evidence.busy),
        ("explicit_human_request", evidence.human_request),
        ("explicit_next_step_request", evidence.explicit_next_step),
        (
            "explicit_decision_authority_statement",
            evidence.decision_authority_statement,
        ),
        ("explicit_timing_preference", evidence.timing_preference),
        ("explicit_objection", evidence.objection),
    ):
        if value is not None:
            changes[target] = ObservedValue(value, provenance)

    if evidence.explicit_next_step is not None:
        inference_changes["preferred_next_step"] = None
    if evidence.decision_authority_statement is not None:
        inference_changes["decision_authority"] = None
    if evidence.objection is not None:
        inference_changes["objection_type"] = None

    return replace(previous, **changes), replace(inferred, **inference_changes)


def _clear_shadowed_inferences(
    observed: ObservedProspectFacts,
    inferred: InferredProspectState,
) -> InferredProspectState:
    """Prevent later inference from shadowing an existing observed fact."""
    changes: dict[str, object] = {}
    if observed.explicit_role is not None:
        changes["likely_role"] = None
    if observed.explicit_decision_authority_statement is not None:
        changes["decision_authority"] = None
    if (
        observed.explicit_current_solution is not None
        and observed.explicit_current_solution.satisfaction
        != SolutionSatisfaction.UNKNOWN
    ):
        changes["current_solution_satisfaction"] = None
    if observed.explicit_pain_points:
        changes["pain_summary"] = None
    if observed.explicit_next_step_request is not None:
        changes["preferred_next_step"] = None
    if observed.explicit_objection is not None:
        changes["objection_type"] = None
    return replace(inferred, **changes) if changes else inferred


def _inferred(
    candidate: InferenceCandidate[ValueT], source_turn_id: str
) -> InferredValue[ValueT]:
    return InferredValue(
        candidate.value,
        candidate.confidence,
        EvidenceProvenance(source_turn_id, EvidenceSourceKind.INFERENCE),
    )
