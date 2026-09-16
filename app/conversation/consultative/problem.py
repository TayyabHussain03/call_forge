"""Pure bounded updates for the current prospect problem model."""

from __future__ import annotations

from dataclasses import replace

from app.conversation.consultative.contracts import (
    ProblemEvidence,
    ProblemEvidenceBasis,
    ProspectProblem,
)


class ProblemModelUpdater:
    """Update a problem hypothesis without inventing missing impact or process."""

    def update(
        self, previous: ProspectProblem, evidence: ProblemEvidence
    ) -> ProspectProblem:
        if evidence.correction:
            return _from_evidence(evidence)
        if (
            previous.evidence_basis == ProblemEvidenceBasis.EXPLICIT
            and evidence.evidence_basis == ProblemEvidenceBasis.INFERRED
        ):
            return previous
        changes = {
            name: value
            for name, value in (
                ("category", evidence.category),
                ("explicit_description", evidence.explicit_description),
                ("requested_solution", evidence.requested_solution),
                ("current_process", evidence.current_process),
                ("friction", evidence.friction),
                ("impact", evidence.impact),
                ("desired_outcome", evidence.desired_outcome),
                ("source_or_channel", evidence.source_or_channel),
            )
            if value is not None
            and not (name == "category" and value.value == "unknown")
        }
        changes["evidence_basis"] = evidence.evidence_basis
        changes["source_turn_id"] = evidence.source_turn_id
        return replace(previous, **changes)


def _from_evidence(evidence: ProblemEvidence) -> ProspectProblem:
    return ProspectProblem(
        category=evidence.category,
        explicit_description=evidence.explicit_description,
        requested_solution=evidence.requested_solution,
        current_process=evidence.current_process,
        friction=evidence.friction,
        impact=evidence.impact,
        desired_outcome=evidence.desired_outcome,
        source_or_channel=evidence.source_or_channel,
        evidence_basis=evidence.evidence_basis,
        source_turn_id=evidence.source_turn_id,
    )
