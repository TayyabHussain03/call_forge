"""Deterministic mapping from validated meaning to prospect evidence."""

from __future__ import annotations

from typing import TypeVar

from app.conversation.prospect_intelligence.contracts import (
    EvidenceSourceKind,
    InferenceCandidate,
    InferredProspectEvidence,
    ObservedProspectEvidence,
    PainEvidence,
    ProspectEvidence,
)
from app.conversation.understanding.contracts import (
    EvidenceBasis,
    FreeTextUnderstanding,
    MeaningObservation,
)

ValueT = TypeVar("ValueT")


class UnderstandingEvidenceMapper:
    """Attach provenance and preserve explicit/inferred separation."""

    def map(
        self, understanding: FreeTextUnderstanding, source_turn_id: str
    ) -> ProspectEvidence | None:
        observed_values = {
            "explicit_role": _explicit(understanding.role_observation),
            "current_solution": _explicit(
                understanding.current_solution_observation
            ),
            "explicit_pain": _explicit(understanding.pain_observation),
            "interest": _explicit(understanding.interest_observation),
            "busy": _explicit(understanding.busy_observation),
            "human_request": _explicit(understanding.explicit_human_request),
            "explicit_next_step": _explicit(understanding.next_step_request),
            "decision_authority_statement": _explicit(
                understanding.authority_observation
            ),
            "timing_preference": _explicit(understanding.timing_observation),
            "objection": _explicit(understanding.objection_observation),
        }
        inferred_values = {
            "likely_role": _inferred(understanding.role_observation),
            "decision_authority": _inferred(understanding.authority_observation),
            "objection_type": _inferred(understanding.objection_observation),
            "pain_summary": _inferred_pain(understanding.pain_observation),
            "preferred_next_step": _inferred(understanding.next_step_request),
        }
        observed = (
            ObservedProspectEvidence(
                source_turn_id=source_turn_id,
                source_kind=EvidenceSourceKind.STRUCTURED_SIGNAL,
                **{
                    key: value
                    for key, value in observed_values.items()
                    if value is not None
                },
            )
            if any(value is not None for value in observed_values.values())
            else None
        )
        inferred = (
            InferredProspectEvidence(
                source_turn_id=source_turn_id,
                **{
                    key: value
                    for key, value in inferred_values.items()
                    if value is not None
                },
            )
            if any(value is not None for value in inferred_values.values())
            else None
        )
        if observed is None and inferred is None:
            return None
        return ProspectEvidence(observed=observed, inferred=inferred)


def _explicit(
    observation: MeaningObservation[ValueT] | None,
) -> ValueT | None:
    if observation is None or observation.basis != EvidenceBasis.EXPLICIT:
        return None
    return observation.value


def _inferred(
    observation: MeaningObservation[ValueT] | None,
) -> InferenceCandidate[ValueT] | None:
    if observation is None or observation.basis != EvidenceBasis.INFERRED:
        return None
    return InferenceCandidate(observation.value, float(observation.confidence))


def _inferred_pain(
    observation: MeaningObservation[PainEvidence] | None,
) -> InferenceCandidate[str] | None:
    candidate = _inferred(observation)
    if candidate is None:
        return None
    return InferenceCandidate(candidate.value.summary, candidate.confidence)
