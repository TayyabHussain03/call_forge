"""Production ExplanationConceptProvider.

Adapts existing trusted structured domain output into a stable semantic
concept ID the participant layer can key coverage against. Reads from
OpportunityGuidance (service-fit) or BrainProposal — NEVER derives
identity from response prose. If no concept exists for the current turn
(e.g. pure greeting, DNC acknowledgement), returns None.
"""

from __future__ import annotations

from typing import Callable


# Stable concept-id prefixes — deterministic, bounded, project-wide
SERVICE_CONCEPT_PREFIX = "service:"
PROBLEM_CONCEPT_PREFIX = "problem:"


class BrainConceptProvider:
    """Resolves `service:<service_id>` from the processor's live
    `playbook_guidance.selected_service_id` after each turn.

    This adapter is a pure read against existing trusted structured
    state. It never invokes an LLM, never parses prose, never fabricates
    identity when no service was selected.
    """

    def __init__(self, processor) -> None:
        self._processor = processor

    def resolve(self) -> str | None:
        guidance = getattr(self._processor, "playbook_guidance", None)
        if guidance is None:
            return None
        service_id = getattr(guidance, "selected_service_id", None)
        if not service_id or not isinstance(service_id, str):
            return None
        return f"{SERVICE_CONCEPT_PREFIX}{service_id}"

    def as_callable(self) -> Callable:
        """Return a processor-ready two-argument provider callable.

        Note: the processor passes `(turn, understanding)` but this
        provider sources from the processor's own live state, so the
        arguments are ignored intentionally — identity comes from
        trusted structured guidance, not from the raw utterance.
        """
        def _provider(turn, understanding):
            return self.resolve()
        return _provider


class StaticConceptProvider:
    """Returns a fixed concept id for every call. For tests only."""

    def __init__(self, concept_id: str | None) -> None:
        self._concept_id = concept_id

    def as_callable(self) -> Callable:
        def _provider(turn, understanding):
            return self._concept_id
        return _provider
