"""Serializers for participant continuity + discussion coverage."""

from __future__ import annotations

from app.application.sales_runtime.serializers.base import (
    PayloadRequirement,
    _coerce_enum,
    build_envelope,
    parse_envelope,
)
from app.conversation.participant.contracts import (
    DiscussionCoverageEntry,
    DiscussionCoverageState,
    ExplanationDepth,
    ParticipantConversationState,
    ParticipantReference,
    ParticipantRoleHint,
)
from app.persistence.errors import InvalidPersistedPayloadError


PARTICIPANT_STATE_KEY = "participant_state"
PARTICIPANT_STATE_VERSION = 1
DISCUSSION_COVERAGE_KEY = "discussion_coverage"
DISCUSSION_COVERAGE_VERSION = 1


class ParticipantStateSerializer:
    @property
    def key(self) -> str:
        return PARTICIPANT_STATE_KEY

    @property
    def version(self) -> int:
        return PARTICIPANT_STATE_VERSION

    @property
    def requirement(self) -> PayloadRequirement:
        return PayloadRequirement.OPTIONAL_DERIVED

    def serialize(
        self,
        active: ParticipantReference | None,
        archived: tuple[ParticipantConversationState, ...] = (),
    ) -> str:
        return build_envelope(self.key, self.version, {
            "active": (self._ref_to_dict(active) if active else None),
            "archived": [self._state_to_dict(s) for s in archived],
        })

    def deserialize(
        self, envelope_json: str,
    ) -> tuple[ParticipantReference | None, tuple[ParticipantConversationState, ...]]:
        payload = parse_envelope(envelope_json, self.key, (self.version,))
        try:
            active_raw = payload.get("active")
            active = self._ref_from_dict(active_raw) if active_raw else None
            archived = tuple(
                self._state_from_dict(s) for s in payload.get("archived", [])
            )
            return active, archived
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidPersistedPayloadError(
                f"participant state payload invalid: {exc}"
            ) from exc

    @staticmethod
    def _ref_to_dict(ref: ParticipantReference) -> dict:
        return {
            "reference_id": ref.reference_id,
            "role_hint": ref.role_hint.value,
            "display_label": ref.display_label,
        }

    @staticmethod
    def _ref_from_dict(d: dict) -> ParticipantReference:
        return ParticipantReference(
            reference_id=d["reference_id"],
            role_hint=_coerce_enum(
                ParticipantRoleHint, d.get("role_hint", "unknown"),
                "participant.role_hint",
            ),
            display_label=d.get("display_label"),
        )

    def _state_to_dict(self, s: ParticipantConversationState) -> dict:
        return {
            "participant": self._ref_to_dict(s.participant),
            "heard_explanation_concepts": sorted(s.heard_explanation_concepts),
            "role_hint": s.role_hint.value,
            "interest_observed": s.interest_observed,
            "objection_concepts": sorted(s.objection_concepts),
            "language_preference": s.language_preference,
            "busy_observed": s.busy_observed,
        }

    def _state_from_dict(self, d: dict) -> ParticipantConversationState:
        return ParticipantConversationState(
            participant=self._ref_from_dict(d["participant"]),
            heard_explanation_concepts=frozenset(d.get("heard_explanation_concepts", [])),
            role_hint=_coerce_enum(
                ParticipantRoleHint, d.get("role_hint", "unknown"),
                "participant_state.role_hint",
            ),
            interest_observed=d.get("interest_observed"),
            objection_concepts=frozenset(d.get("objection_concepts", [])),
            language_preference=d.get("language_preference"),
            busy_observed=d.get("busy_observed"),
        )


class DiscussionCoverageSerializer:
    @property
    def key(self) -> str:
        return DISCUSSION_COVERAGE_KEY

    @property
    def version(self) -> int:
        return DISCUSSION_COVERAGE_VERSION

    @property
    def requirement(self) -> PayloadRequirement:
        return PayloadRequirement.OPTIONAL_DERIVED

    def serialize(self, state: DiscussionCoverageState) -> str:
        return build_envelope(self.key, self.version, {
            "entries": [
                {
                    "participant_reference_id": e.participant_reference_id,
                    "concept_id": e.concept_id,
                    "depth": e.depth.value,
                    "source_turn_id": e.source_turn_id,
                }
                for e in sorted(
                    state.entries,
                    key=lambda e: (
                        e.participant_reference_id, e.concept_id, e.source_turn_id,
                    ),
                )
            ],
            "asked_question_concepts": [
                [p, c] for p, c in sorted(state.asked_question_concepts)
            ],
            "handled_objection_concepts": [
                [p, c] for p, c in sorted(state.handled_objection_concepts)
            ],
            "discussed_evidence_ids": sorted(state.discussed_evidence_ids),
        })

    def deserialize(self, envelope_json: str) -> DiscussionCoverageState:
        payload = parse_envelope(envelope_json, self.key, (self.version,))
        try:
            return DiscussionCoverageState(
                entries=tuple(
                    DiscussionCoverageEntry(
                        participant_reference_id=e["participant_reference_id"],
                        concept_id=e["concept_id"],
                        depth=_coerce_enum(
                            ExplanationDepth, e["depth"], "coverage.depth",
                        ),
                        source_turn_id=e["source_turn_id"],
                    )
                    for e in payload.get("entries", [])
                ),
                asked_question_concepts=tuple(
                    (p, c) for p, c in payload.get("asked_question_concepts", [])
                ),
                handled_objection_concepts=tuple(
                    (p, c) for p, c in payload.get("handled_objection_concepts", [])
                ),
                discussed_evidence_ids=frozenset(
                    payload.get("discussed_evidence_ids", [])
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidPersistedPayloadError(
                f"discussion coverage payload invalid: {exc}"
            ) from exc
