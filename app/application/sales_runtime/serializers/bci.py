"""BusinessConversationSnapshot serializer.

Observed facts remain observed; archived observations preserved; corrections
preserved as hypotheses. Deterministic collection ordering.
"""

from __future__ import annotations

from app.application.sales_runtime.serializers.base import (
    PayloadRequirement,
    _coerce_enum,
    build_envelope,
    parse_envelope,
)
from app.conversation.business_conversation.contracts import (
    BusinessConstraintKind,
    BusinessConversationSnapshot,
    BusinessFactKind,
    BusinessGoalKind,
    BusinessProblemKind,
    ConversationTopic,
    ObservedBusinessFact,
    ObservedBusinessProblem,
    RootCauseHypothesis,
    RootCauseKind,
    UnknownArea,
)
from app.persistence.errors import InvalidPersistedPayloadError


BUSINESS_CONVERSATION_KEY = "bci"
BUSINESS_CONVERSATION_VERSION = 1


class BusinessConversationSerializer:
    """Round-trip serializer for BusinessConversationSnapshot."""

    @property
    def key(self) -> str:
        return BUSINESS_CONVERSATION_KEY

    @property
    def version(self) -> int:
        return BUSINESS_CONVERSATION_VERSION

    @property
    def requirement(self) -> PayloadRequirement:
        return PayloadRequirement.REQUIRED_FOR_ACTIVE_RECOVERY

    def serialize(self, snap: BusinessConversationSnapshot) -> str:
        payload = {
            "facts": [self._fact_to_dict(f) for f in snap.facts],
            "problems": [self._problem_to_dict(p) for p in snap.problems],
            "goals": sorted(g.value for g in snap.goals),
            "constraints": sorted(c.value for c in snap.constraints),
            "unknown_areas": sorted(u.value for u in snap.unknown_areas),
            "hypotheses": [self._hyp_to_dict(h) for h in snap.hypotheses],
            "current_focus": (
                snap.current_focus.value if snap.current_focus else None
            ),
            "pending_topic": (
                snap.pending_topic.value if snap.pending_topic else None
            ),
            "archived_facts": [self._fact_to_dict(f) for f in snap.archived_facts],
            "archived_problems": [
                self._problem_to_dict(p) for p in snap.archived_problems
            ],
        }
        return build_envelope(self.key, self.version, payload)

    def deserialize(self, envelope_json: str) -> BusinessConversationSnapshot:
        payload = parse_envelope(envelope_json, self.key, (self.version,))
        try:
            return BusinessConversationSnapshot(
                facts=tuple(self._fact_from_dict(f) for f in payload.get("facts", [])),
                problems=tuple(
                    self._problem_from_dict(p) for p in payload.get("problems", [])
                ),
                goals=frozenset(
                    _coerce_enum(BusinessGoalKind, v, "goal")
                    for v in payload.get("goals", [])
                ),
                constraints=frozenset(
                    _coerce_enum(BusinessConstraintKind, v, "constraint")
                    for v in payload.get("constraints", [])
                ),
                unknown_areas=frozenset(
                    _coerce_enum(UnknownArea, v, "unknown_area")
                    for v in payload.get("unknown_areas", [])
                ),
                hypotheses=tuple(
                    self._hyp_from_dict(h) for h in payload.get("hypotheses", [])
                ),
                current_focus=(
                    _coerce_enum(
                        ConversationTopic, payload["current_focus"], "current_focus",
                    )
                    if payload.get("current_focus") is not None else None
                ),
                pending_topic=(
                    _coerce_enum(
                        ConversationTopic, payload["pending_topic"], "pending_topic",
                    )
                    if payload.get("pending_topic") is not None else None
                ),
                archived_facts=tuple(
                    self._fact_from_dict(f)
                    for f in payload.get("archived_facts", [])
                ),
                archived_problems=tuple(
                    self._problem_from_dict(p)
                    for p in payload.get("archived_problems", [])
                ),
            )
        except (TypeError, ValueError) as exc:
            raise InvalidPersistedPayloadError(
                f"BCI payload invalid: {exc}"
            ) from exc

    @staticmethod
    def _fact_to_dict(f: ObservedBusinessFact) -> dict:
        return {
            "kind": f.kind.value,
            "value": f.value,
            "source_turn_id": f.source_turn_id,
        }

    @staticmethod
    def _fact_from_dict(d: dict) -> ObservedBusinessFact:
        return ObservedBusinessFact(
            kind=_coerce_enum(BusinessFactKind, d["kind"], "fact.kind"),
            value=d["value"],
            source_turn_id=d["source_turn_id"],
        )

    @staticmethod
    def _problem_to_dict(p: ObservedBusinessProblem) -> dict:
        return {
            "kind": p.kind.value,
            "detail": p.detail,
            "source_turn_id": p.source_turn_id,
            "primary": p.primary,
        }

    @staticmethod
    def _problem_from_dict(d: dict) -> ObservedBusinessProblem:
        return ObservedBusinessProblem(
            kind=_coerce_enum(BusinessProblemKind, d["kind"], "problem.kind"),
            detail=d["detail"],
            source_turn_id=d["source_turn_id"],
            primary=bool(d.get("primary", False)),
        )

    @staticmethod
    def _hyp_to_dict(h: RootCauseHypothesis) -> dict:
        return {
            "kind": h.kind.value,
            "confidence": h.confidence,
            "reason": h.reason,
            "supporting_observations": list(h.supporting_observations),
        }

    @staticmethod
    def _hyp_from_dict(d: dict) -> RootCauseHypothesis:
        return RootCauseHypothesis(
            kind=_coerce_enum(RootCauseKind, d["kind"], "hypothesis.kind"),
            confidence=float(d["confidence"]),
            reason=d["reason"],
            supporting_observations=tuple(d["supporting_observations"]),
        )
