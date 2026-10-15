"""QualificationSnapshot serializer.

Preserves OBSERVED / INFERRED / UNKNOWN per field — restart must never
promote or demote a qualification state.
"""

from __future__ import annotations

from app.application.sales_runtime.serializers.base import (
    PayloadRequirement,
    _coerce_enum,
    build_envelope,
    parse_envelope,
)
from app.conversation.qualification.contracts import (
    QualificationCompleteness,
    QualificationFieldState,
    QualificationObservation,
    QualificationProgress,
    QualificationSnapshot,
    QualificationValueState,
    QualificationEvidenceKind,
)
from app.persistence.errors import InvalidPersistedPayloadError


QUALIFICATION_KEY = "qualification"
QUALIFICATION_VERSION = 1


class QualificationSerializer:
    @property
    def key(self) -> str:
        return QUALIFICATION_KEY

    @property
    def version(self) -> int:
        return QUALIFICATION_VERSION

    @property
    def requirement(self) -> PayloadRequirement:
        return PayloadRequirement.REQUIRED_FOR_ACTIVE_RECOVERY

    def serialize(self, snap: QualificationSnapshot) -> str:
        payload = {
            "configuration_id": list(snap.configuration_id),
            "fields": [self._field_to_dict(f) for f in snap.fields],
            "completeness": snap.completeness.value,
            "progress": snap.progress.value,
            "newly_observed": sorted(snap.newly_observed),
            "corrected": sorted(snap.corrected),
            "replaced": sorted(snap.replaced),
            "archived": [self._obs_to_dict(o) for o in snap.archived],
        }
        return build_envelope(self.key, self.version, payload)

    def deserialize(self, envelope_json: str) -> QualificationSnapshot:
        payload = parse_envelope(envelope_json, self.key, (self.version,))
        try:
            cfg = payload["configuration_id"]
            if not isinstance(cfg, list) or len(cfg) != 3:
                raise ValueError("configuration_id must be a 3-element list")
            return QualificationSnapshot(
                configuration_id=tuple(cfg),
                fields=tuple(
                    self._field_from_dict(f) for f in payload.get("fields", [])
                ),
                completeness=_coerce_enum(
                    QualificationCompleteness, payload["completeness"],
                    "qualification.completeness",
                ),
                progress=_coerce_enum(
                    QualificationProgress, payload["progress"],
                    "qualification.progress",
                ),
                newly_observed=frozenset(payload.get("newly_observed", [])),
                corrected=frozenset(payload.get("corrected", [])),
                replaced=frozenset(payload.get("replaced", [])),
                archived=tuple(
                    self._obs_from_dict(o) for o in payload.get("archived", [])
                ),
            )
        except (TypeError, KeyError, ValueError) as exc:
            raise InvalidPersistedPayloadError(
                f"Qualification payload invalid: {exc}"
            ) from exc

    @staticmethod
    def _field_to_dict(f: QualificationFieldState) -> dict:
        return {
            "field_id": f.field_id,
            "state": f.state.value,
            "value": f.value,
            "evidence_ids": list(f.evidence_ids),
            "source_turn_id": f.source_turn_id,
        }

    @staticmethod
    def _field_from_dict(d: dict) -> QualificationFieldState:
        return QualificationFieldState(
            field_id=d["field_id"],
            state=_coerce_enum(
                QualificationValueState, d["state"], "qualification.field.state",
            ),
            value=d.get("value"),
            evidence_ids=tuple(d.get("evidence_ids", ())),
            source_turn_id=d.get("source_turn_id"),
        )

    @staticmethod
    def _obs_to_dict(o: QualificationObservation) -> dict:
        return {
            "field_id": o.field_id,
            "value": o.value,
            "evidence_kind": o.evidence_kind.value,
            "evidence_ids": list(o.evidence_ids),
            "source_turn_id": o.source_turn_id,
        }

    @staticmethod
    def _obs_from_dict(d: dict) -> QualificationObservation:
        return QualificationObservation(
            field_id=d["field_id"],
            value=d["value"],
            evidence_kind=_coerce_enum(
                QualificationEvidenceKind, d["evidence_kind"],
                "qualification.obs.evidence_kind",
            ),
            evidence_ids=tuple(d["evidence_ids"]),
            source_turn_id=d["source_turn_id"],
        )
