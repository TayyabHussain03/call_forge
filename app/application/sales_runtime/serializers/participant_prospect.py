"""Per-participant ProspectIntelligenceSnapshot archive serializer.

Participant A's prospect intelligence (interest, objection, busy, role,
openness, etc.) must NEVER transfer to Participant B. This serializer
persists the archived snapshots keyed by participant reference ID so a
restart can restore the isolation.
"""

from __future__ import annotations

from app.application.sales_runtime.serializers.base import (
    PayloadRequirement,
    build_envelope,
    parse_envelope,
)
from app.application.sales_runtime.serializers.prospect import (
    ProspectIntelligenceSerializer,
)
from app.conversation.prospect_intelligence.contracts import (
    ProspectIntelligenceSnapshot,
)
from app.persistence.errors import InvalidPersistedPayloadError


PARTICIPANT_PROSPECT_ARCHIVE_KEY = "participant_prospect_archive"
PARTICIPANT_PROSPECT_ARCHIVE_VERSION = 1


class ParticipantProspectArchiveSerializer:
    """Serializes `tuple[(participant_reference_id, snapshot), ...]`.

    Each snapshot is itself encoded via the Slice 48 Prospect Intelligence
    serializer envelope so individual snapshot schema can evolve
    independently of this outer container.
    """

    @property
    def key(self) -> str:
        return PARTICIPANT_PROSPECT_ARCHIVE_KEY

    @property
    def version(self) -> int:
        return PARTICIPANT_PROSPECT_ARCHIVE_VERSION

    @property
    def requirement(self) -> PayloadRequirement:
        return PayloadRequirement.OPTIONAL_DERIVED

    def serialize(
        self, archive: tuple[tuple[str, ProspectIntelligenceSnapshot], ...],
    ) -> str:
        inner = ProspectIntelligenceSerializer()
        payload = {
            "entries": [
                [pid, inner.serialize(snap)]
                for pid, snap in sorted(archive, key=lambda e: e[0])
            ],
        }
        return build_envelope(self.key, self.version, payload)

    def deserialize(
        self, envelope_json: str,
    ) -> tuple[tuple[str, ProspectIntelligenceSnapshot], ...]:
        payload = parse_envelope(envelope_json, self.key, (self.version,))
        inner = ProspectIntelligenceSerializer()
        try:
            return tuple(
                (pid, inner.deserialize(envelope))
                for pid, envelope in payload.get("entries", [])
            )
        except (TypeError, KeyError, ValueError) as exc:
            raise InvalidPersistedPayloadError(
                f"participant prospect archive payload invalid: {exc}"
            ) from exc
