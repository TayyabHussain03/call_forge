"""Framework-agnostic, immutable advisory qualification contracts."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class QualificationEvidenceKind(str, Enum): OBSERVED = "observed"; INFERRED = "inferred"
class QualificationValueState(str, Enum): OBSERVED = "observed"; INFERRED = "inferred"; UNKNOWN = "unknown"
class QualificationCompleteness(str, Enum): VERY_EARLY = "very_early"; PARTIAL = "partial"; MOSTLY_KNOWN = "mostly_known"; SUFFICIENT = "sufficient"
class QualificationProgress(str, Enum): ADVANCING = "advancing"; STABLE = "stable"; REGRESSED = "regressed"; CORRECTED = "corrected"; COMPLETE = "complete"

@dataclass(frozen=True)
class QualificationField:
    field_id: str
    display_label: str
    mandatory: bool
    evidence_kind: QualificationEvidenceKind
    allowed_values: frozenset[str] = frozenset()
    def __post_init__(self):
        if not self.field_id or len(self.field_id) > 80 or not self.display_label or len(self.display_label) > 120: raise ValueError("qualification field identity is bounded")
        if not isinstance(self.mandatory, bool): raise TypeError("mandatory must be boolean")

@dataclass(frozen=True)
class QualificationConfiguration:
    tenant_id: str; business_id: str; campaign_id: str; fields: tuple[QualificationField, ...]
    def __post_init__(self):
        if not all((self.tenant_id, self.business_id, self.campaign_id)) or len(self.fields) > 30 or len({f.field_id for f in self.fields}) != len(self.fields): raise ValueError("qualification configuration is invalid")

@dataclass(frozen=True)
class QualificationObservation:
    field_id: str; value: str; evidence_kind: QualificationEvidenceKind; evidence_ids: tuple[str, ...]; source_turn_id: str
    def __post_init__(self):
        if not self.field_id or not self.value or not self.source_turn_id or len(self.evidence_ids) > 6: raise ValueError("qualification observation is bounded")

@dataclass(frozen=True)
class QualificationEvidence:
    observations: tuple[QualificationObservation, ...] = (); corrections: frozenset[str] = frozenset()

@dataclass(frozen=True)
class QualificationFieldState:
    field_id: str; state: QualificationValueState; value: str | None = None; evidence_ids: tuple[str, ...] = (); source_turn_id: str | None = None

@dataclass(frozen=True)
class QualificationSnapshot:
    configuration_id: tuple[str, str, str]
    fields: tuple[QualificationFieldState, ...]
    completeness: QualificationCompleteness
    progress: QualificationProgress
    newly_observed: frozenset[str] = frozenset(); corrected: frozenset[str] = frozenset(); replaced: frozenset[str] = frozenset(); archived: tuple[QualificationObservation, ...] = ()
    def __post_init__(self):
        forbidden={"service","action","authority","price","execution","question","recommend"}
        if any(any(word in f for word in forbidden) for f in self.__dataclass_fields__): raise ValueError("qualification contains forbidden field")
