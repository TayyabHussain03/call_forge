"""Exact deterministic lookup across already-approved evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.conversation.context.contracts import (
    ApprovedEvidenceItem,
    EvidenceScopeKind,
    EvidenceType,
)


class EvidenceMatchStatus(str, Enum):
    MATCHED = "matched"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class EvidenceMatch:
    status: EvidenceMatchStatus
    items: tuple[ApprovedEvidenceItem, ...] = ()


class ApprovedEvidenceMatcher:
    """Match exact typed keys and scopes without semantic inference."""

    def match(
        self,
        evidence: tuple[ApprovedEvidenceItem, ...],
        *,
        fact_key: str | None,
        evidence_type: EvidenceType | None,
        campaign_id: str | None,
        service_id: str | None,
    ) -> EvidenceMatch:
        if fact_key is None:
            return EvidenceMatch(EvidenceMatchStatus.NOT_FOUND)
        matches = tuple(
            sorted(
                (
                    item
                    for item in evidence
                    if item.fact_key == fact_key
                    and (evidence_type is None or item.evidence_type == evidence_type)
                    and _scope_matches(item, campaign_id, service_id)
                ),
                key=lambda item: item.evidence_id,
            )
        )
        if not matches:
            return EvidenceMatch(EvidenceMatchStatus.NOT_FOUND)
        if len({item.statement for item in matches}) > 1:
            return EvidenceMatch(EvidenceMatchStatus.CONFLICT)
        return EvidenceMatch(EvidenceMatchStatus.MATCHED, matches[:8])


def _scope_matches(
    item: ApprovedEvidenceItem,
    campaign_id: str | None,
    service_id: str | None,
) -> bool:
    if item.scope.kind == EvidenceScopeKind.GLOBAL:
        return True
    if item.scope.kind == EvidenceScopeKind.CAMPAIGN:
        return item.scope.scope_id == campaign_id
    return service_id is not None and item.scope.scope_id == service_id
