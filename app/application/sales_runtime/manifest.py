"""Snapshot compatibility manifest.

Static declaration of what this runtime can serialize/deserialize.
Lets recovery diagnose compatibility BEFORE trying to reconstruct a
processor. Keys use the stable persistence contract names from
app.application.sales_runtime.serializers.

Keys classified as REQUIRED_FOR_ACTIVE_RECOVERY must be present for
mid-conversation recovery; missing or incompatible → fail closed.
OPTIONAL_DERIVED keys may be absent; downstream may recompute.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.application.sales_runtime.serializers import (
    BUSINESS_CONVERSATION_KEY,
    DISCUSSION_COVERAGE_KEY,
    LANGUAGE_PROFILE_KEY,
    PARTICIPANT_STATE_KEY,
    PROSPECT_INTELLIGENCE_KEY,
    QUALIFICATION_KEY,
    PayloadRequirement,
)
from app.application.sales_runtime.snapshot import SNAPSHOT_VERSION_CURRENT


# Reserved stable keys for payloads whose serializers may land later.
STEERING_KEY = "steering"
UNFINISHED_INTENT_KEY = "unfinished_intent"
REPETITION_STATE_KEY = "repetition_state"
BUDGET_STATE_KEY = "budget_state"
EVIDENCE_SATURATION_KEY = "evidence_saturation"


@dataclass(frozen=True)
class SnapshotCompatibilityManifest:
    """What this runtime can roundtrip + what active recovery requires."""

    top_level_snapshot_version: int
    supported_contract_versions: tuple[tuple[str, int], ...]
    required_for_active_recovery: frozenset[str]
    optional_derived: frozenset[str]

    def is_supported(self, key: str, version: int) -> bool:
        for k, v in self.supported_contract_versions:
            if k == key and v == version:
                return True
        return False

    def is_required(self, key: str) -> bool:
        return key in self.required_for_active_recovery

    def missing_required(self, present_keys: frozenset[str]) -> frozenset[str]:
        return frozenset(self.required_for_active_recovery - present_keys)


def default_manifest() -> SnapshotCompatibilityManifest:
    """Current manifest for Slice 46.

    BCI, Qualification, LanguageProfile are fully supported and required.
    Prospect Intelligence, Steering, Unfinished Intent, Repetition, Budget,
    Evidence Saturation are reserved keys — serializers for these land in
    follow-up slices; their absence is tolerated as OPTIONAL_DERIVED today.
    """
    return SnapshotCompatibilityManifest(
        top_level_snapshot_version=SNAPSHOT_VERSION_CURRENT,
        supported_contract_versions=(
            (BUSINESS_CONVERSATION_KEY, 1),
            (QUALIFICATION_KEY, 1),
            (LANGUAGE_PROFILE_KEY, 1),
            (PROSPECT_INTELLIGENCE_KEY, 1),
            (PARTICIPANT_STATE_KEY, 1),
            (DISCUSSION_COVERAGE_KEY, 1),
        ),
        required_for_active_recovery=frozenset({
            BUSINESS_CONVERSATION_KEY,
        }),
        optional_derived=frozenset({
            LANGUAGE_PROFILE_KEY,
            QUALIFICATION_KEY,
            PROSPECT_INTELLIGENCE_KEY,
            PARTICIPANT_STATE_KEY,
            DISCUSSION_COVERAGE_KEY,
            STEERING_KEY,
            UNFINISHED_INTENT_KEY,
            REPETITION_STATE_KEY,
            BUDGET_STATE_KEY,
            EVIDENCE_SATURATION_KEY,
        }),
    )
