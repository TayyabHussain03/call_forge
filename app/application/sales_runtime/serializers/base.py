"""Shared serializer protocol + envelope format.

Envelope: {"contract": <key>, "version": <int>, "payload": {...}}

Rules:
- deterministic JSON (sorted collection ordering)
- explicit enum values
- reject unknown enum values
- reject payload missing required fields
- never pickle, never repr(), never __dict__ introspection
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Protocol, TypeVar

from app.persistence.errors import InvalidPersistedPayloadError


T = TypeVar("T")


class PayloadRequirement(str, Enum):
    """How critical a contract payload is for active-session recovery."""

    REQUIRED_FOR_ACTIVE_RECOVERY = "required_for_active_recovery"
    OPTIONAL_DERIVED = "optional_derived"
    NOT_APPLICABLE = "not_applicable"


class SnapshotContractSerializer(Protocol, Generic[T]):
    """One serializer per domain contract."""

    @property
    def key(self) -> str:
        """Stable, versioned, persistence-contract key (never a class name)."""

    @property
    def version(self) -> int:
        """Payload version for this contract. Bumps on breaking payload change."""

    @property
    def requirement(self) -> PayloadRequirement:
        """How critical restoration of this contract is."""

    def serialize(self, value: T) -> str:
        """Return envelope JSON string. Deterministic."""

    def deserialize(self, envelope_json: str) -> T:
        """Parse envelope JSON string into a domain contract.

        Raises InvalidPersistedPayloadError on malformed/unsupported payload.
        """


def build_envelope(key: str, version: int, payload: dict[str, Any]) -> str:
    """Deterministic JSON envelope. Lists/dicts caller-sorted where needed."""
    return json.dumps(
        {"contract": key, "version": version, "payload": payload},
        ensure_ascii=False, sort_keys=True,
    )


def parse_envelope(
    envelope_json: str, expected_key: str, supported_versions: tuple[int, ...],
) -> dict[str, Any]:
    """Validate envelope and return the payload dict.

    Raises InvalidPersistedPayloadError if JSON is invalid, key mismatches,
    or the stored version is not in supported_versions.
    """
    try:
        envelope = json.loads(envelope_json)
    except json.JSONDecodeError as exc:
        raise InvalidPersistedPayloadError(
            f"{expected_key} envelope JSON invalid: {exc}"
        ) from exc
    if not isinstance(envelope, dict):
        raise InvalidPersistedPayloadError(
            f"{expected_key} envelope must be a JSON object"
        )
    contract = envelope.get("contract")
    if contract != expected_key:
        raise InvalidPersistedPayloadError(
            f"envelope key mismatch: expected {expected_key!r}, got {contract!r}"
        )
    version = envelope.get("version")
    if not isinstance(version, int) or version not in supported_versions:
        raise InvalidPersistedPayloadError(
            f"{expected_key} payload version {version!r} not in supported {supported_versions}"
        )
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise InvalidPersistedPayloadError(
            f"{expected_key} payload must be a JSON object"
        )
    return payload


def _coerce_enum(enum_cls: type[Enum], value: Any, field_name: str):
    """Strict enum coercion. Unknown value → InvalidPersistedPayloadError."""
    try:
        return enum_cls(value)
    except (ValueError, KeyError) as exc:
        raise InvalidPersistedPayloadError(
            f"{field_name} has unknown enum value {value!r}"
        ) from exc
