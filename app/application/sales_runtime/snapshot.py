"""Authoritative conversation runtime snapshot contract.

One committed turn = session revision + runtime snapshot + idempotency row.
The snapshot captures the minimum authoritative state required to continue
an ACTIVE conversation after process restart WITHOUT replaying the transcript.

Design rules:
- deterministic JSON serialization
- explicit snapshot_version
- fail closed on unknown version
- no raw LLM reasoning, no provider objects, no retrieval candidates
- reuse existing immutable domain contracts; do not create competing models
- derived state is recomputed only when deterministic equivalence is proven
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.core.constants import ConversationState


SNAPSHOT_VERSION_CURRENT = 1


class SnapshotStatus(str, Enum):
    """Snapshot export outcome."""

    EXPORTED = "exported"
    EXPORT_NOT_SUPPORTED = "export_not_supported"


@dataclass(frozen=True)
class ConversationRuntimeSnapshot:
    """Minimum authoritative state for deterministic conversation continuation.

    This is the ONLY contract that bridges a running TurnProcessor and
    persistent storage. Processors own export; persistent runtime owns
    atomic commit; the factory owns reconstruction.

    What this snapshot carries today:
        - identity (session_id, source_session_revision)
        - FSM authoritative conversation state
        - sales stage (advisory)
        - active_knowledge_base_id (configuration reference)
        - last_pipeline_outcome marker (bounded enum string)
        - persisted_contract_payloads: a map of domain-contract identities
          to serialized frozen snapshots (BCI, BDE, PQI, prospect intelligence,
          business memory, language, steering) — payload keys are stable
          strings; empty for processors that cannot yet export those contracts

    What this snapshot deliberately omits:
        - raw transcript (observational, already stored on ChatSession)
        - LLM chain-of-thought
        - provider SDK objects
        - RetrievalCandidateSet (ephemeral, re-run after recovery)
        - rejected evidence
        - Python callbacks / network clients
    """

    snapshot_version: int
    session_id: str
    source_session_revision: int
    conversation_state: ConversationState
    sales_stage: str | None = None
    active_knowledge_base_id: str | None = None
    last_pipeline_outcome: str | None = None
    persisted_contract_payloads: tuple[tuple[str, str], ...] = ()
    is_dnc_observed: bool = False
    is_not_interested_observed: bool = False

    def __post_init__(self) -> None:
        if self.snapshot_version < 1:
            raise ValueError("snapshot_version must be >= 1")
        if not self.session_id or not self.session_id.strip():
            raise ValueError("session_id must not be empty")
        if self.source_session_revision < 0:
            raise ValueError("source_session_revision must be non-negative")
        if not isinstance(self.conversation_state, ConversationState):
            raise TypeError("conversation_state must be ConversationState")
        # Deterministic: no duplicate payload keys
        keys = [k for k, _ in self.persisted_contract_payloads]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate persisted_contract_payload keys")

    def payload(self, key: str) -> str | None:
        for k, v in self.persisted_contract_payloads:
            if k == key:
                return v
        return None
