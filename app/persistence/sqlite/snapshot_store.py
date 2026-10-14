"""Persistent store for ConversationRuntimeSnapshot.

Snapshot lives on the chat_sessions row (runtime_snapshot_json,
runtime_snapshot_version) so one atomic write updates session revision
and snapshot together in a single transaction.
"""

from __future__ import annotations

import json

from app.application.sales_runtime.snapshot import (
    SNAPSHOT_VERSION_CURRENT,
    ConversationRuntimeSnapshot,
)
from app.core.constants import ConversationState
from app.persistence.errors import InvalidPersistedPayloadError, NotFoundError
from app.persistence.sqlite.connection import SqliteConnection


def snapshot_to_json(snapshot: ConversationRuntimeSnapshot) -> str:
    return json.dumps(
        {
            "snapshot_version": snapshot.snapshot_version,
            "session_id": snapshot.session_id,
            "source_session_revision": snapshot.source_session_revision,
            "conversation_state": snapshot.conversation_state.value,
            "sales_stage": snapshot.sales_stage,
            "active_knowledge_base_id": snapshot.active_knowledge_base_id,
            "last_pipeline_outcome": snapshot.last_pipeline_outcome,
            "persisted_contract_payloads": [
                [k, v] for k, v in snapshot.persisted_contract_payloads
            ],
            "is_dnc_observed": snapshot.is_dnc_observed,
            "is_not_interested_observed": snapshot.is_not_interested_observed,
        },
        ensure_ascii=False,
    )


def snapshot_from_json(payload: str) -> ConversationRuntimeSnapshot:
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise InvalidPersistedPayloadError(f"snapshot JSON invalid: {exc}") from exc
    try:
        version = int(raw["snapshot_version"])
        if version > SNAPSHOT_VERSION_CURRENT:
            raise InvalidPersistedPayloadError(
                f"snapshot_version {version} is newer than supported "
                f"{SNAPSHOT_VERSION_CURRENT}"
            )
        return ConversationRuntimeSnapshot(
            snapshot_version=version,
            session_id=raw["session_id"],
            source_session_revision=int(raw["source_session_revision"]),
            conversation_state=ConversationState(raw["conversation_state"]),
            sales_stage=raw.get("sales_stage"),
            active_knowledge_base_id=raw.get("active_knowledge_base_id"),
            last_pipeline_outcome=raw.get("last_pipeline_outcome"),
            persisted_contract_payloads=tuple(
                (k, v) for k, v in raw.get("persisted_contract_payloads", [])
            ),
            is_dnc_observed=bool(raw.get("is_dnc_observed", False)),
            is_not_interested_observed=bool(
                raw.get("is_not_interested_observed", False)
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidPersistedPayloadError(f"snapshot payload invalid: {exc}") from exc


class SqliteRuntimeSnapshotStore:
    """Reads and writes ConversationRuntimeSnapshot on chat_sessions row."""

    def __init__(self, connection: SqliteConnection) -> None:
        self._c = connection

    def save(self, snapshot: ConversationRuntimeSnapshot) -> None:
        """Write snapshot alongside session row. Must run inside the
        same transaction as the session revision update."""
        self._c.raw.execute(
            """UPDATE chat_sessions
               SET runtime_snapshot_json = ?, runtime_snapshot_version = ?
               WHERE session_id = ?""",
            (
                snapshot_to_json(snapshot),
                snapshot.snapshot_version,
                snapshot.session_id,
            ),
        )

    def load(self, session_id: str) -> ConversationRuntimeSnapshot | None:
        row = self._c.raw.execute(
            """SELECT runtime_snapshot_json, runtime_snapshot_version
               FROM chat_sessions WHERE session_id = ?""",
            (session_id,),
        ).fetchone()
        if row is None:
            raise NotFoundError(f"session {session_id} not found")
        if row["runtime_snapshot_json"] is None:
            return None
        return snapshot_from_json(row["runtime_snapshot_json"])
