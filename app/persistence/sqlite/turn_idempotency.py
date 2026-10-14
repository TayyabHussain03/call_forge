"""Persistent turn-level idempotency store.

A retry with the same (session_id, turn_request_id) after process restart
must NOT re-execute the turn. The stored committed result is returned.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from app.persistence.errors import IdempotencyConflictError
from app.persistence.sqlite.connection import SqliteConnection


@dataclass(frozen=True)
class CommittedTurnRecord:
    """Previously committed turn result, replayed on duplicate request."""

    session_id: str
    turn_request_id: str
    turn_sequence: int
    committed_revision: int
    agent_response: str
    conversation_terminal: bool
    pipeline_outcome: str | None


class SqliteTurnIdempotencyStore:
    """Durable (session_id, turn_request_id) → committed-result map."""

    def __init__(self, connection: SqliteConnection) -> None:
        self._c = connection

    def record(self, record: CommittedTurnRecord) -> None:
        try:
            self._c.raw.execute(
                """INSERT INTO turn_idempotency (
                    session_id, turn_request_id, turn_sequence,
                    committed_revision, agent_response,
                    conversation_terminal, pipeline_outcome
                ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.session_id, record.turn_request_id,
                    record.turn_sequence, record.committed_revision,
                    record.agent_response,
                    1 if record.conversation_terminal else 0,
                    record.pipeline_outcome,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise IdempotencyConflictError(
                f"turn_request_id {record.turn_request_id} already recorded "
                f"for session {record.session_id}"
            ) from exc

    def lookup(
        self, session_id: str, turn_request_id: str,
    ) -> CommittedTurnRecord | None:
        row = self._c.raw.execute(
            """SELECT * FROM turn_idempotency
               WHERE session_id = ? AND turn_request_id = ?""",
            (session_id, turn_request_id),
        ).fetchone()
        if row is None:
            return None
        return CommittedTurnRecord(
            session_id=row["session_id"],
            turn_request_id=row["turn_request_id"],
            turn_sequence=int(row["turn_sequence"]),
            committed_revision=int(row["committed_revision"]),
            agent_response=row["agent_response"],
            conversation_terminal=bool(row["conversation_terminal"]),
            pipeline_outcome=row["pipeline_outcome"],
        )
