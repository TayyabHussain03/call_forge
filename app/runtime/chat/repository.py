"""In-memory chat session repository with optimistic concurrency.

Sessions are immutable snapshots. Each mutation returns a new snapshot
with an incremented revision. A stale revision is rejected to prevent
lost updates.
"""

from __future__ import annotations

from dataclasses import replace

from app.runtime.chat.contracts import ChatSession, SessionStatus


class StaleRevisionError(Exception):
    """Raised when a session update is attempted with a stale revision."""


class SessionNotFoundError(Exception):
    """Raised when a session_id does not exist in the repository."""


class InMemoryChatSessionRepository:
    """Thread-unsafe in-memory store; sufficient for single-process simulation."""

    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}

    def create(self, session: ChatSession) -> ChatSession:
        if session.session_id in self._sessions:
            raise ValueError(f"session {session.session_id} already exists")
        stored = replace(session, revision=0)
        self._sessions[session.session_id] = stored
        return stored

    def get(self, session_id: str) -> ChatSession:
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"session {session_id} not found")
        return self._sessions[session_id]

    def update(self, session: ChatSession) -> ChatSession:
        current = self.get(session.session_id)
        if session.revision != current.revision:
            raise StaleRevisionError(
                f"expected revision {current.revision}, got {session.revision}"
            )
        updated = replace(session, revision=current.revision + 1)
        self._sessions[session.session_id] = updated
        return updated

    def list_sessions(self, tenant_id: str | None = None) -> tuple[ChatSession, ...]:
        sessions = self._sessions.values()
        if tenant_id is not None:
            sessions = [s for s in sessions if s.tenant_id == tenant_id]
        return tuple(sessions)

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def active_count(self, tenant_id: str | None = None) -> int:
        return sum(
            1 for s in self._sessions.values()
            if s.status == SessionStatus.ACTIVE
            and (tenant_id is None or s.tenant_id == tenant_id)
        )
