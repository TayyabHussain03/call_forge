"""Transactional coordinator for atomic session + queue operations.

Fulfills the Slice 40 atomicity boundary with a real database transaction:
    - create session
    - transition queue entry READY → IN_PROGRESS
both commit together or both roll back.
"""

from __future__ import annotations

from app.leads.campaign.contracts import LeadQueueEntry, LeadQueueStatus
from app.persistence.errors import (
    IdempotencyConflictError,
    StaleRevisionError,
    TransactionFailedError,
)
from app.persistence.sqlite.connection import SqliteConnection
from app.persistence.sqlite.repositories import (
    SqliteCampaignQueueStore,
    SqliteChatSessionRepository,
)
from app.runtime.chat.contracts import ChatSession


class AtomicSessionQueueCoordinator:
    """Starts one SQLite transaction, performs both operations, commits atomically.

    If either operation raises, the transaction is rolled back and NEITHER
    change persists — no orphan session, no falsely consumed lead.
    """

    def __init__(
        self,
        connection: SqliteConnection,
        session_repository: SqliteChatSessionRepository,
        queue_store: SqliteCampaignQueueStore,
    ) -> None:
        self._c = connection
        self._sessions = session_repository
        self._queue = queue_store

    def create_session_and_consume_lead(
        self,
        session: ChatSession,
        tenant_id: str,
        campaign_id: str,
        lead_id: str,
        from_status: LeadQueueStatus = LeadQueueStatus.READY,
        to_status: LeadQueueStatus = LeadQueueStatus.IN_PROGRESS,
    ) -> tuple[ChatSession, LeadQueueEntry]:
        """Commit both operations atomically.

        Raises TransactionFailedError (with the underlying cause chained)
        if either step fails. Neither change is persisted on failure.
        """
        try:
            with self._c.transaction():
                created = self._sessions.create(session)
                transitioned = self._queue.transition(
                    tenant_id, campaign_id, lead_id, from_status, to_status,
                )
            return created, transitioned
        except (IdempotencyConflictError, StaleRevisionError):
            raise
        except Exception as exc:
            raise TransactionFailedError(
                f"session+queue transaction failed: {exc}"
            ) from exc
