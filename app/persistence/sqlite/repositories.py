"""SQLite-backed persistent repository adapters.

Each adapter matches the interface of its in-memory counterpart so that
higher-level services (SessionBootstrapService, SalesSessionLifecycleService)
can be composed with either backing. All adapters share one SqliteConnection
so a cross-repository transaction is possible via connection.transaction().
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace

from app.leads.campaign.contracts import LeadQueueEntry, LeadQueueStatus
from app.leads.contracts import (
    DuplicateClassification,
    LeadImportReadiness,
    LeadRecord,
    NormalizedPhone,
    PhoneNormalizationStatus,
    WebsiteStatus,
)
from app.persistence.errors import (
    IdempotencyConflictError,
    NotFoundError,
    StaleRevisionError,
)
from app.persistence.sqlite.connection import SqliteConnection
from app.persistence.sqlite.serializers import (
    plan_from_json,
    plan_to_json,
    policy_from_json,
    policy_to_json,
    transcript_from_json,
    transcript_to_json,
)
from app.precall.contracts import PreCallDiscoveryPolicy
from app.runtime.chat.contracts import ChatSession, RuntimeMode, SessionStatus


# ---------------------------------------------------------------------------
# ChatSession repository
# ---------------------------------------------------------------------------


class SqliteChatSessionRepository:
    """Persistent ChatSession store with optimistic concurrency on revision."""

    def __init__(self, connection: SqliteConnection) -> None:
        self._c = connection

    def create(self, session: ChatSession) -> ChatSession:
        stored = replace(session, revision=0)
        try:
            self._c.raw.execute(
                """INSERT INTO chat_sessions (
                    session_id, tenant_id, campaign_id, lead_id,
                    status, mode, revision, turn_count,
                    transcript_json, pre_call_plan_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    stored.session_id, stored.tenant_id, stored.campaign_id,
                    stored.lead_id, stored.status.value, stored.mode.value,
                    stored.revision, stored.turn_count,
                    transcript_to_json(stored.transcript),
                    plan_to_json(stored.pre_call_plan) if stored.pre_call_plan else None,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise IdempotencyConflictError(
                f"session {session.session_id} already exists"
            ) from exc
        return stored

    def get(self, session_id: str) -> ChatSession:
        row = self._c.raw.execute(
            "SELECT * FROM chat_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise NotFoundError(f"session {session_id} not found")
        return self._row_to_session(row)

    def update(self, session: ChatSession) -> ChatSession:
        current = self.get(session.session_id)
        if session.revision != current.revision:
            raise StaleRevisionError(
                f"expected revision {current.revision}, got {session.revision}"
            )
        updated = replace(session, revision=current.revision + 1)
        self._c.raw.execute(
            """UPDATE chat_sessions SET
                status = ?, mode = ?, revision = ?, turn_count = ?,
                transcript_json = ?, pre_call_plan_json = ?
             WHERE session_id = ? AND revision = ?""",
            (
                updated.status.value, updated.mode.value, updated.revision,
                updated.turn_count,
                transcript_to_json(updated.transcript),
                plan_to_json(updated.pre_call_plan) if updated.pre_call_plan else None,
                updated.session_id, current.revision,
            ),
        )
        return updated

    def list_sessions(self, tenant_id: str | None = None) -> tuple[ChatSession, ...]:
        if tenant_id is None:
            cur = self._c.raw.execute("SELECT * FROM chat_sessions")
        else:
            cur = self._c.raw.execute(
                "SELECT * FROM chat_sessions WHERE tenant_id = ?",
                (tenant_id,),
            )
        return tuple(self._row_to_session(r) for r in cur.fetchall())

    def exists(self, session_id: str) -> bool:
        row = self._c.raw.execute(
            "SELECT 1 FROM chat_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row is not None

    def active_count(self, tenant_id: str | None = None) -> int:
        if tenant_id is None:
            cur = self._c.raw.execute(
                "SELECT COUNT(*) AS n FROM chat_sessions WHERE status = ?",
                (SessionStatus.ACTIVE.value,),
            )
        else:
            cur = self._c.raw.execute(
                "SELECT COUNT(*) AS n FROM chat_sessions "
                "WHERE status = ? AND tenant_id = ?",
                (SessionStatus.ACTIVE.value, tenant_id),
            )
        return int(cur.fetchone()["n"])

    def delete(self, session_id: str) -> None:
        """Used only by transactional rollback paths."""
        self._c.raw.execute(
            "DELETE FROM chat_sessions WHERE session_id = ?",
            (session_id,),
        )

    def _row_to_session(self, row: sqlite3.Row) -> ChatSession:
        return ChatSession(
            session_id=row["session_id"],
            tenant_id=row["tenant_id"],
            campaign_id=row["campaign_id"],
            lead_id=row["lead_id"],
            status=SessionStatus(row["status"]),
            mode=RuntimeMode(row["mode"]),
            revision=int(row["revision"]),
            turn_count=int(row["turn_count"]),
            transcript=transcript_from_json(row["transcript_json"]),
            pre_call_plan=plan_from_json(row["pre_call_plan_json"]),
        )


# ---------------------------------------------------------------------------
# Lead store
# ---------------------------------------------------------------------------


class SqliteLeadStore:
    """Persistent LeadRecord store scoped by tenant/campaign/lead_id."""

    def __init__(self, connection: SqliteConnection) -> None:
        self._c = connection

    def save(self, lead: LeadRecord) -> None:
        self._c.raw.execute(
            """INSERT OR REPLACE INTO leads (
                lead_id, tenant_id, campaign_id, business_name,
                phone_raw, phone_normalized, phone_status, phone_country_context,
                website, website_status, category, city, state, country, address,
                email, contact_name, source, notes,
                import_batch_id, row_number, readiness, duplicate_status
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                lead.lead_id, lead.tenant_id, lead.campaign_id, lead.business_name,
                lead.phone.raw, lead.phone.normalized, lead.phone.status.value,
                lead.phone.country_context,
                lead.website, lead.website_status.value,
                lead.category, lead.city, lead.state, lead.country, lead.address,
                lead.email, lead.contact_name, lead.source, lead.notes,
                lead.import_batch_id, lead.row_number,
                lead.readiness.value, lead.duplicate_status.value,
            ),
        )

    def get(self, tenant_id: str, campaign_id: str, lead_id: str) -> LeadRecord | None:
        row = self._c.raw.execute(
            "SELECT * FROM leads WHERE lead_id = ? AND tenant_id = ? AND campaign_id = ?",
            (lead_id, tenant_id, campaign_id),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_lead(row)

    def list_by_campaign(self, tenant_id: str, campaign_id: str) -> tuple[LeadRecord, ...]:
        cur = self._c.raw.execute(
            "SELECT * FROM leads WHERE tenant_id = ? AND campaign_id = ?",
            (tenant_id, campaign_id),
        )
        return tuple(self._row_to_lead(r) for r in cur.fetchall())

    def _row_to_lead(self, row: sqlite3.Row) -> LeadRecord:
        phone = NormalizedPhone(
            raw=row["phone_raw"],
            normalized=row["phone_normalized"],
            status=PhoneNormalizationStatus(row["phone_status"]),
            country_context=row["phone_country_context"],
        )
        return LeadRecord(
            lead_id=row["lead_id"],
            tenant_id=row["tenant_id"],
            campaign_id=row["campaign_id"],
            business_name=row["business_name"],
            phone=phone,
            website=row["website"],
            website_status=WebsiteStatus(row["website_status"]),
            category=row["category"],
            city=row["city"],
            state=row["state"],
            country=row["country"],
            address=row["address"],
            email=row["email"],
            contact_name=row["contact_name"],
            source=row["source"],
            notes=row["notes"],
            import_batch_id=row["import_batch_id"],
            row_number=row["row_number"],
            readiness=LeadImportReadiness(row["readiness"]),
            duplicate_status=DuplicateClassification(row["duplicate_status"]),
        )


# ---------------------------------------------------------------------------
# Campaign queue store
# ---------------------------------------------------------------------------


class SqliteCampaignQueueStore:
    """Persistent queue state with optimistic concurrency.

    DNC persistence is enforced by the LeadQueueEntry contract: is_dnc=True
    forces status=DNC. Writers must never regress DNC — guarded here too.
    """

    def __init__(self, connection: SqliteConnection) -> None:
        self._c = connection

    def save(self, entry: LeadQueueEntry) -> LeadQueueEntry:
        """Insert or replace. DNC rows cannot be regressed to non-DNC."""
        existing = self.get(entry.tenant_id, entry.campaign_id, entry.lead_id)
        if existing is not None and existing.is_dnc and not entry.is_dnc:
            # DNC must survive everything — refuse to regress
            raise StaleRevisionError(
                f"cannot regress DNC lead {entry.lead_id} to non-DNC state"
            )
        revision = 0 if existing is None else self._stored_revision(entry) + 1
        self._c.raw.execute(
            """INSERT OR REPLACE INTO queue_entries (
                lead_id, campaign_id, tenant_id, queue_position, status,
                attempt_count, last_attempt_at, is_dnc, revision
            ) VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                entry.lead_id, entry.campaign_id, entry.tenant_id,
                entry.queue_position, entry.status.value,
                entry.attempt_count, entry.last_attempt_at,
                1 if entry.is_dnc else 0,
                revision,
            ),
        )
        return entry

    def get(
        self, tenant_id: str, campaign_id: str, lead_id: str,
    ) -> LeadQueueEntry | None:
        row = self._c.raw.execute(
            """SELECT * FROM queue_entries
               WHERE tenant_id = ? AND campaign_id = ? AND lead_id = ?""",
            (tenant_id, campaign_id, lead_id),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def _stored_revision(self, entry: LeadQueueEntry) -> int:
        row = self._c.raw.execute(
            """SELECT revision FROM queue_entries
               WHERE tenant_id = ? AND campaign_id = ? AND lead_id = ?""",
            (entry.tenant_id, entry.campaign_id, entry.lead_id),
        ).fetchone()
        return int(row["revision"]) if row else -1

    def transition(
        self,
        tenant_id: str,
        campaign_id: str,
        lead_id: str,
        from_status: LeadQueueStatus,
        to_status: LeadQueueStatus,
    ) -> LeadQueueEntry:
        """Atomic state transition. Rejects if current status doesn't match."""
        current = self.get(tenant_id, campaign_id, lead_id)
        if current is None:
            raise NotFoundError(f"queue entry {lead_id} not found")
        if current.is_dnc and to_status != LeadQueueStatus.DNC:
            raise StaleRevisionError(
                f"cannot transition DNC lead {lead_id} to {to_status.value}"
            )
        if current.status != from_status:
            raise StaleRevisionError(
                f"queue entry {lead_id} is {current.status.value}, expected {from_status.value}"
            )
        updated = replace(current, status=to_status)
        new_revision = self._stored_revision(current) + 1
        self._c.raw.execute(
            """UPDATE queue_entries SET status = ?, revision = ?
               WHERE tenant_id = ? AND campaign_id = ? AND lead_id = ?
                 AND status = ?""",
            (
                to_status.value, new_revision,
                tenant_id, campaign_id, lead_id, from_status.value,
            ),
        )
        return updated

    def mark_dnc(
        self, tenant_id: str, campaign_id: str, lead_id: str,
    ) -> LeadQueueEntry:
        """DNC has highest priority — overrides any current status."""
        current = self.get(tenant_id, campaign_id, lead_id)
        if current is None:
            raise NotFoundError(f"queue entry {lead_id} not found")
        updated = replace(current, is_dnc=True, status=LeadQueueStatus.DNC)
        new_revision = self._stored_revision(current) + 1
        self._c.raw.execute(
            """UPDATE queue_entries SET status = ?, is_dnc = 1, revision = ?
               WHERE tenant_id = ? AND campaign_id = ? AND lead_id = ?""",
            (
                LeadQueueStatus.DNC.value, new_revision,
                tenant_id, campaign_id, lead_id,
            ),
        )
        return updated

    def list_by_campaign(
        self, tenant_id: str, campaign_id: str,
    ) -> tuple[LeadQueueEntry, ...]:
        cur = self._c.raw.execute(
            """SELECT * FROM queue_entries
               WHERE tenant_id = ? AND campaign_id = ?
               ORDER BY queue_position""",
            (tenant_id, campaign_id),
        )
        return tuple(self._row_to_entry(r) for r in cur.fetchall())

    def _row_to_entry(self, row: sqlite3.Row) -> LeadQueueEntry:
        return LeadQueueEntry(
            lead_id=row["lead_id"],
            campaign_id=row["campaign_id"],
            tenant_id=row["tenant_id"],
            queue_position=int(row["queue_position"]),
            status=LeadQueueStatus(row["status"]),
            attempt_count=int(row["attempt_count"]),
            last_attempt_at=row["last_attempt_at"],
            is_dnc=bool(row["is_dnc"]),
        )


# ---------------------------------------------------------------------------
# Policy registry
# ---------------------------------------------------------------------------


class SqlitePolicyRegistry:
    """Persistent exact-version policy lookup. No fuzzy, no fallback."""

    def __init__(self, connection: SqliteConnection) -> None:
        self._c = connection

    def register(self, campaign_id: str, policy: PreCallDiscoveryPolicy) -> None:
        self._c.raw.execute(
            """INSERT OR REPLACE INTO policies (
                campaign_id, policy_version, policy_id, is_default, policy_json
            ) VALUES (?, ?, ?, 0, ?)""",
            (campaign_id, policy.version, policy.policy_id, policy_to_json(policy)),
        )

    def register_default(self, campaign_id: str, policy: PreCallDiscoveryPolicy) -> None:
        # Clear any prior default for this campaign
        self._c.raw.execute(
            "UPDATE policies SET is_default = 0 WHERE campaign_id = ?",
            (campaign_id,),
        )
        self._c.raw.execute(
            """INSERT OR REPLACE INTO policies (
                campaign_id, policy_version, policy_id, is_default, policy_json
            ) VALUES (?, ?, ?, 1, ?)""",
            (campaign_id, policy.version, policy.policy_id, policy_to_json(policy)),
        )

    def resolve(
        self, campaign_id: str, version: str | None = None,
    ) -> PreCallDiscoveryPolicy | None:
        if version is not None:
            row = self._c.raw.execute(
                """SELECT policy_json FROM policies
                   WHERE campaign_id = ? AND policy_version = ?""",
                (campaign_id, version),
            ).fetchone()
        else:
            row = self._c.raw.execute(
                """SELECT policy_json FROM policies
                   WHERE campaign_id = ? AND is_default = 1 LIMIT 1""",
                (campaign_id,),
            ).fetchone()
        if row is None:
            return None
        return policy_from_json(row["policy_json"])


# ---------------------------------------------------------------------------
# Bootstrap idempotency store
# ---------------------------------------------------------------------------


class SqliteBootstrapIdempotencyStore:
    """Persistent record of bootstrap request outcomes, keyed by request_id."""

    def __init__(self, connection: SqliteConnection) -> None:
        self._c = connection

    def record(
        self,
        bootstrap_request_id: str,
        tenant_id: str,
        campaign_id: str,
        lead_id: str,
        session_id: str | None,
        status: str,
        result_payload: dict,
    ) -> None:
        try:
            self._c.raw.execute(
                """INSERT INTO bootstrap_idempotency (
                    bootstrap_request_id, tenant_id, campaign_id, lead_id,
                    session_id, status, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    bootstrap_request_id, tenant_id, campaign_id, lead_id,
                    session_id, status, json.dumps(result_payload),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise IdempotencyConflictError(
                f"bootstrap_request_id {bootstrap_request_id} already recorded"
            ) from exc

    def lookup(self, bootstrap_request_id: str) -> dict | None:
        row = self._c.raw.execute(
            """SELECT status, result_json FROM bootstrap_idempotency
               WHERE bootstrap_request_id = ?""",
            (bootstrap_request_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "status": row["status"],
            "result": json.loads(row["result_json"]),
        }
