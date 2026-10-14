"""SQLite connection wrapper with schema initialization and migrations.

Keeps infrastructure concerns (connection, foreign keys, transactions)
in one place. Storage technology must remain replaceable — all SQL is
isolated here and in per-repository modules.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from app.persistence.errors import (
    DatabaseUnavailableError,
    SchemaMismatchError,
    TransactionFailedError,
)


CURRENT_SCHEMA_VERSION = 1


# Schema definition — one place for all persistent tables
# Enum values are stored as their stable string values from the Python enums.
# Complex nested objects (PreCallConversationPlan, transcript) use JSON text.
SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY NOT NULL
);

CREATE TABLE IF NOT EXISTS leads (
    lead_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    business_name TEXT NOT NULL,
    phone_raw TEXT NOT NULL,
    phone_normalized TEXT,
    phone_status TEXT NOT NULL,
    phone_country_context TEXT,
    website TEXT,
    website_status TEXT NOT NULL,
    category TEXT,
    city TEXT,
    state TEXT,
    country TEXT,
    address TEXT,
    email TEXT,
    contact_name TEXT,
    source TEXT,
    notes TEXT,
    import_batch_id TEXT,
    row_number INTEGER,
    readiness TEXT NOT NULL,
    duplicate_status TEXT NOT NULL,
    PRIMARY KEY (tenant_id, campaign_id, lead_id)
);
CREATE INDEX IF NOT EXISTS idx_leads_tenant_campaign
    ON leads(tenant_id, campaign_id);

CREATE TABLE IF NOT EXISTS queue_entries (
    lead_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    queue_position INTEGER NOT NULL,
    status TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_attempt_at TEXT,
    is_dnc INTEGER NOT NULL DEFAULT 0,
    revision INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (tenant_id, campaign_id, lead_id)
);
CREATE INDEX IF NOT EXISTS idx_queue_tenant_campaign_status
    ON queue_entries(tenant_id, campaign_id, status);

CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY NOT NULL,
    tenant_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    lead_id TEXT NOT NULL,
    status TEXT NOT NULL,
    mode TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 0,
    turn_count INTEGER NOT NULL DEFAULT 0,
    transcript_json TEXT NOT NULL DEFAULT '[]',
    pre_call_plan_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_tenant
    ON chat_sessions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_sessions_lead
    ON chat_sessions(tenant_id, campaign_id, lead_id);

CREATE TABLE IF NOT EXISTS policies (
    campaign_id TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    policy_id TEXT NOT NULL,
    is_default INTEGER NOT NULL DEFAULT 0,
    policy_json TEXT NOT NULL,
    PRIMARY KEY (campaign_id, policy_version)
);

CREATE TABLE IF NOT EXISTS bootstrap_idempotency (
    bootstrap_request_id TEXT PRIMARY KEY NOT NULL,
    tenant_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    lead_id TEXT NOT NULL,
    session_id TEXT,
    status TEXT NOT NULL,
    result_json TEXT NOT NULL
);
"""


class SqliteConnection:
    """Thin wrapper enforcing foreign keys, transactions, and schema init.

    Not thread-safe by itself — each process should hold its own instance.
    The database file enforces cross-process isolation via SQLite locking.
    """

    def __init__(self, database_path: str) -> None:
        self._database_path = database_path
        try:
            self._conn = sqlite3.connect(
                database_path,
                isolation_level=None,  # autocommit; we manage transactions explicitly
                check_same_thread=False,
            )
        except sqlite3.Error as exc:
            raise DatabaseUnavailableError(f"cannot open {database_path}: {exc}") from exc
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._initialize_schema()

    @property
    def database_path(self) -> str:
        return self._database_path

    @property
    def raw(self) -> sqlite3.Connection:
        return self._conn

    def _initialize_schema(self) -> None:
        """Create tables and record schema version. Idempotent."""
        self._conn.executescript(SCHEMA_V1)
        cur = self._conn.execute("SELECT version FROM schema_version LIMIT 1")
        row = cur.fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO schema_version(version) VALUES (?)",
                (CURRENT_SCHEMA_VERSION,),
            )
        else:
            stored = int(row["version"])
            if stored > CURRENT_SCHEMA_VERSION:
                raise SchemaMismatchError(
                    f"stored schema version {stored} is newer than supported "
                    f"version {CURRENT_SCHEMA_VERSION}"
                )
            # future: run migrations stored → current

    def schema_version(self) -> int:
        row = self._conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        return int(row["version"]) if row else 0

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Explicit transaction context — commit on success, rollback on exception.

        Yields the raw connection so repositories executing within this
        context share one unit of work.
        """
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except Exception:
            try:
                self._conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        else:
            try:
                self._conn.execute("COMMIT")
            except sqlite3.Error as exc:
                try:
                    self._conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise TransactionFailedError(f"commit failed: {exc}") from exc

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass
