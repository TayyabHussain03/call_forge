"""Session recovery with typed diagnoses.

HONESTY RULE: in-flight ACTIVE sessions carry process-local structured
intelligence state (state machine position, business conversation state,
prospect intelligence, qualification snapshot, business memory) inside
ProductionTurnProcessor. That state is NOT currently serialized to the
ChatSession row — only identity, transcript (observational), turn_count,
revision, and the pre-call plan are persisted.

Therefore mid-conversation recovery (turn_count > 0, ACTIVE) returns
RECOVERY_UNSUPPORTED_FOR_STATE with a diagnosis. Do not rebuild authoritative
state from the transcript — transcript is audit data, not truth.

Terminal/DNC/completed sessions have no live state to recover and can be
re-opened safely for outcome generation and read-only inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.application.sales_runtime.manifest import (
    SnapshotCompatibilityManifest,
    default_manifest,
)
from app.application.sales_runtime.serializers.base import parse_envelope
from app.application.sales_runtime.snapshot import ConversationRuntimeSnapshot
from app.leads.campaign.contracts import LeadQueueEntry, LeadQueueStatus
from app.persistence.errors import InvalidPersistedPayloadError, NotFoundError
from app.persistence.sqlite.repositories import (
    SqliteCampaignQueueStore,
    SqliteChatSessionRepository,
    SqlitePolicyRegistry,
)
from app.persistence.sqlite.snapshot_store import SqliteRuntimeSnapshotStore
from app.precall.contracts import PreCallDiscoveryPolicy
from app.runtime.chat.contracts import ChatSession, SessionStatus


class RecoveryStatus(str, Enum):
    """Typed recovery outcome. Fail-closed over hallucination."""

    RECOVERED_FRESH = "recovered_fresh"
    RECOVERED_ACTIVE = "recovered_active"
    RECOVERED_TERMINAL = "recovered_terminal"
    ALREADY_TERMINAL = "already_terminal"
    SESSION_NOT_FOUND = "session_not_found"
    POLICY_VERSION_MISSING = "policy_version_missing"
    ORPHANED_QUEUE_STATE = "orphaned_queue_state"
    STALE_REVISION = "stale_revision"
    INVALID_PERSISTED_STATE = "invalid_persisted_state"
    RUNTIME_SNAPSHOT_MISSING = "runtime_snapshot_missing"
    RUNTIME_SNAPSHOT_INVALID = "runtime_snapshot_invalid"
    RUNTIME_SNAPSHOT_VERSION_UNSUPPORTED = "runtime_snapshot_version_unsupported"
    REQUIRED_PAYLOAD_MISSING = "required_payload_missing"
    UNSUPPORTED_PAYLOAD_VERSION = "unsupported_payload_version"
    STATE_RESTORE_FAILED = "state_restore_failed"
    RECOVERY_UNSUPPORTED_FOR_STATE = "recovery_unsupported_for_state"
    RECOVERY_BLOCKED = "recovery_blocked"


class OrphanDiagnosis(str, Enum):
    """Specific orphan/inconsistency classifications."""

    NONE = "none"
    QUEUE_IN_PROGRESS_SESSION_MISSING = "queue_in_progress_session_missing"
    ACTIVE_SESSION_QUEUE_NOT_IN_PROGRESS = "active_session_queue_not_in_progress"
    DNC_SESSION_QUEUE_NOT_DNC = "dnc_session_queue_not_dnc"


@dataclass(frozen=True)
class SessionRecoveryDiagnostics:
    """Bounded safe diagnostics — no secrets, no hidden reasoning."""

    session_id: str
    campaign_id: str | None = None
    lead_id: str | None = None
    stored_revision: int | None = None
    policy_id: str | None = None
    policy_version: str | None = None
    queue_status: str | None = None
    turn_count: int | None = None
    orphan_diagnosis: OrphanDiagnosis = OrphanDiagnosis.NONE
    failure_detail: str | None = None
    missing_required_payload: str | None = None
    unsupported_payload: str | None = None


@dataclass(frozen=True)
class SessionRecoveryResult:
    """Typed recovery result."""

    status: RecoveryStatus
    session: ChatSession | None = None
    queue_entry: LeadQueueEntry | None = None
    policy: PreCallDiscoveryPolicy | None = None
    runtime_snapshot: ConversationRuntimeSnapshot | None = None
    diagnostics: SessionRecoveryDiagnostics | None = None


class SessionRecoveryService:
    """Diagnoses whether a persisted session can be safely resumed.

    No automatic repair. No transcript replay. If live structured state
    is required but was not persisted, return RECOVERY_UNSUPPORTED_FOR_STATE
    with diagnosis for an admin workflow to decide.
    """

    def __init__(
        self,
        session_repository: SqliteChatSessionRepository,
        queue_store: SqliteCampaignQueueStore,
        policy_registry: SqlitePolicyRegistry,
        snapshot_store: SqliteRuntimeSnapshotStore | None = None,
        manifest: SnapshotCompatibilityManifest | None = None,
    ) -> None:
        self._sessions = session_repository
        self._queue = queue_store
        self._policies = policy_registry
        self._snapshots = snapshot_store
        self._manifest = manifest or default_manifest()

    def recover(self, session_id: str) -> SessionRecoveryResult:
        try:
            session = self._sessions.get(session_id)
        except NotFoundError:
            return SessionRecoveryResult(
                status=RecoveryStatus.SESSION_NOT_FOUND,
                diagnostics=SessionRecoveryDiagnostics(session_id=session_id),
            )

        queue_entry = self._queue.get(
            session.tenant_id, session.campaign_id, session.lead_id,
        )
        orphan = self._diagnose_orphan(session, queue_entry)
        queue_status = queue_entry.status.value if queue_entry else None

        plan = session.pre_call_plan
        policy_id = plan.policy_id if plan else None
        policy_version = plan.policy_version if plan else None

        base_diag = SessionRecoveryDiagnostics(
            session_id=session_id,
            campaign_id=session.campaign_id,
            lead_id=session.lead_id,
            stored_revision=session.revision,
            policy_id=policy_id,
            policy_version=policy_version,
            queue_status=queue_status,
            turn_count=session.turn_count,
            orphan_diagnosis=orphan,
        )

        # Terminal sessions — safe to recover for read/outcome generation only
        if session.status in (
            SessionStatus.COMPLETED, SessionStatus.TERMINATED,
            SessionStatus.ERROR,
        ):
            return SessionRecoveryResult(
                status=RecoveryStatus.RECOVERED_TERMINAL,
                session=session,
                queue_entry=queue_entry,
                diagnostics=base_diag,
            )

        # Resolve historical policy by exact version — never substitute latest
        policy = None
        if policy_version is not None:
            policy = self._policies.resolve(session.campaign_id, policy_version)
            if policy is None:
                return SessionRecoveryResult(
                    status=RecoveryStatus.POLICY_VERSION_MISSING,
                    session=session,
                    queue_entry=queue_entry,
                    diagnostics=base_diag,
                )

        # Orphan: queue inconsistent with session
        if orphan != OrphanDiagnosis.NONE:
            return SessionRecoveryResult(
                status=RecoveryStatus.ORPHANED_QUEUE_STATE,
                session=session,
                queue_entry=queue_entry,
                policy=policy,
                diagnostics=base_diag,
            )

        # Fresh session (turn_count == 0): no live state yet, safe to continue
        if session.turn_count == 0:
            return SessionRecoveryResult(
                status=RecoveryStatus.RECOVERED_FRESH,
                session=session,
                queue_entry=queue_entry,
                policy=policy,
                diagnostics=base_diag,
            )

        # Mid-conversation ACTIVE session — attempt snapshot-based restore.
        # Transcript replay is forbidden. Fail closed on missing/invalid snapshot.
        if self._snapshots is None:
            return SessionRecoveryResult(
                status=RecoveryStatus.RUNTIME_SNAPSHOT_MISSING,
                session=session,
                queue_entry=queue_entry,
                policy=policy,
                diagnostics=self._with_detail(
                    base_diag, "snapshot store not configured",
                ),
            )
        try:
            snapshot = self._snapshots.load(session_id)
        except InvalidPersistedPayloadError as exc:
            return SessionRecoveryResult(
                status=RecoveryStatus.RUNTIME_SNAPSHOT_INVALID,
                session=session,
                queue_entry=queue_entry,
                policy=policy,
                diagnostics=self._with_detail(base_diag, str(exc)),
            )
        if snapshot is None:
            return SessionRecoveryResult(
                status=RecoveryStatus.RUNTIME_SNAPSHOT_MISSING,
                session=session,
                queue_entry=queue_entry,
                policy=policy,
                diagnostics=self._with_detail(
                    base_diag,
                    "no runtime snapshot persisted for ACTIVE session",
                ),
            )

        # Validate manifest compatibility BEFORE processor reconstruction
        present_keys = frozenset(
            k for k, _ in snapshot.persisted_contract_payloads
        )
        missing = self._manifest.missing_required(present_keys)
        if missing:
            missing_key = sorted(missing)[0]
            return SessionRecoveryResult(
                status=RecoveryStatus.REQUIRED_PAYLOAD_MISSING,
                session=session,
                queue_entry=queue_entry,
                policy=policy,
                runtime_snapshot=snapshot,
                diagnostics=self._with_missing_payload(base_diag, missing_key),
            )

        # Each present payload known to the manifest must carry a supported
        # version. Required payloads MUST be supported; optional-but-present
        # payloads cannot carry an unsupported version either (they would
        # otherwise be silently dropped at restore time, which is unsafe).
        known_keys = set(self._manifest.required_for_active_recovery) | set(
            self._manifest.optional_derived
        )
        for key, envelope_json in snapshot.persisted_contract_payloads:
            if key not in known_keys:
                continue
            try:
                # peek version without full deserialization
                import json as _json
                env = _json.loads(envelope_json)
                version = env.get("version")
            except (ValueError, TypeError):
                return SessionRecoveryResult(
                    status=RecoveryStatus.RUNTIME_SNAPSHOT_INVALID,
                    session=session,
                    queue_entry=queue_entry,
                    policy=policy,
                    runtime_snapshot=snapshot,
                    diagnostics=self._with_detail(
                        base_diag, f"payload {key} envelope malformed",
                    ),
                )
            if not self._manifest.is_supported(key, version):
                return SessionRecoveryResult(
                    status=RecoveryStatus.UNSUPPORTED_PAYLOAD_VERSION,
                    session=session,
                    queue_entry=queue_entry,
                    policy=policy,
                    runtime_snapshot=snapshot,
                    diagnostics=self._with_unsupported_payload(base_diag, key),
                )

        return SessionRecoveryResult(
            status=RecoveryStatus.RECOVERED_ACTIVE,
            session=session,
            queue_entry=queue_entry,
            policy=policy,
            runtime_snapshot=snapshot,
            diagnostics=base_diag,
        )

    @staticmethod
    def _with_detail(
        diag: SessionRecoveryDiagnostics, detail: str,
    ) -> SessionRecoveryDiagnostics:
        return SessionRecoveryDiagnostics(
            session_id=diag.session_id,
            campaign_id=diag.campaign_id,
            lead_id=diag.lead_id,
            stored_revision=diag.stored_revision,
            policy_id=diag.policy_id,
            policy_version=diag.policy_version,
            queue_status=diag.queue_status,
            turn_count=diag.turn_count,
            orphan_diagnosis=diag.orphan_diagnosis,
            failure_detail=detail,
        )

    @staticmethod
    def _with_missing_payload(
        diag: SessionRecoveryDiagnostics, payload_key: str,
    ) -> SessionRecoveryDiagnostics:
        return SessionRecoveryDiagnostics(
            session_id=diag.session_id,
            campaign_id=diag.campaign_id,
            lead_id=diag.lead_id,
            stored_revision=diag.stored_revision,
            policy_id=diag.policy_id,
            policy_version=diag.policy_version,
            queue_status=diag.queue_status,
            turn_count=diag.turn_count,
            orphan_diagnosis=diag.orphan_diagnosis,
            failure_detail=f"required payload missing: {payload_key}",
            missing_required_payload=payload_key,
        )

    @staticmethod
    def _with_unsupported_payload(
        diag: SessionRecoveryDiagnostics, payload_key: str,
    ) -> SessionRecoveryDiagnostics:
        return SessionRecoveryDiagnostics(
            session_id=diag.session_id,
            campaign_id=diag.campaign_id,
            lead_id=diag.lead_id,
            stored_revision=diag.stored_revision,
            policy_id=diag.policy_id,
            policy_version=diag.policy_version,
            queue_status=diag.queue_status,
            turn_count=diag.turn_count,
            orphan_diagnosis=diag.orphan_diagnosis,
            failure_detail=f"unsupported payload version: {payload_key}",
            unsupported_payload=payload_key,
        )

    def _diagnose_orphan(
        self, session: ChatSession, queue_entry: LeadQueueEntry | None,
    ) -> OrphanDiagnosis:
        if queue_entry is None:
            return OrphanDiagnosis.NONE
        if session.status == SessionStatus.ACTIVE:
            if queue_entry.status != LeadQueueStatus.IN_PROGRESS and not queue_entry.is_dnc:
                return OrphanDiagnosis.ACTIVE_SESSION_QUEUE_NOT_IN_PROGRESS
        return OrphanDiagnosis.NONE

    def diagnose_orphan_queue_entry(
        self, tenant_id: str, campaign_id: str, lead_id: str,
    ) -> OrphanDiagnosis:
        """Diagnose queue=IN_PROGRESS with no corresponding session."""
        entry = self._queue.get(tenant_id, campaign_id, lead_id)
        if entry is None:
            return OrphanDiagnosis.NONE
        if entry.status != LeadQueueStatus.IN_PROGRESS:
            return OrphanDiagnosis.NONE
        # Look for any session matching this lead
        sessions = self._sessions.list_sessions(tenant_id)
        matches = [s for s in sessions if s.lead_id == lead_id and s.campaign_id == campaign_id]
        if not matches:
            return OrphanDiagnosis.QUEUE_IN_PROGRESS_SESSION_MISSING
        return OrphanDiagnosis.NONE
