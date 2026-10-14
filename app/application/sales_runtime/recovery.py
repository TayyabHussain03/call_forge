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

from app.leads.campaign.contracts import LeadQueueEntry, LeadQueueStatus
from app.persistence.errors import NotFoundError
from app.persistence.sqlite.repositories import (
    SqliteCampaignQueueStore,
    SqliteChatSessionRepository,
    SqlitePolicyRegistry,
)
from app.precall.contracts import PreCallDiscoveryPolicy
from app.runtime.chat.contracts import ChatSession, SessionStatus


class RecoveryStatus(str, Enum):
    """Typed recovery outcome. Fail-closed over hallucination."""

    RECOVERED_FRESH = "recovered_fresh"
    RECOVERED_TERMINAL = "recovered_terminal"
    ALREADY_TERMINAL = "already_terminal"
    SESSION_NOT_FOUND = "session_not_found"
    POLICY_VERSION_MISSING = "policy_version_missing"
    ORPHANED_QUEUE_STATE = "orphaned_queue_state"
    STALE_REVISION = "stale_revision"
    INVALID_PERSISTED_STATE = "invalid_persisted_state"
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


@dataclass(frozen=True)
class SessionRecoveryResult:
    """Typed recovery result."""

    status: RecoveryStatus
    session: ChatSession | None = None
    queue_entry: LeadQueueEntry | None = None
    policy: PreCallDiscoveryPolicy | None = None
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
    ) -> None:
        self._sessions = session_repository
        self._queue = queue_store
        self._policies = policy_registry

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

        # Mid-conversation ACTIVE session — honest fail-closed diagnosis.
        # Process-local structured intelligence (state machine current_state,
        # BCI, BDE, prospect intel, qualification, business memory, language,
        # repetition saturation, unfinished communicative intent) is not
        # persisted on the session row. Transcript replay is forbidden.
        return SessionRecoveryResult(
            status=RecoveryStatus.RECOVERY_UNSUPPORTED_FOR_STATE,
            session=session,
            queue_entry=queue_entry,
            policy=policy,
            diagnostics=SessionRecoveryDiagnostics(
                session_id=base_diag.session_id,
                campaign_id=base_diag.campaign_id,
                lead_id=base_diag.lead_id,
                stored_revision=base_diag.stored_revision,
                policy_id=base_diag.policy_id,
                policy_version=base_diag.policy_version,
                queue_status=base_diag.queue_status,
                turn_count=base_diag.turn_count,
                orphan_diagnosis=orphan,
                failure_detail=(
                    "mid-conversation structured intelligence "
                    "(state_machine, BCI, BDE, prospect_intel, qualification) "
                    "is process-local and not persisted on ChatSession; "
                    "transcript replay is forbidden"
                ),
            ),
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
