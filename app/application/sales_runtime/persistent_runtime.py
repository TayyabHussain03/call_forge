"""Persistent sales runtime composition root.

Wires SQLite adapters, bootstrap, PCIE, chat runtime, outcome engine,
atomic session+queue coordinator, idempotency stores, and the recovery
service. No business reasoning lives here — pure wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from app.leads.campaign.contracts import (
    CallEligibility,
    CampaignConfig,
    LeadQueueEntry,
    LeadQueueStatus,
)
from app.leads.contracts import LeadRecord
from app.leads.enrichment.reconciler import reconcile
from app.persistence.errors import (
    IdempotencyConflictError,
    StaleRevisionError,
    TransactionFailedError,
)
from app.persistence.sqlite.connection import SqliteConnection
from app.persistence.sqlite.coordinator import AtomicSessionQueueCoordinator
from app.persistence.sqlite.repositories import (
    SqliteBootstrapIdempotencyStore,
    SqliteCampaignQueueStore,
    SqliteChatSessionRepository,
    SqliteLeadStore,
    SqlitePolicyRegistry,
)
from app.persistence.sqlite.snapshot_store import SqliteRuntimeSnapshotStore
from app.persistence.sqlite.turn_idempotency import (
    CommittedTurnRecord,
    SqliteTurnIdempotencyStore,
)
from app.application.sales_runtime.snapshot import (
    SNAPSHOT_VERSION_CURRENT,
    ConversationRuntimeSnapshot,
)
from app.application.sales_runtime.snapshot_protocol import SnapshotCapable
from app.application.sales_runtime.manifest import SnapshotCompatibilityManifest
from app.core.constants import ConversationState
from app.precall.engine import PreCallIntelligenceEngine
from app.runtime.chat.bootstrap.contracts import (
    BootstrapStatus,
    SessionBootstrapInput,
    SessionBootstrapResult,
)
from app.runtime.chat.contracts import (
    ChatSession,
    SessionStatus,
    TranscriptEntry,
    TranscriptRole,
)
from app.runtime.chat.outcome.contracts import (
    ExecutionStatus,
    LeadOutcome,
    OutcomeIntelligenceInput,
    TerminationReason,
)
from app.runtime.chat.outcome.engine import OutcomeIntelligenceEngine
from app.runtime.contracts import CoordinatedUserTurn, TurnProcessor
from app.conversation.response_planning.contracts import (
    AddresseeStatus,
    InterruptionCategory,
    InterruptionContext,
)
from app.application.sales_runtime.processor_factory import TurnProcessorFactory
from app.application.sales_runtime.recovery import (
    RecoveryStatus,
    SessionRecoveryResult,
    SessionRecoveryService,
)


# ---------------------------------------------------------------------------
# Typed results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PersistentTurnResult:
    """Result of a persisted turn commit."""

    session_id: str
    turn_sequence: int
    committed_revision: int
    agent_response: str
    conversation_terminal: bool
    pipeline_outcome: str | None
    replayed_from_idempotency: bool = False


@dataclass(frozen=True)
class PersistentLifecycleResult:
    """End-of-lifecycle result when a session reaches terminal state."""

    session: ChatSession
    lead_outcome: LeadOutcome
    execution_status: ExecutionStatus


# ---------------------------------------------------------------------------
# Composition root
# ---------------------------------------------------------------------------


class PersistentSalesRuntime:
    """Composition root for the persistent sales runtime.

    Infrastructure wiring only. All domain decisions remain in existing
    intelligence layers. Supports:
        - fresh atomic bootstrap (session + queue IN_PROGRESS in one txn)
        - persisted per-turn session updates with turn-level idempotency
        - DNC durable persistence after live conversation
        - terminal outcome generation from trusted structured state
        - session recovery diagnosis (fresh/terminal/unsupported mid-conversation)
    """

    def __init__(
        self,
        connection: SqliteConnection,
        processor_factory: TurnProcessorFactory,
        pcie: PreCallIntelligenceEngine | None = None,
        outcome_engine: OutcomeIntelligenceEngine | None = None,
        snapshot_manifest: SnapshotCompatibilityManifest | None = None,
    ) -> None:
        self._c = connection
        self.leads = SqliteLeadStore(connection)
        self.queue = SqliteCampaignQueueStore(connection)
        self.sessions = SqliteChatSessionRepository(connection)
        self.policies = SqlitePolicyRegistry(connection)
        self.bootstrap_idempotency = SqliteBootstrapIdempotencyStore(connection)
        self.turn_idempotency = SqliteTurnIdempotencyStore(connection)
        self.snapshots = SqliteRuntimeSnapshotStore(connection)
        self.coordinator = AtomicSessionQueueCoordinator(
            connection, self.sessions, self.queue,
        )
        self.recovery = SessionRecoveryService(
            self.sessions, self.queue, self.policies, self.snapshots,
            manifest=snapshot_manifest,
        )
        self._processor_factory = processor_factory
        self._pcie = pcie or PreCallIntelligenceEngine()
        self._outcome = outcome_engine or OutcomeIntelligenceEngine()
        # Process-local processor registry — rebuilt from factory after restart.
        # NOT persisted. TurnProcessor instances are intentionally ephemeral.
        self._processors: dict[str, TurnProcessor] = {}

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def start_session(
        self,
        bootstrap_input: SessionBootstrapInput,
        campaign: CampaignConfig,
    ) -> SessionBootstrapResult:
        """Fresh session bootstrap with persistent atomic transaction.

        Resolves lead from SQLite, builds PreCallBusinessContext, runs PCIE,
        then commits session-create + queue READY→IN_PROGRESS in one txn.
        Records bootstrap_request_id for cross-restart idempotency.
        """
        # 1. Durable idempotency check
        cached = self.bootstrap_idempotency.lookup(
            bootstrap_input.bootstrap_request_id,
        )
        if cached is not None:
            if cached["status"] == BootstrapStatus.CREATED.value:
                # Reconstruct result from persisted session
                sid = cached["result"].get("session_id")
                if sid and self.sessions.exists(sid):
                    session = self.sessions.get(sid)
                    entry = self.queue.get(
                        session.tenant_id, session.campaign_id, session.lead_id,
                    )
                    return SessionBootstrapResult(
                        status=BootstrapStatus.CREATED,
                        session=session,
                        queue_entry=entry,
                        pre_call_plan=session.pre_call_plan,
                    )
            # Non-success cached — replay as-is
            return SessionBootstrapResult(
                status=BootstrapStatus(cached["status"]),
                rejection_reason=cached["result"].get("rejection_reason"),
            )

        # 2. Identity consistency
        if bootstrap_input.tenant_id != campaign.tenant_id:
            return self._reject_bootstrap(
                bootstrap_input, BootstrapStatus.CONTEXT_INVALID,
                "tenant_id mismatch",
            )
        if bootstrap_input.campaign_id != campaign.campaign_id:
            return self._reject_bootstrap(
                bootstrap_input, BootstrapStatus.CONTEXT_INVALID,
                "campaign_id mismatch",
            )

        # 3. Lead resolution
        lead = self.leads.get(
            bootstrap_input.tenant_id,
            bootstrap_input.campaign_id,
            bootstrap_input.lead_id,
        )
        if lead is None:
            return self._reject_bootstrap(
                bootstrap_input, BootstrapStatus.LEAD_NOT_FOUND,
                "lead not found",
            )

        # 4. Queue eligibility
        entry = self.queue.get(
            bootstrap_input.tenant_id,
            bootstrap_input.campaign_id,
            bootstrap_input.lead_id,
        )
        if entry is None:
            return self._reject_bootstrap(
                bootstrap_input, BootstrapStatus.LEAD_NOT_FOUND,
                "queue entry not found",
            )
        if entry.is_dnc:
            return self._reject_bootstrap(
                bootstrap_input, BootstrapStatus.NOT_ELIGIBLE,
                "lead is DNC",
            )
        if entry.status != LeadQueueStatus.READY:
            return self._reject_bootstrap(
                bootstrap_input, BootstrapStatus.NOT_ELIGIBLE,
                f"queue status is {entry.status.value}, expected READY",
            )

        # 5. Build pre-call context
        try:
            context = reconcile(lead)
        except Exception as exc:
            return self._reject_bootstrap(
                bootstrap_input, BootstrapStatus.CONTEXT_INVALID,
                f"context build failed: {exc}",
            )

        # 6. Policy resolution
        policy = self.policies.resolve(bootstrap_input.campaign_id)
        if policy is None:
            return self._reject_bootstrap(
                bootstrap_input, BootstrapStatus.POLICY_NOT_FOUND,
                "no policy registered",
            )

        # 7. PCIE
        try:
            plan = self._pcie.plan(context, campaign, policy)
        except Exception as exc:
            return self._reject_bootstrap(
                bootstrap_input, BootstrapStatus.PCIE_FAILED,
                f"pcie failed: {exc}",
            )

        # 8. Atomic session + queue transition in one SQLite transaction
        session = ChatSession(
            session_id=bootstrap_input.session_id,
            tenant_id=bootstrap_input.tenant_id,
            campaign_id=bootstrap_input.campaign_id,
            lead_id=bootstrap_input.lead_id,
            status=SessionStatus.ACTIVE,
            mode=bootstrap_input.runtime_mode,
            pre_call_plan=plan,
        )
        try:
            created, transitioned = self.coordinator.create_session_and_consume_lead(
                session,
                bootstrap_input.tenant_id,
                bootstrap_input.campaign_id,
                bootstrap_input.lead_id,
            )
        except (IdempotencyConflictError, StaleRevisionError) as exc:
            return self._reject_bootstrap(
                bootstrap_input, BootstrapStatus.CONCURRENCY_CONFLICT,
                str(exc),
            )
        except TransactionFailedError as exc:
            return self._reject_bootstrap(
                bootstrap_input, BootstrapStatus.SESSION_CREATION_FAILED,
                str(exc),
            )

        # 9. Rebuild processor and register locally
        processor = self._processor_factory.create_for_session(
            created, campaign, policy,
        )
        self._processors[created.session_id] = processor

        # 10. Record success in durable idempotency store
        try:
            self.bootstrap_idempotency.record(
                bootstrap_input.bootstrap_request_id,
                bootstrap_input.tenant_id,
                bootstrap_input.campaign_id,
                bootstrap_input.lead_id,
                created.session_id,
                BootstrapStatus.CREATED.value,
                {"session_id": created.session_id},
            )
        except IdempotencyConflictError:
            pass  # concurrent duplicate — safe

        return SessionBootstrapResult(
            status=BootstrapStatus.CREATED,
            session=created,
            queue_entry=transitioned,
            pre_call_plan=plan,
        )

    def _reject_bootstrap(
        self,
        bootstrap_input: SessionBootstrapInput,
        status: BootstrapStatus,
        detail: str,
    ) -> SessionBootstrapResult:
        try:
            self.bootstrap_idempotency.record(
                bootstrap_input.bootstrap_request_id,
                bootstrap_input.tenant_id,
                bootstrap_input.campaign_id,
                bootstrap_input.lead_id,
                None,
                status.value,
                {"rejection_reason": detail},
            )
        except IdempotencyConflictError:
            pass
        return SessionBootstrapResult(
            status=status, rejection_reason=detail,
        )

    # ------------------------------------------------------------------
    # Turn processing — read, process outside txn, short compare-and-update
    # ------------------------------------------------------------------

    def process_turn(
        self,
        session_id: str,
        message: str,
        turn_sequence: int,
        turn_request_id: str,
    ) -> PersistentTurnResult:
        """Persist an accepted turn under optimistic concurrency.

        Pattern: read session, run processor OUTSIDE write transaction (so
        SQLite write locks are never held during long operations), then
        short compare-and-update commits the new revision + idempotency row
        together. If another worker committed meanwhile, StaleRevisionError
        is raised and the caller must retry.
        """
        # 1. Durable turn idempotency check — survives restart
        cached = self.turn_idempotency.lookup(session_id, turn_request_id)
        if cached is not None:
            return PersistentTurnResult(
                session_id=session_id,
                turn_sequence=cached.turn_sequence,
                committed_revision=cached.committed_revision,
                agent_response=cached.agent_response,
                conversation_terminal=cached.conversation_terminal,
                pipeline_outcome=cached.pipeline_outcome,
                replayed_from_idempotency=True,
            )

        # 2. Read session (outside any write txn)
        session = self.sessions.get(session_id)
        if session.status != SessionStatus.ACTIVE:
            raise StaleRevisionError(
                f"session {session_id} is {session.status.value}"
            )
        if turn_sequence <= session.turn_count:
            raise StaleRevisionError(
                f"turn {turn_sequence} already processed "
                f"(session at {session.turn_count})"
            )

        processor = self._processors.get(session_id)
        if processor is None:
            raise StaleRevisionError(
                f"no processor registered for session {session_id} — "
                f"call restore_processor() after restart"
            )

        # 3. Process turn OUTSIDE write transaction
        # Slice 51B — capture pre-turn processor coverage so a failed
        # or stale commit can restore it. Prevents speculative coverage
        # from leaking into a subsequent turn's snapshot export.
        pre_turn_coverage = getattr(processor, "_discussion_coverage", None)
        # Slice 51I — Model B (committed-turn semantics): roll back
        # reasoning-call budget alongside coverage so the same logical
        # history produces the same budget regardless of process
        # lifetime. In-process failure + continue and failure + restart
        # must yield identical remaining allowance.
        pre_turn_reasoning_calls = getattr(processor, "_reasoning_calls", None)
        coordinated = CoordinatedUserTurn(
            turn_id=f"chat_{session_id}_t{turn_sequence}",
            sequence_number=turn_sequence,
            utterance=message,
            interruption=InterruptionContext(),
            addressee_status=AddresseeStatus.ADDRESSED_TO_AGENT,
            conversation_category=InterruptionCategory.OTHER,
        )
        output = processor.process_turn(coordinated)

        new_status = (
            SessionStatus.COMPLETED if output.conversation_terminal
            else SessionStatus.ACTIVE
        )
        prospect_entry = TranscriptEntry(
            turn_sequence=turn_sequence, role=TranscriptRole.PROSPECT,
            text=message,
        )
        agent_entry = TranscriptEntry(
            turn_sequence=turn_sequence, role=TranscriptRole.AGENT,
            text=output.rendered_response.text,
        )
        updated = replace(
            session,
            turn_count=turn_sequence,
            status=new_status,
            transcript=(*session.transcript, prospect_entry, agent_entry),
        )

        # 4. Short write transaction: session update + snapshot + idempotency
        # all commit together or all roll back. The processor exports its
        # authoritative state as a ConversationRuntimeSnapshot — the ONLY
        # bridge between live runtime and persistent storage.
        snapshot = self._export_snapshot_if_capable(
            processor, session_id, updated_revision_target=session.revision + 1,
        )
        try:
            with self._c.transaction():
                committed = self.sessions.update(updated)
                if snapshot is not None:
                    # Rebind snapshot to the committed revision for consistency
                    snapshot = replace(
                        snapshot, source_session_revision=committed.revision,
                    )
                    self.snapshots.save(snapshot)
                pipeline_outcome = (
                    output.pipeline_outcome.value if output.pipeline_outcome else None
                )
                self.turn_idempotency.record(CommittedTurnRecord(
                    session_id=session_id,
                    turn_request_id=turn_request_id,
                    turn_sequence=turn_sequence,
                    committed_revision=committed.revision,
                    agent_response=output.rendered_response.text,
                    conversation_terminal=output.conversation_terminal,
                    pipeline_outcome=pipeline_outcome,
                ))
        except Exception:
            # Slice 51B/51I — commit failed (stale revision, idempotency
            # conflict, DB error). Restore the processor's in-memory
            # discussion coverage AND reasoning-call budget so speculative
            # state cannot leak into the next turn's snapshot export AND
            # the budget semantics (Model B — committed-turn) are
            # consistent whether we continue in-process or crash + restart.
            if pre_turn_coverage is not None and hasattr(
                processor, "_discussion_coverage",
            ):
                processor._discussion_coverage = pre_turn_coverage
            if pre_turn_reasoning_calls is not None and hasattr(
                processor, "_reasoning_calls",
            ):
                processor._reasoning_calls = pre_turn_reasoning_calls
            raise

        return PersistentTurnResult(
            session_id=session_id,
            turn_sequence=turn_sequence,
            committed_revision=committed.revision,
            agent_response=output.rendered_response.text,
            conversation_terminal=output.conversation_terminal,
            pipeline_outcome=pipeline_outcome,
        )

    def _export_snapshot_if_capable(
        self,
        processor: TurnProcessor,
        session_id: str,
        updated_revision_target: int,
    ) -> ConversationRuntimeSnapshot | None:
        """Export a ConversationRuntimeSnapshot from the processor if it
        implements SnapshotCapable. Non-snapshottable processors silently
        skip — recovery will return RUNTIME_SNAPSHOT_MISSING for them.
        """
        if not isinstance(processor, SnapshotCapable):
            return None
        return processor.export_snapshot(updated_revision_target)

    # ------------------------------------------------------------------
    # DNC durable write (triggered by priority logic / admin)
    # ------------------------------------------------------------------

    def mark_lead_dnc(
        self, tenant_id: str, campaign_id: str, lead_id: str,
    ) -> LeadQueueEntry:
        """Durable DNC write — survives restart, cannot be regressed."""
        return self.queue.mark_dnc(tenant_id, campaign_id, lead_id)

    # ------------------------------------------------------------------
    # Session recovery
    # ------------------------------------------------------------------

    def recover_session(self, session_id: str) -> SessionRecoveryResult:
        """Diagnose whether a persisted session can be safely resumed."""
        return self.recovery.recover(session_id)

    def restore_processor_if_possible(
        self,
        session_id: str,
        campaign: CampaignConfig,
    ) -> SessionRecoveryResult:
        """Attempt processor rebuild.

        - RECOVERED_FRESH: processor rebuilt from policy/campaign only
        - RECOVERED_ACTIVE: processor rebuilt from policy + runtime snapshot
          via the factory's create_from_snapshot method
        - RUNTIME_SNAPSHOT_MISSING / INVALID / UNSUPPORTED_FOR_STATE:
          no processor registered — fail-closed
        """
        result = self.recovery.recover(session_id)
        if result.session is None or result.policy is None:
            return result
        if result.status == RecoveryStatus.RECOVERED_FRESH:
            processor = self._processor_factory.create_for_session(
                result.session, campaign, result.policy,
            )
            self._processors[session_id] = processor
        elif result.status == RecoveryStatus.RECOVERED_ACTIVE:
            if result.runtime_snapshot is None:
                return result
            processor = self._processor_factory.create_from_snapshot(
                result.session, campaign, result.policy, result.runtime_snapshot,
            )
            self._processors[session_id] = processor
        return result

    # ------------------------------------------------------------------
    # Outcome generation from trusted terminal state
    # ------------------------------------------------------------------

    def finish_session(
        self,
        session_id: str,
        *,
        is_dnc: bool = False,
        is_not_interested: bool = False,
        session_error: bool = False,
        termination_reason: TerminationReason = TerminationReason.NATURAL_COMPLETION,
    ) -> PersistentLifecycleResult:
        """Generate LeadOutcome from persisted terminal state."""
        session = self.sessions.get(session_id)
        inp = OutcomeIntelligenceInput(
            lead_id=session.lead_id,
            campaign_id=session.campaign_id,
            session_id=session_id,
            tenant_id=session.tenant_id,
            is_dnc=is_dnc,
            is_not_interested=is_not_interested,
            session_error=session_error,
            termination_reason=termination_reason,
            conversation_terminal=True,
        )
        outcome = self._outcome.resolve(inp)
        execution = ExecutionStatus.FAILED if session_error else ExecutionStatus.COMPLETED

        # Durable DNC write for the queue if the session concluded as DNC
        if is_dnc:
            try:
                self.mark_lead_dnc(
                    session.tenant_id, session.campaign_id, session.lead_id,
                )
            except Exception:
                pass  # never let infra failure corrupt customer outcome

        return PersistentLifecycleResult(
            session=session,
            lead_outcome=outcome,
            execution_status=execution,
        )

    # ------------------------------------------------------------------
    # Read access
    # ------------------------------------------------------------------

    def get_session(self, session_id: str) -> ChatSession:
        return self.sessions.get(session_id)
