"""Slice 44 — End-to-end persistent lifecycle integration tests.

Proves the full flow against real SQLite file storage:
    SQLite Lead → Queue → Policy → Bootstrap → PCIE → atomic session
    → Chat Runtime → persisted turns → restart → recover → outcome
    → DNC + idempotency + policy version survive restart.

HONESTY NOTE: mid-conversation ACTIVE recovery returns
RECOVERY_UNSUPPORTED_FOR_STATE because process-local structured
intelligence (state machine, BCI, BDE, prospect intelligence, etc.) is
not currently persisted on ChatSession rows. This slice does NOT pretend
otherwise — see test_mid_conversation_recovery_is_honestly_blocked.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.application.sales_runtime.persistent_runtime import (
    PersistentSalesRuntime,
    PersistentTurnResult,
)
from app.application.sales_runtime.processor_factory import TurnProcessorFactory
from app.application.sales_runtime.recovery import (
    OrphanDiagnosis,
    RecoveryStatus,
)
from app.conversation.response_planning.contracts import (
    AcknowledgementKind,
    AddresseeStatus,
    AuthoritativeResultKind,
    ConversationMove,
    InterruptionCategory,
    InterruptionHandling,
    QuestionStrategy,
    ResponseLength,
    ResponsePlan,
)
from app.conversation.response_rendering.contracts import RenderedResponse
from app.core.constants import Tone
from app.leads.campaign.contracts import (
    CampaignConfig,
    CampaignStatus,
    LeadQueueEntry,
    LeadQueueStatus,
)
from app.leads.contracts import (
    LeadRecord,
    NormalizedPhone,
    PhoneNormalizationStatus,
    WebsiteStatus,
)
from app.persistence.errors import StaleRevisionError
from app.persistence.sqlite.connection import SqliteConnection
from app.precall.contracts import PreCallDiscoveryPolicy, ServiceDiscoveryRule
from app.runtime.chat.bootstrap.contracts import BootstrapStatus, SessionBootstrapInput
from app.runtime.chat.contracts import ChatSession, RuntimeMode, SessionStatus
from app.runtime.chat.outcome.contracts import (
    ExecutionStatus,
    OutcomeStatus,
    TerminationReason,
)
from app.runtime.contracts import (
    CoordinatedTurnOutput,
    CoordinatedUserTurn,
    TurnProcessor,
)


# ---------------------------------------------------------------------------
# Scripted TurnProcessor + factory (deterministic, dumb, transparent)
# ---------------------------------------------------------------------------


def _plan() -> ResponsePlan:
    return ResponsePlan(
        communicative_goal=ConversationMove.COMMUNICATE_RESULT,
        response_length=ResponseLength.SHORT,
        tone=Tone.NEUTRAL,
        acknowledgement=AcknowledgementKind.NONE,
        question_strategy=QuestionStrategy.NONE,
        clarification_required=False,
        interruption_handling=InterruptionHandling.NONE,
        resume_previous_point=False,
        pending_intent=None,
        addressee_status=AddresseeStatus.ADDRESSED_TO_AGENT,
    )


def _rendered(text: str) -> RenderedResponse:
    return RenderedResponse(
        text=text,
        communicative_goal=ConversationMove.COMMUNICATE_RESULT,
        length_class=ResponseLength.SHORT,
        clarification_required=False,
    )


class ScriptedProcessor:
    """Deterministic processor: returns pre-scripted outputs by turn seq.

    Dumb by design — proves lifecycle wiring without exercising the full
    production brain, which is already covered by other integration tests.
    """

    def __init__(
        self,
        outputs: tuple[tuple[str, bool, str | None], ...],
    ) -> None:
        self._outputs = outputs

    def process_turn(self, turn: CoordinatedUserTurn) -> CoordinatedTurnOutput:
        idx = turn.sequence_number - 1
        if idx >= len(self._outputs):
            text, terminal, outcome = ("", False, None)
        else:
            text, terminal, outcome = self._outputs[idx]
        outcome_kind = (
            AuthoritativeResultKind(outcome) if outcome else None
        )
        return CoordinatedTurnOutput(
            turn_id=turn.turn_id,
            response_plan=_plan(),
            rendered_response=_rendered(text or "ok"),
            pipeline_outcome=outcome_kind,
            conversation_terminal=terminal,
        )


class ScriptedFactory:
    """Factory that returns a scripted processor keyed by session_id."""

    def __init__(self) -> None:
        self._scripts: dict[str, tuple[tuple[str, bool, str | None], ...]] = {}
        self._default: tuple[tuple[str, bool, str | None], ...] = (
            ("ack-1", False, None),
            ("ack-2", False, None),
            ("done", True, None),
        )

    def script(
        self, session_id: str,
        outputs: tuple[tuple[str, bool, str | None], ...],
    ) -> None:
        self._scripts[session_id] = outputs

    def create_for_session(
        self, session: ChatSession, campaign, policy,
    ) -> TurnProcessor:
        return ScriptedProcessor(
            self._scripts.get(session.session_id, self._default)
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _phone(raw: str = "+15551234567") -> NormalizedPhone:
    return NormalizedPhone(
        raw=raw, normalized=raw, status=PhoneNormalizationStatus.VALID_E164,
    )


def _lead(lead_id: str = "lead-1") -> LeadRecord:
    return LeadRecord(
        lead_id=lead_id,
        tenant_id="t1", campaign_id="c1",
        business_name="Alpha Inc", phone=_phone(),
        category=None, city=None, state=None, country=None,
        address=None, email=None, contact_name=None,
        website=None, website_status=WebsiteStatus.UNKNOWN,
    )


def _queue_entry(
    lead_id: str = "lead-1",
    status: LeadQueueStatus = LeadQueueStatus.READY,
    is_dnc: bool = False,
) -> LeadQueueEntry:
    return LeadQueueEntry(
        lead_id=lead_id, tenant_id="t1", campaign_id="c1",
        queue_position=0, status=status, is_dnc=is_dnc,
    )


def _campaign() -> CampaignConfig:
    return CampaignConfig(
        campaign_id="c1", tenant_id="t1", name="Test",
        status=CampaignStatus.ACTIVE,
        authorized_service_ids=("svc-web",),
    )


def _policy(version: str = "1.0") -> PreCallDiscoveryPolicy:
    return PreCallDiscoveryPolicy(
        policy_id="pol-1", version=version,
        service_rules=(
            ServiceDiscoveryRule(
                service_id="svc-web",
                required_fact_keys=("website.presence",),
            ),
        ),
    )


def _boot_input(
    session_id: str = "sess-1", request_id: str = "req-1",
) -> SessionBootstrapInput:
    return SessionBootstrapInput(
        tenant_id="t1", campaign_id="c1", lead_id="lead-1",
        session_id=session_id, bootstrap_request_id=request_id,
        trusted_time_iso="2026-10-14T10:00:00Z",
    )


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "runtime.db")


def _make_runtime(db_path: str, factory: ScriptedFactory | None = None) -> PersistentSalesRuntime:
    conn = SqliteConnection(db_path)
    f = factory or ScriptedFactory()
    rt = PersistentSalesRuntime(conn, f)
    # Seed common state only when not already present — never regress DNC
    if rt.leads.get("t1", "c1", "lead-1") is None:
        rt.leads.save(_lead())
    if rt.queue.get("t1", "c1", "lead-1") is None:
        rt.queue.save(_queue_entry())
    if rt.policies.resolve("c1") is None:
        rt.policies.register_default("c1", _policy())
    return rt


# ---------------------------------------------------------------------------
# A. Normal lifecycle
# ---------------------------------------------------------------------------


class TestNormalLifecycle:
    def test_bootstrap_turns_terminal_outcome(self, db_path) -> None:
        rt = _make_runtime(db_path)
        boot = rt.start_session(_boot_input(), _campaign())
        assert boot.status == BootstrapStatus.CREATED
        assert boot.queue_entry.status == LeadQueueStatus.IN_PROGRESS

        r1 = rt.process_turn("sess-1", "hello", 1, "tr-1")
        assert r1.committed_revision == 1
        r2 = rt.process_turn("sess-1", "tell me more", 2, "tr-2")
        assert r2.committed_revision == 2
        r3 = rt.process_turn("sess-1", "sure, done", 3, "tr-3")
        assert r3.conversation_terminal is True

        done = rt.finish_session("sess-1")
        assert done.lead_outcome is not None
        assert done.execution_status == ExecutionStatus.COMPLETED


# ---------------------------------------------------------------------------
# B. Restart mid-conversation — honest diagnosis
# ---------------------------------------------------------------------------


class TestMidConversationRecovery:
    def test_mid_conversation_recovery_without_snapshot_is_honestly_blocked(
        self, db_path,
    ) -> None:
        """Slice 44's ScriptedFactory produces non-SnapshotCapable processors,
        so Slice 45's new RUNTIME_SNAPSHOT_MISSING status fires here —
        still honestly fail-closed, now with a more precise diagnosis."""
        factory = ScriptedFactory()
        rt = _make_runtime(db_path, factory)
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-1", "hello", 1, "tr-1")
        rt.process_turn("sess-1", "more", 2, "tr-2")

        factory2 = ScriptedFactory()
        rt2 = _make_runtime(db_path, factory2)
        result = rt2.recover_session("sess-1")
        assert result.status == RecoveryStatus.RUNTIME_SNAPSHOT_MISSING
        assert result.diagnostics.turn_count == 2

    def test_fresh_session_recovery_allowed(self, db_path) -> None:
        rt = _make_runtime(db_path)
        rt.start_session(_boot_input(), _campaign())
        # No turns processed

        factory2 = ScriptedFactory()
        rt2 = _make_runtime(db_path, factory2)
        result = rt2.recover_session("sess-1")
        assert result.status == RecoveryStatus.RECOVERED_FRESH


# ---------------------------------------------------------------------------
# C. PCIE correction survives restart (persisted plan intact)
# ---------------------------------------------------------------------------


class TestPlanSurvivesRestart:
    def test_plan_persisted_across_restart(self, db_path) -> None:
        rt = _make_runtime(db_path)
        boot = rt.start_session(_boot_input(), _campaign())
        original_plan = boot.session.pre_call_plan

        rt2 = _make_runtime(db_path)
        reloaded = rt2.get_session("sess-1")
        assert reloaded.pre_call_plan is not None
        assert reloaded.pre_call_plan.policy_id == original_plan.policy_id
        assert reloaded.pre_call_plan.policy_version == original_plan.policy_version


# ---------------------------------------------------------------------------
# D. DNC survives restart
# ---------------------------------------------------------------------------


class TestDNCSurvivesRestart:
    def test_dnc_during_conversation_persists(self, db_path) -> None:
        rt = _make_runtime(db_path)
        rt.start_session(_boot_input(), _campaign())
        rt.mark_lead_dnc("t1", "c1", "lead-1")

        # Restart
        rt2 = _make_runtime(db_path)
        entry = rt2.queue.get("t1", "c1", "lead-1")
        assert entry.is_dnc is True
        assert entry.status == LeadQueueStatus.DNC

    def test_dnc_blocks_new_bootstrap_after_restart(self, db_path) -> None:
        rt = _make_runtime(db_path)
        rt.mark_lead_dnc("t1", "c1", "lead-1")

        rt2 = _make_runtime(db_path)
        result = rt2.start_session(
            _boot_input(session_id="sess-X", request_id="req-X"),
            _campaign(),
        )
        assert result.status == BootstrapStatus.NOT_ELIGIBLE

    def test_finish_as_dnc_persists_queue_dnc(self, db_path) -> None:
        rt = _make_runtime(db_path)
        rt.start_session(_boot_input(), _campaign())
        rt.finish_session(
            "sess-1", is_dnc=True,
            termination_reason=TerminationReason.DNC_REQUEST,
        )
        entry = rt.queue.get("t1", "c1", "lead-1")
        assert entry.is_dnc is True


# ---------------------------------------------------------------------------
# E. Turn idempotency survives restart
# ---------------------------------------------------------------------------


class TestTurnIdempotency:
    def test_duplicate_turn_request_id_replays(self, db_path) -> None:
        rt = _make_runtime(db_path)
        rt.start_session(_boot_input(), _campaign())
        r1 = rt.process_turn("sess-1", "hi", 1, "tr-abc")

        # Replay same turn_request_id
        r2 = rt.process_turn("sess-1", "hi again", 1, "tr-abc")
        assert r2.replayed_from_idempotency is True
        assert r2.committed_revision == r1.committed_revision
        assert r2.agent_response == r1.agent_response

        # Only one turn committed
        session = rt.get_session("sess-1")
        assert session.turn_count == 1

    def test_turn_idempotency_survives_restart(self, db_path) -> None:
        rt = _make_runtime(db_path)
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-1", "hi", 1, "tr-persist")

        rt2 = _make_runtime(db_path)
        # Attempt the same turn — must replay, not re-execute
        result = rt2.turn_idempotency.lookup("sess-1", "tr-persist")
        assert result is not None
        assert result.turn_sequence == 1


# ---------------------------------------------------------------------------
# F. Concurrent revisions
# ---------------------------------------------------------------------------


class TestConcurrentRevisions:
    def test_two_connections_stale_revision_rejected(self, db_path) -> None:
        rt_a = _make_runtime(db_path)
        rt_a.start_session(_boot_input(), _campaign())
        rt_a.process_turn("sess-1", "turn 1", 1, "tr-1")

        # Second runtime over same DB — attempt a turn based on stale read.
        # Load current session via rt_b, let rt_a commit turn 2, then rt_b
        # tries to commit turn 2 using its stale snapshot.
        factory_b = ScriptedFactory()
        rt_b = PersistentSalesRuntime(SqliteConnection(db_path), factory_b)
        # rt_b needs a processor for sess-1 — register one
        rt_b.restore_processor_if_possible("sess-1", _campaign())
        # rt_b hasn't processed any turn, but the session is turn_count=1
        # rt_a commits turn 2
        rt_a.process_turn("sess-1", "turn 2", 2, "tr-2")
        # rt_b tries to commit turn 2 — but DB already at turn 2
        with pytest.raises(StaleRevisionError):
            rt_b.process_turn("sess-1", "turn 2 racing", 2, "tr-2-b")


# ---------------------------------------------------------------------------
# G. Atomic bootstrap failure — queue unchanged
# ---------------------------------------------------------------------------


class TestAtomicBootstrapFailure:
    def test_dnc_lead_blocks_bootstrap_queue_unchanged(self, db_path) -> None:
        rt = _make_runtime(db_path)
        rt.mark_lead_dnc("t1", "c1", "lead-1")
        before = rt.queue.get("t1", "c1", "lead-1")

        result = rt.start_session(_boot_input(), _campaign())
        assert result.status == BootstrapStatus.NOT_ELIGIBLE

        after = rt.queue.get("t1", "c1", "lead-1")
        assert after.is_dnc == before.is_dnc
        assert after.status == before.status
        assert not rt.sessions.exists("sess-1")

    def test_duplicate_session_id_rolls_back_queue(self, db_path) -> None:
        rt = _make_runtime(db_path)
        rt.start_session(_boot_input(session_id="sess-dup"), _campaign())
        # Reset the queue back to READY for the second attempt — we want
        # to show that the SECOND bootstrap atomically rolls back
        rt.queue.save(_queue_entry())  # reset to READY

        # Try to bootstrap again with same session_id, different request_id
        result = rt.start_session(
            _boot_input(session_id="sess-dup", request_id="req-other"),
            _campaign(),
        )
        assert result.status == BootstrapStatus.CONCURRENCY_CONFLICT
        # Queue must still be READY — rollback worked
        assert rt.queue.get("t1", "c1", "lead-1").status == LeadQueueStatus.READY


# ---------------------------------------------------------------------------
# H. Historical policy version pinning
# ---------------------------------------------------------------------------


class TestPolicyVersionPinning:
    def test_recovery_uses_pinned_version_not_latest(self, db_path) -> None:
        rt = _make_runtime(db_path)
        # Session bootstrapped with v1.0 (the default)
        rt.start_session(_boot_input(), _campaign())

        # Later: v2.0 registered as new default
        rt.policies.register_default("c1", _policy(version="2.0"))

        # Restart
        rt2 = _make_runtime(db_path)
        result = rt2.recover_session("sess-1")
        # The session's recovered policy MUST be v1.0, not v2.0
        assert result.session.pre_call_plan.policy_version == "1.0"
        if result.policy is not None:
            assert result.policy.version == "1.0"

    def test_missing_policy_version_blocks_recovery(self, db_path) -> None:
        rt = _make_runtime(db_path)
        rt.start_session(_boot_input(), _campaign())

        # Restart with a fresh DB file would lose the policy;
        # here we simulate by connecting to a new DB in which the policy
        # version isn't registered.
        rt2 = PersistentSalesRuntime(
            SqliteConnection(db_path + ".new"), ScriptedFactory(),
        )
        # Copy session into new DB and omit the policy registration
        rt2.sessions.create(rt.get_session("sess-1"))
        rt2.queue.save(rt.queue.get("t1", "c1", "lead-1"))
        # No policy registered in rt2
        result = rt2.recover_session("sess-1")
        assert result.status == RecoveryStatus.POLICY_VERSION_MISSING


# ---------------------------------------------------------------------------
# I. Runtime failure preserves sales facts
# ---------------------------------------------------------------------------


class TestRuntimeFailureSeparate:
    def test_runtime_failure_does_not_overwrite_customer_outcome(self, db_path) -> None:
        rt = _make_runtime(db_path)
        rt.start_session(_boot_input(), _campaign())
        # Terminate with session_error=True while reporting interest observed
        done = rt.finish_session(
            "sess-1", session_error=True,
            termination_reason=TerminationReason.RUNTIME_ERROR,
        )
        assert done.execution_status == ExecutionStatus.FAILED
        # Customer outcome determined independently — not forced to DNC
        assert done.lead_outcome.outcome_status != OutcomeStatus.DNC


# ---------------------------------------------------------------------------
# J. Orphan diagnosis
# ---------------------------------------------------------------------------


class TestOrphanDiagnosis:
    def test_queue_in_progress_but_no_session(self, db_path) -> None:
        rt = _make_runtime(db_path)
        # Directly set queue IN_PROGRESS without a session (orphan state)
        rt.queue.transition(
            "t1", "c1", "lead-1",
            LeadQueueStatus.READY, LeadQueueStatus.IN_PROGRESS,
        )
        diag = rt.recovery.diagnose_orphan_queue_entry("t1", "c1", "lead-1")
        assert diag == OrphanDiagnosis.QUEUE_IN_PROGRESS_SESSION_MISSING

    def test_active_session_but_queue_not_in_progress(self, db_path) -> None:
        rt = _make_runtime(db_path)
        rt.start_session(_boot_input(), _campaign())
        # Someone externally reset the queue to READY while session active
        rt._c.raw.execute(
            "UPDATE queue_entries SET status = ?, is_dnc = 0 "
            "WHERE lead_id = ? AND tenant_id = ? AND campaign_id = ?",
            (LeadQueueStatus.READY.value, "lead-1", "t1", "c1"),
        )
        result = rt.recover_session("sess-1")
        assert result.status == RecoveryStatus.ORPHANED_QUEUE_STATE


# ---------------------------------------------------------------------------
# Bootstrap idempotency across restart
# ---------------------------------------------------------------------------


class TestBootstrapIdempotencyAcrossRestart:
    def test_same_request_id_after_restart_returns_cached(self, db_path) -> None:
        rt = _make_runtime(db_path)
        r1 = rt.start_session(_boot_input(), _campaign())
        assert r1.status == BootstrapStatus.CREATED

        rt2 = _make_runtime(db_path)
        r2 = rt2.start_session(_boot_input(), _campaign())
        # Same request_id — cached success returned, no new session created
        assert r2.status == BootstrapStatus.CREATED
        assert r2.session.session_id == "sess-1"
        # Still only ONE session for this lead
        sessions_for_lead = [
            s for s in rt2.sessions.list_sessions("t1") if s.lead_id == "lead-1"
        ]
        assert len(sessions_for_lead) == 1


# ---------------------------------------------------------------------------
# Terminal session recovery
# ---------------------------------------------------------------------------


class TestTerminalRecovery:
    def test_terminal_session_recovers_cleanly(self, db_path) -> None:
        rt = _make_runtime(db_path)
        rt.start_session(_boot_input(), _campaign())
        # Reach terminal state
        rt.process_turn("sess-1", "hi", 1, "tr-1")
        rt.process_turn("sess-1", "more", 2, "tr-2")
        rt.process_turn("sess-1", "done", 3, "tr-3")

        rt2 = _make_runtime(db_path)
        result = rt2.recover_session("sess-1")
        assert result.status == RecoveryStatus.RECOVERED_TERMINAL


# ---------------------------------------------------------------------------
# Not-interested stays distinct from DNC after restart
# ---------------------------------------------------------------------------


class TestNotInterestedDistinct:
    def test_not_interested_persists_without_becoming_dnc(self, db_path) -> None:
        rt = _make_runtime(db_path)
        rt.start_session(_boot_input(), _campaign())
        rt.finish_session(
            "sess-1", is_not_interested=True,
            termination_reason=TerminationReason.NOT_INTERESTED_EXIT,
        )
        entry = rt.queue.get("t1", "c1", "lead-1")
        assert entry.is_dnc is False
        # Future campaign policy decides re-contact; this slice must
        # preserve the distinction


# ---------------------------------------------------------------------------
# No domain intelligence in persistent runtime
# ---------------------------------------------------------------------------


class TestNoDomainIntelligenceInRuntime:
    def test_runtime_has_no_reasoning_imports(self) -> None:
        import inspect

        from app.application.sales_runtime import (
            persistent_runtime, processor_factory, recovery,
        )
        for module in (persistent_runtime, processor_factory, recovery):
            source = inspect.getsource(module)
            for forbidden in (
                "BrainOrchestrator", "StateMachine", "ActionValidator",
                "ReasoningProvider", "HybridRetrievalEngine",
                "ServiceRelevanceResolver",
            ):
                assert forbidden not in source
