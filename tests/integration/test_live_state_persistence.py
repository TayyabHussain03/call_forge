"""Slice 45 — Live Structured Intelligence Persistence tests.

Proves the end-to-end restart-safe conversation:
    turn committed → session revision + runtime snapshot + idempotency
    row persist atomically → process dies → recover from DB → next turn
    continues with restored authoritative state.

HONESTY NOTE: the real ProductionTurnProcessor's export_snapshot() today
captures the FSM state + stage markers but not the full BCI/BDE/PQI JSON
roundtrip (those contracts need per-field serializers that would be a
large, higher-risk follow-up). This test suite exercises the full
infrastructure via a scripted processor that DOES round-trip a complete
small payload, proving the atomic commit, idempotency, outcome
equivalence, DNC safety, snapshot versioning, and recovery statuses.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from app.application.sales_runtime.persistent_runtime import PersistentSalesRuntime
from app.application.sales_runtime.processor_factory import TurnProcessorFactory
from app.application.sales_runtime.recovery import RecoveryStatus
from app.application.sales_runtime.snapshot import (
    SNAPSHOT_VERSION_CURRENT,
    ConversationRuntimeSnapshot,
)
from app.conversation.response_planning.contracts import (
    AcknowledgementKind,
    AddresseeStatus,
    AuthoritativeResultKind,
    ConversationMove,
    InterruptionHandling,
    QuestionStrategy,
    ResponseLength,
    ResponsePlan,
)
from app.conversation.response_rendering.contracts import RenderedResponse
from app.core.constants import ConversationState, Tone
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
from app.persistence.errors import InvalidPersistedPayloadError
from app.persistence.sqlite.connection import SqliteConnection
from app.persistence.sqlite.snapshot_store import (
    SqliteRuntimeSnapshotStore,
    snapshot_from_json,
    snapshot_to_json,
)
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
# Snapshot-capable scripted processor that fully round-trips state
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


@dataclass
class _StubState:
    """Small, round-trip-safe state proving the restore boundary works."""

    conversation_state: ConversationState = ConversationState.NEW_CALL
    sales_stage: str = "opening"
    discovered_facts: tuple[str, ...] = ()
    objections_voiced: tuple[str, ...] = ()
    language_pref: str = "english"
    discovery_questions_asked: tuple[str, ...] = ()
    budget_turns_consumed: int = 0


class ScriptedSnapshotProcessor:
    """TurnProcessor that fully round-trips its state via snapshot.

    Dumb-by-design: no reasoning. Scripted outputs with per-turn state
    mutations proving that updates performed BEFORE a restart survive
    into the recovered processor's state.
    """

    def __init__(
        self,
        session_id: str,
        outputs: tuple[tuple[str, bool, str | None], ...],
        state: _StubState | None = None,
        state_mutations: tuple[dict, ...] = (),
    ) -> None:
        self._session_id = session_id
        self._outputs = outputs
        self.state = state or _StubState()
        self._state_mutations = state_mutations

    def process_turn(self, turn: CoordinatedUserTurn) -> CoordinatedTurnOutput:
        idx = turn.sequence_number - 1
        # Apply scripted mutation BEFORE emitting the response
        if idx < len(self._state_mutations):
            m = self._state_mutations[idx]
            for key, value in m.items():
                if hasattr(self.state, key):
                    cur = getattr(self.state, key)
                    if isinstance(cur, tuple):
                        setattr(self.state, key, cur + tuple(value))
                    else:
                        setattr(self.state, key, value)
        self.state.budget_turns_consumed += 1

        text, terminal, outcome = (
            self._outputs[idx] if idx < len(self._outputs)
            else ("ok", False, None)
        )
        return CoordinatedTurnOutput(
            turn_id=turn.turn_id,
            response_plan=_plan(),
            rendered_response=_rendered(text),
            pipeline_outcome=(AuthoritativeResultKind(outcome) if outcome else None),
            conversation_terminal=terminal,
        )

    def export_snapshot(self, session_revision: int) -> ConversationRuntimeSnapshot:
        import json
        payload = json.dumps({
            "sales_stage": self.state.sales_stage,
            "discovered_facts": list(self.state.discovered_facts),
            "objections_voiced": list(self.state.objections_voiced),
            "language_pref": self.state.language_pref,
            "discovery_questions_asked": list(self.state.discovery_questions_asked),
            "budget_turns_consumed": self.state.budget_turns_consumed,
        })
        return ConversationRuntimeSnapshot(
            snapshot_version=SNAPSHOT_VERSION_CURRENT,
            session_id=self._session_id,
            source_session_revision=session_revision,
            conversation_state=self.state.conversation_state,
            sales_stage=self.state.sales_stage,
            persisted_contract_payloads=(("_stub_state", payload),),
        )

    @classmethod
    def restore_from_snapshot(
        cls,
        session_id: str,
        outputs: tuple[tuple[str, bool, str | None], ...],
        snapshot: ConversationRuntimeSnapshot,
        state_mutations: tuple[dict, ...] = (),
    ) -> "ScriptedSnapshotProcessor":
        import json
        payload = snapshot.payload("_stub_state")
        if payload is None:
            raise ValueError("snapshot missing _stub_state payload")
        data = json.loads(payload)
        state = _StubState(
            conversation_state=snapshot.conversation_state,
            sales_stage=data["sales_stage"],
            discovered_facts=tuple(data["discovered_facts"]),
            objections_voiced=tuple(data["objections_voiced"]),
            language_pref=data["language_pref"],
            discovery_questions_asked=tuple(data["discovery_questions_asked"]),
            budget_turns_consumed=int(data["budget_turns_consumed"]),
        )
        return cls(session_id, outputs, state=state,
                   state_mutations=state_mutations)


class ScriptedSnapshotFactory:
    """Factory that produces ScriptedSnapshotProcessors.

    Each session's scripts and mutations are registered externally so the
    factory stays stateless about the dialogue plan — same contract the
    future real factory will follow.
    """

    def __init__(self) -> None:
        self._scripts: dict[str, tuple[tuple[str, bool, str | None], ...]] = {}
        self._mutations: dict[str, tuple[dict, ...]] = {}

    def register(
        self,
        session_id: str,
        outputs: tuple[tuple[str, bool, str | None], ...],
        mutations: tuple[dict, ...] = (),
    ) -> None:
        self._scripts[session_id] = outputs
        self._mutations[session_id] = mutations

    def create_for_session(self, session: ChatSession, campaign, policy):
        return ScriptedSnapshotProcessor(
            session.session_id,
            self._scripts.get(session.session_id, (("ok", False, None),)),
            state_mutations=self._mutations.get(session.session_id, ()),
        )

    def create_from_snapshot(
        self, session: ChatSession, campaign, policy,
        snapshot: ConversationRuntimeSnapshot,
    ):
        return ScriptedSnapshotProcessor.restore_from_snapshot(
            session.session_id,
            self._scripts.get(session.session_id, (("ok", False, None),)),
            snapshot,
            state_mutations=self._mutations.get(session.session_id, ()),
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _phone() -> NormalizedPhone:
    return NormalizedPhone(
        raw="+15551234567", normalized="+15551234567",
        status=PhoneNormalizationStatus.VALID_E164,
    )


def _lead() -> LeadRecord:
    return LeadRecord(
        lead_id="lead-1", tenant_id="t1", campaign_id="c1",
        business_name="Alpha Inc", phone=_phone(),
        category=None, city=None, state=None, country=None,
        address=None, email=None, contact_name=None,
        website=None, website_status=WebsiteStatus.UNKNOWN,
    )


def _queue_entry() -> LeadQueueEntry:
    return LeadQueueEntry(
        lead_id="lead-1", tenant_id="t1", campaign_id="c1",
        queue_position=0, status=LeadQueueStatus.READY,
    )


def _campaign() -> CampaignConfig:
    return CampaignConfig(
        campaign_id="c1", tenant_id="t1", name="Test",
        status=CampaignStatus.ACTIVE,
        authorized_service_ids=("svc-web",),
    )


def _policy() -> PreCallDiscoveryPolicy:
    return PreCallDiscoveryPolicy(
        policy_id="pol-1", version="1.0",
        service_rules=(
            ServiceDiscoveryRule(
                service_id="svc-web",
                required_fact_keys=("website.presence",),
            ),
        ),
    )


def _boot_input(session_id: str = "sess-1", req: str = "req-1") -> SessionBootstrapInput:
    return SessionBootstrapInput(
        tenant_id="t1", campaign_id="c1", lead_id="lead-1",
        session_id=session_id, bootstrap_request_id=req,
        trusted_time_iso="2026-10-14T10:00:00Z",
    )


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "runtime.db")


def _make_runtime(
    db_path: str, factory: ScriptedSnapshotFactory | None = None,
) -> PersistentSalesRuntime:
    from app.application.sales_runtime.manifest import SnapshotCompatibilityManifest
    conn = SqliteConnection(db_path)
    f = factory or ScriptedSnapshotFactory()
    # Slice 45 scripted factory emits its own stub payload; use a permissive
    # manifest so the architecture is proven without requiring the Slice 46
    # BCI/LanguageProfile payloads.
    permissive = SnapshotCompatibilityManifest(
        top_level_snapshot_version=SNAPSHOT_VERSION_CURRENT,
        supported_contract_versions=(),
        required_for_active_recovery=frozenset(),
        optional_derived=frozenset(),
    )
    rt = PersistentSalesRuntime(conn, f, snapshot_manifest=permissive)
    if rt.leads.get("t1", "c1", "lead-1") is None:
        rt.leads.save(_lead())
    if rt.queue.get("t1", "c1", "lead-1") is None:
        rt.queue.save(_queue_entry())
    if rt.policies.resolve("c1") is None:
        rt.policies.register_default("c1", _policy())
    return rt


# ---------------------------------------------------------------------------
# Snapshot contract & serialization
# ---------------------------------------------------------------------------


class TestSnapshotContract:
    def test_roundtrip(self) -> None:
        snap = ConversationRuntimeSnapshot(
            snapshot_version=1,
            session_id="s1", source_session_revision=5,
            conversation_state=ConversationState.LISTEN,
            sales_stage="discovery",
            persisted_contract_payloads=(("a", '{"x":1}'),),
        )
        text = snapshot_to_json(snap)
        back = snapshot_from_json(text)
        assert back == snap

    def test_newer_version_rejected(self) -> None:
        snap = ConversationRuntimeSnapshot(
            snapshot_version=SNAPSHOT_VERSION_CURRENT,
            session_id="s1", source_session_revision=0,
            conversation_state=ConversationState.NEW_CALL,
        )
        text = snapshot_to_json(snap).replace(
            f'"snapshot_version": {SNAPSHOT_VERSION_CURRENT}',
            f'"snapshot_version": {SNAPSHOT_VERSION_CURRENT + 99}',
        )
        with pytest.raises(InvalidPersistedPayloadError):
            snapshot_from_json(text)

    def test_duplicate_payload_keys_rejected(self) -> None:
        with pytest.raises(ValueError):
            ConversationRuntimeSnapshot(
                snapshot_version=1,
                session_id="s1", source_session_revision=0,
                conversation_state=ConversationState.NEW_CALL,
                persisted_contract_payloads=(("a", "x"), ("a", "y")),
            )


# ---------------------------------------------------------------------------
# Atomic turn commit — snapshot + revision + idempotency together
# ---------------------------------------------------------------------------


class TestAtomicTurnCommit:
    def test_snapshot_persisted_with_revision(self, db_path) -> None:
        factory = ScriptedSnapshotFactory()
        rt = _make_runtime(db_path, factory)
        factory.register("sess-1", (
            ("turn 1 reply", False, None),
            ("turn 2 reply", False, None),
            ("goodbye", True, None),
        ), mutations=(
            {"discovered_facts": ["website.exists"]},
            {"objections_voiced": ["already_have_agency"]},
            {},
        ))

        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-1", "we have a website", 1, "tr-1")
        snap = rt.snapshots.load("sess-1")
        assert snap is not None
        assert snap.source_session_revision == 1


# ---------------------------------------------------------------------------
# Mid-conversation recovery — RECOVERED_ACTIVE
# ---------------------------------------------------------------------------


class TestMidConversationRecovery:
    def test_recovered_active_after_restart(self, db_path) -> None:
        factory = ScriptedSnapshotFactory()
        rt = _make_runtime(db_path, factory)
        factory.register("sess-1", (
            ("r1", False, None),
            ("r2", False, None),
            ("done", True, None),
        ), mutations=(
            {"discovered_facts": ["website.exists"]},
            {"objections_voiced": ["already_have_agency"]},
            {},
        ))
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-1", "hi", 1, "tr-1")
        rt.process_turn("sess-1", "ok", 2, "tr-2")

        # New runtime over same DB — simulate restart
        factory2 = ScriptedSnapshotFactory()
        factory2.register("sess-1", (
            ("r1", False, None),
            ("r2", False, None),
            ("done", True, None),
        ), mutations=())
        rt2 = _make_runtime(db_path, factory2)
        result = rt2.recover_session("sess-1")
        assert result.status == RecoveryStatus.RECOVERED_ACTIVE
        assert result.runtime_snapshot is not None
        assert result.runtime_snapshot.source_session_revision == 2

    def test_restored_processor_has_saved_state(self, db_path) -> None:
        factory = ScriptedSnapshotFactory()
        rt = _make_runtime(db_path, factory)
        factory.register("sess-1", (
            ("r1", False, None), ("r2", False, None), ("done", True, None),
        ), mutations=(
            {"discovered_facts": ["website.exists"],
             "discovery_questions_asked": ["q.website.presence"]},
            {"objections_voiced": ["already_have_agency"]},
            {},
        ))
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-1", "we have a website", 1, "tr-1")
        rt.process_turn("sess-1", "and an agency", 2, "tr-2")

        factory2 = ScriptedSnapshotFactory()
        factory2.register("sess-1", (
            ("r1", False, None), ("r2", False, None), ("done", True, None),
        ))
        rt2 = _make_runtime(db_path, factory2)
        result = rt2.restore_processor_if_possible("sess-1", _campaign())
        assert result.status == RecoveryStatus.RECOVERED_ACTIVE
        processor = rt2._processors["sess-1"]
        # Restored processor state carries all facts observed before restart
        assert "website.exists" in processor.state.discovered_facts
        assert "already_have_agency" in processor.state.objections_voiced
        assert "q.website.presence" in processor.state.discovery_questions_asked
        assert processor.state.budget_turns_consumed == 2


# ---------------------------------------------------------------------------
# Outcome equivalence — THE key test
# ---------------------------------------------------------------------------


class TestOutcomeEquivalence:
    def test_outcome_identical_with_and_without_restart(self, tmp_path) -> None:
        outputs = (
            ("hello", False, None),
            ("tell me more", False, None),
            ("goodbye", True, None),
        )
        mutations = (
            {"discovered_facts": ["website.exists"]},
            {"objections_voiced": ["already_have_agency"]},
            {},
        )

        # --- Run A: no restart ---
        db_a = str(tmp_path / "a.db")
        fa = ScriptedSnapshotFactory()
        fa.register("sess-A", outputs, mutations=mutations)
        rt_a = _make_runtime(db_a, fa)
        rt_a.start_session(
            _boot_input(session_id="sess-A", req="req-A"), _campaign(),
        )
        rt_a.process_turn("sess-A", "msg1", 1, "trA-1")
        rt_a.process_turn("sess-A", "msg2", 2, "trA-2")
        rt_a.process_turn("sess-A", "msg3", 3, "trA-3")
        outcome_a = rt_a.finish_session("sess-A").lead_outcome

        # --- Run B: restart after turn 2 ---
        db_b = str(tmp_path / "b.db")
        fb = ScriptedSnapshotFactory()
        fb.register("sess-B", outputs, mutations=mutations)
        rt_b = _make_runtime(db_b, fb)
        rt_b.start_session(
            _boot_input(session_id="sess-B", req="req-B"), _campaign(),
        )
        rt_b.process_turn("sess-B", "msg1", 1, "trB-1")
        rt_b.process_turn("sess-B", "msg2", 2, "trB-2")

        # Restart
        fb2 = ScriptedSnapshotFactory()
        fb2.register("sess-B", outputs, mutations=mutations)
        rt_b2 = _make_runtime(db_b, fb2)
        rt_b2.restore_processor_if_possible("sess-B", _campaign())
        rt_b2.process_turn("sess-B", "msg3", 3, "trB-3")
        outcome_b = rt_b2.finish_session("sess-B").lead_outcome

        # Compare authoritative outcome fields — session_id differs by design
        assert outcome_a.outcome_status == outcome_b.outcome_status
        assert outcome_a.interest_level == outcome_b.interest_level
        assert outcome_a.dnc_status == outcome_b.dnc_status
        assert outcome_a.execution_status == outcome_b.execution_status
        assert outcome_a.termination_reason == outcome_b.termination_reason


# ---------------------------------------------------------------------------
# Turn idempotency with snapshot consistency
# ---------------------------------------------------------------------------


class TestTurnIdempotencyWithSnapshot:
    def test_replay_does_not_rerun_or_create_new_snapshot(self, db_path) -> None:
        factory = ScriptedSnapshotFactory()
        rt = _make_runtime(db_path, factory)
        factory.register("sess-1", (
            ("r1", False, None), ("r2", False, None), ("done", True, None),
        ))
        rt.start_session(_boot_input(), _campaign())
        r1 = rt.process_turn("sess-1", "hi", 1, "tr-same")
        snap1 = rt.snapshots.load("sess-1")

        # Replay same turn_request_id
        r2 = rt.process_turn("sess-1", "hi again", 1, "tr-same")
        assert r2.replayed_from_idempotency is True
        assert r2.committed_revision == r1.committed_revision
        snap2 = rt.snapshots.load("sess-1")
        # Snapshot was not re-written (revision didn't advance)
        assert snap1 == snap2


# ---------------------------------------------------------------------------
# DNC safety with snapshot
# ---------------------------------------------------------------------------


class TestDNCSnapshotSafety:
    def test_dnc_still_persisted_at_queue_after_restart(self, db_path) -> None:
        rt = _make_runtime(db_path)
        rt.mark_lead_dnc("t1", "c1", "lead-1")

        rt2 = _make_runtime(db_path)
        entry = rt2.queue.get("t1", "c1", "lead-1")
        assert entry.is_dnc is True
        assert entry.status == LeadQueueStatus.DNC


# ---------------------------------------------------------------------------
# NOT_INTERESTED remains distinct
# ---------------------------------------------------------------------------


class TestNotInterestedDistinct:
    def test_not_interested_not_promoted_to_dnc(self, db_path) -> None:
        rt = _make_runtime(db_path)
        rt.start_session(_boot_input(), _campaign())
        done = rt.finish_session(
            "sess-1", is_not_interested=True,
            termination_reason=TerminationReason.NOT_INTERESTED_EXIT,
        )
        assert done.lead_outcome.outcome_status == OutcomeStatus.NOT_INTERESTED
        entry = rt.queue.get("t1", "c1", "lead-1")
        assert entry.is_dnc is False


# ---------------------------------------------------------------------------
# Runtime failure after restart preserves customer outcome
# ---------------------------------------------------------------------------


class TestRuntimeFailureAfterRestart:
    def test_failure_preserves_prefailure_customer_state(self, db_path) -> None:
        factory = ScriptedSnapshotFactory()
        rt = _make_runtime(db_path, factory)
        factory.register("sess-1", (
            ("interested in services", False, None),
            ("tell me more", False, None),
        ), mutations=(
            {"discovered_facts": ["interest.observed"]},
            {},
        ))
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-1", "hi", 1, "tr-1")
        rt.process_turn("sess-1", "ok", 2, "tr-2")

        # Restart; next finish_session with provider failure
        rt2 = _make_runtime(db_path, ScriptedSnapshotFactory())
        done = rt2.finish_session(
            "sess-1", session_error=True,
            termination_reason=TerminationReason.RUNTIME_ERROR,
        )
        assert done.execution_status == ExecutionStatus.FAILED
        # Customer outcome determined independently — never force-DNC on failure
        assert done.lead_outcome.outcome_status != OutcomeStatus.DNC


# ---------------------------------------------------------------------------
# Snapshot missing / invalid — fail closed
# ---------------------------------------------------------------------------


class TestFailClosedOnBadSnapshot:
    def test_missing_snapshot_for_active_session(self, db_path) -> None:
        # Factory does NOT implement SnapshotCapable export → no snapshot saved
        class _NoSnapshotFactory:
            def create_for_session(self, session, campaign, policy):
                return ScriptedSnapshotProcessor(
                    session.session_id,
                    (("r1", False, None), ("r2", False, None)),
                )
            def create_from_snapshot(self, session, campaign, policy, snapshot):
                raise AssertionError("should not be called")

        rt = PersistentSalesRuntime(SqliteConnection(db_path), _NoSnapshotFactory())
        rt.leads.save(_lead())
        rt.queue.save(_queue_entry())
        rt.policies.register_default("c1", _policy())
        rt.start_session(_boot_input(), _campaign())
        # Suppress snapshot export by using a processor that IS NOT SnapshotCapable
        class _Blind:
            def __init__(self, outputs): self._outputs = outputs
            def process_turn(self, turn):
                idx = turn.sequence_number - 1
                text, terminal, _ = self._outputs[idx] if idx < len(self._outputs) else ("ok", False, None)
                return CoordinatedTurnOutput(
                    turn_id=turn.turn_id, response_plan=_plan(),
                    rendered_response=_rendered(text),
                    conversation_terminal=terminal,
                )
        rt._processors["sess-1"] = _Blind((("r1", False, None),))
        rt.process_turn("sess-1", "hi", 1, "tr-1")

        rt2 = _make_runtime(db_path, ScriptedSnapshotFactory())
        result = rt2.recover_session("sess-1")
        assert result.status == RecoveryStatus.RUNTIME_SNAPSHOT_MISSING

    def test_corrupted_snapshot_payload_rejected(self, db_path) -> None:
        rt = _make_runtime(db_path)
        rt.start_session(_boot_input(), _campaign())
        # Corrupt the stored JSON directly
        rt._c.raw.execute(
            "UPDATE chat_sessions SET runtime_snapshot_json = ?, "
            "runtime_snapshot_version = ?, status = ?, turn_count = ? "
            "WHERE session_id = ?",
            ("{not valid json", 1, SessionStatus.ACTIVE.value, 1, "sess-1"),
        )

        rt2 = _make_runtime(db_path)
        result = rt2.recover_session("sess-1")
        assert result.status == RecoveryStatus.RUNTIME_SNAPSHOT_INVALID


# ---------------------------------------------------------------------------
# Policy version pinning retained
# ---------------------------------------------------------------------------


class TestPolicyPinningRetained:
    def test_pinned_version_used_after_restart(self, db_path) -> None:
        factory = ScriptedSnapshotFactory()
        rt = _make_runtime(db_path, factory)
        factory.register("sess-1", (("r1", False, None),))
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-1", "hi", 1, "tr-1")

        # Register a newer policy as default
        rt.policies.register_default(
            "c1", replace(_policy(), version="2.0"),
        )

        rt2 = _make_runtime(db_path, ScriptedSnapshotFactory())
        result = rt2.recover_session("sess-1")
        assert result.status == RecoveryStatus.RECOVERED_ACTIVE
        # Pinned v1.0 recovered, not v2.0
        assert result.policy.version == "1.0"


# ---------------------------------------------------------------------------
# Terminal session recovery still works
# ---------------------------------------------------------------------------


class TestTerminalRecoveryStillWorks:
    def test_completed_session_recovers_as_terminal(self, db_path) -> None:
        factory = ScriptedSnapshotFactory()
        rt = _make_runtime(db_path, factory)
        factory.register("sess-1", (
            ("r1", False, None), ("done", True, None),
        ))
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-1", "hi", 1, "tr-1")
        rt.process_turn("sess-1", "bye", 2, "tr-2")

        rt2 = _make_runtime(db_path, ScriptedSnapshotFactory())
        result = rt2.recover_session("sess-1")
        assert result.status == RecoveryStatus.RECOVERED_TERMINAL


# ---------------------------------------------------------------------------
# Schema v3 migration test — v2 → v3 preserves existing data
# ---------------------------------------------------------------------------


class TestSchemaV3Migration:
    def test_v2_data_preserved_through_v3_upgrade(self, db_path) -> None:
        import sqlite3
        # Create a v2-schema DB manually
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER PRIMARY KEY NOT NULL);
            INSERT INTO schema_version(version) VALUES (2);
            CREATE TABLE chat_sessions (
                session_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                campaign_id TEXT NOT NULL,
                lead_id TEXT NOT NULL,
                status TEXT NOT NULL,
                mode TEXT NOT NULL,
                revision INTEGER NOT NULL DEFAULT 0,
                turn_count INTEGER NOT NULL DEFAULT 0,
                transcript_json TEXT NOT NULL DEFAULT '[]',
                pre_call_plan_json TEXT,
                policy_id TEXT,
                policy_version TEXT,
                termination_reason TEXT
            );
            INSERT INTO chat_sessions
                (session_id, tenant_id, campaign_id, lead_id, status, mode)
                VALUES ('legacy', 't1', 'c1', 'l1', 'completed', 'interactive');
        """)
        conn.commit()
        conn.close()

        # Opening via SqliteConnection should upgrade v2 → v3
        rt_conn = SqliteConnection(db_path)
        assert rt_conn.schema_version() == 3
        # Legacy row still present
        row = rt_conn.raw.execute(
            "SELECT * FROM chat_sessions WHERE session_id = 'legacy'",
        ).fetchone()
        assert row is not None
        # New column exists and is NULL for legacy data
        assert row["runtime_snapshot_json"] is None


# ---------------------------------------------------------------------------
# No domain intelligence in snapshot layer
# ---------------------------------------------------------------------------


class TestNoDomainIntelligenceInSnapshot:
    def test_snapshot_module_has_no_reasoning_imports(self) -> None:
        import inspect
        from app.application.sales_runtime import (
            snapshot, snapshot_protocol,
        )
        from app.persistence.sqlite import snapshot_store
        for module in (snapshot, snapshot_protocol, snapshot_store):
            source = inspect.getsource(module)
            for forbidden in (
                "BrainOrchestrator", "StateMachine", "ActionValidator",
                "ReasoningProvider", "HybridRetrievalEngine",
                "ServiceRelevanceResolver",
            ):
                assert forbidden not in source
