"""Tests for Slice 43 — SQLite persistent repository adapters.

Proves: domain state survives process restart, transactions are truly
atomic, revisions enforce optimistic concurrency across processes, DNC
survives everything, tenant/campaign isolation is enforced, serialization
is deterministic, and no domain intelligence lives in persistence.
"""

from __future__ import annotations

import inspect
import os

import pytest

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
    InvalidPersistedPayloadError,
    NotFoundError,
    SchemaMismatchError,
    StaleRevisionError,
    TransactionFailedError,
)
from app.persistence.sqlite.connection import (
    CURRENT_SCHEMA_VERSION,
    SqliteConnection,
)
from app.persistence.sqlite.coordinator import AtomicSessionQueueCoordinator
from app.persistence.sqlite.repositories import (
    SqliteBootstrapIdempotencyStore,
    SqliteCampaignQueueStore,
    SqliteChatSessionRepository,
    SqliteLeadStore,
    SqlitePolicyRegistry,
)
from app.persistence.sqlite.serializers import (
    plan_from_json,
    plan_to_json,
    policy_from_json,
    policy_to_json,
    transcript_from_json,
    transcript_to_json,
)
from app.precall.contracts import (
    PreCallDiscoveryPolicy,
    ServiceDiscoveryRule,
    VerificationPriority,
)
from app.runtime.chat.contracts import (
    ChatSession,
    RuntimeMode,
    SessionStatus,
    TranscriptEntry,
    TranscriptRole,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _phone() -> NormalizedPhone:
    return NormalizedPhone(
        raw="+15551234567",
        normalized="+15551234567",
        status=PhoneNormalizationStatus.VALID_E164,
    )


def _session(
    session_id: str = "sess-1",
    tenant_id: str = "t1",
    campaign_id: str = "c1",
    lead_id: str = "lead-1",
    status: SessionStatus = SessionStatus.ACTIVE,
) -> ChatSession:
    return ChatSession(
        session_id=session_id,
        tenant_id=tenant_id,
        campaign_id=campaign_id,
        lead_id=lead_id,
        status=status,
        mode=RuntimeMode.INTERACTIVE,
    )


def _lead(
    lead_id: str = "lead-1",
    tenant_id: str = "t1",
    campaign_id: str = "c1",
) -> LeadRecord:
    return LeadRecord(
        lead_id=lead_id,
        tenant_id=tenant_id,
        campaign_id=campaign_id,
        business_name="Alpha Inc",
        phone=_phone(),
        category="Restaurant",
        city="NYC",
        state=None, country=None, address=None,
        email=None, contact_name=None, website=None,
        website_status=WebsiteStatus.UNKNOWN,
    )


def _queue_entry(
    lead_id: str = "lead-1",
    tenant_id: str = "t1",
    campaign_id: str = "c1",
    status: LeadQueueStatus = LeadQueueStatus.READY,
    is_dnc: bool = False,
) -> LeadQueueEntry:
    return LeadQueueEntry(
        lead_id=lead_id,
        campaign_id=campaign_id,
        tenant_id=tenant_id,
        queue_position=0,
        status=status,
        is_dnc=is_dnc,
    )


def _policy(version: str = "1.0") -> PreCallDiscoveryPolicy:
    return PreCallDiscoveryPolicy(
        policy_id="pol-1",
        version=version,
        service_rules=(
            ServiceDiscoveryRule(
                service_id="svc-web",
                required_fact_keys=("website.presence",),
            ),
        ),
    )


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


# ---------------------------------------------------------------------------
# 1. ChatSession save/load
# ---------------------------------------------------------------------------


class TestChatSessionPersistence:
    def test_save_and_load_session(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        repo = SqliteChatSessionRepository(conn)
        session = _session()
        created = repo.create(session)
        assert created.revision == 0

        loaded = repo.get("sess-1")
        assert loaded.session_id == "sess-1"
        assert loaded.tenant_id == "t1"
        assert loaded.status == SessionStatus.ACTIVE

    def test_session_with_transcript(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        repo = SqliteChatSessionRepository(conn)
        session = _session()
        from dataclasses import replace
        session = replace(session, transcript=(
            TranscriptEntry(1, TranscriptRole.PROSPECT, "hello"),
            TranscriptEntry(1, TranscriptRole.AGENT, "hi there"),
        ), turn_count=1)
        repo.create(session)
        loaded = repo.get("sess-1")
        assert len(loaded.transcript) == 2
        assert loaded.transcript[0].text == "hello"


# ---------------------------------------------------------------------------
# 2. Session revision update
# ---------------------------------------------------------------------------


class TestRevisionUpdate:
    def test_update_increments_revision(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        repo = SqliteChatSessionRepository(conn)
        session = repo.create(_session())
        from dataclasses import replace
        updated = repo.update(replace(session, turn_count=1))
        assert updated.revision == 1
        updated2 = repo.update(replace(updated, turn_count=2))
        assert updated2.revision == 2


# ---------------------------------------------------------------------------
# 3. Stale revision rejected
# ---------------------------------------------------------------------------


class TestStaleRevisionRejected:
    def test_update_with_stale_revision_fails(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        repo = SqliteChatSessionRepository(conn)
        session = repo.create(_session())
        from dataclasses import replace
        repo.update(replace(session, turn_count=1))
        # Now session on disk has revision 1; using original (rev 0) is stale
        with pytest.raises(StaleRevisionError):
            repo.update(replace(session, turn_count=99))


# ---------------------------------------------------------------------------
# 4. LeadRecord persistence
# ---------------------------------------------------------------------------


class TestLeadPersistence:
    def test_save_and_get_lead(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        store = SqliteLeadStore(conn)
        store.save(_lead())
        loaded = store.get("t1", "c1", "lead-1")
        assert loaded is not None
        assert loaded.business_name == "Alpha Inc"
        assert loaded.phone.normalized == "+15551234567"
        assert loaded.category == "Restaurant"

    def test_missing_lead_returns_none(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        store = SqliteLeadStore(conn)
        assert store.get("t1", "c1", "nope") is None

    def test_list_by_campaign(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        store = SqliteLeadStore(conn)
        store.save(_lead("l1"))
        store.save(_lead("l2"))
        store.save(_lead("l3", campaign_id="c2"))
        leads = store.list_by_campaign("t1", "c1")
        assert len(leads) == 2


# ---------------------------------------------------------------------------
# 5. Queue state persistence
# ---------------------------------------------------------------------------


class TestQueuePersistence:
    def test_save_and_get_queue_entry(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        q = SqliteCampaignQueueStore(conn)
        q.save(_queue_entry())
        loaded = q.get("t1", "c1", "lead-1")
        assert loaded is not None
        assert loaded.status == LeadQueueStatus.READY

    def test_transition_updates_status(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        q = SqliteCampaignQueueStore(conn)
        q.save(_queue_entry())
        updated = q.transition(
            "t1", "c1", "lead-1",
            LeadQueueStatus.READY, LeadQueueStatus.IN_PROGRESS,
        )
        assert updated.status == LeadQueueStatus.IN_PROGRESS
        reloaded = q.get("t1", "c1", "lead-1")
        assert reloaded.status == LeadQueueStatus.IN_PROGRESS

    def test_wrong_from_status_rejected(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        q = SqliteCampaignQueueStore(conn)
        q.save(_queue_entry(status=LeadQueueStatus.PENDING))
        with pytest.raises(StaleRevisionError):
            q.transition(
                "t1", "c1", "lead-1",
                LeadQueueStatus.READY, LeadQueueStatus.IN_PROGRESS,
            )


# ---------------------------------------------------------------------------
# 6. DNC survives restart
# ---------------------------------------------------------------------------


class TestDNCSurvivesRestart:
    def test_dnc_lead_still_dnc_after_reopen(self, db_path) -> None:
        conn1 = SqliteConnection(db_path)
        q1 = SqliteCampaignQueueStore(conn1)
        q1.save(_queue_entry(is_dnc=True))
        conn1.close()

        conn2 = SqliteConnection(db_path)
        q2 = SqliteCampaignQueueStore(conn2)
        loaded = q2.get("t1", "c1", "lead-1")
        assert loaded.is_dnc is True
        assert loaded.status == LeadQueueStatus.DNC

    def test_dnc_cannot_be_regressed(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        q = SqliteCampaignQueueStore(conn)
        q.save(_queue_entry(is_dnc=True))
        # Attempt to save a non-DNC version — must refuse
        with pytest.raises(StaleRevisionError):
            q.save(_queue_entry(is_dnc=False, status=LeadQueueStatus.READY))

    def test_mark_dnc_overrides_any_status(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        q = SqliteCampaignQueueStore(conn)
        q.save(_queue_entry(status=LeadQueueStatus.IN_PROGRESS))
        updated = q.mark_dnc("t1", "c1", "lead-1")
        assert updated.is_dnc is True
        assert updated.status == LeadQueueStatus.DNC


# ---------------------------------------------------------------------------
# 7. Completed / exhausted survives restart
# ---------------------------------------------------------------------------


class TestTerminalStatusSurvives:
    def test_completed_persists(self, db_path) -> None:
        conn1 = SqliteConnection(db_path)
        q1 = SqliteCampaignQueueStore(conn1)
        q1.save(_queue_entry(status=LeadQueueStatus.COMPLETED))
        conn1.close()

        conn2 = SqliteConnection(db_path)
        q2 = SqliteCampaignQueueStore(conn2)
        assert q2.get("t1", "c1", "lead-1").status == LeadQueueStatus.COMPLETED

    def test_exhausted_persists(self, db_path) -> None:
        conn1 = SqliteConnection(db_path)
        q1 = SqliteCampaignQueueStore(conn1)
        q1.save(_queue_entry(status=LeadQueueStatus.EXHAUSTED))
        conn1.close()

        conn2 = SqliteConnection(db_path)
        q2 = SqliteCampaignQueueStore(conn2)
        assert q2.get("t1", "c1", "lead-1").status == LeadQueueStatus.EXHAUSTED

    def test_in_progress_preserved_not_reset(self, db_path) -> None:
        conn1 = SqliteConnection(db_path)
        q1 = SqliteCampaignQueueStore(conn1)
        q1.save(_queue_entry(status=LeadQueueStatus.IN_PROGRESS))
        conn1.close()

        conn2 = SqliteConnection(db_path)
        q2 = SqliteCampaignQueueStore(conn2)
        # IN_PROGRESS must be preserved as-is for future recovery policy
        assert q2.get("t1", "c1", "lead-1").status == LeadQueueStatus.IN_PROGRESS


# ---------------------------------------------------------------------------
# 8. Policy exact-version resolution
# ---------------------------------------------------------------------------


class TestPolicyExactVersion:
    def test_exact_version_resolve(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        reg = SqlitePolicyRegistry(conn)
        reg.register("c1", _policy(version="1.0"))
        reg.register("c1", _policy(version="2.0"))

        assert reg.resolve("c1", "1.0").version == "1.0"
        assert reg.resolve("c1", "2.0").version == "2.0"
        assert reg.resolve("c1", "3.0") is None

    def test_default_resolve(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        reg = SqlitePolicyRegistry(conn)
        reg.register_default("c1", _policy(version="1.0"))
        assert reg.resolve("c1").version == "1.0"

    def test_policy_survives_restart(self, db_path) -> None:
        conn1 = SqliteConnection(db_path)
        reg1 = SqlitePolicyRegistry(conn1)
        reg1.register_default("c1", _policy(version="7.5"))
        conn1.close()

        conn2 = SqliteConnection(db_path)
        reg2 = SqlitePolicyRegistry(conn2)
        policy = reg2.resolve("c1", "7.5")
        assert policy is not None
        assert policy.service_rules[0].service_id == "svc-web"


# ---------------------------------------------------------------------------
# 9. Bootstrap idempotency survives restart
# ---------------------------------------------------------------------------


class TestIdempotencyPersistence:
    def test_record_and_lookup(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        store = SqliteBootstrapIdempotencyStore(conn)
        store.record(
            "req-1", "t1", "c1", "lead-1", "sess-1",
            "CREATED", {"status": "CREATED"},
        )
        found = store.lookup("req-1")
        assert found["status"] == "CREATED"

    def test_duplicate_request_rejected(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        store = SqliteBootstrapIdempotencyStore(conn)
        store.record(
            "req-1", "t1", "c1", "lead-1", "sess-1",
            "CREATED", {"status": "CREATED"},
        )
        with pytest.raises(IdempotencyConflictError):
            store.record(
                "req-1", "t1", "c1", "lead-1", "sess-2",
                "CREATED", {"status": "CREATED"},
            )

    def test_idempotency_survives_restart(self, db_path) -> None:
        conn1 = SqliteConnection(db_path)
        store1 = SqliteBootstrapIdempotencyStore(conn1)
        store1.record(
            "req-99", "t1", "c1", "lead-1", "sess-99",
            "CREATED", {"status": "CREATED"},
        )
        conn1.close()

        conn2 = SqliteConnection(db_path)
        store2 = SqliteBootstrapIdempotencyStore(conn2)
        assert store2.lookup("req-99") is not None


# ---------------------------------------------------------------------------
# 10. Duplicate session prevented
# ---------------------------------------------------------------------------


class TestDuplicateSessionPrevented:
    def test_duplicate_session_raises(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        repo = SqliteChatSessionRepository(conn)
        repo.create(_session())
        with pytest.raises(IdempotencyConflictError):
            repo.create(_session())


# ---------------------------------------------------------------------------
# 11. Session + queue transition atomic commit
# ---------------------------------------------------------------------------


class TestAtomicSessionQueue:
    def test_both_commit_on_success(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        sessions = SqliteChatSessionRepository(conn)
        queue = SqliteCampaignQueueStore(conn)
        queue.save(_queue_entry())

        coord = AtomicSessionQueueCoordinator(conn, sessions, queue)
        session, entry = coord.create_session_and_consume_lead(
            _session(), "t1", "c1", "lead-1",
        )
        assert session.session_id == "sess-1"
        assert entry.status == LeadQueueStatus.IN_PROGRESS

        # Both persisted
        assert sessions.exists("sess-1")
        assert queue.get("t1", "c1", "lead-1").status == LeadQueueStatus.IN_PROGRESS


# ---------------------------------------------------------------------------
# 12. Rollback on session failure
# ---------------------------------------------------------------------------


class TestRollbackOnSessionFailure:
    def test_duplicate_session_rolls_back_queue(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        sessions = SqliteChatSessionRepository(conn)
        queue = SqliteCampaignQueueStore(conn)
        queue.save(_queue_entry())
        sessions.create(_session())  # pre-existing session with same id

        coord = AtomicSessionQueueCoordinator(conn, sessions, queue)
        with pytest.raises(IdempotencyConflictError):
            coord.create_session_and_consume_lead(
                _session(), "t1", "c1", "lead-1",
            )
        # Queue entry must NOT be IN_PROGRESS — rolled back
        assert queue.get("t1", "c1", "lead-1").status == LeadQueueStatus.READY


# ---------------------------------------------------------------------------
# 13. Rollback on queue-transition failure
# ---------------------------------------------------------------------------


class TestRollbackOnQueueFailure:
    def test_queue_transition_failure_rolls_back_session(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        sessions = SqliteChatSessionRepository(conn)
        queue = SqliteCampaignQueueStore(conn)
        # Queue entry is PENDING, not READY — transition to IN_PROGRESS will fail
        queue.save(_queue_entry(status=LeadQueueStatus.PENDING))

        coord = AtomicSessionQueueCoordinator(conn, sessions, queue)
        with pytest.raises(StaleRevisionError):
            coord.create_session_and_consume_lead(
                _session(), "t1", "c1", "lead-1",
            )
        # Session must NOT exist — rolled back
        assert not sessions.exists("sess-1")

    def test_dnc_lead_blocks_bootstrap_and_rolls_back(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        sessions = SqliteChatSessionRepository(conn)
        queue = SqliteCampaignQueueStore(conn)
        queue.save(_queue_entry(is_dnc=True))

        coord = AtomicSessionQueueCoordinator(conn, sessions, queue)
        with pytest.raises(StaleRevisionError):
            coord.create_session_and_consume_lead(
                _session(), "t1", "c1", "lead-1",
            )
        assert not sessions.exists("sess-1")


# ---------------------------------------------------------------------------
# 14. Tenant isolation
# ---------------------------------------------------------------------------


class TestTenantIsolation:
    def test_leads_isolated_by_tenant(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        store = SqliteLeadStore(conn)
        store.save(_lead(lead_id="l1", tenant_id="t1"))
        store.save(_lead(lead_id="l1", tenant_id="t2"))
        a = store.get("t1", "c1", "l1")
        b = store.get("t2", "c1", "l1")
        assert a.tenant_id == "t1"
        assert b.tenant_id == "t2"
        assert store.get("t3", "c1", "l1") is None

    def test_sessions_isolated_by_tenant(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        repo = SqliteChatSessionRepository(conn)
        repo.create(_session("sess-a", tenant_id="t1"))
        repo.create(_session("sess-b", tenant_id="t2"))
        assert len(repo.list_sessions("t1")) == 1
        assert len(repo.list_sessions("t2")) == 1
        assert len(repo.list_sessions()) == 2

    def test_active_count_isolated_by_tenant(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        repo = SqliteChatSessionRepository(conn)
        repo.create(_session("sess-a", tenant_id="t1"))
        repo.create(_session("sess-b", tenant_id="t2"))
        assert repo.active_count("t1") == 1
        assert repo.active_count("t2") == 1


# ---------------------------------------------------------------------------
# 15. Campaign isolation
# ---------------------------------------------------------------------------


class TestCampaignIsolation:
    def test_queue_isolated_by_campaign(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        q = SqliteCampaignQueueStore(conn)
        q.save(_queue_entry(campaign_id="c1"))
        q.save(_queue_entry(campaign_id="c2"))
        assert len(q.list_by_campaign("t1", "c1")) == 1
        assert len(q.list_by_campaign("t1", "c2")) == 1


# ---------------------------------------------------------------------------
# 16. Deterministic serialization
# ---------------------------------------------------------------------------


class TestDeterministicSerialization:
    def test_policy_roundtrip(self) -> None:
        p = _policy()
        s = policy_to_json(p)
        roundtripped = policy_from_json(s)
        assert roundtripped == p

    def test_transcript_roundtrip(self) -> None:
        entries = (
            TranscriptEntry(1, TranscriptRole.PROSPECT, "a"),
            TranscriptEntry(1, TranscriptRole.AGENT, "b"),
        )
        s = transcript_to_json(entries)
        assert transcript_from_json(s) == entries

    def test_invalid_payload_raises_typed_error(self) -> None:
        with pytest.raises(InvalidPersistedPayloadError):
            policy_from_json('{"not valid":')
        with pytest.raises(InvalidPersistedPayloadError):
            transcript_from_json('{"not valid":')

    def test_policy_serialization_deterministic(self) -> None:
        p = _policy()
        s1 = policy_to_json(p)
        s2 = policy_to_json(p)
        assert s1 == s2


# ---------------------------------------------------------------------------
# 17. Repository close/reopen recovery
# ---------------------------------------------------------------------------


class TestRestartRecovery:
    def test_full_state_restored_after_reopen(self, db_path) -> None:
        # Process A
        conn1 = SqliteConnection(db_path)
        leads = SqliteLeadStore(conn1)
        queue = SqliteCampaignQueueStore(conn1)
        sessions = SqliteChatSessionRepository(conn1)
        policies = SqlitePolicyRegistry(conn1)

        leads.save(_lead())
        queue.save(_queue_entry(status=LeadQueueStatus.READY))
        sessions.create(_session())
        policies.register_default("c1", _policy())
        conn1.close()

        # Process B
        conn2 = SqliteConnection(db_path)
        leads2 = SqliteLeadStore(conn2)
        queue2 = SqliteCampaignQueueStore(conn2)
        sessions2 = SqliteChatSessionRepository(conn2)
        policies2 = SqlitePolicyRegistry(conn2)

        assert leads2.get("t1", "c1", "lead-1") is not None
        assert queue2.get("t1", "c1", "lead-1").status == LeadQueueStatus.READY
        assert sessions2.get("sess-1").status == SessionStatus.ACTIVE
        assert policies2.resolve("c1") is not None

    def test_terminal_session_survives_restart(self, db_path) -> None:
        from dataclasses import replace
        conn1 = SqliteConnection(db_path)
        sessions = SqliteChatSessionRepository(conn1)
        created = sessions.create(_session())
        sessions.update(replace(created, status=SessionStatus.COMPLETED))
        conn1.close()

        conn2 = SqliteConnection(db_path)
        sessions2 = SqliteChatSessionRepository(conn2)
        assert sessions2.get("sess-1").status == SessionStatus.COMPLETED


# ---------------------------------------------------------------------------
# 18. Schema initialization
# ---------------------------------------------------------------------------


class TestSchemaInit:
    def test_schema_version_set_on_fresh_db(self, db_path) -> None:
        conn = SqliteConnection(db_path)
        assert conn.schema_version() == CURRENT_SCHEMA_VERSION

    def test_schema_init_idempotent(self, db_path) -> None:
        conn1 = SqliteConnection(db_path)
        v1 = conn1.schema_version()
        conn1.close()
        conn2 = SqliteConnection(db_path)
        v2 = conn2.schema_version()
        assert v1 == v2 == CURRENT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# 19. Migration / version handling
# ---------------------------------------------------------------------------


class TestSchemaMismatchHandling:
    def test_newer_schema_version_raises(self, db_path) -> None:
        conn1 = SqliteConnection(db_path)
        # Simulate a future schema version
        conn1.raw.execute(
            "UPDATE schema_version SET version = ?",
            (CURRENT_SCHEMA_VERSION + 10,),
        )
        conn1.close()
        with pytest.raises(SchemaMismatchError):
            SqliteConnection(db_path)


# ---------------------------------------------------------------------------
# 20. No domain intelligence in persistence layer
# ---------------------------------------------------------------------------


class TestNoDomainIntelligence:
    def test_no_reasoning_imports(self) -> None:
        from app.persistence.sqlite import repositories, coordinator, serializers, connection
        for module in (repositories, coordinator, serializers, connection):
            source = inspect.getsource(module)
            for forbidden in (
                "BrainOrchestrator", "ReasoningProvider",
                "StateMachine", "ActionValidator",
                "HybridRetrievalEngine", "ServiceRelevanceResolver",
                "ResponseRenderer", "ResponsePlanner",
                "openai", "anthropic",
            ):
                assert forbidden not in source, f"{forbidden} found in {module.__name__}"

    def test_no_campaign_specific_branches(self) -> None:
        from app.persistence.sqlite import repositories, coordinator
        for module in (repositories, coordinator):
            source = inspect.getsource(module)
            for forbidden in (
                '== "website"', '== "automation"', '== "restaurant"',
                "if industry", "if service ==",
            ):
                assert forbidden not in source


# ---------------------------------------------------------------------------
# Database file persistence check
# ---------------------------------------------------------------------------


class TestDatabaseFile:
    def test_file_created(self, db_path) -> None:
        SqliteConnection(db_path)
        assert os.path.exists(db_path)


# ---------------------------------------------------------------------------
# Plan serialization through session persistence
# ---------------------------------------------------------------------------


class TestPlanPersistence:
    def test_session_with_plan_roundtrip(self, db_path) -> None:
        from dataclasses import replace
        from app.precall.contracts import (
            InitialPosture,
            OpeningObjective,
            PreCallConversationPlan,
            PreCallPlanStatus,
        )
        conn = SqliteConnection(db_path)
        repo = SqliteChatSessionRepository(conn)

        plan = PreCallConversationPlan(
            lead_id="lead-1",
            campaign_id="c1",
            tenant_id="t1",
            policy_id="pol-1",
            policy_version="1.0",
            opening_objective=OpeningObjective.CONFIRM_BUSINESS_CONTEXT,
            initial_posture=InitialPosture.NORMAL_CONTEXT,
            plan_status=PreCallPlanStatus.READY,
        )
        session = replace(_session(), pre_call_plan=plan)
        repo.create(session)

        loaded = repo.get("sess-1")
        assert loaded.pre_call_plan is not None
        assert loaded.pre_call_plan.policy_version == "1.0"
        assert loaded.pre_call_plan.opening_objective == OpeningObjective.CONFIRM_BUSINESS_CONTEXT


# ---------------------------------------------------------------------------
# Cross-process concurrency (simulated via two connections to same file)
# ---------------------------------------------------------------------------


class TestCrossProcessConcurrency:
    def test_two_connections_see_committed_data(self, db_path) -> None:
        conn_a = SqliteConnection(db_path)
        conn_b = SqliteConnection(db_path)
        repo_a = SqliteChatSessionRepository(conn_a)
        repo_b = SqliteChatSessionRepository(conn_b)

        repo_a.create(_session())
        # conn_b must see it (SQLite is single-file)
        assert repo_b.exists("sess-1")

    def test_second_connection_rejects_stale_revision(self, db_path) -> None:
        from dataclasses import replace
        conn_a = SqliteConnection(db_path)
        repo_a = SqliteChatSessionRepository(conn_a)
        created = repo_a.create(_session())

        conn_b = SqliteConnection(db_path)
        repo_b = SqliteChatSessionRepository(conn_b)

        # Process A updates to revision 1
        repo_a.update(replace(created, turn_count=1))

        # Process B tries to update with its stale snapshot (revision 0)
        with pytest.raises(StaleRevisionError):
            repo_b.update(replace(created, turn_count=42))
