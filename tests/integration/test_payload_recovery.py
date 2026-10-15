"""Slice 46 — Enhanced recovery with per-contract payloads.

Proves the integration of:
    - real BCI/LanguageProfile/Qualification serializers
    - SnapshotCompatibilityManifest validation during recovery
    - REQUIRED_PAYLOAD_MISSING and UNSUPPORTED_PAYLOAD_VERSION statuses
    - outcome equivalence with payloads carried across restart
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.application.sales_runtime.manifest import default_manifest
from app.application.sales_runtime.persistent_runtime import PersistentSalesRuntime
from app.application.sales_runtime.recovery import RecoveryStatus
from app.application.sales_runtime.serializers import (
    BUSINESS_CONVERSATION_KEY,
    LANGUAGE_PROFILE_KEY,
    BusinessConversationSerializer,
    LanguageProfileSerializer,
    QualificationSerializer,
)
from app.application.sales_runtime.snapshot import (
    SNAPSHOT_VERSION_CURRENT,
    ConversationRuntimeSnapshot,
)
from app.conversation.business_conversation.contracts import (
    BusinessConversationSnapshot,
    BusinessFactKind,
    ObservedBusinessFact,
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
from app.conversation.understanding.contracts import (
    ConversationalRegister,
    LanguageProfile,
    LanguageScript,
)
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
from app.persistence.sqlite.connection import SqliteConnection
from app.precall.contracts import PreCallDiscoveryPolicy, ServiceDiscoveryRule
from app.runtime.chat.bootstrap.contracts import SessionBootstrapInput
from app.runtime.chat.contracts import ChatSession, RuntimeMode
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
# Scripted processor that emits real payload envelopes
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


class PayloadEmittingProcessor:
    """Processor that emits REAL BCI + LanguageProfile payloads on export.

    Each turn appends a BCI fact that captures observed truth, and the
    LanguageProfile reflects the current response preference. These are
    the two REQUIRED_FOR_ACTIVE_RECOVERY payloads today.
    """

    def __init__(
        self,
        session_id: str,
        outputs: tuple[tuple[str, bool, str | None], ...],
        turn_mutations: tuple[dict, ...] = (),
        bci: BusinessConversationSnapshot | None = None,
        language: LanguageProfile | None = None,
    ) -> None:
        self._session_id = session_id
        self._outputs = outputs
        self._turn_mutations = turn_mutations
        self.bci = bci or BusinessConversationSnapshot()
        self.language = language or LanguageProfile(primary_language="en")
        self._bci_ser = BusinessConversationSerializer()
        self._lang_ser = LanguageProfileSerializer()

    def process_turn(self, turn: CoordinatedUserTurn) -> CoordinatedTurnOutput:
        idx = turn.sequence_number - 1
        if idx < len(self._turn_mutations):
            m = self._turn_mutations[idx]
            if "add_fact" in m:
                kind, value = m["add_fact"]
                new_fact = ObservedBusinessFact(
                    kind=BusinessFactKind(kind),
                    value=value,
                    source_turn_id=turn.turn_id,
                )
                self.bci = replace(self.bci, facts=self.bci.facts + (new_fact,))
            if "language" in m:
                self.language = m["language"]

        text, terminal, outcome = (
            self._outputs[idx] if idx < len(self._outputs)
            else ("ok", False, None)
        )
        return CoordinatedTurnOutput(
            turn_id=turn.turn_id,
            response_plan=_plan(),
            rendered_response=_rendered(text),
            pipeline_outcome=(
                AuthoritativeResultKind(outcome) if outcome else None
            ),
            conversation_terminal=terminal,
        )

    def export_snapshot(self, session_revision: int) -> ConversationRuntimeSnapshot:
        return ConversationRuntimeSnapshot(
            snapshot_version=SNAPSHOT_VERSION_CURRENT,
            session_id=self._session_id,
            source_session_revision=session_revision,
            conversation_state=ConversationState.LISTEN,
            sales_stage="qualification",
            persisted_contract_payloads=(
                (BUSINESS_CONVERSATION_KEY, self._bci_ser.serialize(self.bci)),
                (LANGUAGE_PROFILE_KEY, self._lang_ser.serialize(self.language)),
            ),
        )

    @classmethod
    def restore(
        cls,
        session_id: str,
        outputs: tuple[tuple[str, bool, str | None], ...],
        snapshot: ConversationRuntimeSnapshot,
        turn_mutations: tuple[dict, ...] = (),
    ) -> "PayloadEmittingProcessor":
        bci_env = snapshot.payload(BUSINESS_CONVERSATION_KEY)
        lang_env = snapshot.payload(LANGUAGE_PROFILE_KEY)
        if bci_env is None or lang_env is None:
            raise ValueError("required payloads missing")
        bci = BusinessConversationSerializer().deserialize(bci_env)
        language = LanguageProfileSerializer().deserialize(lang_env)
        return cls(session_id, outputs, turn_mutations=turn_mutations,
                   bci=bci, language=language)


class PayloadFactory:
    def __init__(self) -> None:
        self._scripts: dict = {}

    def register(self, session_id: str, outputs, mutations=()) -> None:
        self._scripts[session_id] = (outputs, mutations)

    def create_for_session(self, session, campaign, policy):
        outputs, muts = self._scripts.get(
            session.session_id, ((("ok", False, None),), ()),
        )
        return PayloadEmittingProcessor(
            session.session_id, outputs, turn_mutations=muts,
        )

    def create_from_snapshot(self, session, campaign, policy, snapshot):
        outputs, muts = self._scripts.get(
            session.session_id, ((("ok", False, None),), ()),
        )
        return PayloadEmittingProcessor.restore(
            session.session_id, outputs, snapshot, turn_mutations=muts,
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
        business_name="ABC Services", phone=_phone(),
        category=None, city=None, state=None, country=None,
        address=None, email=None, contact_name=None,
        website=None, website_status=WebsiteStatus.UNKNOWN,
    )


def _queue() -> LeadQueueEntry:
    return LeadQueueEntry(
        lead_id="lead-1", tenant_id="t1", campaign_id="c1",
        queue_position=0, status=LeadQueueStatus.READY,
    )


def _campaign() -> CampaignConfig:
    return CampaignConfig(
        campaign_id="c1", tenant_id="t1", name="Test",
        status=CampaignStatus.ACTIVE,
        authorized_service_ids=("svc-web", "svc-automation"),
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
        trusted_time_iso="2026-10-15T10:00:00Z",
    )


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "slice46.db")


def _make_runtime(db_path, factory=None):
    conn = SqliteConnection(db_path)
    f = factory or PayloadFactory()
    rt = PersistentSalesRuntime(conn, f)
    if rt.leads.get("t1", "c1", "lead-1") is None:
        rt.leads.save(_lead())
    if rt.queue.get("t1", "c1", "lead-1") is None:
        rt.queue.save(_queue())
    if rt.policies.resolve("c1") is None:
        rt.policies.register_default("c1", _policy())
    return rt


# ---------------------------------------------------------------------------
# BCI survives restart
# ---------------------------------------------------------------------------


class TestBCISurvivesRestart:
    def test_website_fact_remains_observed_after_restart(self, db_path) -> None:
        factory = PayloadFactory()
        rt = _make_runtime(db_path, factory)
        factory.register("sess-1", (
            ("ok, you have a website", False, None),
            ("tell me about workflow", False, None),
            ("bye", True, None),
        ), mutations=(
            {"add_fact": ("website", "we already have a website")},
            {"add_fact": ("manual_step", "excel tracking")},
            {},
        ))
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-1", "we have a website", 1, "tr-1")
        rt.process_turn("sess-1", "manual tracking in excel", 2, "tr-2")

        # Restart
        factory2 = PayloadFactory()
        factory2.register("sess-1", (
            ("ok, you have a website", False, None),
            ("tell me about workflow", False, None),
            ("bye", True, None),
        ))
        rt2 = _make_runtime(db_path, factory2)
        recovery = rt2.restore_processor_if_possible("sess-1", _campaign())
        assert recovery.status == RecoveryStatus.RECOVERED_ACTIVE

        restored = rt2._processors["sess-1"]
        fact_values = {f.value for f in restored.bci.facts}
        assert "we already have a website" in fact_values
        assert "excel tracking" in fact_values

    def test_bci_payload_envelope_present(self, db_path) -> None:
        factory = PayloadFactory()
        rt = _make_runtime(db_path, factory)
        factory.register("sess-1", (("r1", False, None),))
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-1", "hi", 1, "tr-1")

        snap = rt.snapshots.load("sess-1")
        assert snap is not None
        payload_keys = {k for k, _ in snap.persisted_contract_payloads}
        assert BUSINESS_CONVERSATION_KEY in payload_keys


# ---------------------------------------------------------------------------
# Language survives restart
# ---------------------------------------------------------------------------


class TestLanguageSurvivesRestart:
    def test_language_switch_preserved(self, db_path) -> None:
        factory = PayloadFactory()
        rt = _make_runtime(db_path, factory)
        urdu_profile = LanguageProfile(
            primary_language="ur",
            secondary_language="en",
            mixed_language=True,
            script=LanguageScript.LATIN,
            conversational_register=ConversationalRegister.CASUAL,
        )
        factory.register("sess-1", (
            ("ok", False, None),
            ("roman urdu mein", False, None),
        ), mutations=(
            {},
            {"language": urdu_profile},
        ))
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-1", "hi", 1, "tr-1")
        rt.process_turn("sess-1", "roman urdu mein baat karein", 2, "tr-2")

        # Restart and verify language preserved
        factory2 = PayloadFactory()
        factory2.register("sess-1", (("ok", False, None), ("ok", False, None)))
        rt2 = _make_runtime(db_path, factory2)
        rt2.restore_processor_if_possible("sess-1", _campaign())
        restored = rt2._processors["sess-1"]
        assert restored.language.primary_language == "ur"
        assert restored.language.mixed_language is True


# ---------------------------------------------------------------------------
# Required payload missing — fail closed
# ---------------------------------------------------------------------------


class TestRequiredPayloadMissing:
    def test_missing_bci_blocks_recovery(self, db_path) -> None:
        # Processor emits only LanguageProfile, no BCI → manifest violation
        class LangOnlyProcessor:
            def __init__(self, session_id):
                self._sid = session_id
            def process_turn(self, turn):
                return CoordinatedTurnOutput(
                    turn_id=turn.turn_id, response_plan=_plan(),
                    rendered_response=_rendered("ok"),
                )
            def export_snapshot(self, revision):
                lp = LanguageProfile(primary_language="en")
                return ConversationRuntimeSnapshot(
                    snapshot_version=SNAPSHOT_VERSION_CURRENT,
                    session_id=self._sid,
                    source_session_revision=revision,
                    conversation_state=ConversationState.LISTEN,
                    persisted_contract_payloads=(
                        (LANGUAGE_PROFILE_KEY, LanguageProfileSerializer().serialize(lp)),
                    ),
                )

        class _F:
            def create_for_session(self, session, c, p):
                return LangOnlyProcessor(session.session_id)
            def create_from_snapshot(self, session, c, p, s):
                return LangOnlyProcessor(session.session_id)

        rt = _make_runtime(db_path, _F())
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-1", "hi", 1, "tr-1")

        rt2 = _make_runtime(db_path, _F())
        result = rt2.recover_session("sess-1")
        assert result.status == RecoveryStatus.REQUIRED_PAYLOAD_MISSING
        assert result.diagnostics.missing_required_payload == BUSINESS_CONVERSATION_KEY


# ---------------------------------------------------------------------------
# Unsupported payload version — fail closed
# ---------------------------------------------------------------------------


class TestUnsupportedPayloadVersion:
    def test_future_bci_version_rejected(self, db_path) -> None:
        factory = PayloadFactory()
        rt = _make_runtime(db_path, factory)
        factory.register("sess-1", (("r1", False, None),))
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-1", "hi", 1, "tr-1")

        # Corrupt the stored BCI envelope to a future version
        import json
        snap = rt.snapshots.load("sess-1")
        new_payloads = []
        for k, v in snap.persisted_contract_payloads:
            if k == BUSINESS_CONVERSATION_KEY:
                env = json.loads(v)
                env["version"] = 999
                v = json.dumps(env)
            new_payloads.append((k, v))
        from dataclasses import replace as _replace
        rt.snapshots.save(_replace(
            snap, persisted_contract_payloads=tuple(new_payloads),
        ))

        rt2 = _make_runtime(db_path, PayloadFactory())
        result = rt2.recover_session("sess-1")
        assert result.status == RecoveryStatus.UNSUPPORTED_PAYLOAD_VERSION
        assert result.diagnostics.unsupported_payload == BUSINESS_CONVERSATION_KEY


# ---------------------------------------------------------------------------
# Outcome equivalence with real payloads
# ---------------------------------------------------------------------------


class TestOutcomeEquivalenceWithPayloads:
    def test_outcome_identical_across_restart(self, tmp_path) -> None:
        outputs = (
            ("ok", False, None),
            ("tell me", False, None),
            ("done", True, None),
        )
        mutations = (
            {"add_fact": ("website", "has website")},
            {"add_fact": ("manual_step", "manual workflow")},
            {},
        )

        # Run A: no restart
        rt_a = _make_runtime(str(tmp_path / "a.db"))
        fa = PayloadFactory()
        rt_a._processor_factory = fa  # type: ignore[attr-defined]
        fa.register("sess-A", outputs, mutations)
        rt_a.start_session(
            _boot_input(session_id="sess-A", req="req-A"), _campaign(),
        )
        for i, m in enumerate(["m1", "m2", "m3"], start=1):
            rt_a.process_turn("sess-A", m, i, f"trA-{i}")
        outcome_a = rt_a.finish_session("sess-A").lead_outcome

        # Run B: restart after turn 2
        db_b = str(tmp_path / "b.db")
        rt_b = _make_runtime(db_b)
        fb = PayloadFactory()
        rt_b._processor_factory = fb  # type: ignore[attr-defined]
        fb.register("sess-B", outputs, mutations)
        rt_b.start_session(
            _boot_input(session_id="sess-B", req="req-B"), _campaign(),
        )
        rt_b.process_turn("sess-B", "m1", 1, "trB-1")
        rt_b.process_turn("sess-B", "m2", 2, "trB-2")

        # Restart
        rt_b2 = _make_runtime(db_b)
        fb2 = PayloadFactory()
        rt_b2._processor_factory = fb2  # type: ignore[attr-defined]
        fb2.register("sess-B", outputs, mutations)
        rt_b2.restore_processor_if_possible("sess-B", _campaign())
        rt_b2.process_turn("sess-B", "m3", 3, "trB-3")
        outcome_b = rt_b2.finish_session("sess-B").lead_outcome

        assert outcome_a.outcome_status == outcome_b.outcome_status
        assert outcome_a.interest_level == outcome_b.interest_level
        assert outcome_a.dnc_status == outcome_b.dnc_status
        assert outcome_a.execution_status == outcome_b.execution_status
        assert outcome_a.termination_reason == outcome_b.termination_reason


# ---------------------------------------------------------------------------
# Legacy snapshot (no payloads) still blocks ACTIVE recovery
# ---------------------------------------------------------------------------


class TestLegacySnapshotBlocksActiveRecovery:
    def test_legacy_payloadless_snapshot_blocks(self, db_path) -> None:
        from dataclasses import replace as _replace
        factory = PayloadFactory()
        rt = _make_runtime(db_path, factory)
        factory.register("sess-1", (("r1", False, None),))
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-1", "hi", 1, "tr-1")
        snap = rt.snapshots.load("sess-1")
        # Strip all payloads to simulate a Slice-45-only snapshot
        rt.snapshots.save(_replace(snap, persisted_contract_payloads=()))

        rt2 = _make_runtime(db_path, PayloadFactory())
        result = rt2.recover_session("sess-1")
        assert result.status == RecoveryStatus.REQUIRED_PAYLOAD_MISSING


# ---------------------------------------------------------------------------
# DNC still highest priority with payloads present
# ---------------------------------------------------------------------------


class TestDNCWithPayloads:
    def test_dnc_survives_despite_payload_recovery(self, db_path) -> None:
        factory = PayloadFactory()
        rt = _make_runtime(db_path, factory)
        factory.register("sess-1", (("r1", False, None),))
        rt.start_session(_boot_input(), _campaign())
        rt.process_turn("sess-1", "hi", 1, "tr-1")
        rt.mark_lead_dnc("t1", "c1", "lead-1")

        rt2 = _make_runtime(db_path, PayloadFactory())
        assert rt2.queue.get("t1", "c1", "lead-1").is_dnc is True
