"""Comprehensive tests for Chat Sales Runtime & Simulation Harness."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from app.conversation.response_planning.contracts import (
    AcknowledgementKind,
    AddresseeStatus,
    AuthoritativeResultKind,
    ConversationMove,
    InterruptionCategory,
    InterruptionContext,
    InterruptionHandling,
    QuestionStrategy,
    ResponseLength,
    ResponsePlan,
)
from app.conversation.response_rendering.contracts import RenderedResponse
from app.core.constants import Tone
from app.runtime.contracts import CoordinatedTurnOutput, CoordinatedUserTurn, TurnProcessor
from app.runtime.chat.contracts import (
    ChatSession,
    ChatTurnInput,
    ChatTurnResult,
    RuntimeMode,
    SalesSimulationScenario,
    SessionStatus,
    TranscriptEntry,
    TranscriptRole,
    TurnDiagnosticTrace,
)
from app.runtime.chat.engine import (
    ChatRuntime,
    DuplicateTurnError,
    SessionTerminatedError,
)
from app.runtime.chat.repository import (
    InMemoryChatSessionRepository,
    SessionNotFoundError,
    StaleRevisionError,
)


def _mock_rendered(text: str = "Hello") -> RenderedResponse:
    return RenderedResponse(
        text=text,
        communicative_goal=ConversationMove.COMMUNICATE_RESULT,
        length_class=ResponseLength.SHORT,
        clarification_required=False,
    )


def _mock_plan() -> ResponsePlan:
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


# ---------------------------------------------------------------------------
# Scripted mock processor — deterministic, dumb, transparent
# ---------------------------------------------------------------------------


class ScriptedTurnProcessor:
    """Returns scripted responses in order. No intelligence, no authority."""

    def __init__(self, responses: tuple[str, ...] = ("Hello, how can I help?",)) -> None:
        self._responses = responses
        self._call_count = 0

    def process_turn(self, turn: CoordinatedUserTurn) -> CoordinatedTurnOutput:
        text = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        terminal = text.startswith("[TERMINAL]")
        plan = _mock_plan()
        return CoordinatedTurnOutput(
            turn_id=turn.turn_id,
            response_plan=plan,
            rendered_response=_mock_rendered(text),
            pipeline_outcome=AuthoritativeResultKind.AUTHORITY_APPROVED,
            conversation_terminal=terminal,
        )

    @property
    def call_count(self) -> int:
        return self._call_count


class TerminalProcessor:
    """Always returns terminal. Simulates DNC or end-of-conversation."""

    def process_turn(self, turn: CoordinatedUserTurn) -> CoordinatedTurnOutput:
        plan = _mock_plan()
        return CoordinatedTurnOutput(
            turn_id=turn.turn_id,
            response_plan=plan,
            rendered_response=_mock_rendered("Goodbye."),
            pipeline_outcome=AuthoritativeResultKind.AUTHORITY_APPROVED,
            conversation_terminal=True,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session(
    session_id: str = "sess_1",
    **overrides,
) -> ChatSession:
    values = dict(
        session_id=session_id,
        tenant_id="tenant_1",
        campaign_id="campaign_1",
        lead_id="lead_1",
    )
    values.update(overrides)
    return ChatSession(**values)


def _runtime(
    processor: TurnProcessor | None = None,
    session_id: str = "sess_1",
) -> tuple[ChatRuntime, InMemoryChatSessionRepository]:
    repo = InMemoryChatSessionRepository()
    repo.create(_session(session_id=session_id))
    runtime = ChatRuntime(repo)
    if processor is not None:
        runtime.register_processor(session_id, processor)
    return runtime, repo


def _turn(
    session_id: str = "sess_1",
    message: str = "Hi there",
    sequence: int = 1,
) -> ChatTurnInput:
    return ChatTurnInput(
        session_id=session_id,
        message=message,
        turn_sequence=sequence,
    )


# ===========================================================================
# Contract validation
# ===========================================================================


class TestChatSessionContract:
    def test_valid_session(self):
        s = _session()
        assert s.session_id == "sess_1"
        assert s.status == SessionStatus.ACTIVE
        assert s.revision == 0

    def test_empty_session_id_rejected(self):
        with pytest.raises(ValueError, match="session_id"):
            _session(session_id="")

    def test_empty_tenant_rejected(self):
        with pytest.raises(ValueError, match="tenant_id"):
            _session(tenant_id="")

    def test_empty_campaign_rejected(self):
        with pytest.raises(ValueError, match="campaign_id"):
            _session(campaign_id="")

    def test_empty_lead_rejected(self):
        with pytest.raises(ValueError, match="lead_id"):
            _session(lead_id="")

    def test_frozen(self):
        s = _session()
        with pytest.raises(FrozenInstanceError):
            s.status = SessionStatus.COMPLETED


class TestChatTurnInputContract:
    def test_valid_input(self):
        t = _turn()
        assert t.message == "Hi there"

    def test_empty_message_rejected(self):
        with pytest.raises(ValueError, match="message"):
            _turn(message="")

    def test_whitespace_message_rejected(self):
        with pytest.raises(ValueError, match="message"):
            _turn(message="   ")

    def test_long_message_rejected(self):
        with pytest.raises(ValueError, match="maximum length"):
            _turn(message="x" * 2001)

    def test_negative_sequence_rejected(self):
        with pytest.raises(ValueError, match="turn_sequence"):
            _turn(sequence=-1)

    def test_frozen(self):
        t = _turn()
        with pytest.raises(FrozenInstanceError):
            t.message = "changed"


class TestTranscriptEntry:
    def test_valid_entry(self):
        e = TranscriptEntry(turn_sequence=1, role=TranscriptRole.PROSPECT, text="Hi")
        assert e.role == TranscriptRole.PROSPECT

    def test_empty_text_rejected(self):
        with pytest.raises(ValueError, match="text"):
            TranscriptEntry(turn_sequence=0, role=TranscriptRole.AGENT, text="")

    def test_frozen(self):
        e = TranscriptEntry(turn_sequence=0, role=TranscriptRole.AGENT, text="Hi")
        with pytest.raises(FrozenInstanceError):
            e.text = "changed"


class TestTurnDiagnosticTrace:
    def test_valid_trace(self):
        t = TurnDiagnosticTrace(turn_sequence=1, conversation_state="greeting")
        assert t.is_terminal is False

    def test_frozen(self):
        t = TurnDiagnosticTrace(turn_sequence=1, conversation_state="x")
        with pytest.raises(FrozenInstanceError):
            t.is_terminal = True


class TestSalesSimulationScenario:
    def test_valid_scenario(self):
        s = SalesSimulationScenario(
            scenario_id="s1", name="Happy path", prospect_turns=("Hi", "Tell me more"),
        )
        assert len(s.prospect_turns) == 2

    def test_empty_turns_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            SalesSimulationScenario(scenario_id="s1", name="Empty", prospect_turns=())

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="name"):
            SalesSimulationScenario(scenario_id="s1", name="", prospect_turns=("Hi",))


# ===========================================================================
# Repository
# ===========================================================================


class TestRepository:
    def test_create_and_get(self):
        repo = InMemoryChatSessionRepository()
        created = repo.create(_session())
        assert created.revision == 0
        got = repo.get("sess_1")
        assert got.session_id == "sess_1"

    def test_duplicate_create_rejected(self):
        repo = InMemoryChatSessionRepository()
        repo.create(_session())
        with pytest.raises(ValueError, match="already exists"):
            repo.create(_session())

    def test_get_missing_raises(self):
        repo = InMemoryChatSessionRepository()
        with pytest.raises(SessionNotFoundError):
            repo.get("nonexistent")

    def test_update_increments_revision(self):
        repo = InMemoryChatSessionRepository()
        created = repo.create(_session())
        updated = repo.update(replace(created, turn_count=1))
        assert updated.revision == 1

    def test_stale_revision_rejected(self):
        repo = InMemoryChatSessionRepository()
        created = repo.create(_session())
        repo.update(replace(created, turn_count=1))
        with pytest.raises(StaleRevisionError):
            repo.update(replace(created, turn_count=2))

    def test_exists(self):
        repo = InMemoryChatSessionRepository()
        assert repo.exists("sess_1") is False
        repo.create(_session())
        assert repo.exists("sess_1") is True

    def test_list_sessions(self):
        repo = InMemoryChatSessionRepository()
        repo.create(_session("s1"))
        repo.create(_session("s2", tenant_id="other"))
        assert len(repo.list_sessions()) == 2
        assert len(repo.list_sessions(tenant_id="tenant_1")) == 1

    def test_active_count(self):
        repo = InMemoryChatSessionRepository()
        repo.create(_session("s1"))
        s2 = repo.create(_session("s2"))
        repo.update(replace(s2, status=SessionStatus.COMPLETED))
        assert repo.active_count() == 1


# ===========================================================================
# Runtime — single turn
# ===========================================================================


class TestSingleTurn:
    def test_process_turn_returns_response(self):
        proc = ScriptedTurnProcessor()
        runtime, _ = _runtime(proc)
        result = runtime.process_turn(_turn())
        assert result.agent_response == "Hello, how can I help?"
        assert result.conversation_terminal is False
        assert proc.call_count == 1

    def test_turn_updates_transcript(self):
        proc = ScriptedTurnProcessor()
        runtime, _ = _runtime(proc)
        runtime.process_turn(_turn())
        transcript = runtime.get_transcript("sess_1")
        assert len(transcript) == 2
        assert transcript[0].role == TranscriptRole.PROSPECT
        assert transcript[1].role == TranscriptRole.AGENT

    def test_turn_updates_session_turn_count(self):
        proc = ScriptedTurnProcessor()
        runtime, _ = _runtime(proc)
        runtime.process_turn(_turn(sequence=1))
        session = runtime.get_session("sess_1")
        assert session.turn_count == 1

    def test_diagnostics_present(self):
        proc = ScriptedTurnProcessor()
        runtime, _ = _runtime(proc)
        result = runtime.process_turn(_turn())
        assert result.diagnostics is not None
        assert result.diagnostics.turn_sequence == 1


# ===========================================================================
# Runtime — multi-turn
# ===========================================================================


class TestMultiTurn:
    def test_sequential_turns(self):
        proc = ScriptedTurnProcessor(("Response 1", "Response 2"))
        runtime, _ = _runtime(proc)
        r1 = runtime.process_turn(_turn(sequence=1))
        r2 = runtime.process_turn(_turn(sequence=2, message="Next question"))
        assert r1.agent_response == "Response 1"
        assert r2.agent_response == "Response 2"
        assert proc.call_count == 2

    def test_transcript_accumulates(self):
        proc = ScriptedTurnProcessor(("R1", "R2"))
        runtime, _ = _runtime(proc)
        runtime.process_turn(_turn(sequence=1))
        runtime.process_turn(_turn(sequence=2, message="More"))
        transcript = runtime.get_transcript("sess_1")
        assert len(transcript) == 4


# ===========================================================================
# Runtime — terminal handling
# ===========================================================================


class TestTerminalHandling:
    def test_terminal_completes_session(self):
        proc = TerminalProcessor()
        runtime, _ = _runtime(proc)
        result = runtime.process_turn(_turn())
        assert result.conversation_terminal is True
        session = runtime.get_session("sess_1")
        assert session.status == SessionStatus.COMPLETED

    def test_turn_after_terminal_rejected(self):
        proc = TerminalProcessor()
        runtime, _ = _runtime(proc)
        runtime.process_turn(_turn(sequence=1))
        with pytest.raises(SessionTerminatedError):
            runtime.process_turn(_turn(sequence=2, message="More"))


# ===========================================================================
# Runtime — duplicate turn prevention
# ===========================================================================


class TestDuplicatePrevention:
    def test_duplicate_sequence_rejected(self):
        proc = ScriptedTurnProcessor()
        runtime, _ = _runtime(proc)
        runtime.process_turn(_turn(sequence=1))
        with pytest.raises(DuplicateTurnError):
            runtime.process_turn(_turn(sequence=1, message="Duplicate"))

    def test_past_sequence_rejected(self):
        proc = ScriptedTurnProcessor(("R1", "R2"))
        runtime, _ = _runtime(proc)
        runtime.process_turn(_turn(sequence=1))
        runtime.process_turn(_turn(sequence=2, message="Second"))
        with pytest.raises(DuplicateTurnError):
            runtime.process_turn(_turn(sequence=1, message="Old"))


# ===========================================================================
# Runtime — session lifecycle
# ===========================================================================


class TestSessionLifecycle:
    def test_terminate_active_session(self):
        runtime, _ = _runtime(ScriptedTurnProcessor())
        session = runtime.terminate_session("sess_1")
        assert session.status == SessionStatus.TERMINATED

    def test_terminate_already_terminated_is_noop(self):
        runtime, _ = _runtime(ScriptedTurnProcessor())
        runtime.terminate_session("sess_1")
        session = runtime.terminate_session("sess_1")
        assert session.status == SessionStatus.TERMINATED

    def test_get_session(self):
        runtime, _ = _runtime(ScriptedTurnProcessor())
        session = runtime.get_session("sess_1")
        assert session.session_id == "sess_1"

    def test_missing_session_raises(self):
        runtime, _ = _runtime(ScriptedTurnProcessor())
        with pytest.raises(SessionNotFoundError):
            runtime.get_session("nonexistent")


# ===========================================================================
# Runtime — no processor registered
# ===========================================================================


class TestNoProcessor:
    def test_missing_processor_raises(self):
        repo = InMemoryChatSessionRepository()
        repo.create(_session())
        runtime = ChatRuntime(repo)
        with pytest.raises(ValueError, match="no processor"):
            runtime.process_turn(_turn())


# ===========================================================================
# Runtime — simulation scenario
# ===========================================================================


class TestSimulationScenario:
    def test_run_scenario_processes_all_turns(self):
        proc = ScriptedTurnProcessor(("R1", "R2", "R3"))
        runtime, _ = _runtime(proc)
        results = runtime.run_scenario("sess_1", ("Hi", "More", "Done"))
        assert len(results) == 3
        assert proc.call_count == 3

    def test_scenario_stops_on_terminal(self):
        proc = ScriptedTurnProcessor(("[TERMINAL] Goodbye",))
        runtime, _ = _runtime(proc)
        results = runtime.run_scenario("sess_1", ("Hi", "More"))
        assert len(results) == 1
        assert results[0].conversation_terminal is True

    def test_scenario_transcripts(self):
        proc = ScriptedTurnProcessor(("R1", "R2"))
        runtime, _ = _runtime(proc)
        runtime.run_scenario("sess_1", ("T1", "T2"))
        transcript = runtime.get_transcript("sess_1")
        assert len(transcript) == 4
        assert transcript[0].text == "T1"
        assert transcript[1].text == "R1"


# ===========================================================================
# Turn mapping — chat input → CoordinatedUserTurn
# ===========================================================================


class TestTurnMapping:
    def test_turn_id_format(self):
        from app.runtime.chat.engine import _to_coordinated_turn

        coordinated = _to_coordinated_turn(_turn())
        assert coordinated.turn_id == "chat_sess_1_t1"
        assert coordinated.sequence_number == 1

    def test_utterance_preserved(self):
        from app.runtime.chat.engine import _to_coordinated_turn

        coordinated = _to_coordinated_turn(_turn(message="Test message"))
        assert coordinated.utterance == "Test message"

    def test_defaults_for_chat(self):
        from app.runtime.chat.engine import _to_coordinated_turn

        coordinated = _to_coordinated_turn(_turn())
        assert coordinated.addressee_status == AddresseeStatus.ADDRESSED_TO_AGENT
        assert coordinated.conversation_category == InterruptionCategory.OTHER
        assert coordinated.interruption.was_interrupted is False


# ===========================================================================
# Isolation
# ===========================================================================


class TestIsolation:
    def test_sessions_isolated(self):
        repo = InMemoryChatSessionRepository()
        repo.create(_session("s1"))
        repo.create(_session("s2"))
        proc1 = ScriptedTurnProcessor(("From session 1",))
        proc2 = ScriptedTurnProcessor(("From session 2",))
        runtime = ChatRuntime(repo)
        runtime.register_processor("s1", proc1)
        runtime.register_processor("s2", proc2)

        r1 = runtime.process_turn(_turn(session_id="s1", sequence=1))
        r2 = runtime.process_turn(_turn(session_id="s2", sequence=1))
        assert r1.agent_response == "From session 1"
        assert r2.agent_response == "From session 2"

    def test_transcript_isolation(self):
        repo = InMemoryChatSessionRepository()
        repo.create(_session("s1"))
        repo.create(_session("s2"))
        proc = ScriptedTurnProcessor(("R",))
        runtime = ChatRuntime(repo)
        runtime.register_processor("s1", proc)
        runtime.register_processor("s2", ScriptedTurnProcessor(("R2",)))

        runtime.process_turn(_turn(session_id="s1", sequence=1))
        assert len(runtime.get_transcript("s1")) == 2
        assert len(runtime.get_transcript("s2")) == 0


# ===========================================================================
# Determinism
# ===========================================================================


class TestDeterminism:
    def test_same_input_same_turn_id(self):
        from app.runtime.chat.engine import _to_coordinated_turn

        t1 = _to_coordinated_turn(_turn())
        t2 = _to_coordinated_turn(_turn())
        assert t1.turn_id == t2.turn_id

    def test_no_random_in_engine(self):
        from pathlib import Path
        import app.runtime.chat.engine as m

        source = Path(m.__file__).read_text()
        assert "uuid" not in source
        assert "random" not in source
        assert "datetime.now" not in source


# ===========================================================================
# Safety — no telephony, no LLM, no duplicate logic
# ===========================================================================


class TestSafety:
    def test_no_telephony_imports(self):
        from pathlib import Path
        import app.runtime.chat.engine as m

        source = Path(m.__file__).read_text()
        assert "twilio" not in source.lower()
        assert "telnyx" not in source.lower()
        assert "vapi" not in source.lower()

    def test_no_llm_imports(self):
        from pathlib import Path
        import app.runtime.chat.engine as m

        source = Path(m.__file__).read_text()
        assert "openai" not in source.lower()
        assert "gemini" not in source.lower()
        assert "anthropic" not in source.lower()

    def test_no_brain_logic_in_engine(self):
        from pathlib import Path
        import app.runtime.chat.engine as m

        source = Path(m.__file__).read_text()
        assert "BrainOrchestrator" not in source
        assert "StateMachine" not in source
        assert "ActionValidator" not in source

    def test_engine_immutability(self):
        result = ChatTurnResult(
            session_id="s1",
            turn_sequence=1,
            agent_response="Hi",
        )
        with pytest.raises(FrozenInstanceError):
            result.agent_response = "changed"


# ===========================================================================
# Concurrency safety via repository
# ===========================================================================


class TestConcurrency:
    def test_stale_revision_prevents_lost_update(self):
        repo = InMemoryChatSessionRepository()
        s = repo.create(_session())
        snapshot_a = repo.get("sess_1")
        snapshot_b = repo.get("sess_1")
        repo.update(replace(snapshot_a, turn_count=1))
        with pytest.raises(StaleRevisionError):
            repo.update(replace(snapshot_b, turn_count=2))
