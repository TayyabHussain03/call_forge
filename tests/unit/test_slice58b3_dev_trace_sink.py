"""Slice 58B.3 — dev-only typed trace sink.

Opt-in via `DEV_LIVE_TRACE=1`. When enabled, ProductionTurnProcessor
emits a bounded structured record per turn to the configured sink so
the operator can see:
    - the live BrainProposal classification
    - the scope/authority decisions
    - the plan's communicative goal + clarification flag
    - which approved action the engine executed
    - whether the turn committed as terminal
    - the final rendered text

Scope boundaries:
    - sink is opt-in; default is None and no behavior change
    - sink NEVER crosses the HTTP/WS public boundary
    - trace excludes credentials, raw prompts, chain-of-thought
    - sink failure must NOT corrupt the live turn
    - this slice is purely diagnostic — no runtime semantics change
"""

from __future__ import annotations

import inspect
import io
import json
import os
from unittest.mock import patch

import pytest


class TestEnvironmentToggle:
    def test_disabled_by_default(self, monkeypatch):
        from app.observability.dev_trace import is_dev_trace_enabled, resolve_dev_trace_sink
        monkeypatch.delenv("DEV_LIVE_TRACE", raising=False)
        assert is_dev_trace_enabled() is False
        assert resolve_dev_trace_sink() is None

    @pytest.mark.parametrize("value", ["1", "true", "True", "YES", "on"])
    def test_enabled_by_truthy_env(self, monkeypatch, value):
        from app.observability.dev_trace import is_dev_trace_enabled, resolve_dev_trace_sink
        monkeypatch.setenv("DEV_LIVE_TRACE", value)
        assert is_dev_trace_enabled() is True
        assert resolve_dev_trace_sink() is not None

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_falsy_values_disabled(self, monkeypatch, value):
        from app.observability.dev_trace import is_dev_trace_enabled
        monkeypatch.setenv("DEV_LIVE_TRACE", value)
        assert is_dev_trace_enabled() is False


class TestStderrSinkFormat:
    def test_emits_single_json_line_to_stderr(self, capsys):
        from app.observability.dev_trace import make_stderr_dev_trace_sink
        sink = make_stderr_dev_trace_sink()
        sink({"turn_id": "t1", "foo": "bar"})
        captured = capsys.readouterr()
        assert captured.err.startswith("[dev-trace] ")
        body = captured.err[len("[dev-trace] "):].strip()
        parsed = json.loads(body)
        assert parsed["turn_id"] == "t1"
        assert parsed["foo"] == "bar"

    def test_sink_handles_enum_via_str_default(self, capsys):
        """`default=str` on json.dumps keeps enum/path serialization safe."""
        from enum import Enum
        from app.observability.dev_trace import make_stderr_dev_trace_sink

        class _E(str, Enum):
            A = "alpha"
        sink = make_stderr_dev_trace_sink()
        sink({"kind": _E.A})
        captured = capsys.readouterr()
        parsed = json.loads(captured.err[len("[dev-trace] "):])
        # str(_E.A) → "_E.A" works via default; passing as str Enum value
        # also works. Either way, no exception.
        assert "kind" in parsed


class TestBuildDevTrace:
    """The _build_dev_trace helper serializes bounded typed fields.

    We test it via the processor-side helper rather than reconstructing
    a full orchestrator because the point is to prove the record SHAPE
    + absence of forbidden content, not the orchestrator logic."""

    def _build(self, **overrides):
        from app.application.production_turn_processor import _build_dev_trace
        from types import SimpleNamespace

        turn = SimpleNamespace(
            turn_id=overrides.get("turn_id", "chat_sess_t1"),
            sequence_number=overrides.get("sequence_number", 1),
            utterance=overrides.get("utterance", "hello"),
        )
        trace = SimpleNamespace(
            stage_outcome=overrides.get("stage_outcome", None),
            provider_name=overrides.get("provider_name", "gemini"),
            proposal_status=overrides.get("proposal_status", None),
            scope_category=overrides.get("scope_category", None),
            authority_tier=overrides.get("authority_tier", None),
            action_confidence=overrides.get("action_confidence", 0.8),
            notes=overrides.get("notes", None),
            priority_action=overrides.get("priority_action", None),
        )
        s2 = SimpleNamespace(
            outcome=overrides.get("s2_outcome", None),
            conversation_status=overrides.get("s2_status", None),
            trace=trace,
        )
        plan = SimpleNamespace(
            communicative_goal=overrides.get("communicative_goal", None),
            response_length=overrides.get("response_length", None),
            question_strategy=overrides.get("question_strategy", None),
            clarification_required=overrides.get("clarification_required", False),
            addressee_status=overrides.get("addressee_status", None),
            catalog_request=overrides.get("catalog_request", False),
            re_explanation_strategy=overrides.get("re_explanation_strategy", None),
            escalation_decision=overrides.get("escalation_decision", None),
        )
        output = SimpleNamespace(
            # Slice 58B.3b — correct contract: CoordinatedTurnOutput
            # exposes `pipeline_outcome`, not `authoritative_result`.
            pipeline_outcome=overrides.get("authoritative_result", None),
            conversation_terminal=overrides.get("terminal", False),
            rendered_response=SimpleNamespace(
                text=overrides.get("rendered_text", "Hi there."),
            ),
        )
        return _build_dev_trace(
            turn=turn, slice_two=s2, slice_three=None,
            plan=plan, output=output,
            reasoning_calls=overrides.get("reasoning_calls", 1),
        )

    def test_record_shape_includes_all_diagnostic_fields(self):
        rec = self._build()
        # Top-level
        assert rec["turn_id"] == "chat_sess_t1"
        assert rec["turn_sequence"] == 1
        assert rec["utterance"] == "hello"
        assert rec["reasoning_calls_total"] == 1
        # Slice two
        assert "slice_two" in rec
        for key in (
            "outcome", "conversation_status", "stage_outcome",
            "provider_name", "proposal_status", "scope_category",
            "authority_tier", "action_confidence", "notes", "priority_action",
        ):
            assert key in rec["slice_two"]
        # Plan
        for key in (
            "communicative_goal", "response_length", "question_strategy",
            "clarification_required", "addressee_status", "catalog_request",
            "re_explanation_strategy", "has_escalation_decision",
        ):
            assert key in rec["plan"]
        # Output
        assert "authoritative_result" in rec
        assert "conversation_terminal" in rec
        assert "rendered_text" in rec

    def test_utterance_bounded_to_200_chars(self):
        rec = self._build(utterance="x" * 500)
        assert len(rec["utterance"]) == 200

    def test_rendered_text_bounded_to_300_chars(self):
        rec = self._build(rendered_text="y" * 500)
        assert len(rec["rendered_text"]) == 300

    def test_record_contains_no_credentials_or_prompt(self):
        """Record never carries API keys, raw prompts, or chain-of-thought
        — the helper reads only typed fields the orchestrator exposes."""
        rec = self._build()
        serialized = json.dumps(rec, default=str).lower()
        for forbidden in (
            "api_key", "gemini_api_key", "secret",
            "chain_of_thought", "chainofthought",
            "raw_prompt", "system_instruction", "automatic_function_calling",
        ):
            assert forbidden not in serialized


class TestProcessorAcceptsSinkWithoutBreaking:
    def test_init_accepts_dev_trace_sink_param(self):
        import inspect
        from app.application.production_turn_processor import ProductionTurnProcessor
        sig = inspect.signature(ProductionTurnProcessor.__init__)
        assert "dev_trace_sink" in sig.parameters
        # Default is None (opt-in)
        assert sig.parameters["dev_trace_sink"].default is None

    def test_sink_failure_does_not_break_turn(self):
        """A broken sink must NEVER raise into live-turn flow."""
        import inspect
        from app.application import production_turn_processor as mod
        src = inspect.getsource(mod)
        # The sink call site is wrapped via _emit_dev_trace_safely
        assert "_emit_dev_trace_safely" in src
        assert "_emit_dev_trace_error_safely" in src
        assert "except Exception" in src


class TestNoHttpOrWsExposure:
    def test_dev_trace_module_has_no_fastapi_imports(self):
        """§3, §19 — sink is development-only; never crosses the HTTP/WS
        public boundary."""
        from app.observability import dev_trace as mod
        src = inspect.getsource(mod)
        for forbidden in (
            "fastapi", "FastAPI", "starlette", "WebSocket", "APIRouter",
        ):
            assert forbidden.lower() not in src.lower() or forbidden == "FastAPI" and "FastAPI" not in src

    def test_api_route_modules_do_not_import_dev_trace(self):
        from app.api.routes import sessions
        from app.api.websocket import chat
        for mod in (sessions, chat):
            src = inspect.getsource(mod)
            assert "dev_trace" not in src
            assert "observability" not in src


class TestSafeEmitHelpers:
    """Slice 58B.3b — explicit coverage for the two helper methods
    that replaced the ad-hoc try/except around the sink call."""

    def _make_processor_shell(self, sink):
        """Minimal shell exposing just the helpers under test."""
        from app.application.production_turn_processor import ProductionTurnProcessor
        # Build a ProductionTurnProcessor-like shell: we don't need the
        # full orchestrator pipeline for helper-level tests. We bind
        # just the two helpers to a bare object with `_dev_trace_sink`
        # and `_reasoning_calls`.
        class _Shell:
            pass
        shell = _Shell()
        shell._dev_trace_sink = sink
        shell._reasoning_calls = 0
        shell._emit_dev_trace_safely = (
            ProductionTurnProcessor._emit_dev_trace_safely.__get__(shell)
        )
        shell._emit_dev_trace_error_safely = (
            ProductionTurnProcessor._emit_dev_trace_error_safely.__get__(shell)
        )
        return shell

    def _dummy_inputs(self):
        from types import SimpleNamespace
        turn = SimpleNamespace(turn_id="t", sequence_number=1, utterance="hi")
        slice_two = SimpleNamespace(
            outcome=None, conversation_status=None,
            trace=SimpleNamespace(
                stage_outcome=None, provider_name=None,
                proposal_status=None, scope_category=None,
                authority_tier=None, action_confidence=None,
                notes=None, priority_action=None,
            ),
        )
        plan = SimpleNamespace(
            communicative_goal=None, response_length=None,
            question_strategy=None, clarification_required=False,
            addressee_status=None, catalog_request=False,
            re_explanation_strategy=None, escalation_decision=None,
        )
        output = SimpleNamespace(
            pipeline_outcome=None, conversation_terminal=False,
            rendered_response=SimpleNamespace(text="Hi!"),
        )
        return turn, slice_two, plan, output

    def test_sink_none_short_circuits_no_call(self):
        shell = self._make_processor_shell(sink=None)
        turn, s2, plan, output = self._dummy_inputs()
        # No sink → no emission; also no exception
        shell._emit_dev_trace_safely(
            turn=turn, slice_two=s2, slice_three=None,
            plan=plan, output=output,
        )

    def test_normal_emit_delivers_one_record(self):
        received = []
        shell = self._make_processor_shell(sink=received.append)
        turn, s2, plan, output = self._dummy_inputs()
        shell._emit_dev_trace_safely(
            turn=turn, slice_two=s2, slice_three=None,
            plan=plan, output=output,
        )
        assert len(received) == 1
        assert received[0]["turn_id"] == "t"
        # Correct contract field
        assert received[0]["authoritative_result"] is None

    def test_builder_failure_emits_dev_trace_error_not_silence(self):
        """§6, §15 — silent failure under DEV_LIVE_TRACE is unacceptable."""
        received = []
        shell = self._make_processor_shell(sink=received.append)
        turn, s2, plan, output = self._dummy_inputs()
        # Force builder to crash by passing an output missing
        # pipeline_outcome entirely (AttributeError inside builder)
        from types import SimpleNamespace
        bad_output = SimpleNamespace(
            # no pipeline_outcome, no conversation_terminal, no rendered_response
        )
        shell._emit_dev_trace_safely(
            turn=turn, slice_two=s2, slice_three=None,
            plan=plan, output=bad_output,
        )
        assert len(received) == 1
        assert received[0].get("__dev_trace_error__") is True
        assert received[0]["phase"] == "build"
        assert received[0]["error_type"] == "AttributeError"
        # Message bounded
        assert len(received[0]["message"]) <= 200

    def test_sink_failure_emits_dev_trace_error(self):
        received = []
        def _flaky(record):
            received.append(record)
            if not record.get("__dev_trace_error__"):
                raise RuntimeError("downstream sink boom")
        shell = self._make_processor_shell(sink=_flaky)
        turn, s2, plan, output = self._dummy_inputs()
        shell._emit_dev_trace_safely(
            turn=turn, slice_two=s2, slice_three=None,
            plan=plan, output=output,
        )
        # First the normal record, then the error record
        assert len(received) == 2
        assert received[0].get("__dev_trace_error__", False) is False
        assert received[1].get("__dev_trace_error__") is True
        assert received[1]["phase"] == "sink"
        assert received[1]["error_type"] == "RuntimeError"

    def test_error_reporter_failure_is_swallowed_no_recursion(self):
        """§8 — if the error reporter itself fails, swallow. No recursion."""
        calls = []
        def _always_fails(record):
            calls.append(record)
            raise RuntimeError("sink always fails")
        shell = self._make_processor_shell(sink=_always_fails)
        turn, s2, plan, output = self._dummy_inputs()
        # Does not raise even though sink always fails
        shell._emit_dev_trace_safely(
            turn=turn, slice_two=s2, slice_three=None,
            plan=plan, output=output,
        )
        # Exactly two attempts: normal record + error record. No recursion.
        assert len(calls) == 2

    def test_error_record_contains_no_credentials(self):
        received = []
        shell = self._make_processor_shell(sink=received.append)
        turn, s2, plan, output = self._dummy_inputs()
        # Build the error record directly via helper and inspect
        shell._emit_dev_trace_error_safely(
            received.append,
            RuntimeError("some bounded message"),
            phase="test",
        )
        err_rec = received[-1]
        serialized = json.dumps(err_rec).lower()
        for forbidden in (
            "api_key", "secret", "chain_of_thought", "traceback",
        ):
            assert forbidden not in serialized


class TestMainBootstrapWiresSinkConditionally:
    @pytest.fixture(scope="class")
    def main_source(self) -> str:
        from pathlib import Path
        return (Path(__file__).resolve().parents[2] / "app" / "main.py").read_text(
            encoding="utf-8",
        )

    def test_main_imports_resolver(self, main_source):
        assert "from app.observability.dev_trace import resolve_dev_trace_sink" in main_source

    def test_main_wires_sink_via_resolver(self, main_source):
        assert "dev_trace_sink=resolve_dev_trace_sink()" in main_source
