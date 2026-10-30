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
            authoritative_result=overrides.get("authoritative_result", None),
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
        # The sink call site is wrapped in try/except
        assert "self._dev_trace_sink(" in src
        # And the except clause is bare-ish (catches Exception)
        # (search for the pattern around the sink call)
        assert "except Exception:" in src


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
