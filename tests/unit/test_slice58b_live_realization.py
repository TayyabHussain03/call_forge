"""Slice 58B — live Gemini realization wiring.

Proves:
    (A) `create_realization_renderer` builds a `GuardedConversationRealizer`
        from Settings and the deterministic fallback renderer stays attached
    (B) factory fails closed on missing credentials / unsupported provider
    (C) `GeminiRealizationAdapter` now explicitly disables AFC in its
        GenerateContentConfig (same contract as the Slice 58A reasoning fix)
    (D) when the realization provider fails, the realizer falls back to
        the deterministic renderer (no raw Gemini error crosses the boundary)
    (E) when the realization provider succeeds, its text wins — proving
        the pathway is actually live, not short-circuited

Scope note: integration-level assertions that the browser now gets live
prose rely on a live Gemini call and are out of scope for unit tests.
These tests prove the wiring + the invariants that govern live calls.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from app.conversation.realization.adapter import GeminiRealizationAdapter
from app.conversation.realization.live_contracts import LLMProviderConfig
from app.conversation.realization.provider import (
    MockConversationRealizationProvider,
    RealizationError,
    RealizationFailureKind,
)
from app.conversation.realization.realizer import GuardedConversationRealizer
from app.conversation.response_rendering.renderer import (
    DeterministicResponseRenderer,
    ResponseRenderer,
)
from app.llm.realization_factory import create_realization_renderer
from app.providers.contracts import ProviderId


# ---------------------------------------------------------------------------
# Section A — factory composition
# ---------------------------------------------------------------------------


@dataclass
class _FakeSettings:
    llm_provider: str = "gemini"
    gemini_api_key: str = "test-key"
    llm_model: str = "gemini-flash-latest"
    llm_timeout_seconds: float = 15.0
    llm_temperature: float = 0.2
    llm_top_p: float = 0.9
    llm_max_tokens: int = 256
    llm_structured_output: bool = True


class TestRealizationFactory:
    def test_builds_guarded_realizer_from_settings(self):
        fallback = DeterministicResponseRenderer()
        r = create_realization_renderer(_FakeSettings(), fallback_renderer=fallback)
        assert isinstance(r, GuardedConversationRealizer)

    def test_legacy_llm_gemini_provider_id_accepted(self):
        """§6 — historical config spelling also resolves."""
        s = _FakeSettings(llm_provider="llm.gemini")
        r = create_realization_renderer(s, fallback_renderer=DeterministicResponseRenderer())
        assert isinstance(r, GuardedConversationRealizer)

    def test_unsupported_provider_fails_closed(self):
        s = _FakeSettings(llm_provider="openai")
        with pytest.raises(ValueError) as exc:
            create_realization_renderer(
                s, fallback_renderer=DeterministicResponseRenderer(),
            )
        assert "Gemini" in str(exc.value)

    def test_missing_api_key_fails_closed(self):
        """Composition failure, not silent fallback — a misconfigured
        deployment must not look like a dev-only deterministic session."""
        s = _FakeSettings(gemini_api_key="   ")
        with pytest.raises(ValueError) as exc:
            create_realization_renderer(
                s, fallback_renderer=DeterministicResponseRenderer(),
            )
        assert "GEMINI_API_KEY" in str(exc.value)

    def test_api_key_not_in_error_message(self):
        """§3-adjacent — credential value never leaks in errors."""
        secret = "sk-live-leakme-12345"
        s = _FakeSettings(llm_provider="openai", gemini_api_key=secret)
        with pytest.raises(ValueError) as exc:
            create_realization_renderer(
                s, fallback_renderer=DeterministicResponseRenderer(),
            )
        assert secret not in str(exc.value)


# ---------------------------------------------------------------------------
# Section B — Gemini realization adapter AFC-disable contract
# ---------------------------------------------------------------------------


class _Models:
    def __init__(self, *, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = 0
        self.kwargs: dict = {}

    def generate_content(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.response


def _config() -> LLMProviderConfig:
    return LLMProviderConfig(
        provider_id=ProviderId("llm.gemini"),
        model_name="gemini-test",
        timeout_seconds=5.0,
        temperature=0.2,
        top_p=0.9,
        max_tokens=128,
        structured_output=True,
    )


def _plan():
    """Build a minimal ResponsePlan using the actual positional contract
    (same shape used by the existing realization tests)."""
    from app.conversation.response_planning.contracts import (
        AcknowledgementKind, AddresseeStatus, ConversationMove,
        InterruptionHandling, QuestionStrategy, ResponseLength, ResponsePlan,
        Tone,
    )
    return ResponsePlan(
        ConversationMove.CONSULTATIVE_DISCOVERY,
        ResponseLength.SHORT,
        Tone.NEUTRAL,
        AcknowledgementKind.NONE,
        QuestionStrategy.NONE,
        False,
        InterruptionHandling.DROP_STALE_POINT,
        False,
        None,
        AddresseeStatus.ADDRESSED_TO_AGENT,
    )


def _realization_input():
    from app.conversation.realization.contracts import (
        HumanConversationPolicy, LeanContextView, RealizationInput,
    )
    return RealizationInput(
        _plan(),
        HumanConversationPolicy(),
        LeanContextView("hello"),
    )


class TestGeminiRealizationAdapterAfcDisable:
    def test_config_disables_automatic_function_calling(self):
        models = _Models(
            response=SimpleNamespace(text=json.dumps({"text": "Hi there."})),
        )
        adapter = GeminiRealizationAdapter(
            config=_config(), client=SimpleNamespace(models=models),
        )
        from app.conversation.realization.prompt import build_llm_realization_prompt
        adapter.complete(build_llm_realization_prompt(_realization_input()), _config())
        afc = models.kwargs["config"].get("automatic_function_calling")
        assert afc is not None, "realization adapter did not disable AFC"
        disable = (
            afc.get("disable") if isinstance(afc, dict)
            else getattr(afc, "disable", None)
        )
        assert disable is True

    def test_realization_adapter_has_no_warning_suppression(self):
        """Same contract as Slice 58A: fix at the config layer, never
        global warnings/logging suppression."""
        from app.conversation.realization import adapter as adapter_mod
        src = inspect.getsource(adapter_mod)
        for forbidden in (
            "warnings.filterwarnings",
            "warnings.simplefilter",
            "logging.disable",
            'logging.getLogger("google_genai',
            "logging.getLogger('google_genai",
            # And no Chat.send_message migration
            "chats.create", ".send_message(",
        ):
            assert forbidden not in src


# ---------------------------------------------------------------------------
# Section C — GuardedConversationRealizer behavior
# ---------------------------------------------------------------------------


class _SpyRenderer(ResponseRenderer):
    """Fallback spy — captures whether it was invoked."""
    def __init__(self):
        self.called = 0
        self._inner = DeterministicResponseRenderer()

    def render(self, render_input):
        self.called += 1
        return self._inner.render(render_input)


def _render_input(*, authoritative, addressee):
    """Minimal ResponseRenderInput that `_may_realize` can evaluate.
    Reads only `authoritative_result` + `plan.addressee_status`."""
    from app.conversation.response_planning.contracts import (
        AcknowledgementKind, ConversationMove, InterruptionHandling,
        QuestionStrategy, ResponseLength, ResponsePlan, Tone,
    )
    from app.conversation.response_rendering.contracts import (
        ResponseRenderInput, ResponseRenderingBudget, TrustedRenderingContext,
    )
    plan = ResponsePlan(
        ConversationMove.CONSULTATIVE_DISCOVERY,
        ResponseLength.SHORT,
        Tone.NEUTRAL,
        AcknowledgementKind.NONE,
        QuestionStrategy.NONE,
        False,
        InterruptionHandling.DROP_STALE_POINT,
        False,
        None,
        addressee,
    )
    return ResponseRenderInput(
        plan,
        authoritative_result=authoritative,
        budget=ResponseRenderingBudget(5),
        trusted_context=TrustedRenderingContext(),
        current_prospect_message="hello",
    )


class TestGuardedRealizerFallbackContract:
    def test_provider_failure_falls_back_deterministically(self):
        """§13, §14 — RealizationError → deterministic fallback,
        never a raw error, no retry loop."""
        from app.conversation.response_planning.contracts import (
            AddresseeStatus, AuthoritativeResultKind,
        )
        provider = MockConversationRealizationProvider(should_fail=True)
        fallback = _SpyRenderer()
        realizer = GuardedConversationRealizer(provider, fallback)
        inp = _render_input(
            authoritative=AuthoritativeResultKind.EXECUTED,
            addressee=AddresseeStatus.ADDRESSED_TO_AGENT,
        )
        result = realizer.render(inp)
        assert fallback.called == 1
        assert isinstance(result.text, str)
        assert result.text  # non-empty

    def test_valid_live_text_wins(self):
        """§25 — valid provider text passes validation and reaches output."""
        from app.conversation.response_planning.contracts import (
            AddresseeStatus, AuthoritativeResultKind,
        )
        provider = MockConversationRealizationProvider(response="Hi there!")
        fallback = _SpyRenderer()
        realizer = GuardedConversationRealizer(provider, fallback)
        inp = _render_input(
            authoritative=AuthoritativeResultKind.EXECUTED,
            addressee=AddresseeStatus.ADDRESSED_TO_AGENT,
        )
        result = realizer.render(inp)
        # Provider was invoked; fallback was NOT
        assert provider.call_count == 1
        assert fallback.called == 0
        # The live text wins
        assert "Hi there!" in result.text

    def test_non_executed_path_bypasses_realizer(self):
        """§4-adjacent — only AUTHORITY_APPROVED + ADDRESSED_TO_AGENT
        reaches the live provider. Escalate/Redirect/Fallback go directly
        to the deterministic renderer."""
        from app.conversation.response_planning.contracts import (
            AddresseeStatus, AuthoritativeResultKind,
        )
        provider = MockConversationRealizationProvider(response="should-not-be-called")
        fallback = _SpyRenderer()
        realizer = GuardedConversationRealizer(provider, fallback)
        inp = _render_input(
            authoritative=AuthoritativeResultKind.ESCALATE,
            addressee=AddresseeStatus.ADDRESSED_TO_AGENT,
        )
        realizer.render(inp)
        assert provider.call_count == 0
        assert fallback.called == 1

    def test_not_addressed_to_agent_bypasses_realizer(self):
        from app.conversation.response_planning.contracts import (
            AddresseeStatus, AuthoritativeResultKind,
        )
        provider = MockConversationRealizationProvider(response="should-not-be-called")
        fallback = _SpyRenderer()
        realizer = GuardedConversationRealizer(provider, fallback)
        inp = _render_input(
            authoritative=AuthoritativeResultKind.EXECUTED,
            addressee=AddresseeStatus.NOT_ADDRESSED_TO_AGENT,
        )
        realizer.render(inp)
        assert provider.call_count == 0
        assert fallback.called == 1

    def test_one_call_only_no_regeneration(self):
        """§14 — one invocation of realize() per render()."""
        from app.conversation.response_planning.contracts import (
            AddresseeStatus, AuthoritativeResultKind,
        )
        provider = MockConversationRealizationProvider(response="Hi!")
        realizer = GuardedConversationRealizer(provider, _SpyRenderer())
        inp = _render_input(
            authoritative=AuthoritativeResultKind.EXECUTED,
            addressee=AddresseeStatus.ADDRESSED_TO_AGENT,
        )
        realizer.render(inp)
        assert provider.call_count == 1


# ---------------------------------------------------------------------------
# Section D — authority invariants (realization never widens authority)
# ---------------------------------------------------------------------------


class TestRealizationAuthorityInvariants:
    def test_realization_module_cannot_mutate_state_machine(self):
        """§4 — realizer module imports no state-machine mutation path."""
        from app.conversation.realization import adapter, provider, router, realizer
        for mod in (adapter, provider, router, realizer):
            src = inspect.getsource(mod)
            for forbidden in (
                "ConversationStateMachine(",
                "machine.transition(",
                "mark_lead_dnc",
                "finish_session",
                "OutcomeIntelligenceEngine",
                "process_turn",
            ):
                assert forbidden not in src, (
                    f"{forbidden} leaked into realization module {mod.__name__}"
                )

    def test_realization_module_cannot_import_runtime(self):
        """§7, §12 — runtime remains independent of realization provider details."""
        from app.conversation.realization import adapter, router
        for mod in (adapter, router):
            src = inspect.getsource(mod)
            assert "PersistentSalesRuntime" not in src
            assert "from app.application" not in src

    def test_realization_factory_module_is_only_gemini_specific_in_llm(self):
        """`app/llm/realization_factory.py` imports `GeminiRealization
        Adapter` — this is the only Gemini-specific composition in app/llm.
        The runtime, orchestrator, routes stay provider-neutral."""
        from app.application import production_turn_processor
        src = inspect.getsource(production_turn_processor)
        for forbidden in (
            "GeminiRealizationAdapter",
            "google.genai",
            "from google import genai",
        ):
            assert forbidden not in src


# ---------------------------------------------------------------------------
# Section E — composition wiring at main.py
# ---------------------------------------------------------------------------


class TestMainBootstrapWiresLiveRealization:
    """Scanner-level tests: `app.main` has module-level side effects
    (deletes the dev SQLite DB, instantiates the runtime) so we read
    the source as text rather than importing it."""

    @pytest.fixture(scope="class")
    def main_source(self) -> str:
        from pathlib import Path
        return (
            Path(__file__).resolve().parents[2] / "app" / "main.py"
        ).read_text(encoding="utf-8")

    def test_main_imports_realization_factory(self, main_source):
        assert (
            "from app.llm.realization_factory import create_realization_renderer"
            in main_source
        )
        assert "create_realization_renderer(" in main_source

    def test_main_keeps_deterministic_fallback(self, main_source):
        """§13 — deterministic renderer stays attached as the fallback."""
        assert "DeterministicResponseRenderer()" in main_source
        assert "fallback_renderer=" in main_source
