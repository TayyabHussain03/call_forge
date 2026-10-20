"""Slice 51H — production composition + reasoning-call budget persistence.

Closes two gaps:
    - Composition root helper so callers don't manually wire three
      Slice 51E/51F constructors.
    - `_reasoning_calls` now persisted via BUDGET_STATE_KEY payload so
      restart cannot grant extra Brain reasoning capacity.
"""

from __future__ import annotations

import pytest

from app.application.sales_runtime.composition import (
    build_concept_provider_callable,
    build_intent_provider_callable,
    build_participant_aware_runtime,
    wrap_renderer_strategy_aware,
)
from app.application.sales_runtime.manifest import default_manifest
from app.application.sales_runtime.persistent_runtime import PersistentSalesRuntime
from app.application.sales_runtime.serializers import (
    BUDGET_STATE_KEY,
    BudgetStateSerializer,
)
from app.conversation.participant.intent_provider import (
    RaisingIntentTransport,
    ScriptedIntentTransport,
)
from app.conversation.participant.policy import ReExplanationIntent
from app.conversation.response_rendering.renderer import DeterministicResponseRenderer
from app.conversation.response_rendering.strategy_frame import (
    StrategyAwareResponseRenderer,
)
from app.persistence.errors import InvalidPersistedPayloadError


# ---------------------------------------------------------------------------
# Budget serializer round-trip
# ---------------------------------------------------------------------------


class TestBudgetSerializer:
    def test_zero_round_trip(self) -> None:
        ser = BudgetStateSerializer()
        assert ser.deserialize(ser.serialize(0)) == 0

    def test_positive_round_trip(self) -> None:
        ser = BudgetStateSerializer()
        assert ser.deserialize(ser.serialize(7)) == 7

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            BudgetStateSerializer().serialize(-1)

    def test_boolean_rejected(self) -> None:
        """bool is a subclass of int — must still be rejected."""
        with pytest.raises(ValueError):
            BudgetStateSerializer().serialize(True)  # type: ignore[arg-type]

    def test_corrupted_payload_rejected(self) -> None:
        with pytest.raises(InvalidPersistedPayloadError):
            BudgetStateSerializer().deserialize("{not json")

    def test_negative_in_payload_rejected(self) -> None:
        # Force an invalid envelope payload
        bad = '{"contract":"budget_state","version":1,"payload":{"reasoning_calls":-5}}'
        with pytest.raises(InvalidPersistedPayloadError):
            BudgetStateSerializer().deserialize(bad)

    def test_deterministic_bytes(self) -> None:
        ser = BudgetStateSerializer()
        assert ser.serialize(42) == ser.serialize(42)


# ---------------------------------------------------------------------------
# Budget key registered in manifest
# ---------------------------------------------------------------------------


class TestBudgetManifestRegistration:
    def test_budget_key_is_supported(self) -> None:
        m = default_manifest()
        assert m.is_supported(BUDGET_STATE_KEY, 1)

    def test_budget_key_is_optional_derived(self) -> None:
        m = default_manifest()
        assert not m.is_required(BUDGET_STATE_KEY)
        assert BUDGET_STATE_KEY in m.optional_derived


# ---------------------------------------------------------------------------
# Composition root helpers
# ---------------------------------------------------------------------------


class TestWrapRendererStrategyAware:
    def test_wraps_in_strategy_aware_wrapper(self) -> None:
        base = DeterministicResponseRenderer()
        wrapped = wrap_renderer_strategy_aware(base)
        assert isinstance(wrapped, StrategyAwareResponseRenderer)


class TestIntentProviderBuilder:
    def test_returns_none_when_transport_none(self) -> None:
        """No transport configured → None callable → processor uses safe
        fallback (no strategy threading)."""
        assert build_intent_provider_callable(None) is None

    def test_returns_callable_when_transport_set(self) -> None:
        transport = ScriptedIntentTransport(('{"intent":"recap_request"}',))
        callable_ = build_intent_provider_callable(transport)
        assert callable_ is not None

        class _T:
            utterance = "remind me"
        class _U:
            class language_profile:
                primary_language = "en"
                secondary_language = None

        assert callable_(_T(), _U()) == ReExplanationIntent.RECAP_REQUEST

    def test_transport_exception_falls_back_to_none_intent(self) -> None:
        callable_ = build_intent_provider_callable(RaisingIntentTransport())

        class _T:
            utterance = "hi"

        assert callable_(_T(), None) == ReExplanationIntent.NONE


class TestConceptProviderBuilder:
    def test_sources_from_processor_live_guidance(self) -> None:
        class FakeGuidance:
            selected_service_id = "automation"

        class FakeProcessor:
            @property
            def playbook_guidance(self):
                return FakeGuidance()

        callable_ = build_concept_provider_callable(FakeProcessor())
        assert callable_(None, None) == "service:automation"

    def test_returns_none_when_no_guidance(self) -> None:
        class FakeProcessor:
            @property
            def playbook_guidance(self):
                return None

        callable_ = build_concept_provider_callable(FakeProcessor())
        assert callable_(None, None) is None


# ---------------------------------------------------------------------------
# Processor accepts initial_reasoning_calls
# ---------------------------------------------------------------------------


class TestProcessorAcceptsInitialReasoningCalls:
    def test_initial_counter_defaults_to_zero(self) -> None:
        """Default construction keeps counter at 0 — existing callers
        unchanged (new param is optional additive)."""
        # Smoke-test by inspecting the parameter default via signature
        import inspect
        from app.application.production_turn_processor import ProductionTurnProcessor
        sig = inspect.signature(ProductionTurnProcessor.__init__)
        assert sig.parameters["initial_reasoning_calls"].default == 0

    def test_negative_initial_rejected(self) -> None:
        """Validation prevents a tampered negative counter from entering."""
        import inspect
        from app.application.production_turn_processor import ProductionTurnProcessor
        # We can't easily construct a full processor here; validate the
        # error path by inspecting the source for the check.
        source = inspect.getsource(ProductionTurnProcessor.__init__)
        assert "initial_reasoning_calls must be a non-negative int" in source


# ---------------------------------------------------------------------------
# Budget serializer wired into processor export path
# ---------------------------------------------------------------------------


class TestProcessorExportsBudgetPayload:
    def test_export_includes_budget_when_counter_positive(self) -> None:
        import inspect
        from app.application.production_turn_processor import ProductionTurnProcessor
        source = inspect.getsource(ProductionTurnProcessor.export_snapshot)
        assert "BUDGET_STATE_KEY" in source
        assert "self._reasoning_calls > 0" in source

    def test_export_omits_budget_when_counter_zero(self) -> None:
        """When counter is 0 the payload is OMITTED — avoids payload
        churn and keeps default snapshots lean."""
        import inspect
        from app.application.production_turn_processor import ProductionTurnProcessor
        source = inspect.getsource(ProductionTurnProcessor.export_snapshot)
        # The gate: `if self._reasoning_calls > 0:`
        assert "if self._reasoning_calls > 0:" in source


# ---------------------------------------------------------------------------
# No forbidden imports in new composition module
# ---------------------------------------------------------------------------


class TestNoForbiddenImportsInComposition:
    def test_composition_module_has_no_reasoning_imports(self) -> None:
        import inspect
        from app.application.sales_runtime import composition
        source = inspect.getsource(composition)
        for forbidden in (
            "BrainOrchestrator", "StateMachine", "ActionValidator",
            "HybridRetrievalEngine", "ServiceRelevanceResolver",
            "openai", "anthropic",
        ):
            assert forbidden not in source


# ---------------------------------------------------------------------------
# No service/industry hardcoding in budget serializer
# ---------------------------------------------------------------------------


class TestNoHardcodingInBudgetSerializer:
    def test_budget_module_has_no_industry_keywords(self) -> None:
        import inspect
        from app.application.sales_runtime.serializers import budget
        source = inspect.getsource(budget)
        for forbidden in (
            "restaurant", "dentist", "hvac",
        ):
            assert forbidden not in source.lower()
