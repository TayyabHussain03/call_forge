"""Production composition root for the participant-aware sales runtime.

Factory functions that wire the three Slice 51E/51F adapters together with
an existing `TurnProcessorFactory` so callers do not have to remember three
manual constructor wrappers. HTTP/API handlers can depend on this one
facade without seeing participant-policy details.
"""

from __future__ import annotations

from app.application.sales_runtime.persistent_runtime import PersistentSalesRuntime
from app.application.sales_runtime.processor_factory import TurnProcessorFactory
from app.conversation.participant.concept_provider import BrainConceptProvider
from app.conversation.participant.intent_provider import (
    IntentClassifierTransport,
    ReExplanationIntentLLMProvider,
)
from app.conversation.response_rendering.renderer import ResponseRenderer
from app.conversation.response_rendering.strategy_frame import (
    StrategyAwareResponseRenderer,
)
from app.persistence.sqlite.connection import SqliteConnection


def wrap_renderer_strategy_aware(renderer: ResponseRenderer) -> ResponseRenderer:
    """Wrap any ResponseRenderer so plan.re_explanation_strategy produces
    visible, language-adaptive prose. Idempotent — wrapping a wrapper is
    harmless because the inner wrapper only reacts to a plan with a
    non-None strategy, and the outer wrapper does the same.
    """
    return StrategyAwareResponseRenderer(renderer)


def build_intent_provider_callable(transport: IntentClassifierTransport | None):
    """Return a `(turn, understanding) -> ReExplanationIntent` callable
    suitable for `ProductionTurnProcessor.participant_evidence_provider`-
    style constructor params. If transport is None, returns None — the
    processor's existing safe-fallback path takes over and the strategy
    simply does not fire for that turn.
    """
    if transport is None:
        return None
    return ReExplanationIntentLLMProvider(transport).as_callable()


def build_concept_provider_callable(processor):
    """Return a `(turn, understanding) -> str | None` callable that reads
    the live `processor.playbook_guidance.selected_service_id`. The
    processor itself is the source — no new LLM calls, no prose parsing.
    """
    return BrainConceptProvider(processor).as_callable()


def build_participant_aware_runtime(
    connection: SqliteConnection,
    processor_factory: TurnProcessorFactory,
    *,
    pcie=None,
    outcome_engine=None,
    snapshot_manifest=None,
) -> PersistentSalesRuntime:
    """Production composition root. Returns a `PersistentSalesRuntime` that
    is already wired for participant-aware behavior.

    The caller supplies the `processor_factory` (which may itself apply
    `wrap_renderer_strategy_aware`, plug in the intent/concept providers
    at construction time, and build the real `ProductionTurnProcessor`).
    This helper keeps HTTP/API handlers oblivious to any of that: they
    see a `PersistentSalesRuntime` with the standard public API only.
    """
    return PersistentSalesRuntime(
        connection=connection,
        processor_factory=processor_factory,
        pcie=pcie,
        outcome_engine=outcome_engine,
        snapshot_manifest=snapshot_manifest,
    )
