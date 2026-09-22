"""Configured LLM realization routing with content-free operational metrics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from time import perf_counter

from app.conversation.realization.adapter import LLMProviderAdapter
from app.conversation.realization.contracts import RealizationInput
from app.conversation.realization.live_contracts import (
    LLMProviderConfig,
    ProviderCallOutcome,
    RealizationMetric,
    ValidationOutcome,
)
from app.conversation.realization.prompt import build_llm_realization_prompt
from app.conversation.realization.provider import (
    ConversationRealizationProvider,
    RealizationError,
    RealizationFailureKind,
)
from app.providers.contracts import ProviderCapability, ProviderCategory
from app.providers.registry import ProviderRegistry, ProviderRegistryError
from app.providers.resolver import ProviderResolutionError, ProviderResolver

MetricSink = Callable[[RealizationMetric], None]


class LLMProviderRouter(ConversationRealizationProvider):
    """Select one configured adapter and perform exactly one provider call."""

    def __init__(
        self,
        *,
        registry: ProviderRegistry,
        config: LLMProviderConfig,
        resolver: ProviderResolver,
        metric_sink: MetricSink | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if config.provider_id.category != ProviderCategory.LLM:
            raise ValueError("realization provider must be an LLM provider")
        try:
            descriptor = registry.require_active(config.provider_id)
        except ProviderRegistryError as exc:
            raise RealizationError(
                "configured realization provider is unavailable",
                RealizationFailureKind.PROVIDER_UNAVAILABLE,
            ) from exc
        if ProviderCapability.STRUCTURED_OUTPUT not in descriptor.capabilities:
            raise RealizationError(
                "configured realization provider lacks structured output",
                RealizationFailureKind.PROVIDER_UNAVAILABLE,
            )
        try:
            adapter = resolver.resolve(config.provider_id)
        except (ProviderRegistryError, ProviderResolutionError) as exc:
            raise RealizationError(
                "configured realization adapter is unavailable",
                RealizationFailureKind.PROVIDER_UNAVAILABLE,
            ) from exc
        if not isinstance(adapter, LLMProviderAdapter):
            raise RealizationError(
                "configured realization adapter has an invalid type",
                RealizationFailureKind.PROVIDER_UNAVAILABLE,
            )
        self._adapter = adapter
        self._config = config
        self._metric_sink = metric_sink
        self._clock = clock
        self._last_metric: RealizationMetric | None = None

    @property
    def last_metric(self) -> RealizationMetric | None:
        """Expose the latest content-free metric for composition and tests."""
        return self._last_metric

    def realize(self, value: RealizationInput) -> str:
        """Route one bounded prompt with zero automatic retries."""
        started = self._clock()
        try:
            response = self._adapter.complete(
                build_llm_realization_prompt(value), self._config
            )
        except RealizationError as exc:
            self._record(
                started,
                ProviderCallOutcome.FAILURE,
                failure_kind=exc.kind,
            )
            raise
        except Exception as exc:
            self._record(
                started,
                ProviderCallOutcome.FAILURE,
                failure_kind=RealizationFailureKind.UNKNOWN_FAILURE,
            )
            raise RealizationError(
                "realization adapter failed",
                RealizationFailureKind.UNKNOWN_FAILURE,
            ) from exc
        self._record(
            started,
            ProviderCallOutcome.SUCCESS,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        return response.text

    def record_validation(self, accepted: bool) -> None:
        """Attach only validation outcome; never retain response or prompt content."""
        if self._last_metric is None:
            return
        metric = replace(
            self._last_metric,
            validation=(
                ValidationOutcome.ACCEPTED if accepted else ValidationOutcome.REJECTED
            ),
        )
        self._last_metric = metric
        if self._metric_sink is not None:
            self._metric_sink(metric)

    def _record(
        self,
        started: float,
        outcome: ProviderCallOutcome,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        failure_kind: RealizationFailureKind | None = None,
    ) -> None:
        metric = RealizationMetric(
            self._config.provider_id,
            max(0, round((self._clock() - started) * 1000)),
            outcome,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            failure_kind=failure_kind,
        )
        self._last_metric = metric
        if self._metric_sink is not None and outcome == ProviderCallOutcome.FAILURE:
            self._metric_sink(metric)
