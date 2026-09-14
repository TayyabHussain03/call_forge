"""Provider-neutral understanding interface and transparent offline mock."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.conversation.understanding.contracts import (
    FreeTextUnderstanding,
    FreeTextUnderstandingInput,
)
from app.llm.providers.reasoning_provider import ProviderFailureKind


class UnderstandingError(Exception):
    """Sanitized provider failure using the existing provider taxonomy."""

    def __init__(
        self,
        message: str,
        kind: ProviderFailureKind = ProviderFailureKind.TRANSPORT_FAILURE,
    ) -> None:
        super().__init__(message)
        self.kind = kind


class FreeTextUnderstandingProvider(ABC):
    """Interpret current-turn meaning without deciding or executing anything."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def understand(
        self, understanding_input: FreeTextUnderstandingInput
    ) -> FreeTextUnderstanding:
        """Return one strict advisory interpretation or raise once."""
        raise NotImplementedError


class MockFreeTextUnderstandingProvider(FreeTextUnderstandingProvider):
    """Dumb deterministic replay provider with no text rules or hidden inference."""

    def __init__(
        self,
        *,
        default: FreeTextUnderstanding | None = None,
        scripted: tuple[FreeTextUnderstanding, ...] = (),
        should_fail: bool = False,
        provider_name: str = "mock-understanding",
    ) -> None:
        self._default = default
        self._scripted = tuple(scripted)
        self._should_fail = should_fail
        self._provider_name = provider_name
        self._call_count = 0
        self._last_input: FreeTextUnderstandingInput | None = None

    @property
    def name(self) -> str:
        return self._provider_name

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def last_input(self) -> FreeTextUnderstandingInput | None:
        return self._last_input

    def understand(
        self, understanding_input: FreeTextUnderstandingInput
    ) -> FreeTextUnderstanding:
        index = self._call_count
        self._call_count += 1
        self._last_input = understanding_input
        if self._should_fail:
            raise UnderstandingError("mock understanding provider configured to fail")
        if index < len(self._scripted):
            return self._scripted[index]
        if self._default is not None:
            return self._default
        raise UnderstandingError("mock understanding provider has no scripted result")
