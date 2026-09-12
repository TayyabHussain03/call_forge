"""Provider-neutral advisory supervisor interface and deterministic fake."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.conversation.supervisor.contracts import SupervisorInput, SupervisorInsight


class SupervisorError(Exception):
    """Sanitized optional-supervisor failure."""


class SupervisorProvider(ABC):
    """Analyze a completed turn without participating in the fast path."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a stable provider name."""
        raise NotImplementedError

    @abstractmethod
    def analyze(self, supervisor_input: SupervisorInput) -> SupervisorInsight:
        """Return one advisory insight or raise ``SupervisorError``."""
        raise NotImplementedError


class MockSupervisorProvider(SupervisorProvider):
    """Deterministic scripted fake with no classification or policy logic."""

    def __init__(
        self,
        *,
        default: SupervisorInsight | None = None,
        scripted: tuple[SupervisorInsight, ...] = (),
        should_fail: bool = False,
        provider_name: str = "mock_supervisor",
    ) -> None:
        self._default = default
        self._scripted = scripted
        self._should_fail = should_fail
        self._provider_name = provider_name
        self._call_count = 0

    @property
    def name(self) -> str:
        """Return the configured provider name."""
        return self._provider_name

    @property
    def call_count(self) -> int:
        """Return how many explicit analyses were requested."""
        return self._call_count

    def analyze(self, supervisor_input: SupervisorInput) -> SupervisorInsight:
        """Replay one configured result without interpreting the input."""
        index = self._call_count
        self._call_count += 1
        if self._should_fail:
            raise SupervisorError("scripted supervisor failure")
        if index < len(self._scripted):
            return self._scripted[index]
        if self._default is not None:
            return self._default
        raise SupervisorError("mock supervisor has no configured insight")
