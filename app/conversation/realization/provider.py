"""Provider seam for one untrusted language-only realization proposal."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from app.conversation.realization.contracts import RealizationInput


class RealizationFailureKind(str, Enum):
    """Bounded provider outcomes; every failure deterministically falls back."""

    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    MALFORMED_RESPONSE = "malformed_response"
    HTTP_ERROR = "http_error"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION_FAILURE = "authentication_failure"
    UNKNOWN_FAILURE = "unknown_failure"


class RealizationError(RuntimeError):
    """A wording provider failed; callers must fall back without retrying."""

    def __init__(
        self,
        message: str,
        kind: RealizationFailureKind = RealizationFailureKind.UNKNOWN_FAILURE,
    ) -> None:
        super().__init__(message)
        self.kind = kind


class ConversationRealizationProvider(ABC):
    """Return plain-text wording only; no structured command is accepted."""

    @abstractmethod
    def realize(self, value: RealizationInput) -> str:
        """Return one untrusted wording proposal."""
        raise NotImplementedError


class MockConversationRealizationProvider(ConversationRealizationProvider):
    """Scripted transparent test double with no hidden language intelligence."""

    def __init__(self, response: object = "Okay.", *, should_fail: bool = False) -> None:
        self._response = response
        self._should_fail = should_fail
        self.call_count = 0
        self.last_input: RealizationInput | None = None

    def realize(self, value: RealizationInput) -> str:
        self.call_count += 1
        self.last_input = value
        if self._should_fail:
            raise RealizationError("scripted realization provider failure")
        return self._response  # type: ignore[return-value]
