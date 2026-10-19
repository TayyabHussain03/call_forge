"""Production ReExplanationIntentProvider.

Provider-neutral adapter that uses an injected LLM-transport callable to
classify the current utterance into one of the typed ReExplanationIntent
values. The LLM proposes; deterministic policy (ReExplanationPolicy)
decides. No keyword trees in deterministic logic. On any failure the
provider degrades safely to NONE — a failed classification must never
block a sales turn or fabricate a stronger intent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from app.conversation.participant.policy import ReExplanationIntent


# ---------------------------------------------------------------------------
# Transport protocol — any provider that can turn a prompt into a JSON string
# ---------------------------------------------------------------------------


@runtime_checkable
class IntentClassifierTransport(Protocol):
    """Minimal provider-neutral transport. Returns a JSON string that
    this adapter parses. Implementations may be deterministic/scripted
    (for tests) or live LLM clients.
    """

    def classify(self, prompt_input: dict) -> str:
        """Return a JSON string. The adapter never passes transport
        output to the deterministic core without typed validation."""


# ---------------------------------------------------------------------------
# Bounded input/output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntentClassificationInput:
    """Bounded context sent to the transport. No full transcript."""

    utterance: str
    primary_language: str | None = None
    secondary_language: str | None = None
    active_participant_role_hint: str | None = None
    concept_in_focus: str | None = None

    def __post_init__(self) -> None:
        if not self.utterance or len(self.utterance) > 2000:
            raise ValueError("utterance must be 1-2000 chars")

    def to_prompt_dict(self) -> dict:
        return {
            "utterance": self.utterance,
            "primary_language": self.primary_language,
            "secondary_language": self.secondary_language,
            "active_participant_role_hint": self.active_participant_role_hint,
            "concept_in_focus": self.concept_in_focus,
            "allowed_intents": [i.value for i in ReExplanationIntent],
        }


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class ReExplanationIntentLLMProvider:
    """Production intent provider using an injected transport.

    Usage inside ProductionTurnProcessor:
        provider = ReExplanationIntentLLMProvider(transport)
        processor = ProductionTurnProcessor(
            ...,
            re_explanation_intent_provider=provider.as_callable(),
        )

    Tests inject a `ScriptedIntentTransport` so classifications are
    deterministic without live LLM calls.
    """

    def __init__(self, transport: IntentClassifierTransport) -> None:
        self._transport = transport

    def classify(self, inp: IntentClassificationInput) -> ReExplanationIntent:
        try:
            raw = self._transport.classify(inp.to_prompt_dict())
        except Exception:
            return ReExplanationIntent.NONE
        if not isinstance(raw, str) or not raw:
            return ReExplanationIntent.NONE
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            return ReExplanationIntent.NONE
        if not isinstance(parsed, dict):
            return ReExplanationIntent.NONE
        intent_value = parsed.get("intent")
        if not isinstance(intent_value, str):
            return ReExplanationIntent.NONE
        try:
            return ReExplanationIntent(intent_value)
        except ValueError:
            return ReExplanationIntent.NONE

    def as_callable(self) -> Callable:
        """Return the processor-ready two-argument callable."""
        def _provider(turn, understanding):
            lang = understanding.language_profile if understanding else None
            inp = IntentClassificationInput(
                utterance=turn.utterance,
                primary_language=lang.primary_language if lang else None,
                secondary_language=lang.secondary_language if lang else None,
            )
            return self.classify(inp)
        return _provider


# ---------------------------------------------------------------------------
# Scripted transport for tests
# ---------------------------------------------------------------------------


class ScriptedIntentTransport:
    """Deterministic transport that returns pre-scripted JSON blobs.

    Tests use this to prove the adapter's parsing/validation/failure
    behavior without touching a real LLM. Not for production.
    """

    def __init__(self, responses: tuple[str, ...] = ()) -> None:
        self._responses = responses
        self._i = 0
        self.call_count = 0

    def classify(self, prompt_input: dict) -> str:
        self.call_count += 1
        if self._i < len(self._responses):
            r = self._responses[self._i]
            self._i += 1
            return r
        return '{"intent":"none"}'


class RaisingIntentTransport:
    """Transport that always raises — used to prove safe-fallback."""

    def classify(self, prompt_input: dict) -> str:
        raise RuntimeError("transport failure")
