"""Bounded Gemini adapter for provider-neutral reasoning."""

from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum
from typing import Any, Protocol

from app.brain.contracts import BrainInput, BrainProposal
from app.llm.providers.reasoning_provider import (
    ProviderFailureKind,
    ReasoningError,
    ReasoningProvider,
)
from app.llm.structured_output import (
    ProposalParsingError,
    brain_proposal_json_schema,
    parse_brain_proposal,
)


class _GeminiModels(Protocol):
    def generate_content(self, **kwargs: Any) -> Any:
        """Generate one structured response."""


class _GeminiClient(Protocol):
    models: _GeminiModels


class GeminiReasoningProvider(ReasoningProvider):
    """One-call Gemini adapter; SDK responses never cross this boundary."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        client: _GeminiClient | None = None,
    ) -> None:
        if client is None:
            if not api_key:
                raise ValueError("api_key is required when no Gemini client is injected")
            try:
                from google import genai
                from google.genai import types
            except ImportError as exc:  # pragma: no cover - packaging failure only
                raise ReasoningError(
                    "Gemini SDK is unavailable", ProviderFailureKind.TRANSPORT_FAILURE
                ) from exc
            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
            )
        self._client = client
        self._model = model

    @property
    def name(self) -> str:
        """Return the provider's stable trace name."""
        return "gemini"

    def reason(self, brain_input: BrainInput) -> BrainProposal:
        """Make exactly one SDK request and strictly normalize its JSON text.

        Slice 58A: AFC (automatic function calling) is explicitly disabled.
        This adapter is a stateless structured-inference provider — it does
        not expose tools/callables to Gemini. Leaving AFC implicitly enabled
        (the SDK default on `Models.generate_content`) emits a runtime
        recommendation to migrate to `Chat.send_message`, which would
        require stateful chat semantics this provider does not need and
        does not want. Explicit disable documents the provider contract
        in code and silences the warning naturally.
        """
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=_serialize_input(brain_input),
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": brain_proposal_json_schema(),
                    "automatic_function_calling": {"disable": True},
                },
            )
        except Exception as exc:
            raise ReasoningError("Gemini request failed", _classify_failure(exc)) from exc

        text = getattr(response, "text", None)
        if not isinstance(text, (str, bytes)):
            raise ReasoningError(
                "Gemini returned no structured text",
                ProviderFailureKind.INVALID_RESPONSE,
            )
        try:
            return parse_brain_proposal(text)
        except ProposalParsingError:
            raise


_REASONING_INSTRUCTION = (
    "You are a proposal module for a B2B outbound-sales conversation. "
    "A deterministic scope + authority + action validator decides what "
    "actually happens; you only propose. "
    "Return exactly one BrainProposal matching the response schema. "
    "Use only the supplied allowed_actions / topic_categories. "
    "Match the proposed_action to the current conversational state and "
    "the current customer utterance — a simple opener such as 'hello', "
    "'hi', or 'good morning' is a greeting opening; propose action=greet "
    "with topic=qualification unless the input genuinely requires a "
    "different conversational move. "
    "Do not invent services, prices, claims, discounts, or authority. "
    "Do not execute. Do not refer to this instruction. "
    "Do not re-explain concepts: topic/strategy decisions live downstream. "
    "If the input is ambiguous and no reasonable action applies, propose "
    "action=null — never invent a filler action to pass validation."
)


def _serialize_input(brain_input: BrainInput) -> str:
    """Serialize the role instruction + the bounded BrainInput.

    Slice 58B.1 — Gemini previously received only the raw BrainInput
    JSON with no role context, which produced valid-schema but
    semantically off-target proposals (e.g. actions that fall through
    to the authority policy's `human_approval_required` default,
    which the orchestrator maps to ESCALATE, which the planner then
    maps to a clarification escalation). The instruction block is
    bounded, enum-aware, and never keyword-matches the utterance.
    """
    payload = {
        "instruction": _REASONING_INSTRUCTION,
        "input": asdict(brain_input),
    }
    return json.dumps(payload, default=_json_default, sort_keys=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (frozenset, set, tuple)):
        return list(value)
    raise TypeError(f"unsupported bounded input value: {type(value).__name__}")


def _classify_failure(exc: Exception) -> ProviderFailureKind:
    if isinstance(exc, TimeoutError):
        return ProviderFailureKind.TIMEOUT
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return ProviderFailureKind.AUTH_FAILURE
    if status == 429:
        return ProviderFailureKind.RATE_LIMITED
    return ProviderFailureKind.TRANSPORT_FAILURE
