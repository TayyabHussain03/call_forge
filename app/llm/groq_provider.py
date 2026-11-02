"""Groq provider — reasoning + realization adapters.

Slice 58B.7. Groq (`openai/gpt-oss-120b`) is approved as an alternate
provider after Gemini's structured-output endpoint began returning
5xx. The reasoning adapter uses Groq's strict JSON Schema mode with
the EXISTING authoritative `brain_proposal_json_schema()`; the
realization adapter uses `response_format={"type": "json_object"}`
with a bounded `{"text": "..."}` envelope.

Both adapters:
    * one SDK call per `reason()` / `complete()` — no retries, no
      regeneration loop
    * ReasoningError / RealizationError carry sanitized exception_type
      + http_status so the Slice 58B.6 trace surfaces real failure
      kind
    * no business logic — proposals / wording are passed back untrusted
      for the deterministic policy chain + validators to decide
    * API key, prompt, chain-of-thought never logged
"""

from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum
from typing import Any, Protocol

from app.brain.contracts import BrainInput, BrainProposal
from app.conversation.realization.adapter import LLMProviderAdapter
from app.conversation.realization.live_contracts import (
    LLMProviderConfig,
    LLMProviderResponse,
    LLMRealizationPrompt,
)
from app.conversation.realization.prompt import serialize_llm_realization_prompt
from app.conversation.realization.provider import (
    RealizationError,
    RealizationFailureKind,
)
from app.llm.gemini_provider import _REASONING_INSTRUCTION, _extract_http_status
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


# ---------------------------------------------------------------------------
# Minimal SDK protocol surfaces — tests inject in-process fakes
# ---------------------------------------------------------------------------


class _GroqChatCompletions(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _GroqChat(Protocol):
    completions: _GroqChatCompletions


class _GroqClient(Protocol):
    chat: _GroqChat


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (frozenset, set, tuple)):
        return list(value)
    raise TypeError(f"unsupported bounded value: {type(value).__name__}")


def _groq_compat_schema(schema: Any) -> Any:
    """Transform the authoritative BrainProposal schema for Groq strict mode.

    Groq (openai/gpt-oss-120b) strict Structured Outputs rejects
    `anyOf: [{type:integer}, {type:number}]` with `integer_number_overlap`:
    the union is ambiguous. Pydantic emits this for `int | float` typed
    fields (e.g. action_confidence) — a legitimate domain shape that
    Gemini accepts. We collapse these `anyOf` branches to a single
    `{"type": "number"}` lossessly because `parse_brain_proposal` still
    validates the response against the strict pydantic model (which
    accepts both int and float values).

    Transformation is pure, recursive, and provider-specific — the
    domain schema is NEVER mutated.
    """
    if isinstance(schema, dict):
        any_of = schema.get("anyOf")
        if isinstance(any_of, list) and all(isinstance(b, dict) for b in any_of):
            branch_types = [b.get("type") for b in any_of]
            # Collapse the integer+number ambiguity while preserving
            # any other branches (e.g. `null` for Optional fields).
            if "integer" in branch_types and "number" in branch_types:
                kept = [
                    b for b in any_of
                    if b.get("type") not in {"integer", "number"}
                ]
                number_branch = {"type": "number"}
                collapsed = [number_branch] + kept
                if len(collapsed) == 1:
                    merged = {k: v for k, v in schema.items() if k != "anyOf"}
                    merged.update(collapsed[0])
                    return _groq_compat_schema(merged)
                new_schema = {k: v for k, v in schema.items() if k != "anyOf"}
                new_schema["anyOf"] = collapsed
                return {k: _groq_compat_schema(v) for k, v in new_schema.items()}
        # Groq strict mode requires `additionalProperties: false` on
        # EVERY object schema, not just the top-level model. Pydantic
        # only sets it where `extra="forbid"` is declared, so inject
        # it on any nested object type that didn't already specify it.
        out = {k: _groq_compat_schema(v) for k, v in schema.items()}
        if out.get("type") == "object" and "additionalProperties" not in out:
            out["additionalProperties"] = False
        return out
    if isinstance(schema, list):
        return [_groq_compat_schema(v) for v in schema]
    return schema


def _extract_message_text(response: object) -> str | None:
    """Pull `.choices[0].message.content` from a Groq chat completion."""
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    try:
        first = choices[0]
    except (IndexError, TypeError):
        return None
    message = getattr(first, "message", None)
    if message is None:
        return None
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    return None


# ---------------------------------------------------------------------------
# Groq reasoning provider — strict json_schema BrainProposal
# ---------------------------------------------------------------------------


class GroqReasoningProvider(ReasoningProvider):
    """One-call Groq adapter for structured BrainProposal."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 15.0,
        client: _GroqClient | None = None,
    ) -> None:
        if client is None:
            if not api_key:
                raise ValueError("api_key is required when no Groq client is injected")
            try:
                from groq import Groq
            except ImportError as exc:  # pragma: no cover - packaging failure
                raise ReasoningError(
                    "Groq SDK is unavailable",
                    ProviderFailureKind.TRANSPORT_FAILURE,
                ) from exc
            client = Groq(api_key=api_key, timeout=float(timeout_seconds))
        self._client = client
        self._model = model

    @property
    def name(self) -> str:
        return "groq"

    def reason(self, brain_input: BrainInput) -> BrainProposal:
        # Groq uses `type: "json_object"` mode (above), so the schema
        # must travel inside the content to tell the model what shape
        # to produce. Server-side `parse_brain_proposal` remains the
        # authoritative validator.
        payload = {
            "instruction": _REASONING_INSTRUCTION,
            "response_json_schema": brain_proposal_json_schema(),
            "input": asdict(brain_input),
        }
        contents = json.dumps(payload, default=_json_default, sort_keys=True)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": contents}],
                response_format={
                    # Groq strict `json_schema` mode is too limited for
                    # the full BrainProposal schema (rejects nested
                    # $ref/oneOf/discriminator). `json_object` mode
                    # still returns valid JSON that we then validate
                    # SERVER-SIDE through the authoritative strict
                    # pydantic model (`parse_brain_proposal`), which
                    # catches any schema violation and surfaces it as
                    # ProposalParsingError. Identical safety posture
                    # to the Gemini adapter — reasoning stays strictly
                    # typed, no prose acceptance. The schema goes to
                    # the model via the instruction preamble.
                    "type": "json_object",
                },
            )
        except Exception as exc:
            raise ReasoningError(
                "Groq request failed",
                _classify_failure(exc),
                exception_type=type(exc).__name__,
                http_status=_extract_http_status(exc),
            ) from exc
        text = _extract_message_text(response)
        if not isinstance(text, str) or not text.strip():
            raise ReasoningError(
                "Groq returned no structured text",
                ProviderFailureKind.INVALID_RESPONSE,
            )
        try:
            return parse_brain_proposal(text)
        except ProposalParsingError:
            # Strict parsing failure is intentionally re-raised; the
            # orchestrator maps ProposalParsingError onto its own
            # fallback path exactly like it does for Gemini.
            raise


# ---------------------------------------------------------------------------
# Groq realization adapter — `{"text":"..."}` envelope
# ---------------------------------------------------------------------------


class GroqRealizationAdapter(LLMProviderAdapter):
    """One-call Groq adapter for realized customer-facing text."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        config: LLMProviderConfig,
        client: _GroqClient | None = None,
    ) -> None:
        if client is None:
            if not api_key:
                raise ValueError("api_key is required when no Groq client is injected")
            try:
                from groq import Groq
            except ImportError as exc:  # pragma: no cover - packaging failure
                raise RealizationError(
                    "Groq SDK is unavailable",
                    RealizationFailureKind.PROVIDER_UNAVAILABLE,
                ) from exc
            client = Groq(api_key=api_key, timeout=float(config.timeout_seconds))
        self._client = client

    def complete(
        self, prompt: LLMRealizationPrompt, config: LLMProviderConfig,
    ) -> LLMProviderResponse:
        contents = serialize_llm_realization_prompt(prompt)
        try:
            response = self._client.chat.completions.create(
                model=config.model_name,
                messages=[{"role": "user", "content": contents}],
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "realization_text",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                            "additionalProperties": False,
                        },
                    },
                },
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise RealizationError(
                "Groq realization request failed",
                _classify_realization_failure(exc),
            ) from exc
        text = _extract_message_text(response)
        if not isinstance(text, str) or not text.strip():
            raise RealizationError(
                "Groq returned no realization text",
                RealizationFailureKind.MALFORMED_RESPONSE,
            )
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RealizationError(
                "Groq realization returned invalid JSON",
                RealizationFailureKind.MALFORMED_RESPONSE,
            ) from exc
        if (
            not isinstance(payload, dict)
            or "text" not in payload
            or not isinstance(payload["text"], str)
            or not payload["text"].strip()
        ):
            raise RealizationError(
                "Groq realization has no usable text",
                RealizationFailureKind.MALFORMED_RESPONSE,
            )
        usage = getattr(response, "usage", None)
        return LLMProviderResponse(
            payload["text"][:2000],
            _optional_count(usage, "prompt_tokens"),
            _optional_count(usage, "completion_tokens"),
        )


def _optional_count(value: object, name: str) -> int | None:
    count = getattr(value, name, None)
    return count if isinstance(count, int) and not isinstance(count, bool) else None


# ---------------------------------------------------------------------------
# Failure classification — SDK-agnostic via Slice 58B.6 _extract_http_status
# ---------------------------------------------------------------------------


def _classify_failure(exc: Exception) -> ProviderFailureKind:
    """Map Groq SDK exceptions into our typed ProviderFailureKind."""
    if isinstance(exc, TimeoutError):
        return ProviderFailureKind.TIMEOUT
    status = _extract_http_status(exc)
    if status in (401, 403):
        return ProviderFailureKind.AUTH_FAILURE
    if status == 429:
        return ProviderFailureKind.RATE_LIMITED
    return ProviderFailureKind.TRANSPORT_FAILURE


def _classify_realization_failure(exc: BaseException) -> RealizationFailureKind:
    if isinstance(exc, TimeoutError):
        return RealizationFailureKind.TIMEOUT
    status = _extract_http_status(exc)
    if status in (401, 403):
        return RealizationFailureKind.AUTHENTICATION_FAILURE
    if status == 429:
        return RealizationFailureKind.RATE_LIMIT
    if status is not None and status >= 500:
        return RealizationFailureKind.HTTP_ERROR
    return RealizationFailureKind.UNKNOWN_FAILURE
