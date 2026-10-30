"""Provider-neutral adapter interface and one production Gemini implementation."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, Protocol

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


class LLMProviderAdapter(ABC):
    """Synchronous one-call adapter for untrusted wording proposals."""

    @abstractmethod
    def complete(
        self, prompt: LLMRealizationPrompt, config: LLMProviderConfig
    ) -> LLMProviderResponse:
        """Return one normalized response without retrying."""
        raise NotImplementedError


class _GeminiModels(Protocol):
    def generate_content(self, **kwargs: Any) -> Any:
        """Generate one response."""


class _GeminiClient(Protocol):
    models: _GeminiModels


class GeminiRealizationAdapter(LLMProviderAdapter):
    """One-call Gemini wording adapter with injected credentials or client."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        config: LLMProviderConfig,
        client: _GeminiClient | None = None,
    ) -> None:
        if client is None:
            if not api_key:
                raise ValueError("api_key is required when no Gemini client is injected")
            try:
                from google import genai
                from google.genai import types
            except ImportError as exc:  # pragma: no cover - packaging failure only
                raise RealizationError(
                    "Gemini SDK is unavailable",
                    RealizationFailureKind.PROVIDER_UNAVAILABLE,
                ) from exc
            options: dict[str, Any] = {
                "timeout": int(config.timeout_seconds * 1000)
            }
            if config.api_endpoint is not None:
                options["base_url"] = config.api_endpoint
            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(**options),
            )
        self._client = client

    def complete(
        self, prompt: LLMRealizationPrompt, config: LLMProviderConfig
    ) -> LLMProviderResponse:
        """Make exactly one synchronous request and parse its strict JSON envelope."""
        try:
            response = self._client.models.generate_content(
                model=config.model_name,
                contents=serialize_llm_realization_prompt(prompt),
                config={
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "max_output_tokens": config.max_tokens,
                    "response_mime_type": "application/json",
                    "response_json_schema": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                    # Slice 58B — same contract as the reasoning adapter.
                    # This realization adapter is stateless and exposes no
                    # tools; explicitly disable AFC so google-genai does
                    # not emit the Chat.send_message migration warning.
                    "automatic_function_calling": {"disable": True},
                },
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise RealizationError(
                "Gemini realization request failed", _classify_failure(exc)
            ) from exc
        return _parse_response(response)


_MAX_REALIZATION_TEXT_LEN = 2000


def _parse_response(response: object) -> LLMProviderResponse:
    """Slice 58B.5 — resilient parser.

    Gemini's structured-output response surface is model- and SDK-
    version-sensitive. The request-side strict schema
    (`response_json_schema` + `additionalProperties: false`) tells
    Gemini what we want, but the response can arrive in three shapes:

    1. Clean JSON matching the schema: `{"text": "Hi."}`
    2. JSON with extra envelope keys: `{"text": "Hi.", "metadata": {...}}`
    3. Plain prose: `Hi.` (when the model partially drops JSON mode)

    We accept all three — extracting `text` cleanly where possible,
    treating raw prose as the realized text when JSON parsing fails.
    `MALFORMED_RESPONSE` now fires only when the response carries no
    usable text at all. Downstream `validate_realization` continues
    to enforce service/claim/length invariants on whatever we extract.
    """
    raw = getattr(response, "text", None)
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RealizationError(
                "Gemini returned non-UTF-8 response",
                RealizationFailureKind.MALFORMED_RESPONSE,
            ) from exc
    if not isinstance(raw, str) or not raw.strip():
        raise RealizationError(
            "Gemini returned no response text",
            RealizationFailureKind.MALFORMED_RESPONSE,
        )

    text = _extract_realization_text(raw)
    if text is None:
        raise RealizationError(
            "Gemini response has no usable realization text",
            RealizationFailureKind.MALFORMED_RESPONSE,
        )
    usage = getattr(response, "usage_metadata", None)
    return LLMProviderResponse(
        text[:_MAX_REALIZATION_TEXT_LEN],
        _optional_count(usage, "prompt_token_count"),
        _optional_count(usage, "candidates_token_count"),
    )


def _extract_realization_text(raw: str) -> str | None:
    """Return the realization text from the three accepted response shapes.

    Returns None only when NOTHING usable is present.
    """
    stripped = raw.strip()
    # Shape 3: plain prose that doesn't even look like JSON — accept it
    # as the realized text directly. Any `{` or `[` indicates JSON intent.
    if not stripped.startswith("{") and not stripped.startswith("["):
        return stripped
    try:
        payload = json.loads(stripped)
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Shape 3 (edge): looks like JSON but isn't valid — fall back to
        # raw prose if any non-brace content exists. Otherwise fail.
        return stripped if any(ch.isalpha() for ch in stripped) else None
    # Shape 1 or 2: dict with .text — accept.
    if isinstance(payload, dict):
        candidate = payload.get("text")
        if isinstance(candidate, str) and candidate.strip():
            return candidate
        return None
    # Shape unusual: list or scalar JSON — reject.
    return None


def _optional_count(value: object, name: str) -> int | None:
    count = getattr(value, name, None)
    return count if isinstance(count, int) and not isinstance(count, bool) else None


def _classify_failure(exc: BaseException) -> RealizationFailureKind:
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return RealizationFailureKind.TIMEOUT
    if isinstance(exc, asyncio.CancelledError):
        return RealizationFailureKind.CANCELLED
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status in (401, 403):
        return RealizationFailureKind.AUTHENTICATION_FAILURE
    if status == 429:
        return RealizationFailureKind.RATE_LIMIT
    if isinstance(status, int):
        return RealizationFailureKind.HTTP_ERROR
    return RealizationFailureKind.UNKNOWN_FAILURE
