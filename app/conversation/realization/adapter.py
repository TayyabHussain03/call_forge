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
                },
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise RealizationError(
                "Gemini realization request failed", _classify_failure(exc)
            ) from exc
        return _parse_response(response)


def _parse_response(response: object) -> LLMProviderResponse:
    raw = getattr(response, "text", None)
    if not isinstance(raw, (str, bytes)):
        raise RealizationError(
            "Gemini returned no structured response",
            RealizationFailureKind.MALFORMED_RESPONSE,
        )
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RealizationError(
            "Gemini returned malformed JSON",
            RealizationFailureKind.MALFORMED_RESPONSE,
        ) from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"text"}
        or not isinstance(payload["text"], str)
    ):
        raise RealizationError(
            "Gemini response schema is invalid",
            RealizationFailureKind.MALFORMED_RESPONSE,
        )
    usage = getattr(response, "usage_metadata", None)
    return LLMProviderResponse(
        payload["text"],
        _optional_count(usage, "prompt_token_count"),
        _optional_count(usage, "candidates_token_count"),
    )


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
