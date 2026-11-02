from __future__ import annotations

from app.config.settings import Settings
from app.llm.gemini_provider import GeminiReasoningProvider
from app.llm.providers.reasoning_provider import ReasoningProvider


class UnsupportedLLMProviderError(ValueError):
    """Raised when configured LLM provider has no registered adapter."""


def create_reasoning_provider(settings: Settings) -> ReasoningProvider:
    provider = settings.llm_provider.strip().lower()

    # Accept both historical and current config spelling.
    if provider in {"gemini", "llm.gemini"}:
        if not settings.gemini_api_key.strip():
            raise ValueError(
                "GEMINI_API_KEY is required when LLM_PROVIDER=gemini"
            )

        return GeminiReasoningProvider(
            model=settings.llm_model,
            api_key=settings.gemini_api_key,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    # Slice 58B.7 — Groq alternate provider. Fail-closed on missing key.
    # No automatic Gemini fallback and no silent mock substitution; the
    # operator must set GROQ_API_KEY explicitly for groq to run.
    if provider in {"groq", "llm.groq"}:
        if not settings.groq_api_key.strip():
            raise ValueError(
                "GROQ_API_KEY is required when LLM_PROVIDER=groq"
            )
        # Import lazily so Gemini-only deployments do not need the
        # Groq SDK installed (Slice 58B.7 §45).
        from app.llm.groq_provider import GroqReasoningProvider

        return GroqReasoningProvider(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    raise UnsupportedLLMProviderError(
        f"Unsupported LLM provider: {settings.llm_provider!r}"
    )