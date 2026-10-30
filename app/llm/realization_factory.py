"""Composition helper that builds the live Gemini realization renderer.

Wires the existing building blocks:
    GeminiRealizationAdapter → LLMProviderRouter → GuardedConversationRealizer

The caller supplies a deterministic fallback renderer. Any provider
failure, timeout, invalid response, or validation rejection inside the
realizer silently falls back to that deterministic renderer — the one
the system has shipped since Slice 51 and before.

Nothing in this file is Gemini-specific beyond the single
`GeminiRealizationAdapter` adapter factory callback registered with
the resolver. The runtime, orchestrator, routes, and renderer protocol
remain provider-neutral.
"""

from __future__ import annotations

from app.config.settings import Settings
from app.conversation.realization.adapter import GeminiRealizationAdapter
from app.conversation.realization.live_contracts import LLMProviderConfig
from app.conversation.realization.realizer import GuardedConversationRealizer
from app.conversation.realization.router import LLMProviderRouter
from app.conversation.response_rendering.renderer import ResponseRenderer
from app.providers.contracts import (
    ProviderCapability,
    ProviderCategory,
    ProviderDescriptor,
    ProviderId,
    ProviderStatus,
)
from app.providers.registry import ProviderRegistry
from app.providers.resolver import ProviderResolver


_GEMINI_PROVIDER_ID = ProviderId("llm.gemini")


def _descriptor() -> ProviderDescriptor:
    return ProviderDescriptor(
        provider_id=_GEMINI_PROVIDER_ID,
        category=ProviderCategory.LLM,
        display_name="Gemini",
        description="Google Gemini LLM provider for conversation realization.",
        status=ProviderStatus.AVAILABLE,
        capabilities=frozenset({ProviderCapability.STRUCTURED_OUTPUT}),
        credential_requirements=(),
    )


def _config_from_settings(settings: Settings) -> LLMProviderConfig:
    return LLMProviderConfig(
        provider_id=_GEMINI_PROVIDER_ID,
        model_name=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
        temperature=settings.llm_temperature,
        top_p=settings.llm_top_p,
        max_tokens=settings.llm_max_tokens,
        structured_output=settings.llm_structured_output,
    )


def create_realization_renderer(
    settings: Settings,
    fallback_renderer: ResponseRenderer,
) -> ResponseRenderer:
    """Build the live Gemini `GuardedConversationRealizer`.

    Fails closed on missing credentials or unsupported LLM provider —
    this helper is intentionally Gemini-only and will not silently
    fall back to the deterministic renderer at composition time (that
    would mask misconfiguration). Runtime provider failures DO fall
    back to the deterministic renderer; configuration failures do not.
    """
    provider = settings.llm_provider.strip().lower()
    if provider not in {"gemini", "llm.gemini"}:
        raise ValueError(
            f"realization factory supports Gemini only (got '{settings.llm_provider}')"
        )
    if not settings.gemini_api_key.strip():
        raise ValueError(
            "GEMINI_API_KEY is required to build the live realization renderer"
        )
    cfg = _config_from_settings(settings)
    registry = ProviderRegistry([_descriptor()])
    api_key = settings.gemini_api_key
    resolver = ProviderResolver(
        registry,
        {
            _GEMINI_PROVIDER_ID: lambda: GeminiRealizationAdapter(
                api_key=api_key, config=cfg,
            ),
        },
    )
    router = LLMProviderRouter(
        registry=registry,
        config=cfg,
        resolver=resolver,
    )
    return GuardedConversationRealizer(
        provider=router,
        fallback_renderer=fallback_renderer,
    )
