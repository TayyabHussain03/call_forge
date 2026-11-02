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
_GROQ_PROVIDER_ID = ProviderId("llm.groq")


def _descriptor(provider_id: ProviderId, display_name: str, description: str) -> ProviderDescriptor:
    return ProviderDescriptor(
        provider_id=provider_id,
        category=ProviderCategory.LLM,
        display_name=display_name,
        description=description,
        status=ProviderStatus.AVAILABLE,
        capabilities=frozenset({ProviderCapability.STRUCTURED_OUTPUT}),
        credential_requirements=(),
    )


def _config_from_settings(settings: Settings, provider_id: ProviderId) -> LLMProviderConfig:
    return LLMProviderConfig(
        provider_id=provider_id,
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
    """Build the live `GuardedConversationRealizer` for the configured provider.

    Fails closed on missing credentials or unsupported LLM provider —
    composition-time misconfiguration must not silently degrade to the
    deterministic renderer (that would mask config errors). RUNTIME
    provider failures still fall back to the deterministic renderer.

    Slice 58B.7 — supports both Gemini (`gemini`/`llm.gemini`) and Groq
    (`groq`/`llm.groq`). Provider selection follows `LLM_PROVIDER`;
    only the chosen provider's API key is required.
    """
    provider = settings.llm_provider.strip().lower()
    if provider in {"gemini", "llm.gemini"}:
        if not settings.gemini_api_key.strip():
            raise ValueError(
                "GEMINI_API_KEY is required to build the live realization renderer"
            )
        descriptor = _descriptor(
            _GEMINI_PROVIDER_ID, "Gemini",
            "Google Gemini LLM provider for conversation realization.",
        )
        cfg = _config_from_settings(settings, _GEMINI_PROVIDER_ID)
        api_key = settings.gemini_api_key
        adapter_factory = lambda: GeminiRealizationAdapter(
            api_key=api_key, config=cfg,
        )
        provider_id = _GEMINI_PROVIDER_ID
    elif provider in {"groq", "llm.groq"}:
        if not settings.groq_api_key.strip():
            raise ValueError(
                "GROQ_API_KEY is required to build the live realization renderer"
            )
        from app.llm.groq_provider import GroqRealizationAdapter
        descriptor = _descriptor(
            _GROQ_PROVIDER_ID, "Groq",
            "Groq LLM provider for conversation realization.",
        )
        cfg = _config_from_settings(settings, _GROQ_PROVIDER_ID)
        api_key = settings.groq_api_key
        adapter_factory = lambda: GroqRealizationAdapter(
            api_key=api_key, config=cfg,
        )
        provider_id = _GROQ_PROVIDER_ID
    else:
        raise ValueError(
            f"realization factory supports Gemini + Groq only "
            f"(got '{settings.llm_provider}')"
        )

    registry = ProviderRegistry([descriptor])
    resolver = ProviderResolver(registry, {provider_id: adapter_factory})
    router = LLMProviderRouter(
        registry=registry,
        config=cfg,
        resolver=resolver,
    )
    return GuardedConversationRealizer(
        provider=router,
        fallback_renderer=fallback_renderer,
    )
