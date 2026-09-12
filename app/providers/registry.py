"""Immutable injectable provider registry and selection validation."""

from __future__ import annotations

from types import MappingProxyType
from typing import Iterable, Mapping

from app.providers.contracts import (
    ActiveVoiceStack,
    ProviderCapability,
    ProviderCategory,
    ProviderDescriptor,
    ProviderId,
    ProviderSelection,
    ProviderStatus,
)


class ProviderRegistryError(ValueError):
    """Provider metadata or selection is invalid."""


class ProviderRegistry:
    """Read-only registry constructed explicitly during application composition."""

    def __init__(self, descriptors: Iterable[ProviderDescriptor] = ()) -> None:
        values: dict[ProviderId, ProviderDescriptor] = {}
        for descriptor in descriptors:
            if descriptor.provider_id in values:
                raise ProviderRegistryError(
                    f"duplicate provider id: {descriptor.provider_id}"
                )
            values[descriptor.provider_id] = descriptor
        self._descriptors: Mapping[ProviderId, ProviderDescriptor] = MappingProxyType(
            values
        )

    def get(self, provider_id: ProviderId) -> ProviderDescriptor:
        try:
            return self._descriptors[provider_id]
        except KeyError as exc:
            raise ProviderRegistryError(f"unknown provider: {provider_id}") from exc

    def list_by_category(
        self, category: ProviderCategory
    ) -> tuple[ProviderDescriptor, ...]:
        return tuple(
            sorted(
                (
                    item
                    for item in self._descriptors.values()
                    if item.category == category
                ),
                key=lambda item: item.provider_id.value,
            )
        )

    def validate_selection(
        self,
        selection: ProviderSelection,
        required_capabilities: Mapping[
            ProviderCategory, frozenset[ProviderCapability]
        ] | None = None,
    ) -> ActiveVoiceStack:
        required_capabilities = required_capabilities or {}
        resolved = {
            ProviderCategory.TELEPHONY: self.require_active(selection.telephony_provider),
            ProviderCategory.LLM: self.require_active(selection.llm_provider),
            ProviderCategory.STT: self.require_active(selection.stt_provider),
            ProviderCategory.TTS: self.require_active(selection.tts_provider),
        }
        for category, required in required_capabilities.items():
            missing = required - resolved[category].capabilities
            if missing:
                raise ProviderRegistryError(
                    f"provider {resolved[category].provider_id} lacks required capabilities"
                )
        return ActiveVoiceStack(
            resolved[ProviderCategory.TELEPHONY],
            resolved[ProviderCategory.LLM],
            resolved[ProviderCategory.STT],
            resolved[ProviderCategory.TTS],
        )

    def require_active(self, provider_id: ProviderId) -> ProviderDescriptor:
        """Return an enabled descriptor or fail configuration validation."""
        descriptor = self.get(provider_id)
        if descriptor.status == ProviderStatus.DISABLED:
            raise ProviderRegistryError(f"provider is disabled: {provider_id}")
        return descriptor
