"""Vendor-neutral adapter resolution seam."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

from app.providers.contracts import ProviderId
from app.providers.registry import ProviderRegistry


class ProviderResolutionError(ValueError):
    """A selected provider has no usable registered adapter factory."""


class ProviderResolver:
    """Resolve explicitly selected IDs through injected adapter factories."""

    def __init__(
        self,
        registry: ProviderRegistry,
        adapter_factories: Mapping[ProviderId, Callable[[], Any]],
    ) -> None:
        self._registry = registry
        self._factories = MappingProxyType(dict(adapter_factories))

    def resolve(self, provider_id: ProviderId) -> Any:
        self._registry.require_active(provider_id)
        try:
            factory = self._factories[provider_id]
        except KeyError as exc:
            raise ProviderResolutionError(
                f"no adapter registered for provider: {provider_id}"
            ) from exc
        return factory()
