"""Focused Slice 10 tests for dynamic provider metadata and resolution."""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from app.application.production_turn_processor import ProductionTurnProcessor
from app.brain.contracts import BrainProposal
from app.conversation.engine import ConversationEngine
from app.conversation.response_planning.planner import ResponsePlanner
from app.conversation.response_rendering.renderer import ResponseRenderer
from app.providers.contracts import (
    BillingModel,
    CredentialRequirement,
    IncludedUsage,
    PricingItem,
    PricingProvenance,
    PricingSnapshot,
    PricingStatus,
    PricingUnit,
    ProviderCapability,
    ProviderCategory,
    ProviderDescriptor,
    ProviderId,
    ProviderSelection,
    ProviderStatus,
)
from app.providers.registry import ProviderRegistry, ProviderRegistryError
from app.providers.resolver import ProviderResolutionError, ProviderResolver
from app.runtime.turn_coordinator import TurnCoordinator


def _descriptor(
    provider_id: str,
    *,
    status: ProviderStatus = ProviderStatus.AVAILABLE,
    capabilities: frozenset[ProviderCapability] = frozenset(),
    pricing: PricingSnapshot | None = None,
    credentials: tuple[CredentialRequirement, ...] = (),
) -> ProviderDescriptor:
    identity = ProviderId(provider_id)
    return ProviderDescriptor(
        identity,
        identity.category,
        provider_id.split(".")[1].replace("_", " ").title(),
        "Explicit test metadata; no live provider integration.",
        status,
        capabilities,
        pricing,
        credentials,
    )


def _descriptors() -> tuple[ProviderDescriptor, ...]:
    return (
        _descriptor(
            "telephony.fake_a",
            capabilities=frozenset({ProviderCapability.OUTBOUND_CALLING}),
        ),
        _descriptor(
            "telephony.fake_b",
            capabilities=frozenset(
                {ProviderCapability.OUTBOUND_CALLING, ProviderCapability.SIP}
            ),
        ),
        _descriptor(
            "llm.gemini",
            capabilities=frozenset({ProviderCapability.STRUCTURED_OUTPUT}),
        ),
        _descriptor(
            "stt.fake_stt",
            capabilities=frozenset({ProviderCapability.TRANSCRIPTION}),
        ),
        _descriptor(
            "tts.fake_tts",
            capabilities=frozenset({ProviderCapability.SPEECH_SYNTHESIS}),
        ),
    )


def _selection(telephony: str = "telephony.fake_a") -> ProviderSelection:
    return ProviderSelection(
        ProviderId(telephony),
        ProviderId("llm.gemini"),
        ProviderId("stt.fake_stt"),
        ProviderId("tts.fake_tts"),
    )


def _provenance() -> PricingProvenance:
    return PricingProvenance(
        "https://example.test/official-pricing",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        "USD",
        "Official pricing fixture",
    )


def test_provider_descriptor_validates_identity_category_and_capabilities() -> None:
    descriptor = _descriptor(
        "llm.gemini",
        capabilities=frozenset({ProviderCapability.STRUCTURED_OUTPUT}),
    )
    assert descriptor.provider_id.category == ProviderCategory.LLM
    with pytest.raises(ValueError):
        ProviderDescriptor(
            ProviderId("llm.invalid"),
            ProviderCategory.TELEPHONY,
            "Invalid",
            "Invalid category.",
        )


def test_provider_id_rejects_unscoped_or_unknown_category_strings() -> None:
    with pytest.raises(ValueError):
        ProviderId("gemini")
    with pytest.raises(ValueError):
        ProviderId("billing.vendor")
    with pytest.raises(ValueError):
        _descriptor(
            "tts.invalid",
            capabilities=frozenset({ProviderCapability.OUTBOUND_CALLING}),
        )


def test_duplicate_provider_ids_are_rejected() -> None:
    descriptor = _descriptor("llm.gemini")
    with pytest.raises(ProviderRegistryError):
        ProviderRegistry((descriptor, descriptor))


def test_registry_lists_category_in_deterministic_id_order() -> None:
    registry = ProviderRegistry(reversed(_descriptors()))
    telephony = registry.list_by_category(ProviderCategory.TELEPHONY)
    assert [str(item.provider_id) for item in telephony] == [
        "telephony.fake_a",
        "telephony.fake_b",
    ]


def test_valid_telephony_and_llm_selection_resolves_active_stack() -> None:
    stack = ProviderRegistry(_descriptors()).validate_selection(_selection())
    assert stack.telephony.provider_id == ProviderId("telephony.fake_a")
    assert stack.llm.provider_id == ProviderId("llm.gemini")


def test_selection_can_be_built_from_configuration_strings() -> None:
    selection = ProviderSelection.from_mapping(
        {
            "telephony_provider": "telephony.fake_a",
            "llm_provider": "llm.gemini",
            "stt_provider": "stt.fake_stt",
            "tts_provider": "tts.fake_tts",
        }
    )
    assert selection == _selection()


def test_unknown_and_disabled_providers_fail_without_fallback() -> None:
    registry = ProviderRegistry(
        (*_descriptors(), _descriptor("telephony.disabled", status=ProviderStatus.DISABLED))
    )
    with pytest.raises(ProviderRegistryError, match="unknown"):
        registry.require_active(ProviderId("telephony.unknown"))
    with pytest.raises(ProviderRegistryError, match="disabled"):
        registry.require_active(ProviderId("telephony.disabled"))


def test_required_capability_is_checked_before_adapter_use() -> None:
    registry = ProviderRegistry(_descriptors())
    with pytest.raises(ProviderRegistryError, match="lacks required"):
        registry.validate_selection(
            _selection(),
            {ProviderCategory.TELEPHONY: frozenset({ProviderCapability.SIP})},
        )
    stack = registry.validate_selection(
        _selection("telephony.fake_b"),
        {ProviderCategory.TELEPHONY: frozenset({ProviderCapability.SIP})},
    )
    assert ProviderCapability.SIP in stack.telephony.capabilities


@pytest.mark.parametrize(
    ("unit", "model", "amount"),
    [
        (PricingUnit.PER_MINUTE, BillingModel.USAGE, Decimal("0.01")),
        (PricingUnit.PER_MONTH, BillingModel.SUBSCRIPTION, Decimal("20")),
        (PricingUnit.PER_1M_INPUT_TOKENS, BillingModel.USAGE, Decimal("1.25")),
        (PricingUnit.PER_1M_OUTPUT_TOKENS, BillingModel.USAGE, Decimal("5.00")),
    ],
)
def test_pricing_units_represent_distinct_models(
    unit: PricingUnit, model: BillingModel, amount: Decimal
) -> None:
    item = PricingItem("Fixture item", unit, "USD", amount, model)
    snapshot = PricingSnapshot(PricingStatus.VERIFIED, _provenance(), (item,))
    assert snapshot.items[0].unit == unit
    assert snapshot.provenance.official_source_url.startswith("https://")
    assert snapshot.provenance.last_verified_at.tzinfo is not None


def test_unlimited_plan_represents_fair_use_as_bounded_metadata() -> None:
    item = PricingItem(
        "Fixture unlimited plan",
        PricingUnit.UNLIMITED_PLAN,
        "USD",
        Decimal("99"),
        BillingModel.SUBSCRIPTION,
        included_usage=IncludedUsage.UNLIMITED_WITH_LIMITS,
        fair_use_notes="Subject to the configured fair-use policy.",
        concurrency_limit=2,
    )
    assert item.included_usage == IncludedUsage.UNLIMITED_WITH_LIMITS
    assert item.concurrency_limit == 2


def test_unbounded_unlimited_and_invalid_pricing_fail_closed() -> None:
    with pytest.raises(ValueError):
        PricingItem(
            "Unsafe unlimited",
            PricingUnit.UNLIMITED_PLAN,
            "USD",
            Decimal("10"),
            BillingModel.SUBSCRIPTION,
            included_usage=IncludedUsage.UNLIMITED_WITH_LIMITS,
        )
    with pytest.raises(ValueError):
        PricingItem("Bad", "minute", "USD", 0.1, BillingModel.USAGE)  # type: ignore[arg-type]


def test_pricing_snapshot_rejects_currency_mismatch() -> None:
    item = PricingItem(
        "Euro fixture",
        PricingUnit.PER_MINUTE,
        "EUR",
        Decimal("0.01"),
        BillingModel.USAGE,
    )
    with pytest.raises(ValueError, match="currency"):
        PricingSnapshot(PricingStatus.STALE, _provenance(), (item,))


def test_credential_requirements_are_metadata_not_values() -> None:
    requirement = CredentialRequirement("api_key", "API key", secret=True)
    descriptor = _descriptor("llm.gemini", credentials=(requirement,))
    assert descriptor.credential_requirements == (requirement,)
    assert "value" not in {item.name for item in fields(CredentialRequirement)}


def test_dynamic_adapter_resolution_switches_without_vendor_branching() -> None:
    class FakeA:
        pass

    class FakeB:
        pass

    registry = ProviderRegistry(_descriptors())
    resolver = ProviderResolver(
        registry,
        {
            ProviderId("telephony.fake_a"): FakeA,
            ProviderId("telephony.fake_b"): FakeB,
        },
    )
    assert isinstance(resolver.resolve(_selection().telephony_provider), FakeA)
    assert isinstance(
        resolver.resolve(_selection("telephony.fake_b").telephony_provider), FakeB
    )
    with pytest.raises(ProviderResolutionError):
        resolver.resolve(ProviderId("llm.gemini"))


def test_core_domain_and_runtime_classes_have_no_provider_selection_fields() -> None:
    names = {item.name for item in fields(BrainProposal)}
    assert "provider_id" not in names and "pricing" not in names
    assert not hasattr(ConversationEngine, "provider_selection")
    assert not hasattr(ResponsePlanner, "provider_selection")
    assert not hasattr(ResponseRenderer, "provider_selection")
    assert not hasattr(TurnCoordinator, "provider_selection")
    assert not hasattr(ProductionTurnProcessor, "provider_selection")


def test_registry_instances_are_isolated() -> None:
    first = ProviderRegistry((_descriptor("llm.gemini"),))
    second = ProviderRegistry((_descriptor("llm.other"),))
    assert first.list_by_category(ProviderCategory.LLM) != second.list_by_category(
        ProviderCategory.LLM
    )


def test_operational_pricing_cannot_enter_business_authority_contracts() -> None:
    forbidden = {"pricing", "provider_pricing", "operational_cost", "provider_id"}
    assert forbidden.isdisjoint({item.name for item in fields(BrainProposal)})
    from app.brain.authority.models import AuthorityPolicy

    assert forbidden.isdisjoint({item.name for item in fields(AuthorityPolicy)})
