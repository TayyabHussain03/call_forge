"""Provider-neutral configuration, capability, credential, and pricing metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

_PROVIDER_ID = re.compile(r"^(telephony|llm|stt|tts)\.[a-z0-9][a-z0-9_-]{1,63}$")
_KEY = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


class ProviderCategory(str, Enum):
    TELEPHONY = "telephony"
    LLM = "llm"
    STT = "stt"
    TTS = "tts"


@dataclass(frozen=True, order=True)
class ProviderId:
    """Stable validated provider identifier independent of display name."""

    value: str

    def __post_init__(self) -> None:
        if not _PROVIDER_ID.fullmatch(self.value):
            raise ValueError("provider id must use '<category>.<stable_name>'")

    @property
    def category(self) -> ProviderCategory:
        return ProviderCategory(self.value.split(".", 1)[0])

    def __str__(self) -> str:
        return self.value


class ProviderStatus(str, Enum):
    AVAILABLE = "available"
    EXPERIMENTAL = "experimental"
    DISABLED = "disabled"


class ProviderCapability(str, Enum):
    OUTBOUND_CALLING = "outbound_calling"
    INBOUND_CALLING = "inbound_calling"
    US_NUMBERS = "us_numbers"
    SIP = "sip"
    MEDIA_STREAMING = "media_streaming"
    SMS = "sms"
    BARGE_IN_TRANSPORT = "barge_in_transport"
    CONCURRENCY = "concurrency"
    STRUCTURED_OUTPUT = "structured_output"
    TOOL_CALLING = "tool_calling"
    STREAMING = "streaming"
    REASONING = "reasoning"
    TRANSCRIPTION = "transcription"
    SPEECH_SYNTHESIS = "speech_synthesis"


_CAPABILITIES_BY_CATEGORY = {
    ProviderCategory.TELEPHONY: frozenset(
        {
            ProviderCapability.OUTBOUND_CALLING,
            ProviderCapability.INBOUND_CALLING,
            ProviderCapability.US_NUMBERS,
            ProviderCapability.SIP,
            ProviderCapability.MEDIA_STREAMING,
            ProviderCapability.SMS,
            ProviderCapability.BARGE_IN_TRANSPORT,
            ProviderCapability.CONCURRENCY,
        }
    ),
    ProviderCategory.LLM: frozenset(
        {
            ProviderCapability.STRUCTURED_OUTPUT,
            ProviderCapability.TOOL_CALLING,
            ProviderCapability.STREAMING,
            ProviderCapability.REASONING,
        }
    ),
    ProviderCategory.STT: frozenset(
        {ProviderCapability.TRANSCRIPTION, ProviderCapability.STREAMING}
    ),
    ProviderCategory.TTS: frozenset(
        {ProviderCapability.SPEECH_SYNTHESIS, ProviderCapability.STREAMING}
    ),
}


class PricingUnit(str, Enum):
    PER_MINUTE = "per_minute"
    PER_MONTH = "per_month"
    PER_PHONE_NUMBER_MONTH = "per_phone_number_month"
    PER_1M_INPUT_TOKENS = "per_1m_input_tokens"
    PER_1M_OUTPUT_TOKENS = "per_1m_output_tokens"
    PER_CHARACTER = "per_character"
    PER_SECOND = "per_second"
    UNLIMITED_PLAN = "unlimited_plan"
    CUSTOM = "custom"


class BillingModel(str, Enum):
    USAGE = "usage"
    SUBSCRIPTION = "subscription"
    CUSTOM = "custom"


class IncludedUsage(str, Enum):
    METERED = "metered"
    UNLIMITED_WITH_LIMITS = "unlimited_with_limits"
    NOT_APPLICABLE = "not_applicable"


class PricingStatus(str, Enum):
    VERIFIED = "verified"
    STALE = "stale"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PricingItem:
    name: str
    unit: PricingUnit
    currency: str
    amount: Decimal | None
    billing_model: BillingModel
    region: str | None = None
    direction: str | None = None
    tier: str | None = None
    included_usage: IncludedUsage = IncludedUsage.METERED
    fair_use_notes: str | None = None
    concurrency_limit: int | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or len(self.name) > 120:
            raise ValueError("pricing item requires a bounded name")
        if not re.fullmatch(r"[A-Z]{3}", self.currency):
            raise ValueError("pricing currency must be a three-letter ISO code")
        if self.amount is not None and (
            not isinstance(self.amount, Decimal) or self.amount < 0
        ):
            raise ValueError("pricing amount must be a non-negative Decimal")
        if self.unit != PricingUnit.CUSTOM and self.amount is None:
            raise ValueError("non-custom pricing requires an amount")
        if self.included_usage == IncludedUsage.UNLIMITED_WITH_LIMITS and not (
            self.fair_use_notes or self.concurrency_limit is not None
        ):
            raise ValueError("unlimited-with-limits pricing requires explicit limits")
        if self.concurrency_limit is not None and self.concurrency_limit < 1:
            raise ValueError("concurrency limit must be positive")


@dataclass(frozen=True)
class PricingProvenance:
    official_source_url: str
    last_verified_at: datetime
    currency: str
    source_label: str | None = None
    effective_date: str | None = None

    def __post_init__(self) -> None:
        if not self.official_source_url.startswith("https://"):
            raise ValueError("official pricing source must be HTTPS")
        if self.last_verified_at.tzinfo is None:
            raise ValueError("last_verified_at must be timezone-aware")
        if not re.fullmatch(r"[A-Z]{3}", self.currency):
            raise ValueError("provenance currency must be a three-letter ISO code")


@dataclass(frozen=True)
class PricingSnapshot:
    status: PricingStatus
    provenance: PricingProvenance
    items: tuple[PricingItem, ...] = ()

    def __post_init__(self) -> None:
        if any(item.currency != self.provenance.currency for item in self.items):
            raise ValueError("pricing item currency must match snapshot provenance")


@dataclass(frozen=True)
class CredentialRequirement:
    key: str
    display_name: str
    secret: bool = True
    required: bool = True

    def __post_init__(self) -> None:
        if not _KEY.fullmatch(self.key) or not self.display_name.strip():
            raise ValueError("credential requirement metadata is invalid")


@dataclass(frozen=True)
class ProviderDescriptor:
    provider_id: ProviderId
    category: ProviderCategory
    display_name: str
    description: str
    status: ProviderStatus = ProviderStatus.EXPERIMENTAL
    capabilities: frozenset[ProviderCapability] = frozenset()
    pricing: PricingSnapshot | None = None
    credential_requirements: tuple[CredentialRequirement, ...] = ()

    def __post_init__(self) -> None:
        if self.provider_id.category != self.category:
            raise ValueError("provider id category does not match descriptor category")
        if not self.display_name.strip() or len(self.display_name) > 100:
            raise ValueError("provider display name is invalid")
        if not self.description.strip() or len(self.description) > 500:
            raise ValueError("provider description is invalid")
        invalid = self.capabilities - _CAPABILITIES_BY_CATEGORY[self.category]
        if invalid:
            raise ValueError("provider has capabilities invalid for its category")
        keys = [item.key for item in self.credential_requirements]
        if len(keys) != len(set(keys)):
            raise ValueError("credential requirement keys must be unique")


@dataclass(frozen=True)
class ProviderSelection:
    telephony_provider: ProviderId
    llm_provider: ProviderId
    stt_provider: ProviderId
    tts_provider: ProviderId

    def __post_init__(self) -> None:
        expected = (
            (self.telephony_provider, ProviderCategory.TELEPHONY),
            (self.llm_provider, ProviderCategory.LLM),
            (self.stt_provider, ProviderCategory.STT),
            (self.tts_provider, ProviderCategory.TTS),
        )
        if any(provider.category != category for provider, category in expected):
            raise ValueError("provider selection category mismatch")

    @classmethod
    def from_mapping(cls, values: dict[str, str]) -> "ProviderSelection":
        """Build selection from validated configuration strings, never credentials."""
        return cls(
            ProviderId(values["telephony_provider"]),
            ProviderId(values["llm_provider"]),
            ProviderId(values["stt_provider"]),
            ProviderId(values["tts_provider"]),
        )


@dataclass(frozen=True)
class ActiveVoiceStack:
    telephony: ProviderDescriptor
    llm: ProviderDescriptor
    stt: ProviderDescriptor
    tts: ProviderDescriptor
