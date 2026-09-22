"""Immutable contracts for controlled synchronous LLM realization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.conversation.context.contracts import ApprovedEvidenceItem
from app.conversation.realization.contracts import HumanConversationPolicy, LeanContextView
from app.conversation.realization.provider import RealizationFailureKind
from app.conversation.response_planning.contracts import ResponsePlan
from app.conversation.sales_cognition.contracts import SalesConversationGuidance
from app.conversation.understanding.contracts import LanguageProfile
from app.providers.contracts import ProviderId


@dataclass(frozen=True)
class LLMProviderConfig:
    """Non-secret generation configuration injected by application composition."""

    provider_id: ProviderId
    model_name: str
    timeout_seconds: float = 15.0
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 256
    structured_output: bool = True
    api_endpoint: str | None = None

    def __post_init__(self) -> None:
        if not self.model_name.strip() or len(self.model_name) > 120:
            raise ValueError("model name must be bounded")
        if not 0 < self.timeout_seconds <= 120:
            raise ValueError("timeout must be between zero and 120 seconds")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between zero and two")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be between zero and one")
        if not 1 <= self.max_tokens <= 2048:
            raise ValueError("max tokens must be between one and 2048")
        if not self.structured_output:
            raise ValueError("live realization requires structured output")
        if self.api_endpoint is not None and not self.api_endpoint.startswith("https://"):
            raise ValueError("custom API endpoint must use HTTPS")


@dataclass(frozen=True)
class LLMRealizationPrompt:
    """Bounded wording context with no secrets, transcript, or authority internals."""

    plan: ResponsePlan
    policy: HumanConversationPolicy
    language_profile: LanguageProfile | None
    approved_evidence: tuple[ApprovedEvidenceItem, ...]
    lean_context: LeanContextView
    service_explanation: str | None
    current_guidance: SalesConversationGuidance | None

    def __post_init__(self) -> None:
        evidence = tuple(self.approved_evidence)
        if len(evidence) > 8:
            raise ValueError("prompt evidence must be bounded")
        if self.service_explanation is not None and (
            not self.service_explanation.strip() or len(self.service_explanation) > 500
        ):
            raise ValueError("service explanation must be bounded")
        object.__setattr__(self, "approved_evidence", evidence)


@dataclass(frozen=True)
class LLMProviderResponse:
    """Normalized adapter response; SDK objects never cross this boundary."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        for count in (self.input_tokens, self.output_tokens):
            if count is not None and count < 0:
                raise ValueError("token counts cannot be negative")


class ProviderCallOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class ValidationOutcome(str, Enum):
    NOT_EVALUATED = "not_evaluated"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True)
class RealizationMetric:
    """Content-free operational metric for one provider call."""

    provider_id: ProviderId
    latency_ms: int
    outcome: ProviderCallOutcome
    validation: ValidationOutcome = ValidationOutcome.NOT_EVALUATED
    input_tokens: int | None = None
    output_tokens: int | None = None
    failure_kind: RealizationFailureKind | None = None
