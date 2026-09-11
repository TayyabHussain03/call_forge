"""Strict provider-independent parsing for untrusted reasoning output."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    model_validator,
)

from app.brain.business_intelligence import (
    InferredSignal,
    ObservedSignal,
    Provenance,
    SourceKind,
    UnknownSlot,
)
from app.brain.contracts import BrainProposal, CommercialRequest
from app.contracts.contact_info import ContactChannel
from app.contracts.contact_understanding import (
    ContactIntent,
    ContactReference,
    ContactUnderstanding,
)
from app.core.constants import (
    AgentAction,
    CommercialRequestKind,
    Intent,
    Tone,
    TopicCategory,
)
from app.llm.providers.reasoning_provider import ProviderFailureKind, ReasoningError


class ProposalParsingError(ReasoningError):
    """An external response could not be normalized into a BrainProposal."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _CommercialPayload(_StrictModel):
    kind: Literal[tuple(item.value for item in CommercialRequestKind)]  # type: ignore[valid-type]
    requested_discount_percent: StrictFloat | StrictInt | None

    @model_validator(mode="after")
    def _consistent_discount(self) -> "_CommercialPayload":
        if self.kind == CommercialRequestKind.DISCOUNT.value:
            if self.requested_discount_percent is None:
                raise ValueError("discount requires requested_discount_percent")
            if not 0 <= self.requested_discount_percent <= 100:
                raise ValueError("requested_discount_percent must be between 0 and 100")
        elif self.requested_discount_percent is not None:
            raise ValueError("requested_discount_percent is only valid for discount")
        return self


class _ContactPayload(_StrictModel):
    intent: Literal[tuple(item.value for item in ContactIntent)]  # type: ignore[valid-type]
    channel: Literal[tuple(item.value for item in ContactChannel)]  # type: ignore[valid-type]
    value: StrictStr | None
    reference: (
        Literal[tuple(item.value for item in ContactReference)] | None  # type: ignore[valid-type]
    )
    interpretation_confidence: Annotated[StrictFloat | StrictInt, Field(ge=0, le=1)]

    @model_validator(mode="after")
    def _unambiguous_source(self) -> "_ContactPayload":
        if self.value is not None and self.reference is not None:
            raise ValueError("contact value and reference cannot both be supplied")
        return self


class _ProvenancePayload(_StrictModel):
    source_kind: Literal[tuple(item.value for item in SourceKind)]  # type: ignore[valid-type]
    source_turn: StrictInt | None
    detail: StrictStr | None


class _ObservedPayload(_StrictModel):
    kind: Literal["observed"]
    field: StrictStr
    value: StrictStr
    provenance: _ProvenancePayload


class _InferredPayload(_StrictModel):
    kind: Literal["inferred"]
    field: StrictStr
    value: StrictStr
    inference_confidence: Annotated[StrictFloat | StrictInt, Field(ge=0, le=1)]
    basis: list[StrictStr]
    source_turn: StrictInt | None


class _UnknownPayload(_StrictModel):
    kind: Literal["unknown"]
    field: StrictStr


_IntelligencePayload = Annotated[
    _ObservedPayload | _InferredPayload | _UnknownPayload,
    Field(discriminator="kind"),
]


class _BrainProposalPayload(_StrictModel):
    """External schema. Every key is explicit; omitted and extra keys fail."""

    detected_intent: Literal[tuple(item.value for item in Intent)]  # type: ignore[valid-type]
    tone: Literal[tuple(item.value for item in Tone)]  # type: ignore[valid-type]
    topic_category: Literal[tuple(item.value for item in TopicCategory)]  # type: ignore[valid-type]
    proposed_action: (
        Literal[tuple(item.value for item in AgentAction)] | None  # type: ignore[valid-type]
    )
    needs_service_decision: StrictBool
    involves_contact: StrictBool
    commercial_request: _CommercialPayload | None
    contact_understanding: _ContactPayload | None
    intelligence_updates: list[_IntelligencePayload]
    proposed_goal_update: StrictStr | None
    reasoning: StrictStr | None
    action_confidence: Annotated[StrictFloat | StrictInt, Field(ge=0, le=1)]


def brain_proposal_json_schema() -> dict[str, Any]:
    """Return the provider-neutral JSON schema for structured generation."""
    return _BrainProposalPayload.model_json_schema()


def parse_brain_proposal(payload: str | bytes | Mapping[str, Any]) -> BrainProposal:
    """Parse strict external JSON/mapping into the existing untrusted proposal."""
    try:
        raw: Any
        if isinstance(payload, (str, bytes)):
            raw = json.loads(payload, object_pairs_hook=_unique_object)
        elif isinstance(payload, Mapping):
            raw = dict(payload)
        else:
            raise ProposalParsingError(
                "provider response must be a JSON object",
                ProviderFailureKind.INVALID_RESPONSE,
            )
        if not isinstance(raw, dict):
            raise ProposalParsingError(
                "provider response must be a JSON object",
                ProviderFailureKind.INVALID_RESPONSE,
            )
        parsed = _BrainProposalPayload.model_validate(raw)
    except ProposalParsingError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        kind = (
            ProviderFailureKind.SCHEMA_VALIDATION_FAILURE
            if isinstance(exc, ValidationError)
            else ProviderFailureKind.INVALID_RESPONSE
        )
        raise ProposalParsingError("invalid structured provider response", kind) from exc

    return BrainProposal(
        detected_intent=Intent(parsed.detected_intent),
        tone=Tone(parsed.tone),
        topic_category=TopicCategory(parsed.topic_category),
        proposed_action=(
            AgentAction(parsed.proposed_action) if parsed.proposed_action is not None else None
        ),
        needs_service_decision=parsed.needs_service_decision,
        involves_contact=parsed.involves_contact,
        commercial_request=_commercial(parsed.commercial_request),
        contact_understanding=_contact(parsed.contact_understanding),
        intelligence_updates=tuple(_intelligence(item) for item in parsed.intelligence_updates),
        proposed_goal_update=parsed.proposed_goal_update,
        reasoning=parsed.reasoning,
        action_confidence=float(parsed.action_confidence),
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProposalParsingError(
                "duplicate key in provider response",
                ProviderFailureKind.INVALID_RESPONSE,
            )
        result[key] = value
    return result


def _commercial(value: _CommercialPayload | None) -> CommercialRequest | None:
    if value is None:
        return None
    return CommercialRequest(
        CommercialRequestKind(value.kind),
        float(value.requested_discount_percent)
        if value.requested_discount_percent is not None
        else None,
    )


def _contact(value: _ContactPayload | None) -> ContactUnderstanding | None:
    if value is None:
        return None
    return ContactUnderstanding(
        intent=ContactIntent(value.intent),
        channel=ContactChannel(value.channel),
        value=value.value,
        reference=ContactReference(value.reference) if value.reference is not None else None,
        interpretation_confidence=float(value.interpretation_confidence),
    )


def _intelligence(
    value: _ObservedPayload | _InferredPayload | _UnknownPayload,
) -> ObservedSignal | InferredSignal | UnknownSlot:
    if isinstance(value, _ObservedPayload):
        return ObservedSignal(
            field=value.field,
            value=value.value,
            provenance=Provenance(
                source_kind=SourceKind(value.provenance.source_kind),
                source_turn=value.provenance.source_turn,
                detail=value.provenance.detail,
            ),
        )
    if isinstance(value, _InferredPayload):
        return InferredSignal(
            field=value.field,
            value=value.value,
            inference_confidence=float(value.inference_confidence),
            basis=tuple(value.basis),
            source_turn=value.source_turn,
        )
    return UnknownSlot(field=value.field)
