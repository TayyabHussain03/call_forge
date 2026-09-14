"""Strict parsing for untrusted multilingual-understanding output."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt
from pydantic import StrictStr, ValidationError, model_validator

from app.conversation.prospect_intelligence.contracts import (
    CurrentSolutionEvidence,
    DecisionAuthority,
    ObjectionType,
    PainCategory,
    PainEvidence,
    PreferredNextStep,
    ProspectRole,
    SolutionSatisfaction,
)
from app.conversation.understanding.contracts import (
    CommercialRequestMeaning,
    ConversationalRegister,
    EvidenceBasis,
    FreeTextUnderstanding,
    LanguageProfile,
    LanguageScript,
    MeaningObservation,
    SemanticIntent,
)
from app.conversation.understanding.provider import UnderstandingError
from app.core.constants import CommercialRequestKind
from app.llm.providers.reasoning_provider import ProviderFailureKind


class UnderstandingParsingError(UnderstandingError):
    """External output could not be normalized into the strict domain contract."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


BasisLiteral = Literal[tuple(item.value for item in EvidenceBasis)]  # type: ignore[valid-type]
Confidence = Annotated[StrictFloat | StrictInt, Field(ge=0, le=1)]


class _LanguagePayload(_StrictModel):
    primary_language: StrictStr
    secondary_language: StrictStr | None
    mixed_language: StrictBool
    script: Literal[tuple(item.value for item in LanguageScript)]  # type: ignore[valid-type]
    conversational_register: Literal[  # type: ignore[valid-type]
        tuple(item.value for item in ConversationalRegister)
    ]
    preferred_response_language: StrictStr | None
    preferred_script: (
        Literal[tuple(item.value for item in LanguageScript)] | None  # type: ignore[valid-type]
    )


class _RolePayload(_StrictModel):
    value: Literal[tuple(item.value for item in ProspectRole)]  # type: ignore[valid-type]
    basis: BasisLiteral
    confidence: Confidence


class _AuthorityPayload(_StrictModel):
    value: Literal[tuple(item.value for item in DecisionAuthority)]  # type: ignore[valid-type]
    basis: BasisLiteral
    confidence: Confidence


class _BooleanPayload(_StrictModel):
    value: StrictBool
    basis: BasisLiteral
    confidence: Confidence


class _TextPayload(_StrictModel):
    value: StrictStr
    basis: BasisLiteral
    confidence: Confidence


class _NextStepPayload(_StrictModel):
    value: Literal[tuple(item.value for item in PreferredNextStep)]  # type: ignore[valid-type]
    basis: BasisLiteral
    confidence: Confidence


class _ObjectionPayload(_StrictModel):
    value: Literal[tuple(item.value for item in ObjectionType)]  # type: ignore[valid-type]
    basis: BasisLiteral
    confidence: Confidence


class _SolutionValuePayload(_StrictModel):
    name: StrictStr
    category: StrictStr | None
    satisfaction: Literal[  # type: ignore[valid-type]
        tuple(item.value for item in SolutionSatisfaction)
    ]


class _SolutionPayload(_StrictModel):
    value: _SolutionValuePayload
    basis: BasisLiteral
    confidence: Confidence


class _PainValuePayload(_StrictModel):
    category: Literal[tuple(item.value for item in PainCategory)]  # type: ignore[valid-type]
    summary: StrictStr


class _PainPayload(_StrictModel):
    value: _PainValuePayload
    basis: BasisLiteral
    confidence: Confidence


class _CommercialPayload(_StrictModel):
    kind: Literal[tuple(item.value for item in CommercialRequestKind)]  # type: ignore[valid-type]
    requested_discount_percent: StrictFloat | StrictInt | None

    @model_validator(mode="after")
    def _consistent_discount(self) -> "_CommercialPayload":
        if self.kind == CommercialRequestKind.DISCOUNT.value:
            if self.requested_discount_percent is None or not (
                0 <= self.requested_discount_percent <= 100
            ):
                raise ValueError("discount requires a percentage from zero to 100")
        elif self.requested_discount_percent is not None:
            raise ValueError("only discount may contain a percentage")
        return self


class _UnderstandingPayload(_StrictModel):
    """Every external key is explicit; omission and additional keys both fail."""

    language_profile: _LanguagePayload
    semantic_intents: Annotated[
        list[Literal[tuple(item.value for item in SemanticIntent)]],  # type: ignore[valid-type]
        Field(max_length=6),
    ]
    role_observation: _RolePayload | None
    referenced_role_observation: _RolePayload | None
    authority_observation: _AuthorityPayload | None
    current_solution_observation: _SolutionPayload | None
    pain_observation: _PainPayload | None
    interest_observation: _BooleanPayload | None
    busy_observation: _BooleanPayload | None
    objection_observation: _ObjectionPayload | None
    timing_observation: _TextPayload | None
    next_step_request: _NextStepPayload | None
    explicit_human_request: _BooleanPayload | None
    ambiguity: StrictBool
    commercial_request: _CommercialPayload | None
    confidence: Confidence


def free_text_understanding_json_schema() -> dict[str, Any]:
    """Return the provider-neutral strict response schema."""
    return _UnderstandingPayload.model_json_schema()


def parse_free_text_understanding(
    payload: str | bytes | Mapping[str, Any],
) -> FreeTextUnderstanding:
    """Parse one untrusted structured response or fail closed."""
    try:
        raw: Any
        if isinstance(payload, (str, bytes)):
            raw = json.loads(payload, object_pairs_hook=_unique_object)
        elif isinstance(payload, Mapping):
            raw = dict(payload)
        else:
            raise UnderstandingParsingError(
                "understanding response must be a JSON object",
                ProviderFailureKind.INVALID_RESPONSE,
            )
        if not isinstance(raw, dict):
            raise UnderstandingParsingError(
                "understanding response must be a JSON object",
                ProviderFailureKind.INVALID_RESPONSE,
            )
        parsed = _UnderstandingPayload.model_validate(raw)
        return _understanding(parsed)
    except UnderstandingParsingError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError) as exc:
        kind = (
            ProviderFailureKind.SCHEMA_VALIDATION_FAILURE
            if isinstance(exc, ValidationError)
            else ProviderFailureKind.INVALID_RESPONSE
        )
        raise UnderstandingParsingError("invalid understanding response", kind) from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise UnderstandingParsingError(
                "duplicate key in understanding response",
                ProviderFailureKind.INVALID_RESPONSE,
            )
        result[key] = value
    return result


def _understanding(value: _UnderstandingPayload) -> FreeTextUnderstanding:
    language = value.language_profile
    return FreeTextUnderstanding(
        language_profile=LanguageProfile(
            language.primary_language,
            language.secondary_language,
            language.mixed_language,
            LanguageScript(language.script),
            ConversationalRegister(language.conversational_register),
            language.preferred_response_language,
            (
                LanguageScript(language.preferred_script)
                if language.preferred_script is not None
                else None
            ),
        ),
        semantic_intents=tuple(SemanticIntent(item) for item in value.semantic_intents),
        role_observation=_observation(value.role_observation, ProspectRole),
        referenced_role_observation=_observation(
            value.referenced_role_observation, ProspectRole
        ),
        authority_observation=_observation(
            value.authority_observation, DecisionAuthority
        ),
        current_solution_observation=_solution(value.current_solution_observation),
        pain_observation=_pain(value.pain_observation),
        interest_observation=_observation(value.interest_observation, bool),
        busy_observation=_observation(value.busy_observation, bool),
        objection_observation=_observation(value.objection_observation, ObjectionType),
        timing_observation=_observation(value.timing_observation, str),
        next_step_request=_observation(value.next_step_request, PreferredNextStep),
        explicit_human_request=_observation(value.explicit_human_request, bool),
        ambiguity=value.ambiguity,
        commercial_request=(
            CommercialRequestMeaning(
                CommercialRequestKind(value.commercial_request.kind),
                (
                    float(value.commercial_request.requested_discount_percent)
                    if value.commercial_request.requested_discount_percent is not None
                    else None
                ),
            )
            if value.commercial_request is not None
            else None
        ),
        confidence=float(value.confidence),
    )


def _observation(value: Any, value_type: type[Any]) -> MeaningObservation[Any] | None:
    if value is None:
        return None
    raw = value.value
    converted = raw if value_type in {bool, str} else value_type(raw)
    return MeaningObservation(converted, EvidenceBasis(value.basis), float(value.confidence))


def _solution(
    value: _SolutionPayload | None,
) -> MeaningObservation[CurrentSolutionEvidence] | None:
    if value is None:
        return None
    solution = CurrentSolutionEvidence(
        value.value.name,
        value.value.category,
        SolutionSatisfaction(value.value.satisfaction),
    )
    return MeaningObservation(solution, EvidenceBasis(value.basis), float(value.confidence))


def _pain(
    value: _PainPayload | None,
) -> MeaningObservation[PainEvidence] | None:
    if value is None:
        return None
    pain = PainEvidence(PainCategory(value.value.category), value.value.summary)
    return MeaningObservation(pain, EvidenceBasis(value.basis), float(value.confidence))
