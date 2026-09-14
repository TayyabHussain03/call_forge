"""Immutable contracts for advisory multilingual turn understanding."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

from app.conversation.prospect_intelligence.contracts import (
    CurrentSolutionEvidence,
    DecisionAuthority,
    ObjectionType,
    PainEvidence,
    PreferredNextStep,
    ProspectIntelligenceSummary,
    ProspectRole,
)
from app.conversation.response_planning.contracts import (
    AddresseeStatus,
    InterruptionCategory,
    InterruptionContext,
)
from app.conversation.strategy.contracts import ConversationStrategy
from app.core.constants import CommercialRequestKind, ConversationState


class LanguageScript(str, Enum):
    LATIN = "latin"
    URDU_ARABIC = "urdu_arabic"
    DEVANAGARI = "devanagari"
    OTHER = "other"
    UNKNOWN = "unknown"


class ConversationalRegister(str, Enum):
    CASUAL = "casual"
    NEUTRAL = "neutral"
    FORMAL = "formal"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LanguageProfile:
    """Current-turn language metadata with no locale or authority semantics."""

    primary_language: str
    secondary_language: str | None = None
    mixed_language: bool = False
    script: LanguageScript = LanguageScript.UNKNOWN
    conversational_register: ConversationalRegister = ConversationalRegister.UNKNOWN
    preferred_response_language: str | None = None
    preferred_script: LanguageScript | None = None

    def __post_init__(self) -> None:
        _language(self.primary_language, "primary language")
        if self.secondary_language is not None:
            _language(self.secondary_language, "secondary language")
        if self.preferred_response_language is not None:
            _language(self.preferred_response_language, "preferred response language")
            if self.preferred_response_language not in {
                self.primary_language,
                self.secondary_language,
            }:
                raise ValueError("response language must be present in the current turn")
        if not isinstance(self.mixed_language, bool):
            raise TypeError("mixed language flag must be boolean")
        if self.mixed_language != (self.secondary_language is not None):
            raise ValueError("mixed language requires exactly one secondary language")
        if not isinstance(self.script, LanguageScript):
            raise TypeError("script must be a LanguageScript")
        if not isinstance(self.conversational_register, ConversationalRegister):
            raise TypeError("register must be a ConversationalRegister")
        if self.preferred_script is not None and not isinstance(
            self.preferred_script, LanguageScript
        ):
            raise TypeError("preferred script must be a LanguageScript")
        if (self.preferred_response_language is None) != (
            self.preferred_script is None
        ):
            raise ValueError("preferred response language and script must be paired")


class EvidenceBasis(str, Enum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"


ValueT = TypeVar("ValueT")


@dataclass(frozen=True)
class MeaningObservation(Generic[ValueT]):
    """One bounded interpretation that keeps explicitness and confidence separate."""

    value: ValueT
    basis: EvidenceBasis
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.basis, EvidenceBasis):
            raise TypeError("observation basis must be EvidenceBasis")
        if not isinstance(self.confidence, (int, float)) or isinstance(
            self.confidence, bool
        ):
            raise TypeError("observation confidence must be numeric")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("observation confidence must be between zero and one")
        if isinstance(self.value, str) and not isinstance(self.value, Enum):
            _text(self.value, "observation value", 160)


class SemanticIntent(str, Enum):
    ROLE_INFORMATION = "role_information"
    AVAILABILITY = "availability"
    INTEREST = "interest"
    CURRENT_SOLUTION = "current_solution"
    PAIN = "pain"
    OBJECTION = "objection"
    HUMAN_REQUEST = "human_request"
    NEXT_STEP_REQUEST = "next_step_request"
    COMMERCIAL_QUESTION = "commercial_question"
    GENERAL_QUESTION = "general_question"
    OTHER = "other"


@dataclass(frozen=True)
class CommercialRequestMeaning:
    """An interpreted commercial request, never commercial authority."""

    kind: CommercialRequestKind
    requested_discount_percent: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CommercialRequestKind):
            raise TypeError("commercial kind must be CommercialRequestKind")
        if self.requested_discount_percent is not None and (
            not isinstance(self.requested_discount_percent, (int, float))
            or isinstance(self.requested_discount_percent, bool)
        ):
            raise TypeError("requested discount percentage must be numeric")
        if self.kind == CommercialRequestKind.DISCOUNT:
            if self.requested_discount_percent is None or not (
                0 <= self.requested_discount_percent <= 100
            ):
                raise ValueError(
                    "discount request requires a percentage from zero to 100"
                )
        elif self.requested_discount_percent is not None:
            raise ValueError("only discount requests may contain a percentage")


@dataclass(frozen=True)
class FreeTextUnderstanding:
    """Strict advisory meaning; deliberately lacks authority and execution fields."""

    language_profile: LanguageProfile
    semantic_intents: tuple[SemanticIntent, ...] = ()
    role_observation: MeaningObservation[ProspectRole] | None = None
    referenced_role_observation: MeaningObservation[ProspectRole] | None = None
    authority_observation: MeaningObservation[DecisionAuthority] | None = None
    current_solution_observation: (
        MeaningObservation[CurrentSolutionEvidence] | None
    ) = None
    pain_observation: MeaningObservation[PainEvidence] | None = None
    interest_observation: MeaningObservation[bool] | None = None
    busy_observation: MeaningObservation[bool] | None = None
    objection_observation: MeaningObservation[ObjectionType] | None = None
    timing_observation: MeaningObservation[str] | None = None
    next_step_request: MeaningObservation[PreferredNextStep] | None = None
    explicit_human_request: MeaningObservation[bool] | None = None
    ambiguity: bool = False
    commercial_request: CommercialRequestMeaning | None = None
    confidence: float = 0.0

    def __post_init__(self) -> None:
        intents = tuple(self.semantic_intents)
        if len(intents) > 6 or len(set(intents)) != len(intents):
            raise ValueError("semantic intents must be unique and limited to six")
        if any(not isinstance(item, SemanticIntent) for item in intents):
            raise TypeError("semantic intents must contain SemanticIntent values")
        if not isinstance(self.language_profile, LanguageProfile):
            raise TypeError("language profile must be LanguageProfile")
        if not isinstance(self.ambiguity, bool):
            raise TypeError("ambiguity must be boolean")
        if not isinstance(self.confidence, (int, float)) or isinstance(
            self.confidence, bool
        ):
            raise TypeError("confidence must be numeric")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between zero and one")
        if self.commercial_request is not None and not isinstance(
            self.commercial_request, CommercialRequestMeaning
        ):
            raise TypeError("commercial request has an invalid type")
        _typed_observation(self.role_observation, ProspectRole, "role")
        _typed_observation(
            self.referenced_role_observation, ProspectRole, "referenced role"
        )
        _typed_observation(
            self.authority_observation, DecisionAuthority, "authority"
        )
        _typed_observation(
            self.current_solution_observation,
            CurrentSolutionEvidence,
            "current solution",
        )
        _typed_observation(self.pain_observation, PainEvidence, "pain")
        _typed_observation(self.interest_observation, bool, "interest")
        _typed_observation(self.busy_observation, bool, "busy")
        _typed_observation(self.objection_observation, ObjectionType, "objection")
        _typed_observation(self.timing_observation, str, "timing")
        _typed_observation(
            self.next_step_request, PreferredNextStep, "next step request"
        )
        _typed_observation(
            self.explicit_human_request, bool, "explicit human request"
        )
        object.__setattr__(self, "semantic_intents", intents)


@dataclass(frozen=True)
class FreeTextUnderstandingInput:
    """Minimal current-turn projection derived from canonical Lean Context."""

    current_turn_id: str
    current_user_message: str
    current_state: ConversationState
    prospect_summary: ProspectIntelligenceSummary | None = None
    current_strategy: ConversationStrategy | None = None
    interruption: InterruptionContext = InterruptionContext()
    addressee_status: AddresseeStatus = AddresseeStatus.ADDRESSED_TO_AGENT
    conversation_category: InterruptionCategory = InterruptionCategory.OTHER

    def __post_init__(self) -> None:
        _text(self.current_turn_id, "current turn id", 100)
        _text(self.current_user_message, "current user message", 2000)
        if not isinstance(self.current_state, ConversationState):
            raise TypeError("current state must be ConversationState")
        if self.prospect_summary is not None and not isinstance(
            self.prospect_summary, ProspectIntelligenceSummary
        ):
            raise TypeError("prospect summary has an invalid type")
        if self.current_strategy is not None and not isinstance(
            self.current_strategy, ConversationStrategy
        ):
            raise TypeError("current strategy has an invalid type")
        if not isinstance(self.interruption, InterruptionContext):
            raise TypeError("interruption has an invalid type")
        if not isinstance(self.addressee_status, AddresseeStatus):
            raise TypeError("addressee status has an invalid type")
        if not isinstance(self.conversation_category, InterruptionCategory):
            raise TypeError("conversation category has an invalid type")


def _typed_observation(value: object, expected: type[object], name: str) -> None:
    if value is not None and (
        not isinstance(value, MeaningObservation)
        or not isinstance(value.value, expected)
    ):
        raise TypeError(f"{name} observation has an invalid type")


def _language(value: str, name: str) -> None:
    _text(value, name, 35)
    if not value[0].isalpha() or any(
        not (character.isalnum() or character == "-") for character in value
    ):
        raise ValueError(f"{name} must be a bounded language tag")


def _text(value: str, name: str, limit: int) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must contain 1-{limit} characters")
