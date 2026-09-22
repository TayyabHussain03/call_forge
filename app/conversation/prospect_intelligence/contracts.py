"""Bounded immutable contracts for person-scoped prospect intelligence."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, TypeVar


class ProspectRole(str, Enum):
    """Small conversational role taxonomy, not CRM title normalization."""

    OWNER = "owner"
    FOUNDER = "founder"
    DECISION_MAKER = "decision_maker"
    MANAGER = "manager"
    OPERATIONS_MANAGER = "operations_manager"
    TECHNICAL_MANAGER = "technical_manager"
    SALES_MANAGER = "sales_manager"
    FINANCE_CONTACT = "finance_contact"
    ASSISTANT = "assistant"
    ASSISTANT_MANAGER = "assistant_manager"
    RECEPTIONIST = "receptionist"
    GATEKEEPER = "gatekeeper"
    STAFF = "staff"
    UNKNOWN = "unknown"


class DecisionAuthority(str, Enum):
    """Advisory authority description, separate from role and system policy."""

    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    FINAL = "final"


class InformationLevel(str, Enum):
    """Coarse advisory level shared by influence, openness, and urgency."""

    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BuyingStage(str, Enum):
    """Advisory buying context that never maps directly to FSM state."""

    UNKNOWN = "unknown"
    UNAWARE = "unaware"
    EXPLORING = "exploring"
    PROBLEM_AWARE = "problem_aware"
    SOLUTION_AWARE = "solution_aware"
    EVALUATING = "evaluating"
    READY_FOR_NEXT_STEP = "ready_for_next_step"


class ObjectionType(str, Enum):
    """Descriptive objection category without rebuttal or priority authority."""

    NONE = "none"
    PRICE = "price"
    NO_TIME = "no_time"
    ALREADY_HAVE_PROVIDER = "already_have_provider"
    NO_NEED = "no_need"
    NOT_DECISION_MAKER = "not_decision_maker"
    SEND_INFORMATION = "send_information"
    BAD_TIMING = "bad_timing"
    TRUST = "trust"
    IMPLEMENTATION = "implementation"
    INTEGRATION = "integration"
    INTERNAL_TEAM = "internal_team"
    CONTRACT_LOCK = "contract_lock"
    UNKNOWN = "unknown"


class PreferredNextStep(str, Enum):
    """Observed or inferred preference, never a confirmed commitment."""

    UNKNOWN = "unknown"
    CONTINUE_CALL = "continue_call"
    SEND_INFORMATION = "send_information"
    CALLBACK = "callback"
    DEMO = "demo"
    HUMAN_FOLLOW_UP = "human_follow_up"
    EMAIL = "email"
    END_CONVERSATION = "end_conversation"


class SolutionSatisfaction(str, Enum):
    """Bounded current-solution satisfaction, unknown unless evidenced."""

    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PainCategory(str, Enum):
    """Generic bounded pain category."""

    TIME = "time"
    COST = "cost"
    OPERATIONS = "operations"
    GROWTH = "growth"
    QUALITY = "quality"
    IMPLEMENTATION = "implementation"
    INTEGRATION = "integration"
    OTHER = "other"
    UNKNOWN = "unknown"


class EvidenceSourceKind(str, Enum):
    """How person-scoped evidence entered the deterministic boundary."""

    EXPLICIT_STATEMENT = "explicit_statement"
    STRUCTURED_SIGNAL = "structured_signal"
    INFERENCE = "inference"


@dataclass(frozen=True)
class EvidenceProvenance:
    """Bounded provenance without transcript excerpts or chain-of-thought."""

    source_turn_id: str
    source_kind: EvidenceSourceKind

    def __post_init__(self) -> None:
        if not self.source_turn_id or len(self.source_turn_id) > 100:
            raise ValueError("source turn id must contain 1-100 characters")
        if not isinstance(self.source_kind, EvidenceSourceKind):
            raise TypeError("source_kind must be an EvidenceSourceKind")


ValueT = TypeVar("ValueT")


@dataclass(frozen=True)
class ObservedValue(Generic[ValueT]):
    """One explicit/structured value that carries no confidence score."""

    value: ValueT
    provenance: EvidenceProvenance

    def __post_init__(self) -> None:
        if self.provenance.source_kind == EvidenceSourceKind.INFERENCE:
            raise ValueError("observed values cannot use inference provenance")
        _validate_bounded_value(self.value)


@dataclass(frozen=True)
class InferredValue(Generic[ValueT]):
    """One advisory value with confidence that can never become observed."""

    value: ValueT
    confidence: float
    provenance: EvidenceProvenance

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("inference confidence must be between 0 and 1")
        if self.provenance.source_kind != EvidenceSourceKind.INFERENCE:
            raise ValueError("inferred values require inference provenance")
        _validate_bounded_value(self.value)


@dataclass(frozen=True)
class CurrentSolutionContext:
    """Explicit current solution without implied satisfaction."""

    name: str
    category: str | None
    satisfaction: SolutionSatisfaction
    provenance: EvidenceProvenance

    def __post_init__(self) -> None:
        _validate_text(self.name, "solution name", 100)
        if self.category is not None:
            _validate_text(self.category, "solution category", 80)
        if not isinstance(self.satisfaction, SolutionSatisfaction):
            raise TypeError("satisfaction must be SolutionSatisfaction")
        if self.provenance.source_kind == EvidenceSourceKind.INFERENCE:
            raise ValueError("current solution context must be observed")


@dataclass(frozen=True)
class ProspectPain:
    """One explicitly observed, evidence-linked pain point."""

    category: PainCategory
    summary: str
    provenance: EvidenceProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.category, PainCategory):
            raise TypeError("pain category must be PainCategory")
        _validate_text(self.summary, "pain summary", 160)
        if self.provenance.source_kind == EvidenceSourceKind.INFERENCE:
            raise ValueError("observed pain cannot use inference provenance")


@dataclass(frozen=True)
class ObservedProspectFacts:
    """Current person facts grounded only in explicit/structured evidence."""

    explicit_role: ObservedValue[ProspectRole] | None = None
    explicit_current_solution: CurrentSolutionContext | None = None
    explicit_pain_points: tuple[ProspectPain, ...] = ()
    explicit_interest_signal: ObservedValue[bool] | None = None
    explicit_busy_signal: ObservedValue[bool] | None = None
    explicit_human_request: ObservedValue[bool] | None = None
    explicit_next_step_request: ObservedValue[PreferredNextStep] | None = None
    explicit_decision_authority_statement: (
        ObservedValue[DecisionAuthority] | None
    ) = None
    explicit_timing_preference: ObservedValue[str] | None = None
    explicit_objection: ObservedValue[ObjectionType] | None = None

    def __post_init__(self) -> None:
        _validate_observed(self.explicit_role, ProspectRole, "explicit_role")
        _validate_observed(
            self.explicit_interest_signal, bool, "explicit_interest_signal"
        )
        _validate_observed(self.explicit_busy_signal, bool, "explicit_busy_signal")
        _validate_observed(self.explicit_human_request, bool, "explicit_human_request")
        _validate_observed(
            self.explicit_next_step_request,
            PreferredNextStep,
            "explicit_next_step_request",
        )
        _validate_observed(
            self.explicit_decision_authority_statement,
            DecisionAuthority,
            "explicit_decision_authority_statement",
        )
        _validate_observed(
            self.explicit_timing_preference, str, "explicit_timing_preference"
        )
        _validate_observed(self.explicit_objection, ObjectionType, "explicit_objection")
        if self.explicit_current_solution is not None and not isinstance(
            self.explicit_current_solution, CurrentSolutionContext
        ):
            raise TypeError("explicit_current_solution must be CurrentSolutionContext")
        pain = tuple(self.explicit_pain_points)
        if len(pain) > 3 or any(not isinstance(item, ProspectPain) for item in pain):
            raise ValueError("observed pain points must be typed and limited to three")
        object.__setattr__(self, "explicit_pain_points", pain)


@dataclass(frozen=True)
class InferredProspectState:
    """Current advisory inferences, structurally separate from observed facts."""

    likely_role: InferredValue[ProspectRole] | None = None
    decision_authority: InferredValue[DecisionAuthority] | None = None
    influence_level: InferredValue[InformationLevel] | None = None
    buying_stage: InferredValue[BuyingStage] | None = None
    openness: InferredValue[InformationLevel] | None = None
    urgency: InferredValue[InformationLevel] | None = None
    objection_type: InferredValue[ObjectionType] | None = None
    pain_summary: InferredValue[str] | None = None
    current_solution_satisfaction: InferredValue[SolutionSatisfaction] | None = None
    preferred_next_step: InferredValue[PreferredNextStep] | None = None

    def __post_init__(self) -> None:
        for value, expected, name in (
            (self.likely_role, ProspectRole, "likely_role"),
            (self.decision_authority, DecisionAuthority, "decision_authority"),
            (self.influence_level, InformationLevel, "influence_level"),
            (self.buying_stage, BuyingStage, "buying_stage"),
            (self.openness, InformationLevel, "openness"),
            (self.urgency, InformationLevel, "urgency"),
            (self.objection_type, ObjectionType, "objection_type"),
            (self.pain_summary, str, "pain_summary"),
            (
                self.current_solution_satisfaction,
                SolutionSatisfaction,
                "current_solution_satisfaction",
            ),
            (self.preferred_next_step, PreferredNextStep, "preferred_next_step"),
        ):
            _validate_inferred(value, expected, name)


@dataclass(frozen=True)
class ProspectIntelligenceSnapshot:
    """Immutable current person snapshot; no persistence or authority."""

    observed: ObservedProspectFacts = field(default_factory=ObservedProspectFacts)
    inferred: InferredProspectState = field(default_factory=InferredProspectState)

    def __post_init__(self) -> None:
        if not isinstance(self.observed, ObservedProspectFacts):
            raise TypeError("observed must be ObservedProspectFacts")
        if not isinstance(self.inferred, InferredProspectState):
            raise TypeError("inferred must be InferredProspectState")

    def to_brain_summary(self) -> ProspectIntelligenceSummary:
        """Return the bounded current view permitted at the Brain boundary."""
        return ProspectIntelligenceSummary(
            explicit_role=self.observed.explicit_role,
            likely_role=self.inferred.likely_role,
            explicit_decision_authority=(
                self.observed.explicit_decision_authority_statement
            ),
            inferred_decision_authority=self.inferred.decision_authority,
            buying_stage=self.inferred.buying_stage,
            openness=self.inferred.openness,
            urgency=self.inferred.urgency,
            explicit_objection=self.observed.explicit_objection,
            inferred_objection=self.inferred.objection_type,
            current_solution=self.observed.explicit_current_solution,
            observed_pain_points=self.observed.explicit_pain_points,
            inferred_pain_summary=self.inferred.pain_summary,
            explicit_next_step=self.observed.explicit_next_step_request,
            inferred_next_step=self.inferred.preferred_next_step,
            busy=self.observed.explicit_busy_signal,
            interested=self.observed.explicit_interest_signal,
            human_requested=self.observed.explicit_human_request,
        )


@dataclass(frozen=True)
class ProspectIntelligenceSummary:
    """Bounded current-only Brain view preserving fact/inference separation."""

    explicit_role: ObservedValue[ProspectRole] | None = None
    likely_role: InferredValue[ProspectRole] | None = None
    explicit_decision_authority: ObservedValue[DecisionAuthority] | None = None
    inferred_decision_authority: InferredValue[DecisionAuthority] | None = None
    buying_stage: InferredValue[BuyingStage] | None = None
    openness: InferredValue[InformationLevel] | None = None
    urgency: InferredValue[InformationLevel] | None = None
    explicit_objection: ObservedValue[ObjectionType] | None = None
    inferred_objection: InferredValue[ObjectionType] | None = None
    current_solution: CurrentSolutionContext | None = None
    observed_pain_points: tuple[ProspectPain, ...] = ()
    inferred_pain_summary: InferredValue[str] | None = None
    explicit_next_step: ObservedValue[PreferredNextStep] | None = None
    inferred_next_step: InferredValue[PreferredNextStep] | None = None
    busy: ObservedValue[bool] | None = None
    interested: ObservedValue[bool] | None = None
    human_requested: ObservedValue[bool] | None = None

    def __post_init__(self) -> None:
        observed = ObservedProspectFacts(
            explicit_role=self.explicit_role,
            explicit_current_solution=self.current_solution,
            explicit_pain_points=self.observed_pain_points,
            explicit_interest_signal=self.interested,
            explicit_busy_signal=self.busy,
            explicit_human_request=self.human_requested,
            explicit_next_step_request=self.explicit_next_step,
            explicit_decision_authority_statement=self.explicit_decision_authority,
            explicit_objection=self.explicit_objection,
        )
        InferredProspectState(
            likely_role=self.likely_role,
            decision_authority=self.inferred_decision_authority,
            buying_stage=self.buying_stage,
            openness=self.openness,
            urgency=self.urgency,
            objection_type=self.inferred_objection,
            pain_summary=self.inferred_pain_summary,
            preferred_next_step=self.inferred_next_step,
        )
        object.__setattr__(self, "observed_pain_points", observed.explicit_pain_points)


@dataclass(frozen=True)
class CurrentSolutionEvidence:
    """Structured observed solution details before provenance is attached."""

    name: str
    category: str | None = None
    satisfaction: SolutionSatisfaction = SolutionSatisfaction.UNKNOWN

    def __post_init__(self) -> None:
        _validate_text(self.name, "solution name", 100)
        if self.category is not None:
            _validate_text(self.category, "solution category", 80)
        if not isinstance(self.satisfaction, SolutionSatisfaction):
            raise TypeError("satisfaction must be SolutionSatisfaction")


@dataclass(frozen=True)
class PainEvidence:
    """Structured observed pain details before provenance is attached."""

    category: PainCategory
    summary: str

    def __post_init__(self) -> None:
        if not isinstance(self.category, PainCategory):
            raise TypeError("pain category must be PainCategory")
        _validate_text(self.summary, "pain summary", 160)


@dataclass(frozen=True)
class InferenceCandidate(Generic[ValueT]):
    """Typed advisory candidate supplied to the deterministic updater."""

    value: ValueT
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("inference confidence must be between 0 and 1")
        _validate_bounded_value(self.value)


@dataclass(frozen=True)
class ObservedProspectEvidence:
    """One turn of explicit/structured evidence; no inferred values allowed."""

    source_turn_id: str
    source_kind: EvidenceSourceKind = EvidenceSourceKind.EXPLICIT_STATEMENT
    explicit_role: ProspectRole | None = None
    current_solution: CurrentSolutionEvidence | None = None
    explicit_pain: PainEvidence | None = None
    interest: bool | None = None
    busy: bool | None = None
    human_request: bool | None = None
    explicit_next_step: PreferredNextStep | None = None
    decision_authority_statement: DecisionAuthority | None = None
    timing_preference: str | None = None
    objection: ObjectionType | None = None

    def __post_init__(self) -> None:
        _validate_turn_id(self.source_turn_id)
        if not isinstance(self.source_kind, EvidenceSourceKind):
            raise TypeError("source_kind must be EvidenceSourceKind")
        if self.source_kind == EvidenceSourceKind.INFERENCE:
            raise ValueError("observed evidence cannot use inference source kind")
        for value, expected, name in (
            (self.explicit_role, ProspectRole, "explicit_role"),
            (self.current_solution, CurrentSolutionEvidence, "current_solution"),
            (self.explicit_pain, PainEvidence, "explicit_pain"),
            (self.interest, bool, "interest"),
            (self.busy, bool, "busy"),
            (self.human_request, bool, "human_request"),
            (self.explicit_next_step, PreferredNextStep, "explicit_next_step"),
            (
                self.decision_authority_statement,
                DecisionAuthority,
                "decision_authority_statement",
            ),
            (self.timing_preference, str, "timing_preference"),
            (self.objection, ObjectionType, "objection"),
        ):
            if value is not None and not isinstance(value, expected):
                raise TypeError(f"{name} has an invalid type")
        if self.timing_preference is not None:
            _validate_text(self.timing_preference, "timing preference", 120)


@dataclass(frozen=True)
class InferredProspectEvidence:
    """One turn of explicitly advisory candidates with independent confidence."""

    source_turn_id: str
    likely_role: InferenceCandidate[ProspectRole] | None = None
    decision_authority: InferenceCandidate[DecisionAuthority] | None = None
    influence_level: InferenceCandidate[InformationLevel] | None = None
    buying_stage: InferenceCandidate[BuyingStage] | None = None
    openness: InferenceCandidate[InformationLevel] | None = None
    urgency: InferenceCandidate[InformationLevel] | None = None
    objection_type: InferenceCandidate[ObjectionType] | None = None
    pain_summary: InferenceCandidate[str] | None = None
    current_solution_satisfaction: (
        InferenceCandidate[SolutionSatisfaction] | None
    ) = None
    preferred_next_step: InferenceCandidate[PreferredNextStep] | None = None

    def __post_init__(self) -> None:
        _validate_turn_id(self.source_turn_id)
        for value, expected, name in (
            (self.likely_role, ProspectRole, "likely_role"),
            (self.decision_authority, DecisionAuthority, "decision_authority"),
            (self.influence_level, InformationLevel, "influence_level"),
            (self.buying_stage, BuyingStage, "buying_stage"),
            (self.openness, InformationLevel, "openness"),
            (self.urgency, InformationLevel, "urgency"),
            (self.objection_type, ObjectionType, "objection_type"),
            (self.pain_summary, str, "pain_summary"),
            (
                self.current_solution_satisfaction,
                SolutionSatisfaction,
                "current_solution_satisfaction",
            ),
            (self.preferred_next_step, PreferredNextStep, "preferred_next_step"),
        ):
            _validate_candidate(value, expected, name)


@dataclass(frozen=True)
class ProspectEvidence:
    """Explicitly separated observed and inferred evidence for one update."""

    observed: ObservedProspectEvidence | None = None
    inferred: InferredProspectEvidence | None = None

    def __post_init__(self) -> None:
        if self.observed is None and self.inferred is None:
            raise ValueError("prospect evidence requires observed or inferred input")
        if self.observed is not None and not isinstance(
            self.observed, ObservedProspectEvidence
        ):
            raise TypeError("observed evidence has an invalid type")
        if self.inferred is not None and not isinstance(
            self.inferred, InferredProspectEvidence
        ):
            raise TypeError("inferred evidence has an invalid type")


def _validate_turn_id(value: str) -> None:
    if not value or len(value) > 100:
        raise ValueError("source turn id must contain 1-100 characters")


def _validate_text(value: str, name: str, limit: int) -> None:
    if not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must contain 1-{limit} characters")


def _validate_bounded_value(value: object) -> None:
    if isinstance(value, str) and not isinstance(value, Enum):
        _validate_text(value, "evidence value", 160)


def _validate_observed(
    value: object,
    expected: type[object],
    name: str,
) -> None:
    if value is not None and (
        not isinstance(value, ObservedValue) or not isinstance(value.value, expected)
    ):
        raise TypeError(f"{name} has an invalid observed value")


def _validate_inferred(
    value: object,
    expected: type[object],
    name: str,
) -> None:
    if value is not None and (
        not isinstance(value, InferredValue) or not isinstance(value.value, expected)
    ):
        raise TypeError(f"{name} has an invalid inferred value")


def _validate_candidate(
    value: object,
    expected: type[object],
    name: str,
) -> None:
    if value is not None and (
        not isinstance(value, InferenceCandidate)
        or not isinstance(value.value, expected)
    ):
        raise TypeError(f"{name} has an invalid inference candidate")
