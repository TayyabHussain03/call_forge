"""Bounded, evidence-separated business conversation understanding contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BusinessFactKind(str, Enum):
    INDUSTRY = "industry"
    BUSINESS_TYPE = "business_type"
    BRANCHES = "branches"
    EMPLOYEES = "employees"
    CURRENT_WORKFLOW = "current_workflow"
    MANUAL_STEP = "manual_step"
    SOFTWARE = "software"
    INTEGRATION = "integration"
    REPORTING = "reporting"
    COMMUNICATION = "communication"
    WEBSITE = "website"
    CRM = "crm"


class BusinessProblemKind(str, Enum):
    OPERATIONAL = "operational"
    TECHNICAL = "technical"
    MARKETING = "marketing"
    SALES = "sales"
    FINANCIAL = "financial"
    CUSTOMER_EXPERIENCE = "customer_experience"
    TIME = "time"
    QUALITY = "quality"
    ERRORS = "errors"
    FOLLOW_UPS = "follow_ups"


class BusinessGoalKind(str, Enum):
    INCREASE_SALES = "increase_sales"
    REDUCE_TIME = "reduce_time"
    REDUCE_COST = "reduce_cost"
    GROW_BUSINESS = "grow_business"
    SCALE_OPERATIONS = "scale_operations"
    IMPROVE_CUSTOMER_EXPERIENCE = "improve_customer_experience"
    GENERATE_LEADS = "generate_leads"
    VISIBILITY = "visibility"
    EFFICIENCY = "efficiency"


class BusinessConstraintKind(str, Enum):
    BUDGET = "budget"
    TIME = "time"
    RESOURCES = "resources"
    STAFF = "staff"
    TECHNOLOGY = "technology"
    APPROVAL = "approval"
    CURRENT_VENDOR = "current_vendor"
    RISK = "risk"


class UnknownArea(str, Enum):
    ORDER_VOLUME = "order_volume"
    CURRENT_SOFTWARE = "current_software"
    REPORTING_PROCESS = "reporting_process"
    DECISION_MAKER = "decision_maker"
    INVENTORY = "inventory"
    BILLING = "billing"
    CRM = "crm"
    INTEGRATIONS = "integrations"


class ConversationTopic(str, Enum):
    OPERATIONS = "operations"
    WEBSITE = "website"
    CRM = "crm"
    SALES = "sales"
    MARKETING = "marketing"
    REPORTING = "reporting"


class RootCauseKind(str, Enum):
    NO_CENTRAL_SYSTEM = "no_central_system"
    MANUAL_DEPENDENCY = "manual_dependency"
    PROCESS_GAP = "process_gap"
    ADOPTION_GAP = "adoption_gap"
    REPORTING_GAP = "reporting_gap"
    AUTOMATION_GAP = "automation_gap"


@dataclass(frozen=True)
class ObservedBusinessFact:
    kind: BusinessFactKind
    value: str
    source_turn_id: str

    def __post_init__(self) -> None:
        _text(self.value, "fact value", 160)
        _text(self.source_turn_id, "source turn id", 100)


@dataclass(frozen=True)
class ObservedBusinessProblem:
    kind: BusinessProblemKind
    detail: str
    source_turn_id: str
    primary: bool = False

    def __post_init__(self) -> None:
        _text(self.detail, "problem detail", 200)
        _text(self.source_turn_id, "source turn id", 100)


@dataclass(frozen=True)
class RootCauseHypothesis:
    kind: RootCauseKind
    confidence: float
    reason: str
    supporting_observations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.confidence, (int, float)) or isinstance(self.confidence, bool) or not 0 <= self.confidence <= 1:
            raise ValueError("hypothesis confidence must be between zero and one")
        _text(self.reason, "hypothesis reason", 240)
        if not self.supporting_observations or len(self.supporting_observations) > 6:
            raise ValueError("hypothesis requires one to six supporting observations")
        for item in self.supporting_observations:
            _text(item, "supporting observation", 160)


@dataclass(frozen=True)
class BusinessConversationEvidence:
    """Upstream semantic observations; never a recommendation or authority grant."""

    facts: tuple[ObservedBusinessFact, ...] = ()
    problems: tuple[ObservedBusinessProblem, ...] = ()
    goals: frozenset[BusinessGoalKind] = frozenset()
    constraints: frozenset[BusinessConstraintKind] = frozenset()
    unknown_areas: frozenset[UnknownArea] = frozenset()
    current_focus: ConversationTopic | None = None
    corrections: frozenset[BusinessFactKind | BusinessProblemKind] = frozenset()
    pending_topic: ConversationTopic | None = None

    def __post_init__(self) -> None:
        if len(self.facts) > 12 or len(self.problems) > 8:
            raise ValueError("business evidence exceeds bounded observations")
        if any(not isinstance(item, ObservedBusinessFact) for item in self.facts):
            raise TypeError("facts must be observed business facts")
        if any(not isinstance(item, ObservedBusinessProblem) for item in self.problems):
            raise TypeError("problems must be observed business problems")
        if not all(isinstance(item, BusinessGoalKind) for item in self.goals):
            raise TypeError("goals must be business goals")
        if not all(isinstance(item, BusinessConstraintKind) for item in self.constraints):
            raise TypeError("constraints must be business constraints")
        if not all(isinstance(item, UnknownArea) for item in self.unknown_areas):
            raise TypeError("unknown areas must be UnknownArea values")


@dataclass(frozen=True)
class BusinessConversationSnapshot:
    """Immutable current understanding: observed facts never blend with hypotheses."""

    facts: tuple[ObservedBusinessFact, ...] = ()
    problems: tuple[ObservedBusinessProblem, ...] = ()
    goals: frozenset[BusinessGoalKind] = frozenset()
    constraints: frozenset[BusinessConstraintKind] = frozenset()
    unknown_areas: frozenset[UnknownArea] = frozenset()
    hypotheses: tuple[RootCauseHypothesis, ...] = ()
    current_focus: ConversationTopic | None = None
    pending_topic: ConversationTopic | None = None
    archived_facts: tuple[ObservedBusinessFact, ...] = ()
    archived_problems: tuple[ObservedBusinessProblem, ...] = ()

    def __post_init__(self) -> None:
        if len(self.facts) > 24 or len(self.problems) > 16 or len(self.hypotheses) > 6:
            raise ValueError("business snapshot exceeds bounded state")
        if any(not isinstance(item, RootCauseHypothesis) for item in self.hypotheses):
            raise TypeError("hypotheses must be root cause hypotheses")


def _text(value: str, name: str, limit: int) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must contain 1-{limit} characters")
