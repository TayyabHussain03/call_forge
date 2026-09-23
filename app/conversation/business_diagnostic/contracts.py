"""Immutable, non-commercial diagnostics derived only from BCI facts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.conversation.business_conversation.contracts import (
    BusinessConversationSnapshot,
    BusinessConstraintKind,
    BusinessGoalKind,
)
from app.conversation.prospect_intelligence.contracts import ProspectIntelligenceSummary
from app.conversation.strategy.contracts import ConversationStrategy


class DiagnosticConfidence(str, Enum):
    OBSERVED = "observed"
    LIKELY = "likely"
    POSSIBLE = "possible"
    UNKNOWN = "unknown"


class BusinessMaturity(str, Enum):
    EARLY_DIGITAL = "early_digital"
    GROWING = "growing"
    OPTIMIZING = "optimizing"
    ADVANCED = "advanced"
    UNKNOWN = "unknown"


class CapabilityArea(str, Enum):
    ONLINE_PRESENCE = "online_presence"
    WORKFLOW = "workflow"
    LEAD_TRACKING = "lead_tracking"
    CRM = "crm"
    REPORTING = "reporting"
    ANALYTICS = "analytics"


class DiagnosticFocus(str, Enum):
    WORKFLOW = "workflow"
    WEBSITE_PERFORMANCE = "website_performance"
    SALES_PROCESS = "sales_process"
    LEAD_MANAGEMENT = "lead_management"
    REPORTING = "reporting"
    BUSINESS_CONTEXT = "business_context"


class DiagnosticHypothesisKind(str, Enum):
    MANUAL_DEPENDENCY = "manual_dependency"
    PROCESS_GAP = "process_gap"
    CENTRALIZATION_GAP = "centralization_gap"
    REPORTING_GAP = "reporting_gap"


@dataclass(frozen=True)
class DiagnosticFinding:
    area: CapabilityArea
    confidence: DiagnosticConfidence
    supporting_observations: tuple[str, ...]
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.supporting_observations or len(self.supporting_observations) > 6:
            raise ValueError("finding requires one to six supporting observations")
        if len(self.evidence_ids) > 6 or any(not item.strip() or len(item) > 100 for item in self.evidence_ids):
            raise ValueError("finding evidence ids must be bounded")


@dataclass(frozen=True)
class DiagnosticHypothesis:
    kind: DiagnosticHypothesisKind
    confidence: DiagnosticConfidence
    supporting_observations: tuple[str, ...]
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.confidence == DiagnosticConfidence.OBSERVED:
            raise ValueError("a hypothesis may not become observed")
        if not self.supporting_observations:
            raise ValueError("hypothesis requires supporting observations")


@dataclass(frozen=True)
class ConsultantSummary:
    """Typed internal summary only; prose and advice are deliberately absent."""

    current_situation: tuple[CapabilityArea, ...] = ()
    primary_problem: CapabilityArea | None = None
    secondary_problems: tuple[CapabilityArea, ...] = ()
    business_goals: frozenset[BusinessGoalKind] = frozenset()
    current_workflow_known: bool = False
    known_constraints: frozenset[BusinessConstraintKind] = frozenset()
    remaining_unknowns: tuple[CapabilityArea, ...] = ()


@dataclass(frozen=True)
class BusinessDiagnosticInput:
    conversation: BusinessConversationSnapshot
    prospect: ProspectIntelligenceSummary | None = None
    strategy: ConversationStrategy | None = None
    approved_metadata_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.approved_metadata_ids) > 12 or any(not item.strip() or len(item) > 100 for item in self.approved_metadata_ids):
            raise ValueError("approved metadata ids must be bounded")


@dataclass(frozen=True)
class BusinessDiagnosticSnapshot:
    maturity: BusinessMaturity
    maturity_confidence: DiagnosticConfidence
    findings: tuple[DiagnosticFinding, ...]
    capability_gaps: tuple[DiagnosticFinding, ...]
    hypotheses: tuple[DiagnosticHypothesis, ...]
    diagnostic_focus: DiagnosticFocus
    question_concept: DiagnosticFocus
    summary: ConsultantSummary

    def __post_init__(self) -> None:
        if len(self.findings) > 8 or len(self.capability_gaps) > 8 or len(self.hypotheses) > 6:
            raise ValueError("diagnostic snapshot exceeds bounded state")
        forbidden = {"service", "action", "authority", "state", "price", "execution", "recommend"}
        if any(any(word in field for word in forbidden) for field in self.__dataclass_fields__):
            raise ValueError("diagnostic contract contains forbidden authority fields")
