"""Bounded advisory contracts for progressive consultative conversation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.conversation.context.contracts import ApprovedEvidenceItem
from app.conversation.prospect_intelligence.contracts import ProspectIntelligenceSummary
from app.conversation.response_planning.contracts import (
    AddresseeStatus,
    InterruptionCategory,
    PendingConversationIntent,
    ResponseLength,
)
from app.conversation.strategy.contracts import ConversationStrategy
from app.conversation.understanding.contracts import LanguageProfile


class ProblemCategory(str, Enum):
    LEAD_FLOW = "lead_flow"
    FOLLOW_UP = "follow_up"
    CUSTOMER_QUESTIONS = "customer_questions"
    WORKFLOW = "workflow"
    WEBSITE = "website"
    VISIBILITY = "visibility"
    BRANDING = "branding"
    OTHER = "other"
    UNKNOWN = "unknown"


class ProblemEvidenceBasis(str, Enum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"


class ProblemField(str, Enum):
    UNDERLYING_PROBLEM = "underlying_problem"
    CURRENT_PROCESS = "current_process"
    FRICTION = "friction"
    IMPACT = "impact"
    DESIRED_OUTCOME = "desired_outcome"
    SOURCE_OR_CHANNEL = "source_or_channel"
    PROVIDER_SATISFACTION = "provider_satisfaction"
    OBJECTION_REASON = "objection_reason"
    ROLE_ROUTING = "role_routing"
    APPROVED_EVIDENCE = "approved_evidence"
    POLICY_DISCLOSURE = "policy_disclosure"
    ALREADY_OFFERED = "already_offered"


@dataclass(frozen=True)
class ProspectProblem:
    """Current bounded problem hypothesis; unknown fields remain absent."""

    category: ProblemCategory = ProblemCategory.UNKNOWN
    explicit_description: str | None = None
    requested_solution: str | None = None
    current_process: str | None = None
    friction: str | None = None
    impact: str | None = None
    desired_outcome: str | None = None
    source_or_channel: str | None = None
    evidence_basis: ProblemEvidenceBasis = ProblemEvidenceBasis.INFERRED
    source_turn_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.category, ProblemCategory):
            raise TypeError("problem category is invalid")
        if not isinstance(self.evidence_basis, ProblemEvidenceBasis):
            raise TypeError("problem evidence basis is invalid")
        for value, name, limit in (
            (self.explicit_description, "problem description", 200),
            (self.requested_solution, "requested solution", 100),
            (self.current_process, "current process", 200),
            (self.friction, "problem friction", 160),
            (self.impact, "problem impact", 160),
            (self.desired_outcome, "desired outcome", 160),
            (self.source_or_channel, "source or channel", 100),
            (self.source_turn_id, "source turn id", 100),
        ):
            if value is not None:
                _text(value, name, limit)

    @property
    def meaningful(self) -> bool:
        return self.explicit_description is not None or self.friction is not None


@dataclass(frozen=True)
class ProblemEvidence:
    """One structured problem update; correction explicitly replaces stale context."""

    source_turn_id: str
    category: ProblemCategory = ProblemCategory.UNKNOWN
    explicit_description: str | None = None
    requested_solution: str | None = None
    current_process: str | None = None
    friction: str | None = None
    impact: str | None = None
    desired_outcome: str | None = None
    source_or_channel: str | None = None
    evidence_basis: ProblemEvidenceBasis = ProblemEvidenceBasis.EXPLICIT
    correction: bool = False

    def __post_init__(self) -> None:
        ProspectProblem(
            self.category,
            self.explicit_description,
            self.requested_solution,
            self.current_process,
            self.friction,
            self.impact,
            self.desired_outcome,
            self.source_or_channel,
            self.evidence_basis,
            self.source_turn_id,
        )
        if not isinstance(self.correction, bool):
            raise TypeError("correction flag must be boolean")


class ServiceFitStatus(str, Enum):
    INSUFFICIENT_CONTEXT = "insufficient_context"
    NO_AUTHORIZED_FIT = "no_authorized_fit"
    POSSIBLE_FIT = "possible_fit"
    SUPPORTED_FIT = "supported_fit"


@dataclass(frozen=True)
class ServiceFitDecision:
    """Advisory relevance within pre-authorized eligibility boundaries."""

    status: ServiceFitStatus
    service_id: str | None = None
    matched_problem_aspects: tuple[str, ...] = ()
    missing_information: tuple[ProblemField, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    explanation_allowed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.status, ServiceFitStatus):
            raise TypeError("service fit status is invalid")
        aspects = tuple(self.matched_problem_aspects)
        missing = tuple(self.missing_information)
        evidence = tuple(self.evidence_ids)
        if len(aspects) > 4 or any(not item.strip() or len(item) > 100 for item in aspects):
            raise ValueError("matched problem aspects must be bounded")
        if len(missing) > 3 or any(not isinstance(item, ProblemField) for item in missing):
            raise ValueError("missing information must be typed and bounded")
        if len(evidence) > 8 or tuple(sorted(set(evidence))) != evidence:
            raise ValueError("service-fit evidence ids must be unique and ordered")
        if self.service_id is not None:
            _text(self.service_id, "service id", 100)
        if self.status == ServiceFitStatus.SUPPORTED_FIT and (
            self.service_id is None or not evidence or not self.explanation_allowed
        ):
            raise ValueError("supported fit requires a service and approved evidence")
        if self.explanation_allowed and self.status != ServiceFitStatus.SUPPORTED_FIT:
            raise ValueError("only supported fit may allow explanation")
        object.__setattr__(self, "matched_problem_aspects", aspects)
        object.__setattr__(self, "missing_information", missing)
        object.__setattr__(self, "evidence_ids", evidence)


@dataclass(frozen=True)
class ServiceAnswerContext:
    """One grounded relevant service projection, never a catalog dump."""

    service_id: str
    service_name: str
    approved_description: str | None
    approved_capabilities: tuple[str, ...]
    approved_evidence: tuple[ApprovedEvidenceItem, ...]
    problem_summary: str
    current_process_summary: str | None = None
    disclosure_allowed: bool = True
    unresolved_technical_questions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.service_id, "service id", 100)
        _text(self.service_name, "service name", 120)
        _text(self.problem_summary, "problem summary", 200)
        if self.approved_description is not None:
            _text(self.approved_description, "approved description", 240)
        if self.current_process_summary is not None:
            _text(self.current_process_summary, "current process summary", 200)
        capabilities = tuple(self.approved_capabilities)
        evidence = tuple(self.approved_evidence)
        questions = tuple(self.unresolved_technical_questions)
        if len(capabilities) > 4 or any(
            not item.strip() or len(item) > 100 for item in capabilities
        ):
            raise ValueError("approved capabilities must be bounded")
        if not evidence or len(evidence) > 8:
            raise ValueError("service answer requires bounded approved evidence")
        if any(not isinstance(item, ApprovedEvidenceItem) for item in evidence):
            raise TypeError("service answer evidence must be approved evidence")
        if not isinstance(self.disclosure_allowed, bool):
            raise TypeError("disclosure flag must be boolean")
        if len(questions) > 2 or any(
            not item.strip() or len(item) > 160 for item in questions
        ):
            raise ValueError("technical questions must be bounded")
        object.__setattr__(self, "approved_capabilities", capabilities)
        object.__setattr__(self, "approved_evidence", evidence)
        object.__setattr__(self, "unresolved_technical_questions", questions)


class ConsultativeObjective(str, Enum):
    OPEN_TRUTHFULLY = "open_truthfully"
    UNDERSTAND = "understand"
    EXPLAIN = "explain"
    ANSWER = "answer"
    PROGRESS = "progress"
    CLOSE = "close"
    RECOVER = "recover"


class ConsultativeMove(str, Enum):
    OPEN_CONVERSATION = "open_conversation"
    CLARIFY = "clarify"
    DISCOVER_PROBLEM = "discover_problem"
    UNDERSTAND_CURRENT_PROCESS = "understand_current_process"
    UNDERSTAND_IMPACT = "understand_impact"
    UNDERSTAND_EXISTING_SOLUTION = "understand_existing_solution"
    QUALIFY_ROLE = "qualify_role"
    QUALIFY_AUTHORITY = "qualify_authority"
    EXPLAIN_RELEVANT_FIT = "explain_relevant_fit"
    HANDLE_OBJECTION = "handle_objection"
    ANSWER_DIRECT_QUESTION = "answer_direct_question"
    ASK_MICRO_COMMITMENT = "ask_micro_commitment"
    ROUTE_TO_DECISION_MAKER = "route_to_decision_maker"
    LOW_PRESSURE_CALLBACK = "low_pressure_callback"
    SIMPLIFY_EXPLANATION = "simplify_explanation"
    GRACEFUL_CLOSE = "graceful_close"


class QuestionPolicy(str, Enum):
    NONE = "none"
    ONE_PRIMARY = "one_primary"
    SHORT_CONTRAST = "short_contrast"


class AcknowledgementIntent(str, Enum):
    NONE = "none"
    ACKNOWLEDGE = "acknowledge"
    REFLECT = "reflect"
    CONFIRM = "confirm"


@dataclass(frozen=True)
class ConsultativeConversationDecision:
    """Best advisory conversational direction with no execution authority."""

    objective: ConsultativeObjective
    move: ConsultativeMove
    primary_information_gap: ProblemField | None = None
    problem_summary: str | None = None
    service_fit: ServiceFitDecision | None = None
    explanation_depth: ResponseLength = ResponseLength.SHORT
    question_policy: QuestionPolicy = QuestionPolicy.NONE
    acknowledgement_intent: AcknowledgementIntent = AcknowledgementIntent.NONE
    commercial_transparency_required: bool = False
    simplification_required: bool = False
    resume_previous_goal: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.objective, ConsultativeObjective):
            raise TypeError("consultative objective is invalid")
        if not isinstance(self.move, ConsultativeMove):
            raise TypeError("consultative move is invalid")
        if self.primary_information_gap is not None and not isinstance(
            self.primary_information_gap, ProblemField
        ):
            raise TypeError("primary information gap is invalid")
        if self.problem_summary is not None:
            _text(self.problem_summary, "problem summary", 200)
        if self.service_fit is not None and not isinstance(
            self.service_fit, ServiceFitDecision
        ):
            raise TypeError("service fit decision is invalid")
        if not isinstance(self.explanation_depth, ResponseLength):
            raise TypeError("explanation depth is invalid")
        if not isinstance(self.question_policy, QuestionPolicy):
            raise TypeError("question policy is invalid")
        if not isinstance(self.acknowledgement_intent, AcknowledgementIntent):
            raise TypeError("acknowledgement intent is invalid")
        if not isinstance(self.commercial_transparency_required, bool):
            raise TypeError("commercial transparency flag must be boolean")
        if not isinstance(self.simplification_required, bool):
            raise TypeError("simplification flag must be boolean")
        if not isinstance(self.resume_previous_goal, bool):
            raise TypeError("resume flag must be boolean")
        if self.question_policy != QuestionPolicy.NONE and (
            self.primary_information_gap is None
        ):
            raise ValueError("a question policy requires one primary information gap")


@dataclass(frozen=True)
class ConsultativeTurnSignals:
    """Trusted typed current-turn signals; no raw-text matching occurs here."""

    direct_question: bool = False
    commercial_purpose_question: bool = False
    service_information_request: bool = False
    misunderstanding: bool = False
    correction: bool = False
    two_way_ambiguity: bool = False
    explicit_no_problem: bool = False
    buying_signal: bool = False
    explanation_requested: bool = False

    def __post_init__(self) -> None:
        if any(not isinstance(value, bool) for value in self.__dict__.values()):
            raise TypeError("consultative signals must be boolean")


@dataclass(frozen=True)
class ConsultativeDecisionInput:
    problem: ProspectProblem
    service_fit: ServiceFitDecision
    prospect: ProspectIntelligenceSummary | None = None
    strategy: ConversationStrategy | None = None
    language_profile: LanguageProfile | None = None
    signals: ConsultativeTurnSignals = ConsultativeTurnSignals()
    interruption_category: InterruptionCategory = InterruptionCategory.OTHER
    addressee_status: AddresseeStatus = AddresseeStatus.ADDRESSED_TO_AGENT
    pending_intent: PendingConversationIntent | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.problem, ProspectProblem):
            raise TypeError("consultative problem is invalid")
        if not isinstance(self.service_fit, ServiceFitDecision):
            raise TypeError("consultative service fit is invalid")
        if not isinstance(self.signals, ConsultativeTurnSignals):
            raise TypeError("consultative signals are invalid")
        if not isinstance(self.interruption_category, InterruptionCategory):
            raise TypeError("consultative interruption category is invalid")
        if not isinstance(self.addressee_status, AddresseeStatus):
            raise TypeError("consultative addressee status is invalid")


def _text(value: str, name: str, limit: int) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must contain 1-{limit} characters")
