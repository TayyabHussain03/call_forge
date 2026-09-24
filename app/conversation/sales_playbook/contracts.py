"""Immutable bounded contracts for advisory consultative playbooks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from app.conversation.consultative.contracts import (
    ConsultativeConversationDecision,
    ProspectProblem,
)
from app.conversation.prospect_intelligence.contracts import (
    ObjectionType,
    ProspectIntelligenceSummary,
)
from app.conversation.sales_cognition.contracts import SalesConversationGuidance
from app.conversation.strategy.contracts import ConversationStrategy

_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,99}$")


class BusinessSituation(str, Enum):
    NO_ONLINE_PRESENCE = "no_online_presence"
    WEBSITE_EXISTS = "website_exists"
    POOR_TRUST = "poor_trust"
    LOW_VISIBILITY = "low_visibility"
    NO_ONLINE_ORDERS = "no_online_orders"
    POOR_BRANDING = "poor_branding"
    MANUAL_WORKFLOW = "manual_workflow"
    REPEATED_TASKS = "repeated_tasks"
    HUMAN_ERRORS = "human_errors"
    SLOW_WORKFLOW = "slow_workflow"
    LOST_TIME = "lost_time"
    NO_TRAFFIC = "no_traffic"
    POOR_RANKING = "poor_ranking"
    CRM_EXISTS = "crm_exists"
    CRM_UNDERPERFORMING = "crm_underperforming"
    CURRENT_PROVIDER = "current_provider"
    NO_IDENTIFIED_NEED = "no_identified_need"


class DiscoveryTopic(str, Enum):
    CURRENT_WEBSITE = "current_website"
    GOOGLE_PRESENCE = "google_presence"
    ONLINE_ORDERS = "online_orders"
    CUSTOMER_JOURNEY = "customer_journey"
    CURRENT_WORKFLOW = "current_workflow"
    BUSINESS_IMPACT = "business_impact"
    CURRENT_SOFTWARE = "current_software"
    PROVIDER_SATISFACTION = "provider_satisfaction"
    BUSINESS_GOAL = "business_goal"
    OBJECTION_REASON = "objection_reason"


class QualificationSignal(str, Enum):
    DECISION_MAKER = "decision_maker"
    BUDGET_MENTIONED = "budget_mentioned"
    CURRENT_PROCESS_KNOWN = "current_process_known"
    PAIN_CONFIRMED = "pain_confirmed"
    BUSINESS_SIZE_KNOWN = "business_size_known"
    CURRENT_SOFTWARE_KNOWN = "current_software_known"


class BuyingSignal(str, Enum):
    INTERESTED = "interested"
    ASKING_QUESTIONS = "asking_questions"
    FUTURE_PLANNING = "future_planning"
    EXPANSION = "expansion"
    SCALING = "scaling"
    HIRING = "hiring"
    GROWTH = "growth"
    MANUAL_PROBLEMS = "manual_problems"


class BenefitCategory(str, Enum):
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    MARKETING = "marketing"
    CUSTOMER_EXPERIENCE = "customer_experience"
    TIME_SAVINGS = "time_savings"
    AUTOMATION = "automation"
    TRUST = "trust"
    VISIBILITY = "visibility"


class AcknowledgementStrategy(str, Enum):
    ACKNOWLEDGE_CONSTRAINT = "acknowledge_constraint"
    ACKNOWLEDGE_EXISTING_CHOICE = "acknowledge_existing_choice"
    ACKNOWLEDGE_UNCERTAINTY = "acknowledge_uncertainty"
    ACKNOWLEDGE_PREFERENCE = "acknowledge_preference"


class ClarificationStrategy(str, Enum):
    CLARIFY_TIMING = "clarify_timing"
    CLARIFY_BUDGET_CONTEXT = "clarify_budget_context"
    CLARIFY_CURRENT_SATISFACTION = "clarify_current_satisfaction"
    CLARIFY_NEED = "clarify_need"
    CLARIFY_CONCERN = "clarify_concern"


class EducationStrategy(str, Enum):
    EXPLAIN_RELEVANT_OUTCOME = "explain_relevant_outcome"
    EXPLAIN_OPTIMIZATION_PATH = "explain_optimization_path"
    EXPLAIN_WITH_APPROVED_EVIDENCE = "explain_with_approved_evidence"
    DEFER_EDUCATION = "defer_education"


class PitchReadiness(str, Enum):
    DISCOVERY_REQUIRED = "discovery_required"
    DISCOVERY_COMPLETE = "discovery_complete"
    EDUCATION = "education"
    VALUE_DISCUSSION = "value_discussion"
    READY_FOR_MICRO_COMMITMENT = "ready_for_micro_commitment"
    NO_FIT = "no_fit"


class ConsultantStep(str, Enum):
    UNDERSTAND = "understand"
    CLARIFY = "clarify"
    EDUCATE = "educate"
    RELATE = "relate"
    RECOMMEND = "recommend"
    MICRO_COMMITMENT = "micro_commitment"
    GRACEFUL_END = "graceful_end"


class PlaybookRestriction(str, Enum):
    NO_GUARANTEED_RANKINGS = "no_guaranteed_rankings"
    NO_GUARANTEED_REVENUE = "no_guaranteed_revenue"
    NO_GUARANTEED_ROI = "no_guaranteed_roi"
    NO_FAKE_TIMELINES = "no_fake_timelines"
    NO_FAKE_INTEGRATIONS = "no_fake_integrations"
    NO_PRESSURE_LANGUAGE = "no_pressure_language"


@dataclass(frozen=True)
class ObjectionGuidance:
    objection: ObjectionType
    acknowledgement: AcknowledgementStrategy
    clarification: ClarificationStrategy
    education: EducationStrategy

    def __post_init__(self) -> None:
        if not isinstance(self.objection, ObjectionType):
            raise TypeError("objection guidance objection must be typed")
        if not isinstance(self.acknowledgement, AcknowledgementStrategy):
            raise TypeError("acknowledgement strategy must be typed")
        if not isinstance(self.clarification, ClarificationStrategy):
            raise TypeError("clarification strategy must be typed")
        if not isinstance(self.education, EducationStrategy):
            raise TypeError("education strategy must be typed")


@dataclass(frozen=True)
class ServicePlaybook:
    """Structured service reasoning model; contains no scripts or evidence."""

    service_id: str
    name: str
    category: str
    description: str
    industries: frozenset[str]
    solved_situations: frozenset[BusinessSituation]
    discovery_topics: tuple[DiscoveryTopic, ...]
    qualification_signals: frozenset[QualificationSignal] = frozenset()
    buying_signals: frozenset[BuyingSignal] = frozenset()
    objections: tuple[ObjectionGuidance, ...] = ()
    benefits: frozenset[BenefitCategory] = frozenset()
    future_service_ids: tuple[str, ...] = ()
    restrictions: frozenset[PlaybookRestriction] = frozenset(
        {
            PlaybookRestriction.NO_GUARANTEED_REVENUE,
            PlaybookRestriction.NO_GUARANTEED_ROI,
            PlaybookRestriction.NO_PRESSURE_LANGUAGE,
        }
    )
    priority: int = 100

    def __post_init__(self) -> None:
        _identifier(self.service_id, "service id")
        for value, name, limit in (
            (self.name, "service name", 120),
            (self.category, "service category", 80),
            (self.description, "service description", 300),
        ):
            _text(value, name, limit)
        industries = frozenset(item.casefold() for item in self.industries)
        if len(industries) > 20 or any(not _ID.fullmatch(item) for item in industries):
            raise ValueError("playbook industries must be normalized and bounded")
        if not self.solved_situations or len(self.solved_situations) > 12:
            raise ValueError("playbook must solve one to twelve situations")
        if any(not isinstance(item, BusinessSituation) for item in self.solved_situations):
            raise TypeError("solved situations must be typed")
        topics = tuple(self.discovery_topics)
        if (
            not topics
            or len(topics) > 10
            or len(set(topics)) != len(topics)
            or any(not isinstance(item, DiscoveryTopic) for item in topics)
        ):
            raise ValueError("discovery topics must be unique and bounded")
        typed_sets = (
            (self.qualification_signals, QualificationSignal),
            (self.buying_signals, BuyingSignal),
            (self.benefits, BenefitCategory),
            (self.restrictions, PlaybookRestriction),
        )
        if any(
            any(not isinstance(item, expected) for item in values)
            for values, expected in typed_sets
        ):
            raise TypeError("playbook sets must contain their declared enum type")
        future = tuple(self.future_service_ids)
        if len(future) > 6 or len(set(future)) != len(future):
            raise ValueError("future service ids must be unique and bounded")
        for item in future:
            _identifier(item, "future service id")
        if not 0 <= self.priority <= 1_000:
            raise ValueError("playbook priority must be between zero and 1,000")
        if len({item.objection for item in self.objections}) != len(self.objections):
            raise ValueError("objection guidance must be unique")
        if any(not isinstance(item, ObjectionGuidance) for item in self.objections):
            raise TypeError("objection guidance must be typed")
        object.__setattr__(self, "industries", industries)
        object.__setattr__(self, "discovery_topics", topics)
        object.__setattr__(self, "future_service_ids", future)
        object.__setattr__(
            self,
            "restrictions",
            self.restrictions
            | frozenset(
                {
                    PlaybookRestriction.NO_GUARANTEED_REVENUE,
                    PlaybookRestriction.NO_GUARANTEED_ROI,
                    PlaybookRestriction.NO_PRESSURE_LANGUAGE,
                }
            ),
        )


@dataclass(frozen=True)
class BusinessContext:
    industry: str
    situations: frozenset[BusinessSituation] = frozenset()
    known_topics: frozenset[DiscoveryTopic] = frozenset()
    qualification_signals: frozenset[QualificationSignal] = frozenset()
    buying_signals: frozenset[BuyingSignal] = frozenset()
    existing_service_ids: frozenset[str] = frozenset()
    no_fit: bool = False
    true_ambiguity: bool = False

    def __post_init__(self) -> None:
        industry = self.industry.casefold()
        if not _ID.fullmatch(industry):
            raise ValueError("industry must be a normalized bounded identifier")
        if len(self.situations) > 16 or len(self.known_topics) > 12:
            raise ValueError("business context exceeds bounded facts")
        typed_sets = (
            (self.situations, BusinessSituation),
            (self.known_topics, DiscoveryTopic),
            (self.qualification_signals, QualificationSignal),
            (self.buying_signals, BuyingSignal),
        )
        if any(
            any(not isinstance(item, expected) for item in values)
            for values, expected in typed_sets
        ):
            raise TypeError("business context sets must contain typed values")
        if any(not _ID.fullmatch(item) for item in self.existing_service_ids):
            raise ValueError("existing service ids must be normalized")
        if not isinstance(self.no_fit, bool) or not isinstance(self.true_ambiguity, bool):
            raise TypeError("business context flags must be boolean")
        object.__setattr__(self, "industry", industry)


@dataclass(frozen=True)
class SalesPlaybookInput:
    context: BusinessContext
    playbooks: tuple[ServicePlaybook, ...]
    eligible_service_ids: frozenset[str]
    prospect: ProspectIntelligenceSummary
    strategy: ConversationStrategy | None
    problem: ProspectProblem
    consultative_decision: ConsultativeConversationDecision | None
    sales_guidance: SalesConversationGuidance | None
    already_discussed_service_ids: frozenset[str] = frozenset()
    business_conversation: object | None = None
    business_diagnostic: object | None = None
    conversation_priority: object | None = None
    qualification: object | None = None
    business_memory: object | None = None

    def __post_init__(self) -> None:
        playbooks = tuple(self.playbooks)
        if len(playbooks) > 50 or len({item.service_id for item in playbooks}) != len(
            playbooks
        ):
            raise ValueError("playbooks must be unique and bounded")
        if any(not isinstance(item, ServicePlaybook) for item in playbooks):
            raise TypeError("playbooks must be ServicePlaybook values")
        if any(not _ID.fullmatch(item) for item in self.eligible_service_ids):
            raise ValueError("eligible service ids must be normalized")
        object.__setattr__(self, "playbooks", playbooks)
        if self.business_conversation is not None:
            from app.conversation.business_conversation.contracts import BusinessConversationSnapshot
            if not isinstance(self.business_conversation, BusinessConversationSnapshot):
                raise TypeError("business conversation has an invalid type")
        if self.business_diagnostic is not None:
            from app.conversation.business_diagnostic.contracts import BusinessDiagnosticSnapshot
            if not isinstance(self.business_diagnostic, BusinessDiagnosticSnapshot):
                raise TypeError("business diagnostic has an invalid type")
        if self.conversation_priority is not None:
            from app.conversation.conversation_steering.contracts import ConversationPrioritySnapshot
            if not isinstance(self.conversation_priority, ConversationPrioritySnapshot):
                raise TypeError("conversation priority has an invalid type")
        if self.qualification is not None:
            from app.conversation.qualification.contracts import QualificationSnapshot
            if not isinstance(self.qualification, QualificationSnapshot): raise TypeError("qualification has an invalid type")
        if self.business_memory is not None:
            from app.conversation.business_memory.contracts import BusinessMentalModelSnapshot
            if not isinstance(self.business_memory, BusinessMentalModelSnapshot): raise TypeError("business memory has an invalid type")


@dataclass(frozen=True)
class OpportunityGuidance:
    """Advisory reasoning only; never service authorization or execution."""

    selected_service_id: str | None
    matched_situations: tuple[BusinessSituation, ...]
    pitch_readiness: PitchReadiness
    consultant_step: ConsultantStep
    primary_question: DiscoveryTopic | None
    secondary_question: DiscoveryTopic | None
    objection_guidance: ObjectionGuidance | None
    benefit_categories: tuple[BenefitCategory, ...]
    future_opportunity_ids: tuple[str, ...]
    restrictions: frozenset[PlaybookRestriction]
    existing_customer_optimization: bool = False

    def __post_init__(self) -> None:
        if self.selected_service_id is not None:
            _identifier(self.selected_service_id, "selected service id")
        if any(not isinstance(item, BusinessSituation) for item in self.matched_situations):
            raise TypeError("matched situations must be typed")
        if not isinstance(self.pitch_readiness, PitchReadiness):
            raise TypeError("pitch readiness must be typed")
        if not isinstance(self.consultant_step, ConsultantStep):
            raise TypeError("consultant step must be typed")
        questions = tuple(
            item
            for item in (self.primary_question, self.secondary_question)
            if item is not None
        )
        if len(questions) != len(set(questions)) or len(questions) > 2:
            raise ValueError("guidance permits at most two unique questions")
        if self.secondary_question is not None and self.primary_question is None:
            raise ValueError("secondary question requires a primary question")
        if self.selected_service_id is None and self.pitch_readiness not in {
            PitchReadiness.NO_FIT,
            PitchReadiness.DISCOVERY_REQUIRED,
        }:
            raise ValueError("readiness requires an advisory service opportunity")
        if len(self.future_opportunity_ids) > 3:
            raise ValueError("future opportunities must be bounded")
        if len(set(self.future_opportunity_ids)) != len(self.future_opportunity_ids):
            raise ValueError("future opportunities must be unique")
        for item in self.future_opportunity_ids:
            _identifier(item, "future opportunity id")
        if len(set(self.benefit_categories)) != len(self.benefit_categories) or any(
            not isinstance(item, BenefitCategory) for item in self.benefit_categories
        ):
            raise ValueError("benefit categories must be typed and unique")
        if any(not isinstance(item, PlaybookRestriction) for item in self.restrictions):
            raise TypeError("guidance restrictions must be typed")
        if not isinstance(self.existing_customer_optimization, bool):
            raise TypeError("optimization flag must be boolean")


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{name} must be a normalized bounded identifier")


def _text(value: str, name: str, limit: int) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must contain 1-{limit} characters")
