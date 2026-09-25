"""Immutable contracts for deterministic knowledge-retrieval planning."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from app.knowledge.contracts import DocumentStatus, KnowledgePurpose
from app.conversation.business_memory.contracts import BusinessMentalModelSnapshot
from app.conversation.business_diagnostic.contracts import BusinessDiagnosticSnapshot
from app.conversation.conversation_steering.contracts import ConversationPrioritySnapshot
from app.conversation.qualification.contracts import QualificationSnapshot

_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,99}$")
_CONCEPT = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")


class KnowledgeNeedKind(str, Enum):
    NONE="none"; SERVICE_EXPLANATION="service_explanation"; CAPABILITY_EXPLANATION="capability_explanation"; FEATURE_CLARIFICATION="feature_clarification"; PROCESS_EXPLANATION="process_explanation"; TECHNICAL_DETAIL="technical_detail"; COMPARISON="comparison"; CASE_STUDY="case_study"; FAQ="faq"; POLICY_INFORMATION="policy_information"; IMPLEMENTATION_INFORMATION="implementation_information"; COMMERCIAL_INFORMATION="commercial_information"; UNKNOWN="unknown"

class KnowledgeNeedSource(str, Enum):
    STRUCTURED_UNDERSTANDING="structured_understanding"; RESPONSE_PLANNING="response_planning"; CONSULTATIVE_DECISION="consultative_decision"; DIRECT_TYPED_REQUEST="direct_typed_request"

class RequestedDepth(str, Enum): BRIEF="brief"; STANDARD="standard"; DETAILED="detailed"
class AmbiguityState(str, Enum): CLEAR="clear"; UNRESOLVED="unresolved"
class RetrievalRequirement(str, Enum): NONE="none"; OPTIONAL="optional"; REQUIRED="required"
class RetrievalPlanningStatus(str, Enum): READY="ready"; NOT_NEEDED="not_needed"; NEEDS_CLARIFICATION="needs_clarification"; BLOCKED_BY_POLICY="blocked_by_policy"; INVALID_SCOPE="invalid_scope"
class RetrievalPurpose(str, Enum): EDUCATE="educate"; ANSWER_DIRECT_QUESTION="answer_direct_question"; CLARIFY_CAPABILITY="clarify_capability"; COMPARE="compare"; SUPPORT_WITH_EVIDENCE="support_with_evidence"; EXPLAIN_IMPLEMENTATION="explain_implementation"; ANSWER_POLICY_QUESTION="answer_policy_question"; COMMERCIAL_SUPPORT="commercial_support"; NONE="none"
class RetrievalIntent(str, Enum): EXPLANATION="explanation"; CLARIFICATION="clarification"; COMPARISON="comparison"; VALIDATION="validation"; EVIDENCE="evidence"; POLICY_LOOKUP="policy_lookup"; UNKNOWN="unknown"
class KnowledgeScope(str, Enum): TENANT="tenant"; BUSINESS="business"; CAMPAIGN="campaign"; SERVICE="service"; FAQ="faq"; TECHNICAL="technical"; POLICY="policy"; CASE_STUDY="case_study"; GENERAL_COMPANY="general_company"
class KnowledgeDomain(str, Enum): COMPANY="company"; COMMERCIAL_POLICY="commercial_policy"; TECHNICAL="technical"; CASE_STUDY="case_study"; FAQ="faq"; NONE="none"
class RetrievalEvidenceType(str, Enum): SERVICE_DESCRIPTION="service_description"; CAPABILITY="capability"; FEATURE="feature"; LIMITATION="limitation"; REQUIREMENT="requirement"; IMPLEMENTATION="implementation"; FAQ="faq"; CASE_STUDY="case_study"; POLICY="policy"; COMMERCIAL_POLICY="commercial_policy"; INTEGRATION="integration"; PROCESS="process"; GENERAL_COMPANY_FACT="general_company_fact"
class CommercialRetrievalRoute(str, Enum): NOT_COMMERCIAL="not_commercial"; POLICY_GATED="policy_gated"; RETRIEVAL_DISALLOWED="retrieval_disallowed"


@dataclass(frozen=True)
class KnowledgeConcept:
    subject: str
    aspect: str
    secondary_subject: str | None = None
    business_context_category: str | None = None
    requested_depth: RequestedDepth = RequestedDepth.STANDARD
    def __post_init__(self):
        for value in (self.subject,self.aspect,self.secondary_subject,self.business_context_category):
            if value is not None and not _CONCEPT.fullmatch(value): raise ValueError("knowledge concepts must be normalized identifiers")


@dataclass(frozen=True)
class KnowledgeNeed:
    need_id: str; source_turn_id: str; source: KnowledgeNeedSource; need_kind: KnowledgeNeedKind
    primary_concept: KnowledgeConcept | None = None
    explicit_direct_request: bool = False
    requested_evidence: bool = False
    ambiguity: AmbiguityState = AmbiguityState.CLEAR
    service_context_id: str | None = None
    approved_evidence_sufficient: bool = False
    def __post_init__(self):
        if not _ID.fullmatch(self.need_id) or not _ID.fullmatch(self.source_turn_id): raise ValueError("knowledge need identity is invalid")
        if not isinstance(self.source, KnowledgeNeedSource) or not isinstance(self.need_kind, KnowledgeNeedKind): raise TypeError("knowledge need source and kind must be typed")
        if self.need_kind == KnowledgeNeedKind.COMPARISON:
            if self.primary_concept is None or self.primary_concept.secondary_subject is None: raise ValueError("comparison requires a secondary concept")
        elif self.primary_concept is not None and self.primary_concept.secondary_subject is not None: raise ValueError("secondary concept is comparison-only")
        if self.service_context_id is not None and not _ID.fullmatch(self.service_context_id): raise ValueError("service context id is invalid")


@dataclass(frozen=True)
class RetrievalBudget:
    max_documents: int = 3; max_chunks: int = 8; max_context_tokens: int = 3000; max_retrieval_duration_ms: int = 1500
    def __post_init__(self):
        if not (1 <= self.max_documents <= 20 and 1 <= self.max_chunks <= 50 and 100 <= self.max_context_tokens <= 20000 and 50 <= self.max_retrieval_duration_ms <= 10000): raise ValueError("retrieval budget is outside safe bounds")


@dataclass(frozen=True)
class KnowledgeRetrievalPolicy:
    tenant_id: str; business_id: str; campaign_id: str
    knowledge_base_id: str | None = None
    authorized_service_ids: frozenset[str] = frozenset()
    commercial_policy_retrieval_allowed: bool = False
    budget: RetrievalBudget = RetrievalBudget()
    def __post_init__(self):
        for value in (self.tenant_id,self.business_id,self.campaign_id):
            if not _ID.fullmatch(value): raise ValueError("retrieval policy scope is invalid")
        if self.knowledge_base_id is not None and not _ID.fullmatch(self.knowledge_base_id): raise ValueError("knowledge base id is invalid")
        if any(not _ID.fullmatch(value) for value in self.authorized_service_ids): raise ValueError("authorized service ids are invalid")


@dataclass(frozen=True)
class RetrievalPlanningContext:
    business_memory: BusinessMentalModelSnapshot | None = None
    diagnostic: BusinessDiagnosticSnapshot | None = None
    qualification: QualificationSnapshot | None = None
    steering: ConversationPrioritySnapshot | None = None


@dataclass(frozen=True)
class RetrievalFilters:
    tenant_id: str; business_id: str; campaign_id: str
    service_id: str | None; knowledge_base_id: str | None
    purposes: tuple[KnowledgePurpose,...]
    document_status: DocumentStatus = DocumentStatus.PUBLISHED
    active_version_only: bool = True
    preferred_language: str | None = None
    fallback_language: str | None = None
    def __post_init__(self):
        if self.document_status != DocumentStatus.PUBLISHED or self.active_version_only is not True: raise ValueError("retrieval filters must use active published knowledge")
        forbidden={KnowledgePurpose.SALES_KNOWLEDGE,KnowledgePurpose.OBJECTION_HANDLING,KnowledgePurpose.TRAINING_GUIDE}
        if forbidden & set(self.purposes): raise ValueError("permanent sales knowledge cannot be retrieved")


@dataclass(frozen=True)
class RetrievalPlan:
    plan_id: str; source_turn_id: str | None
    requirement: RetrievalRequirement; planning_status: RetrievalPlanningStatus
    purpose: RetrievalPurpose; intent: RetrievalIntent; domain: KnowledgeDomain
    concept: KnowledgeConcept | None; scope: KnowledgeScope | None
    evidence_types: tuple[RetrievalEvidenceType,...]
    filters: RetrievalFilters; budget: RetrievalBudget
    commercial_route: CommercialRetrievalRoute
    def __post_init__(self):
        if not _ID.fullmatch(self.plan_id): raise ValueError("retrieval plan id is invalid")
        if self.requirement == RetrievalRequirement.NONE and self.planning_status == RetrievalPlanningStatus.READY: raise ValueError("empty retrieval cannot be ready")
        forbidden={"document_ids","chunk_ids","search_query","raw_text","transcript","approved_evidence"}
        if forbidden & set(self.__dataclass_fields__): raise ValueError("retrieval plan contains execution data")
