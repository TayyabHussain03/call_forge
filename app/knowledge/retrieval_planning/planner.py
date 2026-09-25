"""Pure planning only: no document, search, embedding, or model dependencies."""
from __future__ import annotations
from hashlib import sha256
from dataclasses import replace
from app.knowledge.contracts import KnowledgePurpose
from app.knowledge.retrieval_planning.contracts import *

class KnowledgeRetrievalPlanner:
    _precedence={KnowledgeNeedSource.DIRECT_TYPED_REQUEST:4,KnowledgeNeedSource.RESPONSE_PLANNING:3,KnowledgeNeedSource.CONSULTATIVE_DECISION:2,KnowledgeNeedSource.STRUCTURED_UNDERSTANDING:1}
    def plan(self, needs: tuple[KnowledgeNeed,...], policy: KnowledgeRetrievalPolicy, *, tenant_id: str, business_id: str, campaign_id: str, context: RetrievalPlanningContext=RetrievalPlanningContext(), preferred_language: str|None=None, fallback_language: str|None=None) -> RetrievalPlan:
        filters=RetrievalFilters(policy.tenant_id,policy.business_id,policy.campaign_id,None,policy.knowledge_base_id,(),preferred_language=preferred_language,fallback_language=fallback_language)
        if (tenant_id,business_id,campaign_id)!=(policy.tenant_id,policy.business_id,policy.campaign_id): return _empty("invalid",RetrievalPlanningStatus.INVALID_SCOPE,filters,policy)
        need=max(needs,key=lambda n:(self._precedence[n.source],n.need_id),default=None)
        if need is None or need.need_kind==KnowledgeNeedKind.NONE: return _empty("none",RetrievalPlanningStatus.NOT_NEEDED,filters,policy)
        if need.need_kind==KnowledgeNeedKind.UNKNOWN or need.ambiguity==AmbiguityState.UNRESOLVED or need.primary_concept is None: return _empty(need.source_turn_id,RetrievalPlanningStatus.NEEDS_CLARIFICATION,filters,policy)
        if need.service_context_id is not None and need.service_context_id not in policy.authorized_service_ids: return _empty(need.source_turn_id,RetrievalPlanningStatus.INVALID_SCOPE,filters,policy)
        concept=_refine_concept(need.primary_concept,context,policy.business_id)
        commercial=need.need_kind==KnowledgeNeedKind.COMMERCIAL_INFORMATION
        if commercial and not policy.commercial_policy_retrieval_allowed: return _empty(need.source_turn_id,RetrievalPlanningStatus.BLOCKED_BY_POLICY,filters,policy,CommercialRetrievalRoute.RETRIEVAL_DISALLOWED)
        purpose,intent,domain,scope,evidence,purposes=_mapping(need)
        requirement=RetrievalRequirement.REQUIRED if need.explicit_direct_request and not need.approved_evidence_sufficient else RetrievalRequirement.OPTIONAL
        if context.steering is not None and context.steering.discussion_timing.value == "premature" and not need.explicit_direct_request: requirement=RetrievalRequirement.NONE
        status=RetrievalPlanningStatus.READY if requirement!=RetrievalRequirement.NONE else RetrievalPlanningStatus.NOT_NEEDED
        route=CommercialRetrievalRoute.POLICY_GATED if commercial else CommercialRetrievalRoute.NOT_COMMERCIAL
        filters=RetrievalFilters(policy.tenant_id,policy.business_id,policy.campaign_id,need.service_context_id,policy.knowledge_base_id,purposes,preferred_language=preferred_language,fallback_language=fallback_language)
        key="|".join((need.need_id,policy.tenant_id,policy.business_id,policy.campaign_id,requirement.value))
        return RetrievalPlan("rp_"+sha256(key.encode()).hexdigest()[:20],need.source_turn_id,requirement,status,purpose,intent,domain,concept,scope,evidence,filters,policy.budget,route)

def _empty(seed,status,filters,policy,route=CommercialRetrievalRoute.NOT_COMMERCIAL):
    return RetrievalPlan("rp_"+sha256((seed+policy.tenant_id+policy.campaign_id).encode()).hexdigest()[:20],None,RetrievalRequirement.NONE,status,RetrievalPurpose.NONE,RetrievalIntent.UNKNOWN,KnowledgeDomain.NONE,None,None,(),filters,policy.budget,route)

def _mapping(need):
    kind=need.need_kind
    if kind==KnowledgeNeedKind.COMPARISON: return RetrievalPurpose.COMPARE,RetrievalIntent.COMPARISON,KnowledgeDomain.COMPANY,KnowledgeScope.BUSINESS,(RetrievalEvidenceType.FEATURE,RetrievalEvidenceType.LIMITATION),(KnowledgePurpose.SERVICE_KNOWLEDGE,KnowledgePurpose.FAQ)
    if kind==KnowledgeNeedKind.CASE_STUDY: return RetrievalPurpose.SUPPORT_WITH_EVIDENCE,RetrievalIntent.EVIDENCE,KnowledgeDomain.CASE_STUDY,KnowledgeScope.CASE_STUDY,(RetrievalEvidenceType.CASE_STUDY,),(KnowledgePurpose.CASE_STUDY,)
    if kind==KnowledgeNeedKind.COMMERCIAL_INFORMATION: return RetrievalPurpose.COMMERCIAL_SUPPORT,RetrievalIntent.POLICY_LOOKUP,KnowledgeDomain.COMMERCIAL_POLICY,KnowledgeScope.POLICY,(RetrievalEvidenceType.COMMERCIAL_POLICY,),(KnowledgePurpose.PRICING_POLICY,)
    if kind==KnowledgeNeedKind.POLICY_INFORMATION: return RetrievalPurpose.ANSWER_POLICY_QUESTION,RetrievalIntent.POLICY_LOOKUP,KnowledgeDomain.COMPANY,KnowledgeScope.POLICY,(RetrievalEvidenceType.POLICY,),(KnowledgePurpose.LEGAL_POLICY,KnowledgePurpose.INTERNAL_SOP)
    if kind==KnowledgeNeedKind.IMPLEMENTATION_INFORMATION: return RetrievalPurpose.EXPLAIN_IMPLEMENTATION,RetrievalIntent.EXPLANATION,KnowledgeDomain.TECHNICAL,KnowledgeScope.TECHNICAL,(RetrievalEvidenceType.IMPLEMENTATION,RetrievalEvidenceType.REQUIREMENT),(KnowledgePurpose.TECHNICAL_DOCUMENTATION,KnowledgePurpose.SERVICE_KNOWLEDGE)
    if kind==KnowledgeNeedKind.FAQ: return RetrievalPurpose.ANSWER_DIRECT_QUESTION,RetrievalIntent.EXPLANATION,KnowledgeDomain.FAQ,KnowledgeScope.FAQ,(RetrievalEvidenceType.FAQ,),(KnowledgePurpose.FAQ,)
    if kind in {KnowledgeNeedKind.TECHNICAL_DETAIL,KnowledgeNeedKind.FEATURE_CLARIFICATION}: return RetrievalPurpose.CLARIFY_CAPABILITY,RetrievalIntent.CLARIFICATION,KnowledgeDomain.TECHNICAL,KnowledgeScope.TECHNICAL,(RetrievalEvidenceType.FEATURE,RetrievalEvidenceType.INTEGRATION,RetrievalEvidenceType.LIMITATION),(KnowledgePurpose.TECHNICAL_DOCUMENTATION,)
    return RetrievalPurpose.ANSWER_DIRECT_QUESTION if need.explicit_direct_request else RetrievalPurpose.EDUCATE,RetrievalIntent.EXPLANATION,KnowledgeDomain.COMPANY,KnowledgeScope.SERVICE if need.service_context_id else KnowledgeScope.BUSINESS,(RetrievalEvidenceType.SERVICE_DESCRIPTION,RetrievalEvidenceType.CAPABILITY),(KnowledgePurpose.SERVICE_KNOWLEDGE,KnowledgePurpose.FAQ)

def _refine_concept(concept,context,business_id):
    if concept is None: return None
    category=concept.business_context_category
    if category is None and context.business_memory is not None:
        industry=next((node.value for node in context.business_memory.nodes if node.identity_key=="bci:fact:industry"),None)
        if industry:
            normalized=industry.casefold().replace(" ","_")
            if normalized.replace("_","").isalnum() and len(normalized)<=80: category=normalized
    aspect=concept.aspect
    if aspect=="general" and context.diagnostic is not None: aspect=context.diagnostic.diagnostic_focus.value
    return replace(concept,aspect=aspect,business_context_category=category)
