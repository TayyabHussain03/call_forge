"""Focused tests for planning-only, tenant-safe knowledge retrieval intent."""
from dataclasses import FrozenInstanceError, fields
import pytest
from app.knowledge.contracts import DocumentStatus, KnowledgePurpose
from app.knowledge.retrieval_planning.contracts import *
from app.knowledge.retrieval_planning.planner import KnowledgeRetrievalPlanner
from app.conversation.conversation_steering.contracts import (
    ConversationPrioritySnapshot, CuriosityBudgetState, CuriosityMemory,
    CustomerFatigue, DiagnosticPriority, DiscoveryStatus, DiscussionTiming,
    PriorityEvolution, QuestionDecision, QuestionReason, ConversationReadiness,
)
from app.conversation.business_memory.contracts import (
    BusinessGraphNode, BusinessMemoryScope, BusinessMentalModelSnapshot,
    BusinessMentalSummary, GraphEvidenceBasis, GraphNodeType, GraphSourceComponent,
)

def _policy(**changes):
    values=dict(tenant_id="tenant",business_id="business",campaign_id="campaign",knowledge_base_id="kb",authorized_service_ids=frozenset({"seo"}))
    values.update(changes); return KnowledgeRetrievalPolicy(**values)
def _need(kind=KnowledgeNeedKind.SERVICE_EXPLANATION, *, source=KnowledgeNeedSource.DIRECT_TYPED_REQUEST, direct=True, subject="seo", secondary=None, **changes):
    values=dict(need_id="n1",source_turn_id="t1",source=source,need_kind=kind,primary_concept=KnowledgeConcept(subject,"overview",secondary),explicit_direct_request=direct)
    values.update(changes); return KnowledgeNeed(**values)
def _plan(needs=(), policy=None, context=RetrievalPlanningContext(), **changes):
    scope=dict(tenant_id="tenant",business_id="business",campaign_id="campaign",context=context); scope.update(changes)
    return KnowledgeRetrievalPlanner().plan(tuple(needs),policy or _policy(),**scope)
def _steering(timing):
    return ConversationPrioritySnapshot(DiagnosticPriority.WORKFLOW,PriorityEvolution.RAISED,None,None,DiscoveryStatus.DISCOVERY_REQUIRED,ConversationReadiness.UNDERSTAND_MORE,CuriosityBudgetState.OPEN,QuestionDecision.ASK_ONE,QuestionReason.MISSING_WORKFLOW,timing,CustomerFatigue.NEUTRAL,CuriosityMemory())

def test_no_need_and_ambiguity_fail_closed():
    assert _plan().requirement==RetrievalRequirement.NONE
    ambiguous=_need(KnowledgeNeedKind.UNKNOWN,subject="topic",ambiguity=AmbiguityState.UNRESOLVED)
    assert _plan((ambiguous,)).planning_status==RetrievalPlanningStatus.NEEDS_CLARIFICATION
def test_direct_factual_question_is_required_even_when_premature():
    context=RetrievalPlanningContext(steering=_steering(DiscussionTiming.PREMATURE))
    plan=_plan((_need(service_context_id="seo"),),context=context)
    assert plan.requirement==RetrievalRequirement.REQUIRED and plan.filters.service_id=="seo"
def test_non_direct_education_is_optional_then_suppressed_when_premature():
    need=_need(source=KnowledgeNeedSource.CONSULTATIVE_DECISION,direct=False)
    assert _plan((need,)).requirement==RetrievalRequirement.OPTIONAL
    context=RetrievalPlanningContext(steering=_steering(DiscussionTiming.PREMATURE))
    assert _plan((need,),context=context).requirement==RetrievalRequirement.NONE
def test_source_precedence_selects_explicit_request_without_merging():
    lower=_need(source=KnowledgeNeedSource.STRUCTURED_UNDERSTANDING,subject="website")
    direct=_need(kind=KnowledgeNeedKind.FAQ,subject="seo",need_id="n2")
    assert _plan((lower,direct)).concept.subject=="seo"
def test_comparison_requires_exactly_one_secondary_concept():
    with pytest.raises(ValueError): _need(KnowledgeNeedKind.COMPARISON)
    plan=_plan((_need(KnowledgeNeedKind.COMPARISON,secondary="website"),))
    assert plan.purpose==RetrievalPurpose.COMPARE and plan.intent==RetrievalIntent.COMPARISON
def test_non_comparison_rejects_secondary_concept():
    with pytest.raises(ValueError): _need(secondary="website")
def test_scope_mismatch_and_unauthorized_service_fail_closed():
    assert _plan((_need(),),tenant_id="other").planning_status==RetrievalPlanningStatus.INVALID_SCOPE
    assert _plan((_need(service_context_id="crm"),)).requirement==RetrievalRequirement.NONE
def test_commercial_retrieval_is_policy_gated_but_grants_no_authority():
    plan=_plan((_need(KnowledgeNeedKind.COMMERCIAL_INFORMATION,subject="pricing"),),_policy(commercial_policy_retrieval_allowed=True))
    assert plan.commercial_route==CommercialRetrievalRoute.POLICY_GATED
    assert plan.evidence_types==(RetrievalEvidenceType.COMMERCIAL_POLICY,)
    assert not any("authority" in item.name or "price" in item.name for item in fields(RetrievalPlan))
def test_commercial_policy_absence_blocks_retrieval():
    plan=_plan((_need(KnowledgeNeedKind.COMMERCIAL_INFORMATION,subject="pricing"),))
    assert plan.planning_status==RetrievalPlanningStatus.BLOCKED_BY_POLICY
    assert plan.commercial_route==CommercialRetrievalRoute.RETRIEVAL_DISALLOWED
def test_published_active_filters_and_company_sales_separation():
    plan=_plan((_need(),))
    assert plan.filters.document_status==DocumentStatus.PUBLISHED and plan.filters.active_version_only
    assert KnowledgePurpose.SALES_KNOWLEDGE not in plan.filters.purposes
    with pytest.raises(ValueError):
        replace_filters=RetrievalFilters("tenant","business","campaign",None,None,(KnowledgePurpose.SALES_KNOWLEDGE,))
def test_purpose_aware_filters():
    assert _plan((_need(KnowledgeNeedKind.CASE_STUDY,subject="restaurant"),)).filters.purposes==(KnowledgePurpose.CASE_STUDY,)
    assert _plan((_need(KnowledgeNeedKind.FAQ),)).domain==KnowledgeDomain.FAQ
    assert KnowledgePurpose.TECHNICAL_DOCUMENTATION in _plan((_need(KnowledgeNeedKind.TECHNICAL_DETAIL),)).filters.purposes
def test_budget_validation_and_configured_budget_passthrough():
    with pytest.raises(ValueError): RetrievalBudget(max_documents=0)
    budget=RetrievalBudget(2,4,1200,500)
    assert _plan((_need(),),_policy(budget=budget)).budget==budget
def test_language_only_populates_language_filters():
    plan=_plan((_need(),),preferred_language="ur",fallback_language="en")
    assert (plan.filters.preferred_language,plan.filters.fallback_language)==("ur","en")
    assert not hasattr(plan.filters,"country") and not hasattr(plan.filters,"authority")
def test_business_memory_refines_context_without_creating_service_scope():
    scope=BusinessMemoryScope("tenant","business","campaign")
    industry=BusinessGraphNode("node","bci:fact:industry","tenant","business","campaign",None,GraphNodeType.BUSINESS,"Restaurant",GraphSourceComponent.BCI,"t1","t1",("t1",),GraphEvidenceBasis.OBSERVED)
    memory=BusinessMentalModelSnapshot(scope,(industry,),(),(),(),BusinessMentalSummary())
    plan=_plan((_need(subject="automation"),),context=RetrievalPlanningContext(business_memory=memory))
    assert plan.concept.business_context_category=="restaurant"
    assert plan.filters.service_id is None
def test_context_without_typed_need_cannot_create_retrieval_intent():
    context=RetrievalPlanningContext(steering=_steering(DiscussionTiming.NATURAL))
    assert _plan(context=context).requirement==RetrievalRequirement.NONE
def test_plan_identity_replay_immutability_and_no_execution_payload():
    one=_plan((_need(),)); two=_plan((_need(),))
    assert one==two
    with pytest.raises(FrozenInstanceError): one.requirement=RetrievalRequirement.NONE
    forbidden={"document_ids","chunk_ids","search_query","raw_text","transcript","approved_evidence"}
    assert not forbidden & set(RetrievalPlan.__dataclass_fields__)
def test_unknown_source_is_rejected():
    with pytest.raises(TypeError): KnowledgeNeed("n","t","random",KnowledgeNeedKind.NONE)  # type: ignore[arg-type]
