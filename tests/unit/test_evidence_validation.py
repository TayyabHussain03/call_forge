"""Focused policy, conflict, provenance, and fail-closed EVAE tests."""
from dataclasses import FrozenInstanceError, replace
import pytest
from app.knowledge.contracts import DocumentStatus, KnowledgePurpose
from app.knowledge.evidence_claims import *
from app.knowledge.evidence_validation.contracts import *
from app.knowledge.evidence_validation.engine import EvidenceValidationEngine
from app.knowledge.hybrid_retrieval.contracts import *
from app.knowledge.processing.contracts import DocumentLanguage
from app.knowledge.retrieval_planning.contracts import RetrievalFilters

CHECK="a"*64
def _claim(value="true",**changes):
    values=dict(claim_id="claim-1",claim_key="integration.whatsapp.supported",claim_subject="whatsapp",claim_predicate="supported",claim_value=value,claim_scope=ClaimScope.SERVICE,claim_type=ClaimType.FEATURE,visibility=EvidenceVisibility.CUSTOMER_VISIBLE,service_scope="automation",language=DocumentLanguage.ENGLISH,document_version=1,checksum=CHECK)
    values.update(changes); return EvidenceClaim(**values)
def _candidate(chunk="c1",claim=None,**changes):
    validation=CandidateValidationMetadata("tenant","business","campaign","kb","automation",DocumentStatus.PUBLISHED,True)
    values=dict(document_id="doc",version_id="v1",document_version=1,chunk_id=chunk,section="Features",language=DocumentLanguage.ENGLISH,purpose=KnowledgePurpose.SERVICE_KNOWLEDGE,chunk_order=0,document_priority=5,metadata_match=3,source=RetrievalSource.HYBRID,lexical_rank=1,semantic_rank=1,rrf_score=.03,rrf_rank=1,text="WhatsApp integration is supported.",content_status=CandidateContentStatus.UNAPPROVED,provenance=CandidateProvenance("snap",CHECK,"idx",lexical_provider="lex",vector_provider="vec"),validation=validation,claims=(_claim() if claim is None else claim,))
    values.update(changes); return RetrievalCandidate(**values)
def _set(*items):
    filters=RetrievalFilters("tenant","business","campaign","automation","kb",(KnowledgePurpose.SERVICE_KNOWLEDGE,))
    trace=RetrievalTrace(filters,RetrievalMode.HYBRID,BackendHealth.HEALTHY,BackendHealth.HEALTHY,"rrf","lex","vec",("idx",),1000)
    return RetrievalCandidateSet("plan",CandidateSetStatus.READY,tuple(items),RetrievalStatistics(1,len(items),len(items),len(items),len(items),len(items)),trace)
def _policy(**changes):
    values=dict(tenant_id="tenant",business_id="business",campaign_id="campaign",knowledge_base_id="kb",allowed_service_ids=frozenset({"automation"}),allowed_languages=frozenset({DocumentLanguage.ENGLISH}),allowed_purposes=frozenset({KnowledgePurpose.SERVICE_KNOWLEDGE}),allowed_index_versions=frozenset({"idx"}))
    values.update(changes); return EvidenceValidationPolicy(**values)
def _validate(*items,policy=None,commercial=CommercialDisclosurePolicy()): return EvidenceValidationEngine().validate(_set(*items),policy or _policy(),commercial)

def test_valid_claim_is_approved_with_stable_identity_reasons_and_provenance():
    one=_validate(_candidate()); two=_validate(_candidate())
    assert one==two and len(one.approved.items)==1
    evidence=one.approved.items[0]
    assert evidence.quality==EvidenceQuality.DIRECT and ApprovalReason.NO_CONFLICT in evidence.approval_reasons
    assert evidence.source_snapshot_id=="snap" and evidence.checksum==CHECK
def test_missing_claim_metadata_fails_closed():
    result=_validate(_candidate(claims=()))
    assert not result.approved.items and result.rejections.records[0].reason==RejectionReason.MISSING_METADATA
@pytest.mark.parametrize("field,value,reason",[("tenant_id","other",RejectionReason.WRONG_TENANT),("business_id","other",RejectionReason.WRONG_BUSINESS),("campaign_id","other",RejectionReason.WRONG_CAMPAIGN),("knowledge_base_id","other",RejectionReason.WRONG_KNOWLEDGE_BASE),("service_id","other",RejectionReason.WRONG_SERVICE)])
def test_scope_isolation(field,value,reason):
    candidate=_candidate(); metadata=replace(candidate.validation,**{field:value})
    assert _validate(replace(candidate,validation=metadata)).rejections.records[0].reason==reason
@pytest.mark.parametrize("status,active,reason",[(DocumentStatus.DRAFT,True,RejectionReason.DRAFT),(DocumentStatus.ARCHIVED,False,RejectionReason.ARCHIVED),(DocumentStatus.PUBLISHED,False,RejectionReason.ARCHIVED)])
def test_status_and_active_version_enforced(status,active,reason):
    candidate=_candidate(); metadata=replace(candidate.validation,document_status=status,active_version=active)
    assert _validate(replace(candidate,validation=metadata)).rejections.records[0].reason==reason
def test_checksum_version_language_purpose_and_index_are_validated():
    assert _validate(_candidate(claim=_claim(checksum="b"*64))).rejections.records[0].reason==RejectionReason.CHECKSUM_FAILED
    assert _validate(_candidate(claim=_claim(document_version=2))).rejections.records[0].reason==RejectionReason.VERSION_MISMATCH
    assert _validate(_candidate(language=DocumentLanguage.URDU)).rejections.records[0].reason==RejectionReason.WRONG_LANGUAGE
    assert _validate(_candidate(purpose=KnowledgePurpose.FAQ)).rejections.records[0].reason==RejectionReason.POLICY_BLOCKED
    assert _validate(_candidate(provenance=replace(_candidate().provenance,index_version="old"))).rejections.records[0].reason==RejectionReason.VERSION_MISMATCH
def test_visibility_and_commercial_policy_fail_closed():
    internal=_candidate(claim=_claim(visibility=EvidenceVisibility.INTERNAL_ONLY))
    assert _validate(internal).rejections.records[0].reason==RejectionReason.POLICY_BLOCKED
    commercial=_candidate(claim=_claim(claim_type=ClaimType.COMMERCIAL_POLICY,claim_key="pricing.standard"))
    assert _validate(commercial).rejections.records[0].reason==RejectionReason.COMMERCIAL_RESTRICTED
    allowed=_validate(commercial,commercial=CommercialDisclosurePolicy(frozenset({"pricing.standard"})))
    assert allowed.approved.items[0].group==EvidenceGroup.COMMERCIAL_POLICY
def test_typed_conflict_reject_all_and_does_not_inspect_prose():
    yes=_candidate("c1",_claim("true",claim_id="yes"),text="same prose")
    no=_candidate("c2",_claim("false",claim_id="no"),text="same prose",rrf_rank=2)
    result=_validate(yes,no)
    assert not result.approved.items and len(result.conflicts.records)==1
    assert all(item.reason==RejectionReason.CONFLICT for item in result.rejections.records)
def test_conflict_latest_version_and_highest_priority_are_deterministic():
    old=_candidate("old",_claim("true",claim_id="old",document_version=1),document_version=1)
    new=_candidate("new",_claim("false",claim_id="new",document_version=2),document_version=2,version_id="v2")
    latest=_validate(old,new,policy=_policy(conflict_resolution=ConflictResolutionMode.LATEST_VERSION))
    assert latest.approved.items[0].chunk_id=="new"
    low=_candidate("low",_claim("true",claim_id="low"),document_priority=1)
    high=_candidate("high",_claim("false",claim_id="high"),document_priority=9,rrf_rank=2)
    ranked=_validate(low,high,policy=_policy(conflict_resolution=ConflictResolutionMode.HIGHEST_PRIORITY))
    assert ranked.approved.items[0].chunk_id=="high"
def test_duplicate_normalization_and_diversity_limits():
    duplicate=_candidate("c2",_claim(claim_id="claim-2"),rrf_rank=2)
    result=_validate(_candidate(),duplicate)
    assert len(result.approved.items)==1 and any(r.reason==RejectionReason.DUPLICATE for r in result.rejections.records)
    second=_candidate("c3",_claim(claim_id="benefit",claim_key="benefit.speed",claim_type=ClaimType.BENEFIT),rrf_rank=3)
    limited=_validate(_candidate(),second,policy=_policy(max_evidence_per_document=1))
    assert len(limited.approved.items)==1 and any(r.validation_step==ValidationStep.APPROVAL for r in limited.rejections.records)
def test_result_is_immutable_and_has_no_retrieval_scores_or_authority():
    result=_validate(_candidate()); evidence=result.approved.items[0]
    with pytest.raises(FrozenInstanceError): result.approved=ApprovedEvidenceSet(())
    assert not hasattr(evidence,"rrf_score") and not hasattr(evidence,"authority") and not hasattr(evidence,"price")
