"""Focused tests for metadata-first, provider-neutral hybrid retrieval."""
from dataclasses import FrozenInstanceError
import pytest
from app.knowledge.contracts import DocumentStatus, KnowledgeCategory, KnowledgePurpose
from app.knowledge.processing.contracts import DocumentLanguage, DocumentSection, KnowledgeChunk, KnowledgeSnapshot, ProcessingStatus
from app.knowledge.retrieval_planning.contracts import *
from app.knowledge.hybrid_retrieval.contracts import *
from app.knowledge.hybrid_retrieval.engine import HybridRetrievalEngine

CHECK="a"*64
def _snapshot(doc="doc",kb="kb",language=DocumentLanguage.ENGLISH,purpose=KnowledgePurpose.SERVICE_KNOWLEDGE,chunks=2):
    parts=tuple(KnowledgeChunk(f"{doc}-c{i}","business",kb,doc,"v1",1,1,"Section",1,KnowledgeCategory.SERVICES,purpose,language,i,CHECK,f"{doc}#c{i}",("word "*350)+str(i)) for i in range(chunks))
    return KnowledgeSnapshot("business",kb,doc,"v1",1,CHECK,language,(DocumentSection("Section",1,0,1,"body"),),parts,ProcessingStatus.READY_FOR_INDEXING,(ProcessingStatus.UPLOADED,ProcessingStatus.PROCESSING,ProcessingStatus.PROCESSED,ProcessingStatus.READY_FOR_INDEXING))
def _indexed(**changes):
    values=dict(tenant_id="tenant",business_id="business",campaign_id="campaign",service_id=None,document_status=DocumentStatus.PUBLISHED,active_version=True,document_priority=5,index_version="idx1",snapshot_id="snap1",snapshot=_snapshot())
    values.update(changes); return IndexedKnowledgeSnapshot(**values)
def _plan(**changes):
    filters=RetrievalFilters("tenant","business","campaign",None,"kb",(KnowledgePurpose.SERVICE_KNOWLEDGE,),preferred_language="en",fallback_language=None)
    values=dict(plan_id="rp_plan",source_turn_id="t1",requirement=RetrievalRequirement.REQUIRED,planning_status=RetrievalPlanningStatus.READY,purpose=RetrievalPurpose.EDUCATE,intent=RetrievalIntent.EXPLANATION,domain=KnowledgeDomain.COMPANY,concept=KnowledgeConcept("automation","implementation"),scope=KnowledgeScope.BUSINESS,evidence_types=(RetrievalEvidenceType.CAPABILITY,),filters=filters,budget=RetrievalBudget(3,8,3000,1000),commercial_route=CommercialRetrievalRoute.NOT_COMMERCIAL)
    values.update(changes); return RetrievalPlan(**values)
class Repo:
    def __init__(self,items): self.items=items; self.calls=0
    def enumerate_snapshots(self,filters): self.calls+=1; return tuple(self.items)
class Lexical:
    name="lexical-test"
    def __init__(self,hits=(),error=None): self.hits=hits; self.error=error; self.tokenizer=None; self.calls=0
    def search(self,query,tokenizer,limit,timeout_ms):
        self.calls+=1; self.tokenizer=tokenizer; self.timeout_ms=timeout_ms
        if self.error: raise self.error
        return self.hits
class Embed:
    def __init__(self,error=None): self.error=error; self.calls=0
    def embed(self,concept,configuration):
        self.calls+=1
        if self.error: raise self.error
        return EmbeddingVector((0.1,0.2),configuration.provider_name,configuration.model_id,2,configuration.version)
class Vector:
    name="vector-test"
    def __init__(self,hits=(),error=None): self.hits=hits; self.error=error; self.calls=0
    def search(self,vector,eligible_chunk_ids,limit,timeout_ms):
        self.calls+=1; self.timeout_ms=timeout_ms
        if self.error: raise self.error
        return self.hits
def _config(mode=RetrievalMode.HYBRID,**changes):
    values=dict(mode=mode,tokenizer=TokenizerConfiguration(False,("the",),"english",True),embedding=EmbeddingConfiguration("embed-test","model",2,"e1") if mode!=RetrievalMode.LEXICAL else None,lexical_index_version="idx1",vector_index_version="idx1" if mode!=RetrievalMode.LEXICAL else None)
    values.update(changes); return HybridRetrievalConfiguration(**values)
def _engine(items=None,lex=None,vec=None,embed=None,config=None): return HybridRetrievalEngine(Repo([_indexed()] if items is None else items),lex,embed,vec,config or _config())
def _hit(cid,rank=1,version="idx1",checksum=CHECK): return RankedChunk(cid,rank,version,checksum)

def test_lexical_only_calls_provider_and_honors_tokenizer():
    lex=Lexical((_hit("doc-c0"),)); engine=_engine(lex=lex,config=_config(RetrievalMode.LEXICAL))
    result=engine.retrieve(_plan())
    assert result.status==CandidateSetStatus.READY and lex.calls==1 and lex.tokenizer.stop_words==("the",)
    assert lex.timeout_ms==1000 and result.trace.timeout_budget_ms==1000
    assert result.candidates[0].source==RetrievalSource.LEXICAL
def test_semantic_only_retains_embedding_and_index_provenance():
    vec=Vector((_hit("doc-c0"),)); result=_engine(lex=None,vec=vec,embed=Embed(),config=_config(RetrievalMode.SEMANTIC)).retrieve(_plan())
    p=result.candidates[0].provenance
    assert p.embedding_provider=="embed-test" and p.embedding_model=="model" and p.embedding_dimension==2 and p.vector_provider=="vector-test"
def test_hybrid_uses_rrf_deduplicates_and_is_deterministic():
    lex=Lexical((_hit("doc-c0",1),_hit("doc-c1",2))); vec=Vector((_hit("doc-c1",1),_hit("doc-c0",2)))
    engine=_engine(lex=lex,vec=vec,embed=Embed()); first=engine.retrieve(_plan()); second=engine.retrieve(_plan())
    assert first==second and len(first.candidates)==2
    assert all(c.source==RetrievalSource.HYBRID for c in first.candidates)
    assert [c.chunk_id for c in first.candidates]==["doc-c0","doc-c1"]
def test_metadata_filtering_precedes_search_and_fails_closed():
    lex=Lexical((_hit("doc-c0"),)); wrong=_indexed(tenant_id="other")
    result=_engine(items=[wrong],lex=lex,vec=Vector(),embed=Embed()).retrieve(_plan())
    assert result.status==CandidateSetStatus.NO_CANDIDATES and lex.calls==0
@pytest.mark.parametrize("change",[{"business_id":"other"},{"campaign_id":"other"},{"active_version":False},{"document_status":DocumentStatus.DRAFT},{"index_version":"old"}])
def test_scope_status_active_and_index_version_mismatches_are_excluded(change):
    assert _engine(items=[_indexed(**change)],lex=Lexical()).retrieve(_plan()).status==CandidateSetStatus.NO_CANDIDATES
def test_kb_purpose_language_and_service_filters_fail_closed():
    service_filters=RetrievalFilters("tenant","business","campaign","seo","kb",(KnowledgePurpose.CASE_STUDY,),preferred_language="ur")
    assert _engine(items=[_indexed(service_id="crm")],lex=Lexical()).retrieve(_plan(filters=service_filters)).status==CandidateSetStatus.NO_CANDIDATES
def test_language_fallback_is_used_only_when_preferred_is_absent():
    english=_indexed(snapshot_id="en",snapshot=_snapshot(doc="english",language=DocumentLanguage.ENGLISH))
    urdu=_indexed(snapshot_id="ur",snapshot=_snapshot(doc="urdu",language=DocumentLanguage.URDU))
    filters=RetrievalFilters("tenant","business","campaign",None,"kb",(KnowledgePurpose.SERVICE_KNOWLEDGE,),preferred_language="ur",fallback_language="en")
    lex=Lexical((_hit("urdu-c0"),))
    assert _engine(items=[english,urdu],lex=lex,config=_config(RetrievalMode.LEXICAL)).retrieve(_plan(filters=filters)).statistics.eligible_documents==1
    fallback_lex=Lexical((_hit("english-c0"),))
    result=_engine(items=[english],lex=fallback_lex,config=_config(RetrievalMode.LEXICAL)).retrieve(_plan(filters=filters))
    assert result.statistics.eligible_documents==1 and result.candidates[0].document_id=="english"
def test_backend_failure_degrades_and_both_failure_returns_empty():
    good=Lexical((_hit("doc-c0"),)); bad_vector=Vector(error=RuntimeError("down"))
    partial=_engine(lex=good,vec=bad_vector,embed=Embed()).retrieve(_plan())
    assert partial.status==CandidateSetStatus.PARTIAL and partial.trace.semantic_health==BackendHealth.FAILED
    empty=_engine(lex=Lexical(error=RetrievalTimeout()),vec=bad_vector,embed=Embed()).retrieve(_plan())
    assert empty.status==CandidateSetStatus.NO_CANDIDATES and not empty.candidates
def test_corrupt_checksum_or_index_is_rejected_without_filesystem_fallback():
    result=_engine(lex=Lexical((_hit("doc-c0",checksum="b"*64),)),vec=Vector(),embed=Embed()).retrieve(_plan())
    assert result.trace.lexical_health==BackendHealth.CORRUPT and not result.candidates
def test_diversity_limits_text_bounds_and_unapproved_content():
    lex=Lexical(tuple(_hit(f"doc-c{i}",i+1) for i in range(2)))
    config=_config(RetrievalMode.LEXICAL,max_chunks_per_document=1,max_chunk_characters=50,max_chunk_tokens=5,max_chunk_lines=1)
    result=_engine(lex=lex,config=config).retrieve(_plan())
    assert len(result.candidates)==1 and len(result.candidates[0].text.split())<=5
    assert result.candidates[0].content_status==CandidateContentStatus.UNAPPROVED
def test_invalid_plan_and_empty_repository_do_not_call_backends():
    lex=Lexical(); invalid=_plan(requirement=RetrievalRequirement.NONE,planning_status=RetrievalPlanningStatus.NOT_NEEDED,concept=None)
    assert _engine(lex=lex).retrieve(invalid).status==CandidateSetStatus.INVALID_PLAN and lex.calls==0
    assert _engine(items=[],lex=lex).retrieve(_plan()).status==CandidateSetStatus.NO_CANDIDATES
def test_candidate_set_is_immutable_and_contains_no_approved_evidence():
    result=_engine(lex=Lexical((_hit("doc-c0"),)),config=_config(RetrievalMode.LEXICAL)).retrieve(_plan())
    with pytest.raises(FrozenInstanceError): result.status=CandidateSetStatus.READY
    assert not hasattr(result,"approved_evidence") and not hasattr(result.candidates[0],"approved")
