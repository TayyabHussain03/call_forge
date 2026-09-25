"""Bounded metadata-first retrieval with deterministic reciprocal-rank fusion."""
from __future__ import annotations
from collections import defaultdict
from app.knowledge.contracts import DocumentStatus
from app.knowledge.hybrid_retrieval.contracts import *
from app.knowledge.retrieval_planning.contracts import RetrievalPlanningStatus, RetrievalRequirement

class HybridRetrievalEngine:
    def __init__(self,repository:KnowledgeRepository,lexical:LexicalIndexProvider|None,embedding:EmbeddingProvider|None,vector:VectorIndexProvider|None,configuration:HybridRetrievalConfiguration):
        self._repository=repository; self._lexical=lexical; self._embedding=embedding; self._vector=vector; self._config=configuration
    def retrieve(self,plan:RetrievalPlan)->RetrievalCandidateSet:
        if plan.requirement==RetrievalRequirement.NONE or plan.planning_status!=RetrievalPlanningStatus.READY or plan.concept is None: return self._empty(plan,CandidateSetStatus.INVALID_PLAN,BackendHealth.NOT_RUN,BackendHealth.NOT_RUN,0,0)
        try: snapshots=self._repository.enumerate_snapshots(plan.filters)
        except Exception: return self._empty(plan,CandidateSetStatus.INDEX_UNAVAILABLE,BackendHealth.FAILED,BackendHealth.FAILED,0,0)
        eligible=_eligible(snapshots,plan,self._config)
        if not eligible: return self._empty(plan,CandidateSetStatus.NO_CANDIDATES,BackendHealth.NOT_RUN,BackendHealth.NOT_RUN,0,0)
        chunks={chunk.chunk_id:(item,chunk) for item in eligible for chunk in item.snapshot.chunks}
        query=IndexQuery(plan.concept,tuple(sorted(chunks)))
        lexical_hits=(); semantic_hits=(); lh=BackendHealth.NOT_RUN; sh=BackendHealth.NOT_RUN; vector=None
        if self._config.mode in {RetrievalMode.LEXICAL,RetrievalMode.HYBRID}:
            try:
                if self._lexical is None: raise CorruptIndexError()
                lexical_hits=self._lexical.search(query,self._config.tokenizer,plan.budget.max_chunks,plan.budget.max_retrieval_duration_ms); _validate_hits(lexical_hits,chunks,self._config.lexical_index_version); lh=BackendHealth.HEALTHY
            except RetrievalTimeout: lexical_hits=(); lh=BackendHealth.TIMED_OUT
            except CorruptIndexError: lexical_hits=(); lh=BackendHealth.CORRUPT
            except Exception: lexical_hits=(); lh=BackendHealth.FAILED
        if self._config.mode in {RetrievalMode.SEMANTIC,RetrievalMode.HYBRID}:
            try:
                if self._embedding is None or self._vector is None or self._config.embedding is None: raise CorruptIndexError()
                vector=self._embedding.embed(plan.concept,self._config.embedding)
                if (vector.provider_name,vector.model_id,vector.dimension,vector.embedding_version)!=(self._config.embedding.provider_name,self._config.embedding.model_id,self._config.embedding.dimension,self._config.embedding.version): raise CorruptIndexError()
                semantic_hits=self._vector.search(vector,query.eligible_chunk_ids,plan.budget.max_chunks,plan.budget.max_retrieval_duration_ms); _validate_hits(semantic_hits,chunks,self._config.vector_index_version or ""); sh=BackendHealth.HEALTHY
            except RetrievalTimeout: semantic_hits=(); sh=BackendHealth.TIMED_OUT
            except CorruptIndexError: semantic_hits=(); sh=BackendHealth.CORRUPT
            except Exception: semantic_hits=(); sh=BackendHealth.FAILED
        candidates=_fuse(lexical_hits,semantic_hits,chunks,eligible,plan,self._config,vector,self._lexical,self._vector)
        health=[h for h in (lh,sh) if h!=BackendHealth.NOT_RUN]
        status=CandidateSetStatus.NO_CANDIDATES if not candidates else CandidateSetStatus.READY if all(h==BackendHealth.HEALTHY for h in health) else CandidateSetStatus.PARTIAL
        stats=RetrievalStatistics(len(eligible),len(chunks),len(lexical_hits),len(semantic_hits),len(set(h.chunk_id for h in (*lexical_hits,*semantic_hits))),len(candidates))
        return RetrievalCandidateSet(plan.plan_id,status,candidates,stats,_trace(plan,self._config,lh,sh,self._lexical,self._vector,eligible))
    def _empty(self,plan,status,lh,sh,docs,chunks):
        return RetrievalCandidateSet(plan.plan_id,status,(),RetrievalStatistics(docs,chunks,0,0,0,0),_trace(plan,self._config,lh,sh,self._lexical,self._vector,()))

def _eligible(items,plan,config):
    result=[]
    aliases={"en":"english","ur":"urdu","hi":"hindi","roman-urdu":"roman_urdu"}
    for item in items:
        if item.tenant_id!=plan.filters.tenant_id or item.business_id!=plan.filters.business_id or item.campaign_id!=plan.filters.campaign_id: continue
        if plan.filters.knowledge_base_id and item.snapshot.knowledge_base_id!=plan.filters.knowledge_base_id: continue
        if plan.filters.service_id and item.service_id!=plan.filters.service_id: continue
        if item.document_status!=DocumentStatus.PUBLISHED or not item.active_version: continue
        expected={config.lexical_index_version}
        if config.vector_index_version is not None: expected.add(config.vector_index_version)
        if item.index_version not in expected: continue
        if plan.filters.purposes and not any(chunk.purpose in plan.filters.purposes for chunk in item.snapshot.chunks): continue
        result.append(item)
    preferred=aliases.get(plan.filters.preferred_language,plan.filters.preferred_language)
    fallback=aliases.get(plan.filters.fallback_language,plan.filters.fallback_language)
    if preferred:
        preferred_items=tuple(item for item in result if item.snapshot.language.value==preferred)
        if preferred_items: return preferred_items
    if fallback: return tuple(item for item in result if item.snapshot.language.value==fallback)
    return tuple(result) if not preferred else ()

def _validate_hits(hits,chunks,index_version):
    seen=set()
    for hit in hits:
        if hit.chunk_id in seen or hit.chunk_id not in chunks or hit.index_version!=index_version or chunks[hit.chunk_id][1].checksum!=hit.checksum: raise CorruptIndexError()
        seen.add(hit.chunk_id)

def _fuse(lexical,semantic,chunks,eligible,plan,config,vector,lexical_provider,vector_provider):
    lr={hit.chunk_id:hit.rank for hit in lexical}; sr={hit.chunk_id:hit.rank for hit in semantic}; scores=defaultdict(float)
    for cid,rank in lr.items(): scores[cid]+=1/(config.rrf_constant+rank)
    for cid,rank in sr.items(): scores[cid]+=1/(config.rrf_constant+rank)
    rows=[]
    for cid,score in scores.items():
        item,chunk=chunks[cid]; metadata=1+int(chunk.purpose in plan.filters.purposes)+int(chunk.language.value==plan.filters.preferred_language)
        rows.append((cid,score,metadata,item.document_priority,chunk.chunk_order))
    rows.sort(key=lambda row:(-row[1],-row[2],-row[3],row[4],row[0]))
    counts=defaultdict(int); docs=set(); result=[]
    for cid,score,metadata,priority,order in rows:
        item,chunk=chunks[cid]
        if counts[chunk.document_id]>=config.max_chunks_per_document: continue
        if chunk.document_id not in docs and len(docs)>=plan.budget.max_documents: continue
        if len(result)>=min(config.max_candidates,plan.budget.max_chunks): break
        counts[chunk.document_id]+=1; docs.add(chunk.document_id)
        source=RetrievalSource.HYBRID if cid in lr and cid in sr else RetrievalSource.LEXICAL if cid in lr else RetrievalSource.SEMANTIC
        text=_bound_text(chunk.text,config)
        provenance=CandidateProvenance(item.snapshot_id,chunk.checksum,(config.vector_index_version if cid in sr else config.lexical_index_version) or "",getattr(lexical_provider,"name",None) if cid in lr else None,getattr(vector_provider,"name",None) if cid in sr else None,vector.provider_name if vector and cid in sr else None,vector.model_id if vector and cid in sr else None,vector.dimension if vector and cid in sr else None,vector.embedding_version if vector and cid in sr else None)
        validation=CandidateValidationMetadata(item.tenant_id,item.business_id,item.campaign_id,item.snapshot.knowledge_base_id,item.service_id,item.document_status,item.active_version)
        claims=dict(item.claims_by_chunk).get(cid,())
        result.append(RetrievalCandidate(chunk.document_id,chunk.version_id,chunk.document_version,cid,chunk.section_title,chunk.language,chunk.purpose,order,priority,metadata,source,lr.get(cid),sr.get(cid),score,len(result)+1,text,CandidateContentStatus.UNAPPROVED,provenance,validation,claims))
    return tuple(result)

def _bound_text(text,config):
    lines=text.splitlines()[:config.max_chunk_lines]; bounded="\n".join(lines)[:config.max_chunk_characters]; words=bounded.split()
    return " ".join(words[:config.max_chunk_tokens])

def _trace(plan,config,lh,sh,lexical,vector,eligible):
    return RetrievalTrace(plan.filters,config.mode,lh,sh,"rrf",getattr(lexical,"name",None),getattr(vector,"name",None),tuple(sorted({item.index_version for item in eligible})),plan.budget.max_retrieval_duration_ms)
