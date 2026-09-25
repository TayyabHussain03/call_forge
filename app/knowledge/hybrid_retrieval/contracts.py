"""Provider-neutral contracts for bounded hybrid candidate retrieval."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Protocol
from app.knowledge.contracts import DocumentStatus, KnowledgePurpose
from app.knowledge.processing.contracts import DocumentLanguage, KnowledgeSnapshot, ProcessingStatus
from app.knowledge.retrieval_planning.contracts import KnowledgeConcept, RetrievalFilters, RetrievalPlan
from app.knowledge.evidence_claims import EvidenceClaim

class RetrievalMode(str,Enum): LEXICAL="lexical"; SEMANTIC="semantic"; HYBRID="hybrid"
class RetrievalSource(str,Enum): LEXICAL="lexical"; SEMANTIC="semantic"; HYBRID="hybrid"
class BackendHealth(str,Enum): NOT_RUN="not_run"; HEALTHY="healthy"; FAILED="failed"; TIMED_OUT="timed_out"; CORRUPT="corrupt"
class CandidateSetStatus(str,Enum): READY="ready"; PARTIAL="partial"; NO_CANDIDATES="no_candidates"; INVALID_PLAN="invalid_plan"; SCOPE_REJECTED="scope_rejected"; INDEX_UNAVAILABLE="index_unavailable"
class CandidateContentStatus(str,Enum): UNAPPROVED="unapproved"

@dataclass(frozen=True)
class TokenizerConfiguration:
    stemming: bool; stop_words: tuple[str,...]; language: str; case_normalization: bool

@dataclass(frozen=True)
class EmbeddingConfiguration:
    provider_name: str; model_id: str; dimension: int; version: str
    def __post_init__(self):
        if not self.provider_name or not self.model_id or not self.version or not 1 <= self.dimension <= 16384: raise ValueError("embedding configuration is invalid")

@dataclass(frozen=True)
class HybridRetrievalConfiguration:
    mode: RetrievalMode; tokenizer: TokenizerConfiguration; embedding: EmbeddingConfiguration | None
    lexical_index_version: str; vector_index_version: str | None; rrf_constant: int=60
    max_candidates: int=12; max_chunks_per_document: int=3; max_chunk_characters: int=1200; max_chunk_tokens: int=300; max_chunk_lines: int=20
    def __post_init__(self):
        if self.mode in {RetrievalMode.SEMANTIC,RetrievalMode.HYBRID} and (self.embedding is None or self.vector_index_version is None): raise ValueError("semantic retrieval requires embedding and vector index configuration")
        if not 1 <= self.rrf_constant <= 1000 or not 1 <= self.max_candidates <= 50 or not 1 <= self.max_chunks_per_document <= 10: raise ValueError("retrieval configuration limits are invalid")

@dataclass(frozen=True)
class IndexedKnowledgeSnapshot:
    tenant_id: str; business_id: str; campaign_id: str; service_id: str|None
    document_status: DocumentStatus; active_version: bool; document_priority: int
    index_version: str; snapshot_id: str; snapshot: KnowledgeSnapshot
    claims_by_chunk: tuple[tuple[str,tuple[EvidenceClaim,...]],...] = ()
    def __post_init__(self):
        if self.snapshot.status != ProcessingStatus.READY_FOR_INDEXING: raise ValueError("only ready snapshots may be indexed")
        if len({chunk_id for chunk_id,_ in self.claims_by_chunk})!=len(self.claims_by_chunk): raise ValueError("chunk claim metadata must be unique")

@dataclass(frozen=True)
class CandidateValidationMetadata:
    tenant_id:str; business_id:str; campaign_id:str; knowledge_base_id:str
    service_id:str|None; document_status:DocumentStatus; active_version:bool

@dataclass(frozen=True)
class IndexQuery:
    concept: KnowledgeConcept; eligible_chunk_ids: tuple[str,...]

@dataclass(frozen=True)
class RankedChunk:
    chunk_id: str; rank: int; index_version: str; checksum: str
    def __post_init__(self):
        if self.rank < 1: raise ValueError("rank must be positive")

@dataclass(frozen=True)
class EmbeddingVector:
    values: tuple[float,...]; provider_name: str; model_id: str; dimension: int; embedding_version: str
    def __post_init__(self):
        if len(self.values)!=self.dimension: raise ValueError("embedding dimension mismatch")

class KnowledgeRepository(Protocol):
    def enumerate_snapshots(self, filters: RetrievalFilters) -> tuple[IndexedKnowledgeSnapshot,...]: ...
class LexicalIndexProvider(Protocol):
    name: str
    def search(self, query: IndexQuery, tokenizer: TokenizerConfiguration, limit: int, timeout_ms: int) -> tuple[RankedChunk,...]: ...
class EmbeddingProvider(Protocol):
    def embed(self, concept: KnowledgeConcept, configuration: EmbeddingConfiguration) -> EmbeddingVector: ...
class VectorIndexProvider(Protocol):
    name: str
    def search(self, vector: EmbeddingVector, eligible_chunk_ids: tuple[str,...], limit: int, timeout_ms: int) -> tuple[RankedChunk,...]: ...

@dataclass(frozen=True)
class CandidateProvenance:
    snapshot_id: str; chunk_checksum: str; index_version: str
    lexical_provider: str|None=None; vector_provider: str|None=None
    embedding_provider: str|None=None; embedding_model: str|None=None; embedding_dimension: int|None=None; embedding_version: str|None=None

@dataclass(frozen=True)
class RetrievalCandidate:
    document_id: str; version_id: str; document_version: int; chunk_id: str
    section: str; language: DocumentLanguage; purpose: KnowledgePurpose
    chunk_order: int; document_priority: int; metadata_match: int
    source: RetrievalSource; lexical_rank: int|None; semantic_rank: int|None; rrf_score: float; rrf_rank: int
    text: str; content_status: CandidateContentStatus; provenance: CandidateProvenance
    validation: CandidateValidationMetadata|None=None
    claims: tuple[EvidenceClaim,...]=()

@dataclass(frozen=True)
class RetrievalStatistics:
    eligible_documents: int; eligible_chunks: int; lexical_hits: int; semantic_hits: int; deduplicated_hits: int; returned_candidates: int

@dataclass(frozen=True)
class RetrievalTrace:
    filters: RetrievalFilters; mode: RetrievalMode; lexical_health: BackendHealth; semantic_health: BackendHealth
    fusion_strategy: str; lexical_provider: str|None; vector_provider: str|None; index_versions: tuple[str,...]; timeout_budget_ms: int

@dataclass(frozen=True)
class RetrievalCandidateSet:
    plan_id: str; status: CandidateSetStatus; candidates: tuple[RetrievalCandidate,...]; statistics: RetrievalStatistics; trace: RetrievalTrace

class RetrievalBackendError(RuntimeError): pass
class RetrievalTimeout(RetrievalBackendError): pass
class CorruptIndexError(RetrievalBackendError): pass
