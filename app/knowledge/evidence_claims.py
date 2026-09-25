"""Typed claim metadata shared by indexing, retrieval, and evidence validation."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from app.knowledge.processing.contracts import DocumentLanguage

class ClaimScope(str,Enum): TENANT="tenant"; BUSINESS="business"; CAMPAIGN="campaign"; SERVICE="service"; DOCUMENT="document"
class ClaimType(str,Enum): FEATURE="feature"; BENEFIT="benefit"; LIMITATION="limitation"; FAQ="faq"; PROCESS="process"; IMPLEMENTATION="implementation"; CASE_STUDY="case_study"; COMMERCIAL_POLICY="commercial_policy"; COMPLIANCE="compliance"
class EvidenceVisibility(str,Enum): CUSTOMER_VISIBLE="customer_visible"; INTERNAL_ONLY="internal_only"; COMMERCIAL_ONLY="commercial_only"; ENGINE_ONLY="engine_only"

@dataclass(frozen=True)
class EvidenceClaim:
    claim_id:str; claim_key:str; claim_subject:str; claim_predicate:str; claim_value:str
    claim_scope:ClaimScope; claim_type:ClaimType; visibility:EvidenceVisibility
    service_scope:str|None; language:DocumentLanguage; document_version:int; checksum:str
    def __post_init__(self):
        if not all((self.claim_id,self.claim_key,self.claim_subject,self.claim_predicate,self.claim_value)) or self.document_version<1 or len(self.checksum)!=64: raise ValueError("claim metadata is invalid")
