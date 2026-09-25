"""Immutable contracts for deterministic validation of retrieved evidence."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from app.knowledge.contracts import KnowledgePurpose
from app.knowledge.evidence_claims import ClaimType, EvidenceClaim
from app.knowledge.processing.contracts import DocumentLanguage

class CandidateLifecycle(str,Enum): UNVALIDATED="unvalidated"; VALIDATED="validated"; REJECTED="rejected"; APPROVED="approved"
class ConflictResolutionMode(str,Enum): REJECT_ALL="reject_all"; LATEST_VERSION="latest_version"; HIGHEST_PRIORITY="highest_priority"
class EvidenceQuality(str,Enum): DIRECT="direct"; SUPPORTED="supported"; LIMITED="limited"
class EvidenceGroup(str,Enum): FEATURE="feature"; BENEFIT="benefit"; LIMITATION="limitation"; FAQ="faq"; PROCESS="process"; IMPLEMENTATION="implementation"; CASE_STUDY="case_study"; COMMERCIAL_POLICY="commercial_policy"; COMPLIANCE="compliance"
class ApprovalReason(str,Enum): VALID_SCOPE="valid_scope"; LATEST_VERSION="latest_version"; VISIBLE="visible"; POLICY_ALLOWED="policy_allowed"; NO_CONFLICT="no_conflict"; CHECKSUM_VALID="checksum_valid"
class RejectionReason(str,Enum): WRONG_TENANT="wrong_tenant"; WRONG_BUSINESS="wrong_business"; WRONG_CAMPAIGN="wrong_campaign"; WRONG_KNOWLEDGE_BASE="wrong_knowledge_base"; WRONG_SERVICE="wrong_service"; WRONG_LANGUAGE="wrong_language"; ARCHIVED="archived"; DRAFT="draft"; CHECKSUM_FAILED="checksum_failed"; VERSION_MISMATCH="version_mismatch"; POLICY_BLOCKED="policy_blocked"; COMMERCIAL_RESTRICTED="commercial_restricted"; CONFLICT="conflict"; DUPLICATE="duplicate"; MISSING_METADATA="missing_metadata"
class ValidationStep(str,Enum): TENANT="tenant"; BUSINESS="business"; CAMPAIGN="campaign"; KNOWLEDGE_BASE="knowledge_base"; STATUS="status"; VERSION="version"; CHECKSUM="checksum"; INDEX_VERSION="index_version"; VISIBILITY="visibility"; PURPOSE="purpose"; SERVICE="service"; LANGUAGE="language"; CLAIMS="claims"; CONFLICT="conflict"; COMMERCIAL="commercial"; APPROVAL="approval"

@dataclass(frozen=True)
class EvidenceValidationPolicy:
    tenant_id:str; business_id:str; campaign_id:str; knowledge_base_id:str|None
    allowed_service_ids:frozenset[str]; allowed_languages:frozenset[DocumentLanguage]
    allowed_purposes:frozenset[KnowledgePurpose]; allowed_index_versions:frozenset[str]
    conflict_resolution:ConflictResolutionMode=ConflictResolutionMode.REJECT_ALL
    max_approved_evidence:int=8; max_evidence_per_document:int=3; max_evidence_per_section:int=2
    validator_version:str="evae-v1"
    def __post_init__(self):
        if not all((self.tenant_id,self.business_id,self.campaign_id,self.validator_version)) or not 1<=self.max_approved_evidence<=30 or not 1<=self.max_evidence_per_document<=10 or not 1<=self.max_evidence_per_section<=10: raise ValueError("evidence validation policy is invalid")

@dataclass(frozen=True)
class CommercialDisclosurePolicy:
    customer_visible_commercial_claims:frozenset[str]=frozenset()

@dataclass(frozen=True)
class ApprovedEvidence:
    approval_id:str; claim_id:str; document_id:str; version_id:str; document_version:int; chunk_id:str; section:str
    approved_text:str; group:EvidenceGroup; quality:EvidenceQuality; approval_reasons:tuple[ApprovalReason,...]
    source_snapshot_id:str; checksum:str; validator_version:str

@dataclass(frozen=True)
class ApprovedEvidenceSet:
    items:tuple[ApprovedEvidence,...]

@dataclass(frozen=True)
class RejectionRecord:
    candidate_id:str; claim_id:str|None; reason:RejectionReason; validation_step:ValidationStep
    document_id:str; version_id:str; section:str

@dataclass(frozen=True)
class RejectionTrace:
    records:tuple[RejectionRecord,...]

@dataclass(frozen=True)
class ConflictRecord:
    claim_key:str; conflicting_candidate_ids:tuple[str,...]; resolution:ConflictResolutionMode; winner_candidate_id:str|None

@dataclass(frozen=True)
class ConflictTrace:
    records:tuple[ConflictRecord,...]

@dataclass(frozen=True)
class CandidateValidationRecord:
    candidate_id:str; state:CandidateLifecycle

@dataclass(frozen=True)
class EvidenceValidationResult:
    approved:ApprovedEvidenceSet; rejections:RejectionTrace; conflicts:ConflictTrace; lifecycle:tuple[CandidateValidationRecord,...]
