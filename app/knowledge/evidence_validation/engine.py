"""Policy-first approval of typed claims; retrieved prose is never interpreted."""
from __future__ import annotations
from collections import Counter,defaultdict
from hashlib import sha256
from app.knowledge.contracts import DocumentStatus
from app.knowledge.evidence_claims import ClaimType,EvidenceVisibility
from app.knowledge.evidence_validation.contracts import *
from app.knowledge.hybrid_retrieval.contracts import RetrievalCandidate,RetrievalCandidateSet

class EvidenceValidationEngine:
    def validate(self,candidates:RetrievalCandidateSet,policy:EvidenceValidationPolicy,commercial:CommercialDisclosurePolicy)->EvidenceValidationResult:
        rejections=[]; accepted=[]
        for candidate in candidates.candidates:
            rejection=_validate_candidate(candidate,policy,commercial)
            if rejection: rejections.append(rejection)
            else:
                for claim in candidate.claims: accepted.append((candidate,claim))
        accepted,rejected_duplicates=_deduplicate(accepted); rejections.extend(rejected_duplicates)
        accepted,rejected_stale=_freshness(accepted); rejections.extend(rejected_stale)
        accepted,conflicts,rejected_conflicts=_conflicts(accepted,policy.conflict_resolution); rejections.extend(rejected_conflicts)
        counts=Counter((claim.claim_key,claim.claim_scope,claim.service_scope,claim.claim_value) for _,claim in accepted)
        approved=[]; docs=defaultdict(int); sections=defaultdict(int)
        for candidate,claim in sorted(accepted,key=lambda item:(item[0].rrf_rank,item[0].chunk_id,item[1].claim_id)):
            if len(approved)>=policy.max_approved_evidence or docs[candidate.document_id]>=policy.max_evidence_per_document or sections[(candidate.document_id,candidate.section)]>=policy.max_evidence_per_section:
                rejections.append(_reject(candidate,claim,RejectionReason.POLICY_BLOCKED,ValidationStep.APPROVAL)); continue
            quality=EvidenceQuality.SUPPORTED if counts[(claim.claim_key,claim.claim_scope,claim.service_scope,claim.claim_value)]>1 else EvidenceQuality.LIMITED if claim.claim_type==ClaimType.CASE_STUDY else EvidenceQuality.DIRECT
            identity="|".join((claim.claim_id,candidate.document_id,candidate.version_id,candidate.chunk_id,policy.validator_version))
            approved.append(ApprovedEvidence("ae_"+sha256(identity.encode()).hexdigest()[:24],claim.claim_id,candidate.document_id,candidate.version_id,candidate.document_version,candidate.chunk_id,candidate.section,candidate.text,_group(claim.claim_type),quality,(ApprovalReason.VALID_SCOPE,ApprovalReason.LATEST_VERSION,ApprovalReason.VISIBLE,ApprovalReason.POLICY_ALLOWED,ApprovalReason.NO_CONFLICT,ApprovalReason.CHECKSUM_VALID),candidate.provenance.snapshot_id,candidate.provenance.chunk_checksum,policy.validator_version)); docs[candidate.document_id]+=1; sections[(candidate.document_id,candidate.section)]+=1
        rejected_ids={item.candidate_id for item in rejections}; approved_ids={item.chunk_id for item in approved}
        lifecycle=tuple(CandidateValidationRecord(candidate.chunk_id,CandidateLifecycle.APPROVED if candidate.chunk_id in approved_ids else CandidateLifecycle.REJECTED if candidate.chunk_id in rejected_ids else CandidateLifecycle.VALIDATED) for candidate in candidates.candidates)
        return EvidenceValidationResult(ApprovedEvidenceSet(tuple(approved)),RejectionTrace(tuple(rejections)),ConflictTrace(tuple(conflicts)),lifecycle)

def _validate_candidate(c,p,commercial):
    m=c.validation
    if m is None or not c.claims: return _reject(c,None,RejectionReason.MISSING_METADATA,ValidationStep.CLAIMS)
    checks=((m.tenant_id!=p.tenant_id,RejectionReason.WRONG_TENANT,ValidationStep.TENANT),(m.business_id!=p.business_id,RejectionReason.WRONG_BUSINESS,ValidationStep.BUSINESS),(m.campaign_id!=p.campaign_id,RejectionReason.WRONG_CAMPAIGN,ValidationStep.CAMPAIGN),(p.knowledge_base_id is not None and m.knowledge_base_id!=p.knowledge_base_id,RejectionReason.WRONG_KNOWLEDGE_BASE,ValidationStep.KNOWLEDGE_BASE),(m.document_status==DocumentStatus.DRAFT,RejectionReason.DRAFT,ValidationStep.STATUS),(m.document_status==DocumentStatus.ARCHIVED or not m.active_version,RejectionReason.ARCHIVED,ValidationStep.STATUS),(c.provenance.index_version not in p.allowed_index_versions,RejectionReason.VERSION_MISMATCH,ValidationStep.INDEX_VERSION),(c.purpose not in p.allowed_purposes,RejectionReason.POLICY_BLOCKED,ValidationStep.PURPOSE),(m.service_id is not None and m.service_id not in p.allowed_service_ids,RejectionReason.WRONG_SERVICE,ValidationStep.SERVICE),(c.language not in p.allowed_languages,RejectionReason.WRONG_LANGUAGE,ValidationStep.LANGUAGE))
    for failed,reason,step in checks:
        if failed:return _reject(c,None,reason,step)
    for claim in c.claims:
        if claim.document_version!=c.document_version:return _reject(c,claim,RejectionReason.VERSION_MISMATCH,ValidationStep.VERSION)
        if claim.checksum!=c.provenance.chunk_checksum:return _reject(c,claim,RejectionReason.CHECKSUM_FAILED,ValidationStep.CHECKSUM)
        if claim.language!=c.language or claim.service_scope!=m.service_id:return _reject(c,claim,RejectionReason.MISSING_METADATA,ValidationStep.CLAIMS)
        if claim.visibility!=EvidenceVisibility.CUSTOMER_VISIBLE:return _reject(c,claim,RejectionReason.COMMERCIAL_RESTRICTED if claim.visibility==EvidenceVisibility.COMMERCIAL_ONLY else RejectionReason.POLICY_BLOCKED,ValidationStep.VISIBILITY)
        if claim.claim_type==ClaimType.COMMERCIAL_POLICY and claim.claim_key not in commercial.customer_visible_commercial_claims:return _reject(c,claim,RejectionReason.COMMERCIAL_RESTRICTED,ValidationStep.COMMERCIAL)
    return None

def _deduplicate(items):
    kept={}; rejected=[]
    for candidate,claim in sorted(items,key=lambda item:(item[0].rrf_rank,item[0].chunk_id)):
        key=(claim.claim_key,claim.claim_scope,claim.document_version,claim.service_scope,claim.claim_value)
        if key in kept: rejected.append(_reject(candidate,claim,RejectionReason.DUPLICATE,ValidationStep.CLAIMS))
        else: kept[key]=(candidate,claim)
    return list(kept.values()),rejected

def _freshness(items):
    newest={}
    for candidate,claim in items:
        key=(claim.claim_key,claim.claim_scope,claim.service_scope); newest[key]=max(newest.get(key,0),claim.document_version)
    kept=[]; rejected=[]
    for candidate,claim in items:
        if claim.document_version<newest[(claim.claim_key,claim.claim_scope,claim.service_scope)]: rejected.append(_reject(candidate,claim,RejectionReason.VERSION_MISMATCH,ValidationStep.VERSION))
        else: kept.append((candidate,claim))
    return kept,rejected

def _conflicts(items,mode):
    groups=defaultdict(list)
    for item in items: groups[(item[1].claim_key,item[1].claim_scope,item[1].service_scope)].append(item)
    kept=[]; traces=[]; rejected=[]
    for (key,_,_),group in groups.items():
        if len({claim.claim_value for _,claim in group})<=1: kept.extend(group); continue
        winner=None
        if mode==ConflictResolutionMode.LATEST_VERSION: winner=max(group,key=lambda item:(item[1].document_version,-item[0].rrf_rank,item[0].chunk_id))
        elif mode==ConflictResolutionMode.HIGHEST_PRIORITY: winner=max(group,key=lambda item:(item[0].document_priority,-item[0].rrf_rank,item[0].chunk_id))
        for item in group:
            if winner is not None and item==winner: kept.append(item)
            else: rejected.append(_reject(item[0],item[1],RejectionReason.CONFLICT,ValidationStep.CONFLICT))
        traces.append(ConflictRecord(key,tuple(sorted(item[0].chunk_id for item in group)),mode,winner[0].chunk_id if winner else None))
    return kept,traces,rejected

def _reject(candidate,claim,reason,step): return RejectionRecord(candidate.chunk_id,claim.claim_id if claim else None,reason,step,candidate.document_id,candidate.version_id,candidate.section)
def _group(claim_type): return EvidenceGroup(claim_type.value)
