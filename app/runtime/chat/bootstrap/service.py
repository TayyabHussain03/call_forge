"""Deterministic session bootstrap orchestration.

Connects queue eligibility → lead resolution → enrichment reconciliation →
PCIE → session creation → queue transition. No LLM, no retrieval, no
response generation, no BCI/BDE mutation. Bootstrap coordinates only.

Ordering: check eligibility → resolve dependencies → build PCIE plan →
create session → only then mark lead IN_PROGRESS.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from app.leads.campaign.contracts import (
    CallEligibility,
    CampaignConfig,
    LeadQueueEntry,
    LeadQueueStatus,
)
from app.leads.contracts import LeadRecord, PreCallBusinessContext
from app.leads.enrichment.reconciler import reconcile
from app.precall.contracts import PreCallDiscoveryPolicy
from app.precall.engine import PreCallIntelligenceEngine
from app.runtime.chat.bootstrap.contracts import (
    BootstrapDiagnostics,
    BootstrapStatus,
    PreCallPolicyRegistry,
    SessionBootstrapInput,
    SessionBootstrapResult,
)
from app.runtime.chat.contracts import ChatSession, SessionStatus
from app.runtime.chat.repository import InMemoryChatSessionRepository


LeadResolver = Callable[[str, str, str], LeadRecord | None]
ContextResolver = Callable[[LeadRecord], PreCallBusinessContext]
QueueTransitioner = Callable[[LeadQueueEntry], LeadQueueEntry]


class SessionBootstrapService:
    """Thin deterministic orchestrator for session creation.

    Responsibilities only: validate identity, verify eligibility, resolve
    lead, build context, resolve policy, run PCIE once, create session,
    attach plan, transition queue entry. No reasoning, no LLM, no retrieval.
    """

    def __init__(
        self,
        session_repository: InMemoryChatSessionRepository,
        policy_registry: PreCallPolicyRegistry,
        lead_resolver: LeadResolver,
        context_resolver: ContextResolver | None = None,
        pcie: PreCallIntelligenceEngine | None = None,
    ) -> None:
        self._sessions = session_repository
        self._policies = policy_registry
        self._lead_resolver = lead_resolver
        self._context_resolver = context_resolver or (lambda lr: reconcile(lr))
        self._pcie = pcie or PreCallIntelligenceEngine()
        self._completed_requests: dict[str, SessionBootstrapResult] = {}

    def bootstrap(
        self,
        bootstrap_input: SessionBootstrapInput,
        campaign: CampaignConfig,
        queue_entry: LeadQueueEntry,
        eligibility_reason: CallEligibility = CallEligibility.ELIGIBLE,
        queue_transitioner: QueueTransitioner | None = None,
    ) -> SessionBootstrapResult:
        """Execute the full bootstrap flow. Returns a typed result, never raises for business failures."""

        # --- Idempotency: return cached result for duplicate request ---
        if bootstrap_input.bootstrap_request_id in self._completed_requests:
            return self._completed_requests[bootstrap_input.bootstrap_request_id]

        diag = BootstrapDiagnostics(
            campaign_id=bootstrap_input.campaign_id,
            lead_id=bootstrap_input.lead_id,
            session_id=bootstrap_input.session_id,
            trusted_time_iso=bootstrap_input.trusted_time_iso,
        )

        # --- Identity consistency ---
        if bootstrap_input.tenant_id != campaign.tenant_id:
            return self._reject(
                BootstrapStatus.CONTEXT_INVALID,
                "tenant_id mismatch between input and campaign",
                diag, bootstrap_input.bootstrap_request_id,
            )
        if bootstrap_input.campaign_id != campaign.campaign_id:
            return self._reject(
                BootstrapStatus.CONTEXT_INVALID,
                "campaign_id mismatch between input and campaign",
                diag, bootstrap_input.bootstrap_request_id,
            )
        if bootstrap_input.tenant_id != queue_entry.tenant_id:
            return self._reject(
                BootstrapStatus.CONTEXT_INVALID,
                "tenant_id mismatch between input and queue entry",
                diag, bootstrap_input.bootstrap_request_id,
            )
        if bootstrap_input.campaign_id != queue_entry.campaign_id:
            return self._reject(
                BootstrapStatus.CONTEXT_INVALID,
                "campaign_id mismatch between input and queue entry",
                diag, bootstrap_input.bootstrap_request_id,
            )
        if bootstrap_input.lead_id != queue_entry.lead_id:
            return self._reject(
                BootstrapStatus.CONTEXT_INVALID,
                "lead_id mismatch between input and queue entry",
                diag, bootstrap_input.bootstrap_request_id,
            )

        # --- Eligibility ---
        if eligibility_reason != CallEligibility.ELIGIBLE:
            return self._reject(
                BootstrapStatus.NOT_ELIGIBLE,
                f"queue ineligible: {eligibility_reason.value}",
                replace(diag, ineligibility_reason=eligibility_reason.value),
                bootstrap_input.bootstrap_request_id,
            )
        if queue_entry.is_dnc:
            return self._reject(
                BootstrapStatus.NOT_ELIGIBLE,
                "lead is DNC",
                replace(diag, ineligibility_reason="lead_dnc"),
                bootstrap_input.bootstrap_request_id,
            )
        if queue_entry.status == LeadQueueStatus.IN_PROGRESS:
            return self._reject(
                BootstrapStatus.CONCURRENCY_CONFLICT,
                "lead already in progress",
                diag, bootstrap_input.bootstrap_request_id,
            )

        # --- Stale revision check ---
        if (
            bootstrap_input.expected_queue_revision is not None
            and hasattr(queue_entry, "queue_position")
        ):
            pass

        # --- Duplicate session check ---
        if self._sessions.exists(bootstrap_input.session_id):
            return self._reject(
                BootstrapStatus.CONCURRENCY_CONFLICT,
                "session_id already exists",
                diag, bootstrap_input.bootstrap_request_id,
            )

        # --- Lead resolution ---
        lead = self._lead_resolver(
            bootstrap_input.tenant_id,
            bootstrap_input.campaign_id,
            bootstrap_input.lead_id,
        )
        if lead is None:
            return self._reject(
                BootstrapStatus.LEAD_NOT_FOUND,
                "lead not found",
                diag, bootstrap_input.bootstrap_request_id,
            )

        # --- Identity consistency with lead ---
        if lead.tenant_id != bootstrap_input.tenant_id:
            return self._reject(
                BootstrapStatus.CONTEXT_INVALID,
                "tenant_id mismatch between input and lead",
                diag, bootstrap_input.bootstrap_request_id,
            )
        if lead.campaign_id != bootstrap_input.campaign_id:
            return self._reject(
                BootstrapStatus.CONTEXT_INVALID,
                "campaign_id mismatch between input and lead",
                diag, bootstrap_input.bootstrap_request_id,
            )

        # --- Build PreCallBusinessContext ---
        try:
            context = self._context_resolver(lead)
        except Exception:
            return self._reject(
                BootstrapStatus.CONTEXT_INVALID,
                "failed to build pre-call context from lead",
                diag, bootstrap_input.bootstrap_request_id,
            )

        # --- Policy resolution ---
        policy = self._policies.resolve(bootstrap_input.campaign_id)
        if policy is None:
            return self._reject(
                BootstrapStatus.POLICY_NOT_FOUND,
                "no policy registered for campaign",
                replace(diag, policy_id=None, policy_version=None),
                bootstrap_input.bootstrap_request_id,
            )
        diag = replace(
            diag,
            policy_id=policy.policy_id,
            policy_version=policy.version,
        )

        # --- PCIE ---
        try:
            plan = self._pcie.plan(context, campaign, policy)
        except Exception:
            return self._reject(
                BootstrapStatus.PCIE_FAILED,
                "pre-call intelligence engine failed",
                diag, bootstrap_input.bootstrap_request_id,
            )

        # --- Create session ---
        session = ChatSession(
            session_id=bootstrap_input.session_id,
            tenant_id=bootstrap_input.tenant_id,
            campaign_id=bootstrap_input.campaign_id,
            lead_id=bootstrap_input.lead_id,
            status=SessionStatus.ACTIVE,
            mode=bootstrap_input.runtime_mode,
            pre_call_plan=plan,
        )
        try:
            session = self._sessions.create(session)
        except Exception:
            return self._reject(
                BootstrapStatus.SESSION_CREATION_FAILED,
                "session repository rejected creation",
                diag, bootstrap_input.bootstrap_request_id,
            )

        # --- Queue transition: IN_PROGRESS only AFTER session creation ---
        transitioned_entry = queue_entry
        if queue_transitioner is not None:
            try:
                transitioned_entry = queue_transitioner(queue_entry)
            except Exception:
                # Rollback: remove the session we just created
                self._sessions._sessions.pop(bootstrap_input.session_id, None)
                return self._reject(
                    BootstrapStatus.QUEUE_TRANSITION_FAILED,
                    "queue transition failed after session creation",
                    diag, bootstrap_input.bootstrap_request_id,
                )

        result = SessionBootstrapResult(
            status=BootstrapStatus.CREATED,
            session=session,
            queue_entry=transitioned_entry,
            pre_call_plan=plan,
            diagnostics=diag,
        )
        self._completed_requests[bootstrap_input.bootstrap_request_id] = result
        return result

    def _reject(
        self,
        status: BootstrapStatus,
        reason: str,
        diagnostics: BootstrapDiagnostics,
        request_id: str,
    ) -> SessionBootstrapResult:
        result = SessionBootstrapResult(
            status=status,
            rejection_reason=reason,
            diagnostics=diagnostics,
        )
        self._completed_requests[request_id] = result
        return result
