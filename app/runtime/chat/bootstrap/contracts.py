"""Immutable contracts for session bootstrap orchestration.

Bootstrap coordinates. It does not reason. It connects queue eligibility,
pre-call intelligence, and session creation into one deterministic flow.

TRUST BOUNDARY: Bootstrap input carries trusted organizational identity.
It never generates responses, calls an LLM, or mutates conversation state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.leads.campaign.contracts import LeadQueueEntry
from app.precall.contracts import PreCallConversationPlan, PreCallDiscoveryPolicy
from app.runtime.chat.contracts import ChatSession, RuntimeMode


# ---------------------------------------------------------------------------
# Bootstrap status
# ---------------------------------------------------------------------------


class BootstrapStatus(str, Enum):
    """Outcome of a session bootstrap attempt."""

    CREATED = "created"
    NOT_ELIGIBLE = "not_eligible"
    LEAD_NOT_FOUND = "lead_not_found"
    POLICY_NOT_FOUND = "policy_not_found"
    CONTEXT_INVALID = "context_invalid"
    PCIE_FAILED = "pcie_failed"
    CONCURRENCY_CONFLICT = "concurrency_conflict"
    STALE_REVISION = "stale_revision"
    DUPLICATE_REQUEST = "duplicate_request"
    SESSION_CREATION_FAILED = "session_creation_failed"
    QUEUE_TRANSITION_FAILED = "queue_transition_failed"


# ---------------------------------------------------------------------------
# Bootstrap input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionBootstrapInput:
    """Typed input for session bootstrap. All identity is explicit."""

    tenant_id: str
    campaign_id: str
    lead_id: str
    session_id: str
    bootstrap_request_id: str
    trusted_time_iso: str
    runtime_mode: RuntimeMode = RuntimeMode.INTERACTIVE
    expected_queue_revision: int | None = None

    def __post_init__(self) -> None:
        for name in ("tenant_id", "campaign_id", "lead_id",
                      "session_id", "bootstrap_request_id", "trusted_time_iso"):
            val = getattr(self, name)
            if not val or not val.strip():
                raise ValueError(f"{name} must not be empty")


# ---------------------------------------------------------------------------
# Bootstrap result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapDiagnostics:
    """Bounded audit metadata — no secrets, no phone/email."""

    policy_id: str | None = None
    policy_version: str | None = None
    campaign_id: str | None = None
    lead_id: str | None = None
    session_id: str | None = None
    trusted_time_iso: str | None = None
    queue_revision: int | None = None
    ineligibility_reason: str | None = None


@dataclass(frozen=True)
class SessionBootstrapResult:
    """Immutable result of a bootstrap attempt."""

    status: BootstrapStatus
    session: ChatSession | None = None
    queue_entry: LeadQueueEntry | None = None
    pre_call_plan: PreCallConversationPlan | None = None
    rejection_reason: str | None = None
    diagnostics: BootstrapDiagnostics | None = None

    def __post_init__(self) -> None:
        if self.status == BootstrapStatus.CREATED:
            if self.session is None:
                raise ValueError("CREATED result must include session")
            if self.pre_call_plan is None:
                raise ValueError("CREATED result must include pre_call_plan")


# ---------------------------------------------------------------------------
# Policy registry protocol
# ---------------------------------------------------------------------------


class PreCallPolicyRegistry:
    """Deterministic policy lookup by campaign identity.

    No fuzzy matching. No fallback to closest. Missing → fail closed
    unless campaign explicitly allows a safe default.
    """

    def __init__(self) -> None:
        self._policies: dict[tuple[str, str], PreCallDiscoveryPolicy] = {}
        self._defaults: dict[str, PreCallDiscoveryPolicy] = {}

    def register(self, campaign_id: str, policy: PreCallDiscoveryPolicy) -> None:
        self._policies[(campaign_id, policy.version)] = policy

    def register_default(self, campaign_id: str, policy: PreCallDiscoveryPolicy) -> None:
        self._defaults[campaign_id] = policy
        self.register(campaign_id, policy)

    def resolve(
        self,
        campaign_id: str,
        version: str | None = None,
    ) -> PreCallDiscoveryPolicy | None:
        if version is not None:
            return self._policies.get((campaign_id, version))
        return self._defaults.get(campaign_id)
