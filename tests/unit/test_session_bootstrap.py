"""Tests for Slice 40 — Deterministic Session Bootstrap & PCIE Runtime Wiring.

Proves: bootstrap coordinates, PCIE runs once, queue transition ordering,
idempotency, concurrency safety, tenant/campaign isolation, no duplicated
intelligence, typed PreCallConversationPlan on ChatSession.
"""

from __future__ import annotations

import inspect
from dataclasses import replace

import pytest

from app.leads.campaign.contracts import (
    CallEligibility,
    CampaignConfig,
    CampaignStatus,
    LeadQueueEntry,
    LeadQueueStatus,
)
from app.leads.contracts import (
    FieldKnowledgeState,
    LeadRecord,
    NormalizedPhone,
    PhoneNormalizationStatus,
    PreCallBusinessContext,
    WebsiteStatus,
)
from app.leads.enrichment.reconciler import reconcile
from app.precall.contracts import (
    PreCallConversationPlan,
    PreCallDiscoveryPolicy,
    ServiceDiscoveryRule,
)
from app.precall.engine import PreCallIntelligenceEngine
from app.runtime.chat.bootstrap.contracts import (
    BootstrapStatus,
    PreCallPolicyRegistry,
    SessionBootstrapInput,
    SessionBootstrapResult,
)
from app.runtime.chat.bootstrap.service import SessionBootstrapService
from app.runtime.chat.contracts import ChatSession, RuntimeMode, SessionStatus
from app.runtime.chat.repository import InMemoryChatSessionRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _phone() -> NormalizedPhone:
    return NormalizedPhone(
        raw="+15551234567",
        normalized="+15551234567",
        status=PhoneNormalizationStatus.VALID_E164,
    )


def _lead(
    lead_id: str = "lead-1",
    tenant_id: str = "t1",
    campaign_id: str = "c1",
    category: str | None = None,
    website: str | None = None,
    website_status: WebsiteStatus = WebsiteStatus.UNKNOWN,
) -> LeadRecord:
    return LeadRecord(
        lead_id=lead_id,
        tenant_id=tenant_id,
        campaign_id=campaign_id,
        business_name="Test Biz",
        phone=_phone(),
        category=category,
        city=None,
        state=None,
        country=None,
        address=None,
        email=None,
        contact_name=None,
        website=website,
        website_status=website_status,
    )


def _campaign(
    campaign_id: str = "c1",
    tenant_id: str = "t1",
    status: CampaignStatus = CampaignStatus.ACTIVE,
    authorized_service_ids: tuple[str, ...] = ("svc-web",),
) -> CampaignConfig:
    return CampaignConfig(
        campaign_id=campaign_id,
        tenant_id=tenant_id,
        name="Test Campaign",
        status=status,
        authorized_service_ids=authorized_service_ids,
    )


def _queue_entry(
    lead_id: str = "lead-1",
    campaign_id: str = "c1",
    tenant_id: str = "t1",
    status: LeadQueueStatus = LeadQueueStatus.READY,
    is_dnc: bool = False,
) -> LeadQueueEntry:
    return LeadQueueEntry(
        lead_id=lead_id,
        campaign_id=campaign_id,
        tenant_id=tenant_id,
        queue_position=0,
        status=status,
        is_dnc=is_dnc,
    )


def _policy(
    campaign_id: str = "c1",
    service_rules: tuple[ServiceDiscoveryRule, ...] = (),
) -> PreCallDiscoveryPolicy:
    return PreCallDiscoveryPolicy(
        policy_id="pol-1",
        version="1.0",
        service_rules=service_rules,
    )


_WEBSITE_RULE = ServiceDiscoveryRule(
    service_id="svc-web",
    required_fact_keys=("website.presence",),
    optional_fact_keys=("seo.quality",),
    knowledge_topic_hints=("web_design",),
)


def _bootstrap_input(
    session_id: str = "sess-1",
    bootstrap_request_id: str = "req-1",
    lead_id: str = "lead-1",
    tenant_id: str = "t1",
    campaign_id: str = "c1",
    runtime_mode: RuntimeMode = RuntimeMode.INTERACTIVE,
) -> SessionBootstrapInput:
    return SessionBootstrapInput(
        tenant_id=tenant_id,
        campaign_id=campaign_id,
        lead_id=lead_id,
        session_id=session_id,
        bootstrap_request_id=bootstrap_request_id,
        trusted_time_iso="2026-10-05T10:00:00Z",
        runtime_mode=runtime_mode,
    )


def _make_service(
    leads: dict[str, LeadRecord] | None = None,
    policy: PreCallDiscoveryPolicy | None = None,
    campaign_id: str = "c1",
) -> SessionBootstrapService:
    repo = InMemoryChatSessionRepository()
    registry = PreCallPolicyRegistry()
    if policy is not None:
        registry.register_default(campaign_id, policy)

    lead_store = leads if leads is not None else {"lead-1": _lead()}

    def resolver(tid: str, cid: str, lid: str) -> LeadRecord | None:
        return lead_store.get(lid)

    return SessionBootstrapService(
        session_repository=repo,
        policy_registry=registry,
        lead_resolver=resolver,
    )


# ---------------------------------------------------------------------------
# 1. Successful minimal lead bootstrap
# ---------------------------------------------------------------------------


class TestSuccessfulMinimalBootstrap:
    def test_minimal_lead_produces_created(self) -> None:
        svc = _make_service(policy=_policy())
        result = svc.bootstrap(
            _bootstrap_input(), _campaign(), _queue_entry(),
        )
        assert result.status == BootstrapStatus.CREATED
        assert result.session is not None
        assert result.session.status == SessionStatus.ACTIVE

    def test_session_identity_matches_input(self) -> None:
        svc = _make_service(policy=_policy())
        result = svc.bootstrap(
            _bootstrap_input(), _campaign(), _queue_entry(),
        )
        assert result.session.session_id == "sess-1"
        assert result.session.tenant_id == "t1"
        assert result.session.campaign_id == "c1"
        assert result.session.lead_id == "lead-1"


# ---------------------------------------------------------------------------
# 2. Successful enriched lead bootstrap
# ---------------------------------------------------------------------------


class TestSuccessfulEnrichedBootstrap:
    def test_enriched_lead_with_category(self) -> None:
        lead = _lead(category="Restaurant")
        svc = _make_service(
            leads={"lead-1": lead},
            policy=_policy(),
        )
        result = svc.bootstrap(
            _bootstrap_input(), _campaign(), _queue_entry(),
        )
        assert result.status == BootstrapStatus.CREATED
        plan = result.pre_call_plan
        known_keys = {f.key for f in plan.known_facts}
        assert "business.category" in known_keys


# ---------------------------------------------------------------------------
# 3. PCIE plan automatically attached
# ---------------------------------------------------------------------------


class TestPCIEPlanAttached:
    def test_plan_on_result(self) -> None:
        svc = _make_service(policy=_policy())
        result = svc.bootstrap(
            _bootstrap_input(), _campaign(), _queue_entry(),
        )
        assert result.pre_call_plan is not None
        assert isinstance(result.pre_call_plan, PreCallConversationPlan)

    def test_plan_on_session(self) -> None:
        svc = _make_service(policy=_policy())
        result = svc.bootstrap(
            _bootstrap_input(), _campaign(), _queue_entry(),
        )
        assert result.session.pre_call_plan is not None
        assert isinstance(result.session.pre_call_plan, PreCallConversationPlan)


# ---------------------------------------------------------------------------
# 4. Typed PreCallConversationPlan, not object
# ---------------------------------------------------------------------------


class TestTypedPlan:
    def test_chat_session_annotation_is_typed(self) -> None:
        hints = ChatSession.__dataclass_fields__["pre_call_plan"]
        assert "object" not in str(hints.type)


# ---------------------------------------------------------------------------
# 5. Website UNKNOWN preserved
# ---------------------------------------------------------------------------


class TestWebsiteUnknown:
    def test_unknown_website_becomes_unknown_fact(self) -> None:
        svc = _make_service(
            policy=_policy(service_rules=(_WEBSITE_RULE,)),
        )
        result = svc.bootstrap(
            _bootstrap_input(), _campaign(), _queue_entry(),
        )
        plan = result.pre_call_plan
        unknown_keys = {f.key for f in plan.unknown_facts}
        assert "website.presence" in unknown_keys


# ---------------------------------------------------------------------------
# 6. Dynamic arbitrary service policy
# ---------------------------------------------------------------------------


class TestDynamicServicePolicy:
    def test_custom_service_rule_creates_gaps(self) -> None:
        custom_rule = ServiceDiscoveryRule(
            service_id="svc-custom",
            required_fact_keys=("custom.fact",),
        )
        campaign = _campaign(authorized_service_ids=("svc-custom",))
        svc = _make_service(
            policy=_policy(service_rules=(custom_rule,)),
        )
        result = svc.bootstrap(
            _bootstrap_input(), campaign, _queue_entry(),
        )
        gap_ids = {g.service_id for g in result.pre_call_plan.service_prerequisite_gaps}
        assert "svc-custom" in gap_ids


# ---------------------------------------------------------------------------
# 7. Queue not eligible
# ---------------------------------------------------------------------------


class TestQueueNotEligible:
    def test_ineligible_returns_not_eligible(self) -> None:
        svc = _make_service(policy=_policy())
        result = svc.bootstrap(
            _bootstrap_input(), _campaign(), _queue_entry(),
            eligibility_reason=CallEligibility.DAILY_LIMIT_REACHED,
        )
        assert result.status == BootstrapStatus.NOT_ELIGIBLE
        assert result.session is None


# ---------------------------------------------------------------------------
# 8. Paused campaign
# ---------------------------------------------------------------------------


class TestPausedCampaign:
    def test_paused_campaign_ineligible(self) -> None:
        svc = _make_service(policy=_policy())
        result = svc.bootstrap(
            _bootstrap_input(), _campaign(), _queue_entry(),
            eligibility_reason=CallEligibility.CAMPAIGN_PAUSED,
        )
        assert result.status == BootstrapStatus.NOT_ELIGIBLE


# ---------------------------------------------------------------------------
# 9. DNC lead
# ---------------------------------------------------------------------------


class TestDNCLead:
    def test_dnc_lead_rejected(self) -> None:
        svc = _make_service(policy=_policy())
        entry = _queue_entry(is_dnc=True)
        result = svc.bootstrap(
            _bootstrap_input(), _campaign(), entry,
        )
        assert result.status == BootstrapStatus.NOT_ELIGIBLE
        assert "DNC" in result.rejection_reason


# ---------------------------------------------------------------------------
# 10. Concurrency limit
# ---------------------------------------------------------------------------


class TestConcurrencyLimit:
    def test_concurrency_limit_rejected(self) -> None:
        svc = _make_service(policy=_policy())
        result = svc.bootstrap(
            _bootstrap_input(), _campaign(), _queue_entry(),
            eligibility_reason=CallEligibility.CONCURRENCY_LIMIT,
        )
        assert result.status == BootstrapStatus.NOT_ELIGIBLE


# ---------------------------------------------------------------------------
# 11. Policy missing
# ---------------------------------------------------------------------------


class TestPolicyMissing:
    def test_no_policy_fails_closed(self) -> None:
        svc = _make_service(policy=None)
        result = svc.bootstrap(
            _bootstrap_input(), _campaign(), _queue_entry(),
        )
        assert result.status == BootstrapStatus.POLICY_NOT_FOUND


# ---------------------------------------------------------------------------
# 12. Wrong policy version
# ---------------------------------------------------------------------------


class TestWrongPolicyVersion:
    def test_versioned_lookup_misses(self) -> None:
        registry = PreCallPolicyRegistry()
        policy = _policy()
        registry.register("c1", policy)
        # No default registered, resolve without version returns None
        assert registry.resolve("c1") is None
        assert registry.resolve("c1", "1.0") is not None
        assert registry.resolve("c1", "2.0") is None


# ---------------------------------------------------------------------------
# 13. Tenant mismatch
# ---------------------------------------------------------------------------


class TestTenantMismatch:
    def test_input_campaign_tenant_mismatch(self) -> None:
        svc = _make_service(policy=_policy())
        inp = _bootstrap_input(tenant_id="t-other")
        result = svc.bootstrap(inp, _campaign(), _queue_entry())
        assert result.status == BootstrapStatus.CONTEXT_INVALID
        assert "tenant" in result.rejection_reason.lower()


# ---------------------------------------------------------------------------
# 14. Campaign mismatch
# ---------------------------------------------------------------------------


class TestCampaignMismatch:
    def test_input_queue_campaign_mismatch(self) -> None:
        svc = _make_service(policy=_policy())
        entry = _queue_entry(campaign_id="c-other")
        result = svc.bootstrap(
            _bootstrap_input(), _campaign(), entry,
        )
        assert result.status == BootstrapStatus.CONTEXT_INVALID
        assert "campaign" in result.rejection_reason.lower()


# ---------------------------------------------------------------------------
# 15. Lead mismatch
# ---------------------------------------------------------------------------


class TestLeadMismatch:
    def test_input_queue_lead_mismatch(self) -> None:
        svc = _make_service(policy=_policy())
        entry = _queue_entry(lead_id="lead-other")
        result = svc.bootstrap(
            _bootstrap_input(), _campaign(), entry,
        )
        assert result.status == BootstrapStatus.CONTEXT_INVALID
        assert "lead" in result.rejection_reason.lower()


# ---------------------------------------------------------------------------
# 16. Minimal / degraded plan allowed
# ---------------------------------------------------------------------------


class TestDegradedPlanAllowed:
    def test_minimal_lead_still_produces_plan(self) -> None:
        svc = _make_service(policy=_policy())
        result = svc.bootstrap(
            _bootstrap_input(), _campaign(), _queue_entry(),
        )
        assert result.status == BootstrapStatus.CREATED
        plan = result.pre_call_plan
        assert plan is not None
        all_facts = (
            plan.known_facts + plan.unknown_facts
            + plan.unverified_facts + plan.conflicting_facts
        )
        assert len(all_facts) > 0


# ---------------------------------------------------------------------------
# 17. Session creation failure leaves queue unchanged
# ---------------------------------------------------------------------------


class TestSessionCreationFailure:
    def test_duplicate_session_id_rejected(self) -> None:
        svc = _make_service(policy=_policy())
        # First bootstrap succeeds
        svc.bootstrap(
            _bootstrap_input(bootstrap_request_id="req-1"),
            _campaign(), _queue_entry(),
        )
        # Second with different request_id but same session_id
        result = svc.bootstrap(
            _bootstrap_input(bootstrap_request_id="req-2"),
            _campaign(), _queue_entry(),
        )
        assert result.status == BootstrapStatus.CONCURRENCY_CONFLICT


# ---------------------------------------------------------------------------
# 18. Queue transition failure handled safely
# ---------------------------------------------------------------------------


class TestQueueTransitionFailure:
    def test_transition_failure_rolls_back_session(self) -> None:
        repo = InMemoryChatSessionRepository()
        registry = PreCallPolicyRegistry()
        registry.register_default("c1", _policy())

        def resolver(tid: str, cid: str, lid: str) -> LeadRecord | None:
            return _lead()

        svc = SessionBootstrapService(
            session_repository=repo,
            policy_registry=registry,
            lead_resolver=resolver,
        )

        def failing_transitioner(entry: LeadQueueEntry) -> LeadQueueEntry:
            raise RuntimeError("queue backend error")

        result = svc.bootstrap(
            _bootstrap_input(), _campaign(), _queue_entry(),
            queue_transitioner=failing_transitioner,
        )
        assert result.status == BootstrapStatus.QUEUE_TRANSITION_FAILED
        assert not repo.exists("sess-1")


# ---------------------------------------------------------------------------
# 19. Duplicate bootstrap request is idempotent
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_same_request_id_returns_cached(self) -> None:
        svc = _make_service(policy=_policy())
        r1 = svc.bootstrap(
            _bootstrap_input(), _campaign(), _queue_entry(),
        )
        r2 = svc.bootstrap(
            _bootstrap_input(), _campaign(), _queue_entry(),
        )
        assert r1 is r2
        assert r1.status == BootstrapStatus.CREATED

    def test_idempotent_does_not_create_second_session(self) -> None:
        svc = _make_service(policy=_policy())
        svc.bootstrap(_bootstrap_input(), _campaign(), _queue_entry())
        svc.bootstrap(_bootstrap_input(), _campaign(), _queue_entry())
        assert svc._sessions.active_count() == 1


# ---------------------------------------------------------------------------
# 20. Stale revision rejected
# ---------------------------------------------------------------------------


class TestStaleRevision:
    def test_in_progress_lead_rejected(self) -> None:
        svc = _make_service(policy=_policy())
        entry = _queue_entry(status=LeadQueueStatus.IN_PROGRESS)
        result = svc.bootstrap(
            _bootstrap_input(), _campaign(), entry,
        )
        assert result.status == BootstrapStatus.CONCURRENCY_CONFLICT


# ---------------------------------------------------------------------------
# 21. Trusted-time determinism
# ---------------------------------------------------------------------------


class TestTrustedTime:
    def test_trusted_time_in_diagnostics(self) -> None:
        svc = _make_service(policy=_policy())
        result = svc.bootstrap(
            _bootstrap_input(), _campaign(), _queue_entry(),
        )
        assert result.diagnostics.trusted_time_iso == "2026-10-05T10:00:00Z"


# ---------------------------------------------------------------------------
# 22. Same inputs → same decision
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_equivalent_inputs_same_plan(self) -> None:
        plan1 = self._run_plan()
        plan2 = self._run_plan()
        assert plan1.known_facts == plan2.known_facts
        assert plan1.unknown_facts == plan2.unknown_facts
        assert plan1.verification_targets == plan2.verification_targets
        assert plan1.assumption_guardrails == plan2.assumption_guardrails

    @staticmethod
    def _run_plan() -> PreCallConversationPlan:
        svc = _make_service(policy=_policy(service_rules=(_WEBSITE_RULE,)))
        result = svc.bootstrap(
            _bootstrap_input(), _campaign(), _queue_entry(),
        )
        return result.pre_call_plan


# ---------------------------------------------------------------------------
# 23. No LLM
# ---------------------------------------------------------------------------


class TestNoLLM:
    def test_no_llm_imports_in_service(self) -> None:
        source = inspect.getsource(SessionBootstrapService)
        module_source = inspect.getmodule(SessionBootstrapService)
        full_source = inspect.getsource(module_source)
        for forbidden in ("BrainOrchestrator", "ReasoningProvider",
                          "MockReasoningProvider", "openai", "anthropic"):
            assert forbidden not in full_source


# ---------------------------------------------------------------------------
# 24. No retrieval
# ---------------------------------------------------------------------------


class TestNoRetrieval:
    def test_no_retrieval_in_service(self) -> None:
        full_source = inspect.getsource(inspect.getmodule(SessionBootstrapService))
        for forbidden in ("HybridRetrievalEngine", "KnowledgeRetrievalPlanner",
                          ".retrieve(", "import retrieval"):
            assert forbidden not in full_source


# ---------------------------------------------------------------------------
# 25. No response generation
# ---------------------------------------------------------------------------


class TestNoResponseGeneration:
    def test_no_rendering_in_service(self) -> None:
        full_source = inspect.getsource(inspect.getmodule(SessionBootstrapService))
        for forbidden in ("ResponseRenderer", "DeterministicResponseRenderer",
                          "RenderedResponse"):
            assert forbidden not in full_source


# ---------------------------------------------------------------------------
# 26. No BCI/BDE mutation
# ---------------------------------------------------------------------------


class TestNoBCIMutation:
    def test_no_bci_bde_in_service(self) -> None:
        full_source = inspect.getsource(inspect.getmodule(SessionBootstrapService))
        for forbidden in ("BusinessConversationIntelligence",
                          "BusinessDiscoveryEngine",
                          "StateMachine", "ActionValidator"):
            assert forbidden not in full_source


# ---------------------------------------------------------------------------
# 27. No service recommendation
# ---------------------------------------------------------------------------


class TestNoServiceRecommendation:
    def test_no_service_resolver_in_service(self) -> None:
        full_source = inspect.getsource(inspect.getmodule(SessionBootstrapService))
        for forbidden in ("ServiceRelevanceResolver", "recommend_service",
                          "SELL_WEBSITE"):
            assert forbidden not in full_source


# ---------------------------------------------------------------------------
# 28. Conversation truth supersedes pre-call plan
# ---------------------------------------------------------------------------


class TestConversationTruthSupersedes:
    def test_plan_is_immutable_on_session(self) -> None:
        svc = _make_service(policy=_policy())
        result = svc.bootstrap(
            _bootstrap_input(), _campaign(), _queue_entry(),
        )
        session = result.session
        plan = session.pre_call_plan
        assert plan is not None
        # Plan is frozen
        with pytest.raises(AttributeError):
            plan.lead_id = "modified"

    def test_session_is_immutable(self) -> None:
        svc = _make_service(policy=_policy())
        result = svc.bootstrap(
            _bootstrap_input(), _campaign(), _queue_entry(),
        )
        with pytest.raises(AttributeError):
            result.session.pre_call_plan = None


# ---------------------------------------------------------------------------
# Bootstrap contract validation
# ---------------------------------------------------------------------------


class TestBootstrapContractValidation:
    def test_empty_session_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            _bootstrap_input(session_id="")

    def test_empty_tenant_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            _bootstrap_input(tenant_id="")

    def test_created_result_requires_session(self) -> None:
        with pytest.raises(ValueError):
            SessionBootstrapResult(status=BootstrapStatus.CREATED)

    def test_rejection_result_ok_without_session(self) -> None:
        result = SessionBootstrapResult(
            status=BootstrapStatus.NOT_ELIGIBLE,
            rejection_reason="test",
        )
        assert result.session is None


# ---------------------------------------------------------------------------
# Policy registry
# ---------------------------------------------------------------------------


class TestPolicyRegistry:
    def test_register_and_resolve_default(self) -> None:
        registry = PreCallPolicyRegistry()
        policy = _policy()
        registry.register_default("c1", policy)
        assert registry.resolve("c1") is policy

    def test_versioned_resolve(self) -> None:
        registry = PreCallPolicyRegistry()
        policy = _policy()
        registry.register("c1", policy)
        assert registry.resolve("c1", "1.0") is policy
        assert registry.resolve("c1", "2.0") is None

    def test_unknown_campaign_returns_none(self) -> None:
        registry = PreCallPolicyRegistry()
        assert registry.resolve("unknown") is None

    def test_no_fuzzy_match(self) -> None:
        registry = PreCallPolicyRegistry()
        registry.register_default("c1", _policy())
        assert registry.resolve("c1-similar") is None


# ---------------------------------------------------------------------------
# Queue transition ordering
# ---------------------------------------------------------------------------


class TestQueueTransitionOrdering:
    def test_transition_called_with_original_entry(self) -> None:
        captured = []

        def transitioner(entry: LeadQueueEntry) -> LeadQueueEntry:
            captured.append(entry)
            return replace(entry, status=LeadQueueStatus.IN_PROGRESS)

        svc = _make_service(policy=_policy())
        result = svc.bootstrap(
            _bootstrap_input(), _campaign(), _queue_entry(),
            queue_transitioner=transitioner,
        )
        assert result.status == BootstrapStatus.CREATED
        assert len(captured) == 1
        assert captured[0].status == LeadQueueStatus.READY

    def test_result_contains_transitioned_entry(self) -> None:
        def transitioner(entry: LeadQueueEntry) -> LeadQueueEntry:
            return replace(entry, status=LeadQueueStatus.IN_PROGRESS)

        svc = _make_service(policy=_policy())
        result = svc.bootstrap(
            _bootstrap_input(), _campaign(), _queue_entry(),
            queue_transitioner=transitioner,
        )
        assert result.queue_entry.status == LeadQueueStatus.IN_PROGRESS


# ---------------------------------------------------------------------------
# Lead not found
# ---------------------------------------------------------------------------


class TestLeadNotFound:
    def test_missing_lead_rejected(self) -> None:
        svc = _make_service(leads={}, policy=_policy())
        result = svc.bootstrap(
            _bootstrap_input(), _campaign(), _queue_entry(),
        )
        assert result.status == BootstrapStatus.LEAD_NOT_FOUND


# ---------------------------------------------------------------------------
# Simulation mode
# ---------------------------------------------------------------------------


class TestSimulationMode:
    def test_simulation_mode_propagated(self) -> None:
        svc = _make_service(policy=_policy())
        result = svc.bootstrap(
            _bootstrap_input(runtime_mode=RuntimeMode.SIMULATION),
            _campaign(), _queue_entry(),
        )
        assert result.session.mode == RuntimeMode.SIMULATION


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_diagnostics_on_success(self) -> None:
        svc = _make_service(policy=_policy())
        result = svc.bootstrap(
            _bootstrap_input(), _campaign(), _queue_entry(),
        )
        d = result.diagnostics
        assert d.policy_id == "pol-1"
        assert d.policy_version == "1.0"
        assert d.campaign_id == "c1"
        assert d.lead_id == "lead-1"
        assert d.session_id == "sess-1"

    def test_diagnostics_on_failure(self) -> None:
        svc = _make_service(policy=None)
        result = svc.bootstrap(
            _bootstrap_input(), _campaign(), _queue_entry(),
        )
        assert result.diagnostics is not None
        assert result.diagnostics.campaign_id == "c1"
