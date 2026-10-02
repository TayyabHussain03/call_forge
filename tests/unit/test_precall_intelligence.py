"""Comprehensive tests for Pre-Call Intelligence Engine (PCIE)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from app.leads.campaign.contracts import CampaignConfig, CampaignStatus
from app.leads.contracts import (
    EnrichedField,
    EnrichmentFieldSource,
    EnrichmentProvenance,
    FieldKnowledgeState,
    NormalizedPhone,
    PhoneNormalizationStatus,
    PreCallBusinessContext,
    WebsiteStatus,
)
from app.precall.contracts import (
    AssumptionGuardrail,
    InitialPosture,
    OpeningObjective,
    PreCallConversationPlan,
    PreCallDiscoveryPolicy,
    PreCallFact,
    PreCallPlanStatus,
    RiskFlag,
    ServiceDiscoveryRule,
    ServicePrerequisiteGap,
    VerificationPriority,
    VerificationTarget,
)
from app.precall.engine import (
    FACT_BUSINESS_CATEGORY,
    FACT_WEBSITE_PRESENCE,
    PreCallIntelligenceEngine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ENGINE = PreCallIntelligenceEngine()


def _phone(e164: str = "+12125551234") -> NormalizedPhone:
    return NormalizedPhone(
        raw=e164, normalized=e164,
        status=PhoneNormalizationStatus.VALID_E164,
    )


def _unknown_field() -> EnrichedField:
    return EnrichedField(
        value=None,
        state=FieldKnowledgeState.UNKNOWN,
        source=EnrichmentFieldSource.UNKNOWN,
    )


def _known_field(value: str, source: EnrichmentFieldSource = EnrichmentFieldSource.USER_PROVIDED) -> EnrichedField:
    return EnrichedField(
        value=value,
        state=FieldKnowledgeState.KNOWN,
        source=source,
    )


def _conflicting_field(value: str) -> EnrichedField:
    return EnrichedField(
        value=value,
        state=FieldKnowledgeState.CONFLICTING,
        source=EnrichmentFieldSource.ENRICHMENT_CANDIDATE,
    )


def _context(
    website_status: WebsiteStatus = WebsiteStatus.UNKNOWN,
    category: EnrichedField | None = None,
    contact_name: EnrichedField | None = None,
    email: EnrichedField | None = None,
    tenant_id: str = "tenant_1",
    campaign_id: str = "campaign_1",
    lead_id: str = "lead_1",
    campaign_context: str | None = None,
) -> PreCallBusinessContext:
    return PreCallBusinessContext(
        lead_id=lead_id,
        tenant_id=tenant_id,
        campaign_id=campaign_id,
        business_name="Acme Pizza",
        phone=_phone(),
        website_status=website_status,
        website=_unknown_field() if website_status == WebsiteStatus.UNKNOWN else _known_field("https://acme.com"),
        category=category or _unknown_field(),
        city=_unknown_field(),
        state=_unknown_field(),
        country=_unknown_field(),
        address=_unknown_field(),
        contact_name=contact_name or _unknown_field(),
        email=email or _unknown_field(),
        lead_source=None,
        campaign_context=campaign_context,
    )


def _campaign(
    authorized: tuple[str, ...] = ("website_development",),
    tenant_id: str = "tenant_1",
    campaign_id: str = "campaign_1",
) -> CampaignConfig:
    return CampaignConfig(
        campaign_id=campaign_id,
        tenant_id=tenant_id,
        name="Test Campaign",
        status=CampaignStatus.ACTIVE,
        authorized_service_ids=authorized,
    )


def _policy(
    service_rules: tuple[ServiceDiscoveryRule, ...] = (),
    version: str = "1.0.0",
    global_required: tuple[str, ...] = (),
    global_optional: tuple[str, ...] = (),
    priority_overrides: tuple[tuple[str, VerificationPriority], ...] = (),
) -> PreCallDiscoveryPolicy:
    return PreCallDiscoveryPolicy(
        policy_id="test_policy",
        version=version,
        service_rules=service_rules,
        global_required_facts=global_required,
        global_optional_facts=global_optional,
        priority_overrides=priority_overrides,
    )


_WEBSITE_RULE = ServiceDiscoveryRule(
    service_id="website_development",
    required_fact_keys=(FACT_WEBSITE_PRESENCE,),
    optional_fact_keys=("digital_presence.current", "customer_acquisition.primary_channel"),
    knowledge_topic_hints=("website_benefits",),
)

_AUTOMATION_RULE = ServiceDiscoveryRule(
    service_id="ai_automation",
    required_fact_keys=("workflow.current_process",),
    optional_fact_keys=("workflow.manual_steps", "software.current_tools"),
    knowledge_topic_hints=("automation_benefits",),
)


# ===========================================================================
# Contract validation
# ===========================================================================


class TestContractValidation:
    def test_fact_unknown_cannot_have_value(self):
        with pytest.raises(ValueError, match="UNKNOWN"):
            PreCallFact(
                key=FACT_WEBSITE_PRESENCE,
                state=FieldKnowledgeState.UNKNOWN,
                value="x",
            )

    def test_fact_known_requires_value(self):
        with pytest.raises(ValueError, match="KNOWN"):
            PreCallFact(
                key=FACT_WEBSITE_PRESENCE,
                state=FieldKnowledgeState.KNOWN,
            )

    def test_fact_conflicting_requires_detail(self):
        with pytest.raises(ValueError, match="CONFLICTING"):
            PreCallFact(
                key="x.y",
                state=FieldKnowledgeState.CONFLICTING,
                value="v",
            )

    def test_policy_requires_version(self):
        with pytest.raises(ValueError, match="version"):
            PreCallDiscoveryPolicy(policy_id="p", version="")

    def test_policy_duplicate_service_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            _policy(service_rules=(_WEBSITE_RULE, _WEBSITE_RULE))

    def test_rule_fact_cannot_be_both_required_and_optional(self):
        with pytest.raises(ValueError, match="both required"):
            ServiceDiscoveryRule(
                service_id="x",
                required_fact_keys=("a",),
                optional_fact_keys=("a",),
            )

    def test_plan_bucket_state_mismatch_rejected(self):
        with pytest.raises(ValueError, match="wrong state"):
            PreCallConversationPlan(
                lead_id="l", campaign_id="c", tenant_id="t",
                policy_id="p", policy_version="1",
                known_facts=(PreCallFact("x", FieldKnowledgeState.UNKNOWN),),
            )

    def test_plan_primary_must_be_in_targets(self):
        target = VerificationTarget("x", VerificationPriority.HIGH, "r", 0)
        other = VerificationTarget("y", VerificationPriority.LOW, "r", 1)
        with pytest.raises(ValueError, match="primary target"):
            PreCallConversationPlan(
                lead_id="l", campaign_id="c", tenant_id="t",
                policy_id="p", policy_version="1",
                primary_initial_discovery_target=target,
                verification_targets=(other,),
            )

    def test_plan_frozen(self):
        p = PreCallConversationPlan(
            lead_id="l", campaign_id="c", tenant_id="t",
            policy_id="p", policy_version="1",
        )
        with pytest.raises(FrozenInstanceError):
            p.plan_status = PreCallPlanStatus.DEGRADED


# ===========================================================================
# Minimal lead
# ===========================================================================


class TestMinimalLead:
    def test_minimal_lead_still_produces_plan(self):
        plan = ENGINE.plan(
            _context(),
            _campaign(),
            _policy(service_rules=(_WEBSITE_RULE,)),
        )
        assert plan.plan_status == PreCallPlanStatus.READY
        assert plan.lead_id == "lead_1"
        assert plan.initial_posture == InitialPosture.LOW_CONTEXT

    def test_no_authorized_services_degraded(self):
        plan = ENGINE.plan(
            _context(),
            _campaign(authorized=()),
            _policy(),
        )
        assert plan.plan_status == PreCallPlanStatus.DEGRADED
        assert RiskFlag.NO_CAMPAIGN_SERVICE_AUTHORIZED in plan.risk_flags


# ===========================================================================
# Website behavior — the core "unknown ≠ negative" test suite
# ===========================================================================


class TestWebsiteBehavior:
    def test_website_unknown_becomes_verification_target(self):
        plan = ENGINE.plan(
            _context(website_status=WebsiteStatus.UNKNOWN),
            _campaign(),
            _policy(service_rules=(_WEBSITE_RULE,)),
        )
        website_fact = next(f for f in plan.unknown_facts if f.key == FACT_WEBSITE_PRESENCE)
        assert website_fact.state == FieldKnowledgeState.UNKNOWN
        assert website_fact.value is None
        target_keys = [t.fact_key for t in plan.verification_targets]
        assert FACT_WEBSITE_PRESENCE in target_keys

    def test_website_unknown_triggers_no_website_guardrail(self):
        plan = ENGINE.plan(
            _context(website_status=WebsiteStatus.UNKNOWN),
            _campaign(),
            _policy(service_rules=(_WEBSITE_RULE,)),
        )
        assert AssumptionGuardrail.DO_NOT_ASSUME_NO_WEBSITE in plan.assumption_guardrails

    def test_website_verified_present_is_known_absent_from_targets(self):
        plan = ENGINE.plan(
            _context(website_status=WebsiteStatus.VERIFIED_PRESENT),
            _campaign(),
            _policy(service_rules=(_WEBSITE_RULE,)),
        )
        website_fact = next(f for f in plan.known_facts if f.key == FACT_WEBSITE_PRESENCE)
        assert website_fact.state == FieldKnowledgeState.KNOWN
        assert website_fact.value == "present"
        target_keys = [t.fact_key for t in plan.verification_targets]
        assert FACT_WEBSITE_PRESENCE not in target_keys

    def test_website_verified_absent_is_known_not_a_pitch(self):
        plan = ENGINE.plan(
            _context(website_status=WebsiteStatus.VERIFIED_ABSENT),
            _campaign(),
            _policy(service_rules=(_WEBSITE_RULE,)),
        )
        website_fact = next(f for f in plan.known_facts if f.key == FACT_WEBSITE_PRESENCE)
        assert website_fact.value == "absent"
        # PCIE does NOT emit any SELL_WEBSITE signal
        import app.precall.engine as m
        from pathlib import Path
        source = Path(m.__file__).read_text()
        assert "SELL_WEBSITE" not in source

    def test_website_unverified_candidate(self):
        plan = ENGINE.plan(
            _context(website_status=WebsiteStatus.UNVERIFIED_CANDIDATE),
            _campaign(),
            _policy(service_rules=(_WEBSITE_RULE,)),
        )
        website_fact = next(
            f for f in plan.unverified_facts if f.key == FACT_WEBSITE_PRESENCE
        )
        assert website_fact.state == FieldKnowledgeState.UNVERIFIED

    def test_website_never_assumed_no_from_unknown(self):
        plan = ENGINE.plan(
            _context(website_status=WebsiteStatus.UNKNOWN),
            _campaign(),
            _policy(service_rules=(_WEBSITE_RULE,)),
        )
        for fact in plan.known_facts:
            assert not (fact.key == FACT_WEBSITE_PRESENCE and fact.value == "absent")


# ===========================================================================
# Conflicting information
# ===========================================================================


class TestConflicting:
    def test_conflicting_website_is_critical_target(self):
        plan = ENGINE.plan(
            _context(
                website_status=WebsiteStatus.VERIFIED_PRESENT,
                category=_conflicting_field("Restaurant"),
            ),
            _campaign(),
            _policy(service_rules=(_WEBSITE_RULE,)),
        )
        critical_targets = [
            t for t in plan.verification_targets
            if t.priority == VerificationPriority.CRITICAL
        ]
        assert len(critical_targets) >= 1
        assert any(t.fact_key == FACT_BUSINESS_CATEGORY for t in critical_targets)
        assert plan.initial_posture == InitialPosture.CONFLICTED_CONTEXT
        assert RiskFlag.CONFLICTING_ENRICHMENT in plan.risk_flags


# ===========================================================================
# Service configurations — website and automation
# ===========================================================================


class TestServiceConfiguration:
    def test_website_campaign_configures_website_prerequisites(self):
        plan = ENGINE.plan(
            _context(website_status=WebsiteStatus.UNKNOWN),
            _campaign(authorized=("website_development",)),
            _policy(service_rules=(_WEBSITE_RULE, _AUTOMATION_RULE)),
        )
        assert "website_benefits" in plan.knowledge_topics_likely_needed
        assert "automation_benefits" not in plan.knowledge_topics_likely_needed
        website_gap = next(
            g for g in plan.service_prerequisite_gaps
            if g.service_id == "website_development"
        )
        assert FACT_WEBSITE_PRESENCE in website_gap.missing_fact_keys

    def test_automation_campaign_configures_workflow_prerequisites(self):
        plan = ENGINE.plan(
            _context(),
            _campaign(authorized=("ai_automation",)),
            _policy(service_rules=(_WEBSITE_RULE, _AUTOMATION_RULE)),
        )
        assert "automation_benefits" in plan.knowledge_topics_likely_needed
        gap = next(g for g in plan.service_prerequisite_gaps if g.service_id == "ai_automation")
        assert "workflow.current_process" in gap.missing_fact_keys

    def test_arbitrary_new_service_works_from_config(self):
        novel_rule = ServiceDiscoveryRule(
            service_id="chatbot_integration",
            required_fact_keys=("customer_support.channels",),
            optional_fact_keys=("business.primary_goal",),
            knowledge_topic_hints=("chatbot_benefits",),
        )
        plan = ENGINE.plan(
            _context(),
            _campaign(authorized=("chatbot_integration",)),
            _policy(service_rules=(novel_rule,)),
        )
        assert "chatbot_benefits" in plan.knowledge_topics_likely_needed
        assert any(
            g.service_id == "chatbot_integration"
            and "customer_support.channels" in g.missing_fact_keys
            for g in plan.service_prerequisite_gaps
        )

    def test_no_python_branches_for_specific_business_types(self):
        from pathlib import Path
        import app.precall.engine as m
        source = Path(m.__file__).read_text().lower()
        assert "restaurant" not in source
        assert "pizza" not in source
        assert "retail" not in source
        assert "ecommerce" not in source
        assert 'if "website"' not in source
        assert 'if website_campaign' not in source


# ===========================================================================
# Prerequisites — missing and satisfied
# ===========================================================================


class TestPrerequisites:
    def test_missing_prerequisite_appears_as_gap(self):
        plan = ENGINE.plan(
            _context(website_status=WebsiteStatus.UNKNOWN),
            _campaign(authorized=("website_development",)),
            _policy(service_rules=(_WEBSITE_RULE,)),
        )
        assert len(plan.service_prerequisite_gaps) == 1
        assert plan.service_prerequisite_gaps[0].service_id == "website_development"

    def test_satisfied_prerequisite_no_gap(self):
        plan = ENGINE.plan(
            _context(website_status=WebsiteStatus.VERIFIED_PRESENT),
            _campaign(authorized=("website_development",)),
            _policy(service_rules=(_WEBSITE_RULE,)),
        )
        assert len(plan.service_prerequisite_gaps) == 0

    def test_unverified_counts_as_unresolved(self):
        plan = ENGINE.plan(
            _context(website_status=WebsiteStatus.UNVERIFIED_CANDIDATE),
            _campaign(authorized=("website_development",)),
            _policy(service_rules=(_WEBSITE_RULE,)),
        )
        gap_keys = [g.missing_fact_keys for g in plan.service_prerequisite_gaps]
        assert any(FACT_WEBSITE_PRESENCE in keys for keys in gap_keys)


# ===========================================================================
# One primary initial discovery target
# ===========================================================================


class TestPrimaryDiscoveryTarget:
    def test_exactly_one_primary(self):
        plan = ENGINE.plan(
            _context(website_status=WebsiteStatus.UNKNOWN),
            _campaign(),
            _policy(
                service_rules=(_WEBSITE_RULE, _AUTOMATION_RULE),
                global_required=("business.primary_goal",),
            ),
        )
        assert plan.primary_initial_discovery_target is not None
        assert isinstance(plan.primary_initial_discovery_target, VerificationTarget)

    def test_primary_is_in_targets(self):
        plan = ENGINE.plan(
            _context(website_status=WebsiteStatus.UNKNOWN),
            _campaign(),
            _policy(service_rules=(_WEBSITE_RULE,)),
        )
        assert plan.primary_initial_discovery_target in plan.verification_targets

    def test_no_primary_when_no_targets(self):
        plan = ENGINE.plan(
            _context(website_status=WebsiteStatus.VERIFIED_PRESENT),
            _campaign(authorized=()),
            _policy(),
        )
        assert plan.primary_initial_discovery_target is None

    def test_primary_is_highest_priority(self):
        plan = ENGINE.plan(
            _context(
                website_status=WebsiteStatus.VERIFIED_PRESENT,
                category=_conflicting_field("Pizzeria"),
            ),
            _campaign(),
            _policy(service_rules=(_WEBSITE_RULE,)),
        )
        assert plan.primary_initial_discovery_target.priority == VerificationPriority.CRITICAL


# ===========================================================================
# Deterministic ordering & tie-breaking
# ===========================================================================


class TestDeterministicOrdering:
    def test_priority_ranking(self):
        plan = ENGINE.plan(
            _context(
                website_status=WebsiteStatus.UNKNOWN,
                category=_conflicting_field("Pizza"),
            ),
            _campaign(),
            _policy(service_rules=(_WEBSITE_RULE,)),
        )
        priorities = [t.priority for t in plan.verification_targets]
        ranks = [["critical", "high", "normal", "low"].index(p.value) for p in priorities]
        assert ranks == sorted(ranks)

    def test_tie_breaking_is_deterministic(self):
        p1 = ENGINE.plan(
            _context(website_status=WebsiteStatus.UNKNOWN),
            _campaign(),
            _policy(service_rules=(_WEBSITE_RULE,), global_optional=("extra.a", "extra.b")),
        )
        p2 = ENGINE.plan(
            _context(website_status=WebsiteStatus.UNKNOWN),
            _campaign(),
            _policy(service_rules=(_WEBSITE_RULE,), global_optional=("extra.a", "extra.b")),
        )
        assert [t.fact_key for t in p1.verification_targets] == [t.fact_key for t in p2.verification_targets]


# ===========================================================================
# Assumption safety — no restaurant/manual-workflow inference
# ===========================================================================


class TestAssumptionSafety:
    def test_restaurant_category_does_not_imply_manual_workflow(self):
        plan = ENGINE.plan(
            _context(
                website_status=WebsiteStatus.UNKNOWN,
                category=_known_field("Restaurant"),
            ),
            _campaign(authorized=("ai_automation",)),
            _policy(service_rules=(_AUTOMATION_RULE,)),
        )
        workflow_known = [
            f for f in plan.known_facts if f.key == "workflow.current_process"
        ]
        assert len(workflow_known) == 0
        workflow_gap = next(
            (g for g in plan.service_prerequisite_gaps if g.service_id == "ai_automation"),
            None,
        )
        assert workflow_gap is not None
        assert "workflow.current_process" in workflow_gap.missing_fact_keys

    def test_no_budget_inference(self):
        plan = ENGINE.plan(
            _context(),
            _campaign(),
            _policy(service_rules=(_WEBSITE_RULE,)),
        )
        assert AssumptionGuardrail.DO_NOT_ASSUME_BUDGET in plan.assumption_guardrails
        budget_facts = [f for f in plan.known_facts if "budget" in f.key]
        assert len(budget_facts) == 0

    def test_no_decision_maker_inference(self):
        plan = ENGINE.plan(
            _context(),
            _campaign(),
            _policy(service_rules=(_WEBSITE_RULE,)),
        )
        assert AssumptionGuardrail.DO_NOT_ASSUME_DECISION_MAKER in plan.assumption_guardrails

    def test_no_language_inference(self):
        plan = ENGINE.plan(
            _context(),
            _campaign(),
            _policy(service_rules=(_WEBSITE_RULE,)),
        )
        assert AssumptionGuardrail.DO_NOT_ASSUME_LANGUAGE in plan.assumption_guardrails


# ===========================================================================
# No service recommendation, no pitch generation
# ===========================================================================


class TestNoServiceRecommendation:
    def test_no_recommendation_field(self):
        plan = ENGINE.plan(
            _context(website_status=WebsiteStatus.VERIFIED_ABSENT),
            _campaign(),
            _policy(service_rules=(_WEBSITE_RULE,)),
        )
        assert not hasattr(plan, "recommended_service")
        assert not hasattr(plan, "pitch")
        assert not hasattr(plan, "sell_service")

    def test_no_pitch_text_in_output(self):
        plan = ENGINE.plan(
            _context(website_status=WebsiteStatus.VERIFIED_ABSENT),
            _campaign(),
            _policy(service_rules=(_WEBSITE_RULE,)),
        )
        # No long prose in the plan
        for target in plan.verification_targets:
            assert len(target.reason) < 200


# ===========================================================================
# No LLM, no retrieval, no mutation
# ===========================================================================


class TestNoLLMNoRetrieval:
    def test_no_llm_imports(self):
        from pathlib import Path
        import app.precall.engine as m
        source = Path(m.__file__).read_text()
        assert "openai" not in source.lower()
        assert "gemini" not in source.lower()
        assert "anthropic" not in source.lower()

    def test_no_retrieval_calls(self):
        from pathlib import Path
        import app.precall.engine as m
        source = Path(m.__file__).read_text()
        assert "HybridRetrievalEngine" not in source
        assert "KnowledgeRetrievalPlanner" not in source
        assert ".retrieve(" not in source

    def test_no_fsm_authority_mutation(self):
        from pathlib import Path
        import app.precall.engine as m
        source = Path(m.__file__).read_text()
        assert "BrainOrchestrator" not in source
        assert "StateMachine" not in source
        assert "ActionValidator" not in source
        assert "AuthorityPolicy" not in source


# ===========================================================================
# Determinism / replay
# ===========================================================================


class TestDeterminism:
    def test_same_inputs_same_plan(self):
        ctx = _context(website_status=WebsiteStatus.UNKNOWN)
        camp = _campaign()
        pol = _policy(service_rules=(_WEBSITE_RULE, _AUTOMATION_RULE))
        p1 = ENGINE.plan(ctx, camp, pol)
        p2 = ENGINE.plan(ctx, camp, pol)
        assert p1 == p2

    def test_no_random_in_engine(self):
        from pathlib import Path
        import app.precall.engine as m
        source = Path(m.__file__).read_text()
        assert "uuid" not in source
        assert "random" not in source
        assert "datetime.now" not in source
        assert "time.time" not in source


# ===========================================================================
# Policy versioning
# ===========================================================================


class TestPolicyVersioning:
    def test_plan_carries_policy_identity(self):
        plan = ENGINE.plan(
            _context(),
            _campaign(),
            _policy(service_rules=(_WEBSITE_RULE,), version="2.5.1"),
        )
        assert plan.policy_id == "test_policy"
        assert plan.policy_version == "2.5.1"

    def test_different_versions_tracked_separately(self):
        ctx = _context(website_status=WebsiteStatus.UNKNOWN)
        camp = _campaign()
        p1 = ENGINE.plan(ctx, camp, _policy(service_rules=(_WEBSITE_RULE,), version="1.0.0"))
        p2 = ENGINE.plan(ctx, camp, _policy(service_rules=(_WEBSITE_RULE,), version="2.0.0"))
        assert p1.policy_version != p2.policy_version


# ===========================================================================
# Conversation truth supersedes
# ===========================================================================


class TestConversationTruthSupersedes:
    def test_plan_is_read_only_frozen(self):
        plan = ENGINE.plan(
            _context(),
            _campaign(),
            _policy(service_rules=(_WEBSITE_RULE,)),
        )
        with pytest.raises(FrozenInstanceError):
            plan.verification_targets = ()

    def test_plan_does_not_mutate_inputs(self):
        ctx = _context(website_status=WebsiteStatus.UNKNOWN)
        camp = _campaign()
        pol = _policy(service_rules=(_WEBSITE_RULE,))

        ENGINE.plan(ctx, camp, pol)

        assert ctx.website_status == WebsiteStatus.UNKNOWN
        assert camp.authorized_service_ids == ("website_development",)
        assert pol.version == "1.0.0"


# ===========================================================================
# Tenant / campaign isolation
# ===========================================================================


class TestIsolation:
    def test_plan_carries_tenant(self):
        plan = ENGINE.plan(
            _context(tenant_id="t_abc", campaign_id="c_xyz"),
            _campaign(tenant_id="t_abc", campaign_id="c_xyz"),
            _policy(),
        )
        assert plan.tenant_id == "t_abc"
        assert plan.campaign_id == "c_xyz"

    def test_tenant_mismatch_rejected(self):
        with pytest.raises(ValueError, match="tenant"):
            ENGINE.plan(
                _context(tenant_id="t1"),
                _campaign(tenant_id="t2", campaign_id="campaign_1"),
                _policy(),
            )

    def test_campaign_mismatch_rejected(self):
        with pytest.raises(ValueError, match="campaign"):
            ENGINE.plan(
                _context(campaign_id="c1"),
                _campaign(campaign_id="c2"),
                _policy(),
            )


# ===========================================================================
# Opening objective
# ===========================================================================


class TestOpeningObjective:
    def test_resume_existing_context_when_campaign_context_present(self):
        plan = ENGINE.plan(
            _context(campaign_context="Follow-up from last week"),
            _campaign(),
            _policy(service_rules=(_WEBSITE_RULE,)),
        )
        assert plan.opening_objective in {
            OpeningObjective.RESUME_EXISTING_CONTEXT,
            OpeningObjective.VERIFY_CRITICAL_FACT,
        }

    def test_establish_contact_default_minimal(self):
        plan = ENGINE.plan(
            _context(),
            _campaign(),
            _policy(),
        )
        assert plan.opening_objective in {
            OpeningObjective.ESTABLISH_CONTACT,
            OpeningObjective.EARN_PERMISSION,
        }


# ===========================================================================
# Chat runtime integration
# ===========================================================================


class TestChatRuntimeIntegration:
    def test_chat_session_accepts_pre_call_plan(self):
        from app.runtime.chat.contracts import ChatSession
        plan = ENGINE.plan(_context(), _campaign(), _policy(service_rules=(_WEBSITE_RULE,)))
        session = ChatSession(
            session_id="s1",
            tenant_id="tenant_1",
            campaign_id="campaign_1",
            lead_id="lead_1",
            pre_call_plan=plan,
        )
        assert session.pre_call_plan is plan

    def test_chat_session_without_plan_still_valid(self):
        from app.runtime.chat.contracts import ChatSession
        session = ChatSession(
            session_id="s1",
            tenant_id="tenant_1",
            campaign_id="campaign_1",
            lead_id="lead_1",
        )
        assert session.pre_call_plan is None
