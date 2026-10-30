from __future__ import annotations

from pathlib import Path

from app.api.app import create_app
from app.api.auth import DevelopmentHeaderPrincipalResolver

from app.application.production_turn_processor import ProductionTurnProcessor
from app.application.sales_runtime.composition import (
    ProductionSalesRuntimeFacade,
    build_production_sales_runtime,
)

from app.brain.authority.models import load_authority_policies
from app.brain.authority.validator import AuthorityPolicyValidator
from app.brain.budget.evaluator import BudgetPolicyEvaluator
from app.brain.budget.models import load_budget_policies
from app.brain.orchestrator.orchestrator import BrainOrchestrator
from app.brain.scope.models import load_scope_policies
from app.brain.scope.validator import ScopePolicyValidator

from app.config.settings import get_settings
from app.contracts.conversation_context import ConversationContext

from app.conversation.contact.resolver import ContactResolver
from app.conversation.engine import ConversationEngine
from app.conversation.guardrails.action_validator import ActionValidator
from app.conversation.guardrails.clarification import ClarificationEngine
from app.conversation.guardrails.fallbacks import FallbackEngine
from app.conversation.guardrails.priority import TrustedPriorityOutcome
from app.conversation.participant.intent_provider import ScriptedIntentTransport
from app.conversation.response_planning.planner import ResponsePlanner
from app.conversation.response_rendering.renderer import DeterministicResponseRenderer
from app.conversation.state_machine.machine import ConversationStateMachine
from app.conversation.state_machine.states import load_config

from app.core.constants import ConversationState

from app.leads.campaign.contracts import (
    CampaignConfig,
    CampaignStatus,
    LeadQueueEntry,
    LeadQueueStatus,
)

from app.leads.contracts import (
    LeadRecord,
    NormalizedPhone,
    PhoneNormalizationStatus,
    WebsiteStatus,
)

from app.llm.provider_factory import create_reasoning_provider
from app.llm.providers.reasoning_provider import ReasoningProvider

from app.persistence.sqlite.connection import SqliteConnection

from app.precall.contracts import (
    PreCallDiscoveryPolicy,
    ServiceDiscoveryRule,
)


# ---------------------------------------------------------------------------
# DEV-ONLY RUNTIME
# ---------------------------------------------------------------------------

DEV_DB = Path("dev-ui.sqlite3")


class DevTurnFactory:
    def __init__(
        self,
        facade: ProductionSalesRuntimeFacade | None,
        reasoning_provider: ReasoningProvider,
    ) -> None:
        self._facade = facade
        self._reasoning_provider = reasoning_provider

    def _build(
        self,
        session,
        *,
        initial_state,
        **kwargs,
    ):
        settings = get_settings()

        config = load_config(
            settings.conversation_config_path
        )

        machine = ConversationStateMachine(
            config,
            initial_state,
        )

        engine = ConversationEngine(
            machine,
            ActionValidator(config),
            fallback_engine=FallbackEngine(config),
            clarification_engine=ClarificationEngine(config),
            contact_resolver=ContactResolver(),
        )

        orchestrator = BrainOrchestrator(
            engine,
            BudgetPolicyEvaluator(),
            reasoning_provider=self._reasoning_provider,
            scope_validator=ScopePolicyValidator(),
            authority_validator=AuthorityPolicyValidator(),
        )

        renderer = DeterministicResponseRenderer()

        if self._facade is not None:
            renderer = self._facade.wrap_renderer(renderer)

        processor = ProductionTurnProcessor(
            orchestrator=orchestrator,
            response_planner=ResponsePlanner(),
            response_renderer=renderer,
            initial_context=ConversationContext(session.session_id),

            budget_policy=load_budget_policies(
                "app/config/budget_policy.yaml"
            )["standard"],

            scope_policy=load_scope_policies(
                "app/config/scope_policy.yaml"
            )["business_general"],

            authority_policy=load_authority_policies(
                "app/config/authority_policy.yaml"
            )["standard"],

            trusted_priority_provider=lambda _:
                TrustedPriorityOutcome.NONE,

            re_explanation_intent_provider=(
                self._facade.intent_callable
                if self._facade is not None
                else None
            ),

            **kwargs,
        )

        if self._facade is not None:
            processor._explanation_concept_provider = (
                self._facade.concept_callable_for(processor)
            )

        return processor

    def create_for_session(
        self,
        session,
        campaign,
        policy,
    ):
        return self._build(
            session,
            initial_state=ConversationState.NEW_CALL,
        )

    def create_from_snapshot(
        self,
        session,
        campaign,
        policy,
        snapshot,
    ):
        return self._build(
            session,
            initial_state=snapshot.conversation_state,
        )


# ---------------------------------------------------------------------------
# DEV DATA
# ---------------------------------------------------------------------------

def make_phone() -> NormalizedPhone:
    return NormalizedPhone(
        raw="+15551234567",
        normalized="+15551234567",
        status=PhoneNormalizationStatus.VALID_E164,
    )


def make_lead() -> LeadRecord:
    return LeadRecord(
        lead_id="lead-1",
        tenant_id="t1",
        campaign_id="c1",
        business_name="ABC Services",
        phone=make_phone(),

        category=None,
        city=None,
        state=None,
        country=None,
        address=None,
        email=None,
        contact_name=None,

        website=None,
        website_status=WebsiteStatus.UNKNOWN,
    )


def make_queue() -> LeadQueueEntry:
    return LeadQueueEntry(
        lead_id="lead-1",
        tenant_id="t1",
        campaign_id="c1",
        queue_position=0,
        status=LeadQueueStatus.READY,
    )


def make_campaign() -> CampaignConfig:
    return CampaignConfig(
        campaign_id="c1",
        tenant_id="t1",
        name="Local UI Demo",
        status=CampaignStatus.ACTIVE,
        authorized_service_ids=(
            "svc-automation",
        ),
    )


def make_policy() -> PreCallDiscoveryPolicy:
    return PreCallDiscoveryPolicy(
        policy_id="pol-1",
        version="1.0",
        service_rules=(
            ServiceDiscoveryRule(
                service_id="svc-automation",
                required_fact_keys=(
                    "website.presence",
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# APP BOOTSTRAP
# ---------------------------------------------------------------------------

settings = get_settings()


# ---------------------------------------------------------------------------
# REAL GEMINI REASONING PROVIDER
# ---------------------------------------------------------------------------

reasoning_provider = create_reasoning_provider(
    settings
)


# ---------------------------------------------------------------------------
# DEV DATABASE
#
# Intentionally reset on local import/reload so the single-use demo queue
# always starts READY.
# ---------------------------------------------------------------------------

if DEV_DB.exists():
    DEV_DB.unlink()

connection = SqliteConnection(
    str(DEV_DB)
)


# ---------------------------------------------------------------------------
# SEMANTIC INTENT TRANSPORT
#
# TEMPORARY:
# Brain reasoning is now real Gemini.
#
# ReExplanation / Catalog intent classification remains scripted until
# GeminiIntentClassifierTransport is introduced in the next bounded step.
# ---------------------------------------------------------------------------

intent_transport = ScriptedIntentTransport(
    (
        '{"intent":"none"}',
        '{"intent":"catalog_request"}',
        '{"intent":"direct_service_question"}',
        '{"intent":"none"}',
        '{"intent":"none"}',
        '{"intent":"none"}',
        '{"intent":"none"}',
        '{"intent":"none"}',
    )
)


# ---------------------------------------------------------------------------
# CANONICAL SALES RUNTIME
# ---------------------------------------------------------------------------

factory = DevTurnFactory(
    facade=None,
    reasoning_provider=reasoning_provider,
)

facade = build_production_sales_runtime(
    connection=connection,
    processor_factory=factory,
    intent_transport=intent_transport,
)

# Factory needs facade only after canonical runtime composition exists.
factory._facade = facade


# ---------------------------------------------------------------------------
# DEV SEED DATA
# ---------------------------------------------------------------------------

facade.runtime.leads.save(
    make_lead()
)

facade.runtime.queue.save(
    make_queue()
)

facade.runtime.policies.register_default(
    "c1",
    make_policy(),
)

campaigns = {
    "c1": make_campaign(),
}


# ---------------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------------

app = create_app(
    runtime=facade.runtime,
    campaign_lookup=campaigns.get,

    # DEV ONLY.
    # Production mode correctly refuses this resolver.
    principal_resolver=DevelopmentHeaderPrincipalResolver(),

    app_env="development",
)