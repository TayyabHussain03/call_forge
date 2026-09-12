"""Slice 9 in-memory composition through runtime, domain, and response layers."""

from __future__ import annotations

from app.application.production_turn_processor import ProductionTurnProcessor
from app.brain.authority.models import load_authority_policies
from app.brain.authority.validator import AuthorityPolicyValidator
from app.brain.budget.evaluator import BudgetPolicyEvaluator
from app.brain.budget.models import load_budget_policies
from app.brain.contracts import BrainInput, BrainProposal, CommercialRequest
from app.brain.orchestrator.orchestrator import BrainOrchestrator
from app.brain.scope.models import load_scope_policies
from app.brain.scope.validator import ScopePolicyValidator
from app.catalog.claim_validator import ClaimValidator
from app.catalog.loader import load_catalog
from app.catalog.scoped_catalog import ScopedCatalog
from app.catalog.selection.selection_service import ServiceSelectionService
from app.catalog.selection.selector import MockServiceSelector
from app.catalog.selection.selector_validator import SelectorValidator
from app.config.settings import get_settings
from app.contracts.contact_understanding import ContactIntent
from app.contracts.conversation_context import ConversationContext
from app.conversation.contact.resolver import ContactResolver
from app.conversation.engine import ConversationEngine
from app.conversation.guardrails.action_validator import ActionValidator
from app.conversation.guardrails.clarification import ClarificationEngine
from app.conversation.guardrails.fallbacks import FallbackEngine
from app.conversation.guardrails.priority import TrustedPriorityOutcome
from app.conversation.response_planning.contracts import (
    AddresseeStatus,
    AuthoritativeResultKind,
    InterruptionCategory,
)
from app.conversation.response_planning.planner import ResponsePlanner
from app.conversation.response_rendering.renderer import (
    DeterministicResponseRenderer,
    ResponseRenderer,
)
from app.conversation.strategy.contracts import ConversationMode, SalesStage
from app.conversation.state_machine.machine import ConversationStateMachine
from app.conversation.state_machine.states import load_config
from app.core.constants import (
    AgentAction,
    CommercialRequestKind,
    ConversationState,
    Intent,
    TopicCategory,
)
from app.llm.providers.contact_understanding_provider import (
    MockContactUnderstandingProvider,
    scripted,
)
from app.llm.providers.reasoning_provider import MockReasoningProvider
from app.runtime.contracts import (
    CoordinationOutcome,
    DeliveryAction,
    RuntimeEvent,
    RuntimeEventType,
)
from app.runtime.turn_coordinator import TurnCoordinator
from app.services.service_offering_service import ServiceOfferingService


class CapturingReasoningProvider(MockReasoningProvider):
    def __init__(self, proposal: BrainProposal) -> None:
        super().__init__(default=proposal)
        self.last_input: BrainInput | None = None

    def reason(self, brain_input: BrainInput) -> BrainProposal:
        self.last_input = brain_input
        return super().reason(brain_input)


def _proposal(
    action: AgentAction = AgentAction.GREET,
    topic: TopicCategory = TopicCategory.QUALIFICATION,
    commercial: CommercialRequest | None = None,
) -> BrainProposal:
    return BrainProposal(
        detected_intent=Intent.INTERESTED,
        proposed_action=action,
        topic_category=topic,
        commercial_request=commercial,
    )


def _processor(
    provider: MockReasoningProvider,
    *,
    initial_state: ConversationState = ConversationState.NEW_CALL,
    context: ConversationContext | None = None,
    priority=TrustedPriorityOutcome.NONE,  # type: ignore[no-untyped-def]
    contact_provider=None,  # type: ignore[no-untyped-def]
    renderer: ResponseRenderer | None = None,
    with_offering: bool = False,
    scope_validator: ScopePolicyValidator | None = None,
) -> ProductionTurnProcessor:
    config = load_config(get_settings().conversation_config_path)
    machine = ConversationStateMachine(config, initial_state)
    offering = None
    if with_offering:
        scoped = ScopedCatalog(load_catalog("app/config/service_config.yaml"), "campaign_a")
        offering = ServiceOfferingService(
            scoped,
            ServiceSelectionService(
                MockServiceSelector(), SelectorValidator(ClaimValidator(scoped))
            ),
            max_offers=3,
        )
    engine = ConversationEngine(
        machine,
        ActionValidator(config),
        fallback_engine=FallbackEngine(config),
        clarification_engine=ClarificationEngine(config),
        contact_resolver=ContactResolver(),
        offering_service=offering,
    )
    orchestrator = BrainOrchestrator(
        engine,
        BudgetPolicyEvaluator(),
        reasoning_provider=provider,
        scope_validator=scope_validator or ScopePolicyValidator(),
        authority_validator=AuthorityPolicyValidator(),
    )
    return ProductionTurnProcessor(
        orchestrator=orchestrator,
        response_planner=ResponsePlanner(),
        response_renderer=renderer or DeterministicResponseRenderer(),
        initial_context=context or ConversationContext("call"),
        budget_policy=load_budget_policies("app/config/budget_policy.yaml")["standard"],
        scope_policy=load_scope_policies("app/config/scope_policy.yaml")["business_general"],
        authority_policy=load_authority_policies("app/config/authority_policy.yaml")["standard"],
        trusted_priority_provider=lambda _: priority,
        contact_understanding_provider=contact_provider,
    )


def _final(
    sequence: int = 1,
    turn_id: str = "t1",
    **metadata,  # type: ignore[no-untyped-def]
) -> RuntimeEvent:
    return RuntimeEvent(
        f"e{sequence}",
        "call",
        sequence,
        RuntimeEventType.USER_UTTERANCE_FINAL,
        turn_id,
        "final prospect utterance",
        **metadata,
    )


def test_normal_runtime_turn_composes_full_existing_pipeline_once() -> None:
    provider = MockReasoningProvider(default=_proposal())
    coordinator = TurnCoordinator("call", _processor(provider))
    result = coordinator.handle(_final())
    assert result.outcome == CoordinationOutcome.TURN_PROCESSED
    assert result.instructions[0].action == DeliveryAction.SPEAK
    assert provider.call_count == 1
    assert coordinator.state.active_delivery is not None
    assert coordinator.state.active_delivery.rendered_response.text


def test_runtime_supplies_typed_strategy_guidance_to_brain() -> None:
    provider = CapturingReasoningProvider(_proposal())
    TurnCoordinator("call", _processor(provider)).handle(
        _final(conversation_category=InterruptionCategory.QUESTION)
    )

    assert provider.last_input is not None
    strategy = provider.last_input.conversation_strategy
    assert strategy is not None
    assert strategy.sales_stage == SalesStage.OPENING
    assert strategy.conversation_mode == ConversationMode.QUESTION_DETOUR


def test_trusted_dnc_and_not_interested_are_distinct_and_skip_brain() -> None:
    for priority, expected_fact in (
        (TrustedPriorityOutcome.DNC, "do-not-call"),
        (TrustedPriorityOutcome.NOT_INTERESTED, "not continue"),
    ):
        provider = MockReasoningProvider(default=_proposal())
        processor = _processor(provider, priority=priority)
        result = TurnCoordinator("call", processor).handle(_final())
        assert provider.call_count == 0
        assert result.state.active_delivery is not None
        assert result.state.active_delivery.rendered_response.text
        assert processor.context.dnc_pending is (priority == TrustedPriorityOutcome.DNC)
        assert expected_fact in result.state.active_delivery.unfinished_point_summary


def test_provider_failure_out_of_scope_and_escalation_map_to_safe_responses() -> None:
    cases = (
        (MockReasoningProvider(should_fail=True), AuthoritativeResultKind.FALLBACK),
        (
            MockReasoningProvider(default=_proposal(topic=TopicCategory.OFF_TOPIC)),
            AuthoritativeResultKind.REDIRECT,
        ),
        (
            MockReasoningProvider(
                default=_proposal(
                    commercial=CommercialRequest(CommercialRequestKind.CUSTOM_PRICING)
                )
            ),
            AuthoritativeResultKind.ESCALATE,
        ),
    )
    for provider, _expected in cases:
        result = TurnCoordinator("call", _processor(provider)).handle(_final())
        assert result.state.active_delivery is not None
        assert result.state.active_delivery.rendered_response.text
        assert result.state.active_delivery.response_plan.communicative_goal.value
        assert provider.call_count == 1
        assert result.state.active_delivery is not None
        assert result.processed_turn is not None
        assert result.turn_output is not None
        assert result.turn_output.pipeline_outcome == _expected


def test_contact_candidate_is_unconfirmed_and_confirmation_creates_no_db_write() -> None:
    provider = MockReasoningProvider(
        default=_proposal(AgentAction.ASK_EMAIL, TopicCategory.CONTACT_COLLECTION)
    )
    contact = MockContactUnderstandingProvider(
        scripts=(scripted("final", value="person@example.com"),)
    )
    processor = _processor(
        provider,
        initial_state=ConversationState.COLLECT_EMAIL,
        context=ConversationContext("call", current_contact_id="contact-1"),
        contact_provider=contact,
    )
    result = TurnCoordinator("call", processor).handle(_final())
    assert processor.context.contact_candidate == "person@example.com"
    assert not processor.context.contact_confirmed
    assert result.state.active_delivery is not None
    assert "is it correct" in result.state.active_delivery.rendered_response.text.lower()


def test_contact_confirmation_uses_existing_contact_path_without_persistence_execution() -> None:
    provider = MockReasoningProvider(
        scripted=(
            _proposal(AgentAction.ASK_EMAIL, TopicCategory.CONTACT_COLLECTION),
            _proposal(AgentAction.CONFIRM_CONTACT, TopicCategory.CONTACT_COLLECTION),
        )
    )
    contact = MockContactUnderstandingProvider(
        scripts=(
            scripted("confirm", intent=ContactIntent.CONFIRM_CONTACT),
            scripted("final", value="person@example.com"),
        )
    )
    processor = _processor(
        provider,
        initial_state=ConversationState.COLLECT_EMAIL,
        context=ConversationContext("call", current_contact_id="contact-1"),
        contact_provider=contact,
    )
    coordinator = TurnCoordinator("call", processor)
    coordinator.handle(_final())
    confirmed = coordinator.handle(
        RuntimeEvent(
            "e2",
            "call",
            2,
            RuntimeEventType.USER_UTTERANCE_FINAL,
            "t2",
            "yes, confirm that contact",
        )
    )
    assert processor.context.contact_confirmed
    assert processor.current_state == ConversationState.END_CALL
    assert confirmed.state.active_delivery is not None
    assert "is confirmed" in confirmed.state.active_delivery.rendered_response.text.lower()


def test_service_offer_uses_existing_catalog_selection() -> None:
    provider = MockReasoningProvider(
        default=_proposal(AgentAction.OFFER_SERVICE, TopicCategory.SERVICE_DISCUSSION)
    )
    processor = _processor(
        provider,
        initial_state=ConversationState.LISTEN,
        context=ConversationContext(
            "call", campaign_id="campaign_a", known_signals=frozenset({"existing_website"})
        ),
        with_offering=True,
    )
    TurnCoordinator("call", processor).handle(_final())
    assert len(processor.context.offered_service_ids) == 1
    assert processor.context.offered_service_ids[0] in {"website_development", "seo"}


def test_barge_in_background_duplicate_and_disconnect_compose_safely() -> None:
    provider = MockReasoningProvider(default=_proposal())
    coordinator = TurnCoordinator("call", _processor(provider))
    first = _final()
    coordinator.handle(first)
    coordinator.handle(
        RuntimeEvent("start", "call", 2, RuntimeEventType.AGENT_DELIVERY_STARTED, "t1")
    )
    cancel = coordinator.handle(
        RuntimeEvent("speech", "call", 3, RuntimeEventType.USER_SPEECH_STARTED, barge_in=True)
    )
    second = coordinator.handle(
        _final(
            4,
            "t2",
            possible_background_speech=True,
            addressee_status=AddresseeStatus.ADDRESSEE_UNCERTAIN,
            conversation_category=InterruptionCategory.ADDRESSEE_UNCERTAIN,
        )
    )
    duplicate = coordinator.handle(_final(5, "t2"))
    disconnected = coordinator.handle(
        RuntimeEvent("disconnect", "call", 6, RuntimeEventType.CALL_DISCONNECTED)
    )
    late = coordinator.handle(_final(7, "t3"))
    assert cancel.instructions[0].action == DeliveryAction.CANCEL_CURRENT
    assert second.processed_turn is not None and second.processed_turn.interruption.was_interrupted
    assert duplicate.outcome == CoordinationOutcome.DUPLICATE_IGNORED
    assert disconnected.state.disconnected
    assert late.outcome == CoordinationOutcome.DISCONNECTED_IGNORED
    assert provider.call_count == 2


def test_render_failure_does_not_rollback_successful_transition_or_retry_brain() -> None:
    class FailingRenderer(ResponseRenderer):
        def render(self, render_input):  # type: ignore[no-untyped-def]
            raise RuntimeError("render failed")

    provider = MockReasoningProvider(default=_proposal())
    processor = _processor(provider, renderer=FailingRenderer())
    result = TurnCoordinator("call", processor).handle(_final())
    assert result.outcome == CoordinationOutcome.FAILED
    assert provider.call_count == 1
    assert processor.current_state == ConversationState.GREETING


def test_failure_before_execution_leaves_state_and_context_unchanged() -> None:
    class FailingScope(ScopePolicyValidator):
        def validate(self, check):  # type: ignore[no-untyped-def]
            raise RuntimeError("scope failed")

    provider = MockReasoningProvider(default=_proposal())
    processor = _processor(provider, scope_validator=FailingScope())
    original = processor.context
    result = TurnCoordinator("call", processor).handle(_final())
    assert result.outcome == CoordinationOutcome.FAILED
    assert processor.context is original
    assert processor.current_state == ConversationState.NEW_CALL
