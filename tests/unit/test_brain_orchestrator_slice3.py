"""Focused behavioral tests for deterministic BrainOrchestrator Slice 3."""

from __future__ import annotations

from app.brain.budget.evaluator import BudgetPolicyEvaluator
from app.brain.orchestrator.orchestrator import (
    BrainOrchestrator,
    ConversationTerminationStatus,
    EffectiveAuthorityBounds,
    SliceThreeOutcome,
    TurnDecisionTrace,
    TurnResult,
    TurnStageOutcome,
)
from app.catalog.claim_validator import ClaimValidator
from app.catalog.loader import load_catalog
from app.catalog.scoped_catalog import ScopedCatalog
from app.catalog.selection.selection_service import ServiceSelectionService
from app.catalog.selection.selector import MockServiceSelector
from app.catalog.selection.selector_validator import SelectorValidator
from app.config.settings import get_settings
from app.contracts.contact_info import ContactChannel
from app.contracts.contact_understanding import ContactIntent, ContactUnderstanding
from app.contracts.conversation_context import ConversationContext
from app.conversation.contact.resolver import ContactResolver
from app.conversation.engine import ConversationEngine
from app.conversation.guardrails.action_validator import ActionValidator
from app.conversation.guardrails.clarification import ClarificationEngine
from app.conversation.state_machine.machine import ConversationStateMachine
from app.conversation.state_machine.states import load_config
from app.core.constants import AgentAction, ConversationState
from app.core.exceptions import StateTransitionError
from app.services.service_offering_service import ServiceOfferingService


class CountingEngine(ConversationEngine):
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.slice3_calls = 0

    def execute_authority_approved_action(
        self, context, action, contact_understanding=None  # type: ignore[no-untyped-def]
    ):
        self.slice3_calls += 1
        return super().execute_authority_approved_action(
            context, action, contact_understanding
        )


class RejectingMachine(ConversationStateMachine):
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.apply_calls = 0

    def apply_transition(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.apply_calls += 1
        raise StateTransitionError("simulated machine rejection")


class RecordingMachine(ConversationStateMachine):
    def __init__(self, events, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.events = events

    def apply_transition(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.events.append("state_machine")
        return super().apply_transition(*args, **kwargs)


class RecordingValidator(ActionValidator):
    def __init__(self, events, config):  # type: ignore[no-untyped-def]
        super().__init__(config)
        self.events = events

    def validate(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.events.append("action_validator")
        return super().validate(*args, **kwargs)


class RecordingResolver(ContactResolver):
    def __init__(self, events):  # type: ignore[no-untyped-def]
        self.events = events

    def resolve(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.events.append("specialized_resolution")
        return super().resolve(*args, **kwargs)


def _config():
    return load_config(get_settings().conversation_config_path)


def _engine(
    state: ConversationState = ConversationState.NEW_CALL,
    *,
    offering: ServiceOfferingService | None = None,
    contact: bool = False,
    machine: ConversationStateMachine | None = None,
) -> CountingEngine:
    config = _config()
    return CountingEngine(
        machine or ConversationStateMachine(config, initial_state=state),
        ActionValidator(config),
        contact_resolver=ContactResolver() if contact else None,
        clarification_engine=ClarificationEngine(config) if contact else None,
        offering_service=offering,
    )


def _orchestrator(engine: ConversationEngine) -> BrainOrchestrator:
    return BrainOrchestrator(engine, BudgetPolicyEvaluator())


def _slice_two(
    outcome: TurnStageOutcome,
    *,
    action: AgentAction | None = None,
    context: ConversationContext | None = None,
    bounds: EffectiveAuthorityBounds | None = None,
) -> TurnResult:
    return TurnResult(
        outcome=outcome,
        conversation_status=ConversationTerminationStatus.NOT_TERMINAL,
        trace=TurnDecisionTrace(stage_outcome=outcome),
        execution=None,
        updated_context=context or ConversationContext("call"),
        action=action,
        effective_bounds=bounds,
        scope_policy_id="scope",
        authority_policy_id="authority",
    )


def _service_offering(forced_id: str | None = None) -> ServiceOfferingService:
    scoped = ScopedCatalog(load_catalog("app/config/service_config.yaml"), "campaign_a")
    validator = SelectorValidator(ClaimValidator(scoped))
    selection = ServiceSelectionService(
        MockServiceSelector(forced_id=forced_id), validator
    )
    return ServiceOfferingService(scoped, selection)


def test_simple_approved_action_validates_and_transitions() -> None:
    engine = _engine()
    result = _orchestrator(engine).execute_slice_three(
        _slice_two(TurnStageOutcome.AUTHORITY_APPROVED, action=AgentAction.GREET)
    )

    assert result.outcome == SliceThreeOutcome.EXECUTED
    assert result.execution is not None
    assert result.execution.approved_action == AgentAction.GREET
    assert engine.machine.current_state == ConversationState.GREETING


def test_action_validator_rejection_does_not_call_state_machine() -> None:
    config = _config()
    machine = RejectingMachine(config)
    engine = CountingEngine(machine, ActionValidator(config))
    result = _orchestrator(engine).execute_slice_three(
        _slice_two(TurnStageOutcome.AUTHORITY_APPROVED, action=AgentAction.ASK_EMAIL)
    )

    assert result.outcome == SliceThreeOutcome.FALLBACK
    assert machine.apply_calls == 0
    assert machine.current_state == ConversationState.NEW_CALL


def test_invalid_state_transition_is_not_forced() -> None:
    config = _config()
    machine = RejectingMachine(config)
    engine = CountingEngine(machine, ActionValidator(config))
    result = _orchestrator(engine).execute_slice_three(
        _slice_two(TurnStageOutcome.AUTHORITY_APPROVED, action=AgentAction.GREET)
    )

    assert result.outcome == SliceThreeOutcome.FALLBACK
    assert machine.apply_calls == 1
    assert machine.current_state == ConversationState.NEW_CALL


def test_service_action_recomputes_eligibility_and_commits_after_transition() -> None:
    context = ConversationContext(
        "call",
        campaign_id="campaign_a",
        known_signals=frozenset({"existing_website"}),
    )
    engine = _engine(ConversationState.LISTEN, offering=_service_offering())
    result = _orchestrator(engine).execute_slice_three(
        _slice_two(
            TurnStageOutcome.AUTHORITY_APPROVED,
            action=AgentAction.OFFER_SERVICE,
            context=context,
        )
    )

    assert result.outcome == SliceThreeOutcome.EXECUTED
    assert result.execution is not None
    assert result.execution.service_selection is not None
    assert result.execution.updated_context.offered_service_ids == (
        "seo",
    )
    assert context.offered_service_ids == ()


def test_unauthorized_service_is_rejected_without_mutation() -> None:
    context = ConversationContext(
        "call",
        campaign_id="campaign_a",
        known_signals=frozenset({"existing_website"}),
    )
    engine = _engine(
        ConversationState.LISTEN,
        offering=_service_offering(forced_id="invented_service"),
    )
    result = _orchestrator(engine).execute_slice_three(
        _slice_two(
            TurnStageOutcome.AUTHORITY_APPROVED,
            action=AgentAction.OFFER_SERVICE,
            context=context,
        )
    )

    assert result.outcome == SliceThreeOutcome.FALLBACK
    assert engine.machine.current_state == ConversationState.LISTEN
    assert result.execution is not None
    assert result.execution.updated_context is context
    assert context.offered_service_ids == ()


def test_already_offered_service_is_not_repeated() -> None:
    context = ConversationContext(
        "call",
        campaign_id="campaign_a",
        known_signals=frozenset({"existing_website"}),
        offered_service_ids=("website_development", "seo"),
        service_offers_made=2,
    )
    engine = _engine(ConversationState.LISTEN, offering=_service_offering())
    result = _orchestrator(engine).execute_slice_three(
        _slice_two(
            TurnStageOutcome.AUTHORITY_APPROVED,
            action=AgentAction.OFFER_SERVICE,
            context=context,
        )
    )

    assert result.outcome == SliceThreeOutcome.FALLBACK
    assert result.execution is not None
    assert result.execution.updated_context.offered_service_ids == (
        "website_development",
        "seo",
    )


def test_contact_action_resolves_then_validates_then_transitions() -> None:
    events = []
    config = _config()
    machine = RecordingMachine(events, config, initial_state=ConversationState.COLLECT_EMAIL)
    engine = CountingEngine(
        machine,
        RecordingValidator(events, config),
        contact_resolver=RecordingResolver(events),
        clarification_engine=ClarificationEngine(config),
    )
    understanding = ContactUnderstanding(
        intent=ContactIntent.PROVIDE_CONTACT,
        channel=ContactChannel.EMAIL,
        value="person@example.com",
    )
    result = _orchestrator(engine).execute_slice_three(
        _slice_two(TurnStageOutcome.AUTHORITY_APPROVED, action=AgentAction.ASK_EMAIL),
        understanding,
    )

    assert result.outcome == SliceThreeOutcome.EXECUTED
    assert events == ["specialized_resolution", "action_validator", "state_machine"]
    assert result.execution is not None
    assert result.execution.updated_context.contact_candidate == "person@example.com"
    assert not result.execution.updated_context.contact_confirmed
    assert result.execution.contact_to_persist is None


def test_unconfirmed_contact_never_produces_persistence_intent() -> None:
    context = ConversationContext("call", current_contact_id="contact")
    engine = _engine(ConversationState.CONFIRM_CONTACT, contact=True)
    understanding = ContactUnderstanding(
        intent=ContactIntent.CONFIRM_CONTACT,
        channel=ContactChannel.EMAIL,
    )
    result = _orchestrator(engine).execute_slice_three(
        _slice_two(
            TurnStageOutcome.AUTHORITY_APPROVED,
            action=AgentAction.CONFIRM_CONTACT,
            context=context,
        ),
        understanding,
    )

    assert result.outcome == SliceThreeOutcome.FALLBACK
    assert result.execution is not None
    assert result.execution.contact_to_persist is None
    assert not context.contact_confirmed


def test_effective_bounds_are_preserved_and_fail_closed() -> None:
    bounds = EffectiveAuthorityBounds(max_discount_percent=10)
    engine = _engine()
    result = _orchestrator(engine).execute_slice_three(
        _slice_two(
            TurnStageOutcome.AUTHORITY_APPROVED,
            action=AgentAction.GREET,
            bounds=bounds,
        )
    )

    assert result.outcome == SliceThreeOutcome.FALLBACK
    assert result.effective_bounds is bounds
    assert result.execution is None
    assert engine.slice3_calls == 0


def test_non_approved_slice_two_outcomes_never_enter_specialized_execution() -> None:
    for outcome, expected in (
        (TurnStageOutcome.ESCALATE, SliceThreeOutcome.FALLBACK),
        (TurnStageOutcome.REDIRECT, SliceThreeOutcome.REDIRECT),
        (TurnStageOutcome.FALLBACK, SliceThreeOutcome.FALLBACK),
    ):
        engine = _engine()
        result = _orchestrator(engine).execute_slice_three(_slice_two(outcome))
        assert result.outcome == expected
        assert engine.slice3_calls == 0


def test_specialized_failure_stops_before_validator_and_machine() -> None:
    config = _config()
    machine = RejectingMachine(config, initial_state=ConversationState.LISTEN)

    class CountingValidator(ActionValidator):
        calls = 0

        def validate(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            return super().validate(*args, **kwargs)

    validator = CountingValidator(config)
    engine = CountingEngine(machine, validator, offering_service=None)
    result = _orchestrator(engine).execute_slice_three(
        _slice_two(TurnStageOutcome.AUTHORITY_APPROVED, action=AgentAction.OFFER_SERVICE)
    )

    assert result.outcome == SliceThreeOutcome.FALLBACK
    assert validator.calls == 0
    assert machine.apply_calls == 0


def test_brain_cannot_control_next_state() -> None:
    engine = _engine()
    approval = _slice_two(
        TurnStageOutcome.AUTHORITY_APPROVED,
        action=AgentAction.GREET,
    )
    object.__setattr__(approval, "proposed_next_state", ConversationState.END_CALL)
    result = _orchestrator(engine).execute_slice_three(approval)

    assert result.outcome == SliceThreeOutcome.EXECUTED
    assert engine.machine.current_state == ConversationState.GREETING


def test_failure_does_not_partially_mutate_context_or_state() -> None:
    context = ConversationContext(
        "call",
        offered_service_ids=("website_development",),
        contact_confirmed=False,
    )
    engine = _engine()
    result = _orchestrator(engine).execute_slice_three(
        _slice_two(
            TurnStageOutcome.AUTHORITY_APPROVED,
            action=AgentAction.ASK_EMAIL,
            context=context,
        )
    )

    assert result.outcome == SliceThreeOutcome.FALLBACK
    assert engine.machine.current_state == ConversationState.NEW_CALL
    assert result.execution is not None
    assert result.execution.updated_context is context
    assert context.offered_service_ids == ("website_development",)
    assert not context.contact_confirmed
    assert result.execution.contact_to_persist is None


def test_deterministic_replay_for_same_trusted_inputs() -> None:
    approval = _slice_two(
        TurnStageOutcome.AUTHORITY_APPROVED,
        action=AgentAction.GREET,
    )
    first = _orchestrator(_engine()).execute_slice_three(approval)
    second = _orchestrator(_engine()).execute_slice_three(approval)

    assert first == second
