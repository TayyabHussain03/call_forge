"""Offline runner that composes the existing deterministic conversation system."""

from __future__ import annotations

from app.brain.authority.models import load_authority_policies
from app.brain.authority.validator import AuthorityPolicyValidator
from app.brain.budget.evaluator import BudgetPolicyEvaluator, TrustedExitSignals
from app.brain.budget.models import BudgetPolicy, load_budget_policies
from app.brain.contracts import BudgetState
from app.brain.orchestrator.orchestrator import (
    BrainOrchestrator,
    BrainSnapshotInput,
    ConversationTerminationStatus,
    SliceThreeOutcome,
    TurnStageOutcome,
)
from app.brain.scope.models import load_scope_policies
from app.brain.scope.validator import ScopePolicyValidator
from app.catalog.claim_validator import ClaimValidator
from app.catalog.loader import load_catalog
from app.catalog.scoped_catalog import ScopedCatalog
from app.catalog.selection.selection_service import ServiceSelectionService
from app.catalog.selection.selector import MockServiceSelector
from app.catalog.selection.selector_validator import SelectorValidator
from app.config.settings import get_settings
from app.conversation.contact.resolver import ContactResolver
from app.conversation.engine import ConversationEngine
from app.conversation.guardrails.action_validator import ActionValidator
from app.conversation.guardrails.clarification import ClarificationEngine
from app.conversation.guardrails.fallbacks import FallbackEngine
from app.conversation.response_planning.contracts import (
    AuthoritativeResultKind,
    ResponsePlanningInput,
)
from app.conversation.response_planning.planner import ResponsePlanner
from app.conversation.response_rendering.contracts import (
    ResponseRenderInput,
    ResponseRenderingBudget,
)
from app.conversation.response_rendering.renderer import DeterministicResponseRenderer
from app.conversation.state_machine.machine import ConversationStateMachine
from app.conversation.state_machine.states import load_config
from app.llm.providers.reasoning_provider import MockReasoningProvider
from app.services.service_offering_service import ServiceOfferingService
from app.simulations.contracts import (
    ExpectedTurnOutcome,
    SimulationResult,
    SimulationScenario,
    SpecializedResolutionCategory,
    TurnTrace,
)


def run_scenario(scenario: SimulationScenario) -> SimulationResult:
    """Run a typed scenario through the real deterministic architecture."""
    settings = get_settings()
    machine_config = load_config(settings.conversation_config_path)
    scope_policies = load_scope_policies("app/config/scope_policy.yaml")
    authority_policies = load_authority_policies("app/config/authority_policy.yaml")
    budget_policies = load_budget_policies("app/config/budget_policy.yaml")

    scope_policy = scope_policies[scenario.scope_policy_id]
    authority_policy = authority_policies[scenario.authority_policy_id]
    budget_policy = budget_policies[scenario.budget_policy_id]

    machine = ConversationStateMachine(machine_config, scenario.initial_state)
    offering = _build_offering(scenario.initial_context.campaign_id, machine_config)
    engine = ConversationEngine(
        machine,
        ActionValidator(machine_config),
        fallback_engine=FallbackEngine(machine_config),
        clarification_engine=ClarificationEngine(machine_config),
        contact_resolver=ContactResolver(),
        offering_service=offering,
    )

    context = scenario.initial_context
    traces: list[TurnTrace] = []
    reasoning_calls = 0
    response_planner = ResponsePlanner()
    response_renderer = DeterministicResponseRenderer()

    for turn_index, turn in enumerate(scenario.turns):
        if machine.is_terminal():
            break

        starting_state = machine.current_state
        provider = MockReasoningProvider(
            default=turn.proposal,
            should_fail=turn.provider_failure,
            provider_name="simulation_mock",
        )
        orchestrator = BrainOrchestrator(
            engine,
            BudgetPolicyEvaluator(),
            reasoning_provider=provider,
            scope_validator=ScopePolicyValidator(),
            authority_validator=AuthorityPolicyValidator(),
        )
        budget_state = _budget_state(budget_policy, turn_index, reasoning_calls)
        slice_two = orchestrator.process_turn(
            context,
            turn.trusted_priority,
            budget_state,
            budget_policy,
            TrustedExitSignals(),
            BrainSnapshotInput(
                current_utterance=turn.utterance,
                business_intelligence=scenario.business_intelligence,
            ),
            scope_policy,
            authority_policy,
        )
        reasoning_calls += provider.call_count
        context = slice_two.updated_context

        slice_three = None
        if slice_two.outcome == TurnStageOutcome.AUTHORITY_APPROVED:
            slice_three = orchestrator.execute_slice_three(
                slice_two, turn.contact_understanding
            )
            if slice_three.execution is not None:
                context = slice_three.execution.updated_context

        execution = slice_three.execution if slice_three is not None else slice_two.execution
        termination = (
            slice_three.conversation_status
            if slice_three is not None
            else slice_two.conversation_status
        )
        response_plan = response_planner.plan(
            ResponsePlanningInput(
                authoritative_state=machine.current_state,
                authoritative_result=_response_result(slice_two.outcome, slice_three),
                current_prospect_message=turn.utterance[:2000],
                approved_action=(execution.approved_action if execution is not None else None),
                conversation_category=turn.conversation_category,
                addressee_status=turn.addressee_status,
                interruption=turn.interruption,
                explanation_need=turn.explanation_need,
                previous_acknowledgement=turn.previous_acknowledgement,
            )
        )
        rendered_response = response_renderer.render(
            ResponseRenderInput(
                plan=response_plan,
                authoritative_result=_response_result(slice_two.outcome, slice_three),
                budget=ResponseRenderingBudget(
                    max_sentences=budget_policy.limits.max_response_sentences
                ),
                trusted_context=turn.trusted_rendering_context,
                variation_seed=f"{scenario.scenario_id}:{turn_index}",
            )
        )
        trace = TurnTrace(
            turn_index=turn_index,
            starting_state=starting_state,
            trusted_priority=turn.trusted_priority,
            brain_called=provider.call_count > 0,
            proposed_action=turn.proposal.proposed_action if turn.proposal else None,
            scope_result=(
                slice_two.trace.scope_category.value
                if slice_two.trace.scope_category is not None
                else None
            ),
            authority_result=(
                slice_two.trace.authority_tier.value
                if slice_two.trace.authority_tier is not None
                else None
            ),
            specialized_resolution=_specialized_category(
                slice_two.action,
                slice_three is not None and slice_three.execution is not None,
            ),
            validation_result=(
                execution.validation.category
                if execution is not None and execution.validation is not None
                else None
            ),
            resulting_state=machine.current_state,
            pipeline_outcome=slice_two.outcome,
            slice_three_outcome=slice_three.outcome if slice_three else None,
            termination_status=termination,
            fallback_used=execution.fallback_used if execution is not None else False,
            selected_service_id=(
                execution.service_selection.approved_service_id
                if execution is not None and execution.service_selection is not None
                else None
            ),
            contact_channel=(
                context.contact_candidate_channel
                if execution is not None and execution.path == "contact"
                else None
            ),
            persistence_intent_created=(
                execution.contact_to_persist is not None if execution is not None else False
            ),
            response_plan=response_plan,
            rendered_response=rendered_response,
            expectation_met=_matches(turn.expected, slice_two.outcome, slice_three, machine),
        )
        traces.append(trace)

    expectation_values = [t.expectation_met for t in traces if t.expectation_met is not None]
    return SimulationResult(
        scenario_id=scenario.scenario_id,
        turns=tuple(traces),
        final_context=context,
        final_state=machine.current_state,
        terminal=machine.is_terminal(),
        expectations_met=all(expectation_values),
        actor_role=scenario.actor_role,
    )


def _response_result(pipeline, slice_three) -> AuthoritativeResultKind:
    """Normalize existing pipeline results for read-only response planning."""
    if slice_three is not None:
        return AuthoritativeResultKind(slice_three.outcome.value)
    mapping = {
        TurnStageOutcome.FALLBACK: AuthoritativeResultKind.FALLBACK,
        TurnStageOutcome.REDIRECT: AuthoritativeResultKind.REDIRECT,
        TurnStageOutcome.ESCALATE: AuthoritativeResultKind.ESCALATE,
        TurnStageOutcome.PIPELINE_STOPPED: AuthoritativeResultKind.PIPELINE_STOPPED,
        TurnStageOutcome.AUTHORITY_APPROVED: AuthoritativeResultKind.AUTHORITY_APPROVED,
    }
    return mapping[pipeline]


def _build_offering(campaign_id: str | None, machine_config) -> ServiceOfferingService | None:
    """Build the existing deterministic offering stack for a known campaign."""
    if campaign_id is None:
        return None
    scoped = ScopedCatalog(load_catalog("app/config/service_config.yaml"), campaign_id)
    selection = ServiceSelectionService(
        MockServiceSelector(),
        SelectorValidator(ClaimValidator(scoped)),
    )
    max_offers = int(machine_config.limits.get("service_offer_max_attempts", 3))
    return ServiceOfferingService(scoped, selection, max_offers=max_offers)


def _specialized_category(
    action,
    slice_three_attempted: bool,
) -> SpecializedResolutionCategory | None:
    """Classify the deterministic Slice 3 path without exposing proposal text."""
    if not slice_three_attempted or action is None:
        return None
    if action.value == "offer_service":
        return SpecializedResolutionCategory.SERVICE
    if action.value in {"ask_email", "confirm_contact", "clarify_contact"}:
        return SpecializedResolutionCategory.CONTACT
    return SpecializedResolutionCategory.GENERIC


def _budget_state(
    policy: BudgetPolicy,
    turn_index: int,
    reasoning_calls: int,
) -> BudgetState:
    """Derive remaining deterministic budgets without mutating scenario context."""
    limits = policy.limits
    return BudgetState(
        turns_remaining=limits.max_turns - turn_index,
        seconds_remaining=limits.max_call_seconds,
        reasoning_calls_remaining=limits.max_reasoning_calls - reasoning_calls,
        max_response_sentences=limits.max_response_sentences,
    )


def _matches(expected: ExpectedTurnOutcome | None, pipeline, slice_three, machine) -> bool | None:
    """Compare an optional declarative expectation to authoritative results."""
    if expected is None:
        return None
    terminal_status = (
        ConversationTerminationStatus.TERMINAL
        if machine.is_terminal()
        else ConversationTerminationStatus.NOT_TERMINAL
    )
    return (
        pipeline == expected.pipeline_outcome
        and machine.current_state == expected.resulting_state
        and (
            expected.slice_three_outcome is None
            or (slice_three is not None and slice_three.outcome == expected.slice_three_outcome)
        )
        and (
            expected.terminal_status is None
            or terminal_status == expected.terminal_status
        )
    )
