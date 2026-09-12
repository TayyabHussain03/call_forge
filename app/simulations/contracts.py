"""Typed contracts for deterministic offline conversation scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.brain.contracts import BrainProposal, BusinessIntelligenceSnapshot
from app.brain.orchestrator.orchestrator import (
    ConversationTerminationStatus,
    SliceThreeOutcome,
    TurnStageOutcome,
)
from app.contracts.contact_understanding import ContactUnderstanding
from app.contracts.conversation_context import ConversationContext
from app.contracts.validation import ValidationCategory
from app.conversation.guardrails.priority import TrustedPriorityOutcome
from app.conversation.response_planning.contracts import (
    AddresseeStatus,
    AcknowledgementKind,
    ExplanationNeed,
    InterruptionCategory,
    InterruptionContext,
    ResponsePlan,
)
from app.conversation.response_rendering.contracts import (
    RenderedResponse,
    TrustedRenderingContext,
)
from app.core.constants import AgentAction, ConversationState


class ScenarioActorRole(str, Enum):
    """Neutral scenario metadata; it grants no authority or policy change."""

    OWNER = "owner"
    MANAGER = "manager"
    RECEPTIONIST = "receptionist"
    GATEKEEPER = "gatekeeper"
    DECISION_MAKER = "decision_maker"
    UNKNOWN = "unknown"


class SpecializedResolutionCategory(str, Enum):
    """Deterministic execution path selected after authority approval."""

    GENERIC = "generic"
    SERVICE = "service"
    CONTACT = "contact"


@dataclass(frozen=True)
class ExpectedTurnOutcome:
    """Optional declarative expectation stored with a scenario turn."""

    pipeline_outcome: TurnStageOutcome
    resulting_state: ConversationState
    slice_three_outcome: SliceThreeOutcome | None = None
    terminal_status: ConversationTerminationStatus | None = None


@dataclass(frozen=True)
class SimulationTurn:
    """One deterministic prospect turn and its scripted provider behavior."""

    utterance: str
    trusted_priority: TrustedPriorityOutcome = TrustedPriorityOutcome.NONE
    proposal: BrainProposal | None = None
    provider_failure: bool = False
    contact_understanding: ContactUnderstanding | None = None
    conversation_category: InterruptionCategory = InterruptionCategory.OTHER
    addressee_status: AddresseeStatus = AddresseeStatus.ADDRESSED_TO_AGENT
    interruption: InterruptionContext = InterruptionContext()
    explanation_need: ExplanationNeed = ExplanationNeed.STANDARD
    previous_acknowledgement: AcknowledgementKind = AcknowledgementKind.NONE
    trusted_rendering_context: TrustedRenderingContext = TrustedRenderingContext()
    expected: ExpectedTurnOutcome | None = None


@dataclass(frozen=True)
class SimulationScenario:
    """Immutable scenario definition referencing trusted repository policies."""

    scenario_id: str
    initial_context: ConversationContext
    turns: tuple[SimulationTurn, ...]
    initial_state: ConversationState = ConversationState.NEW_CALL
    scope_policy_id: str = "business_general"
    authority_policy_id: str = "standard"
    budget_policy_id: str = "standard"
    actor_role: ScenarioActorRole = ScenarioActorRole.UNKNOWN
    business_intelligence: BusinessIntelligenceSnapshot | None = None


@dataclass(frozen=True)
class TurnTrace:
    """Minimal safe trace of authoritative decisions for one simulated turn."""

    turn_index: int
    starting_state: ConversationState
    trusted_priority: TrustedPriorityOutcome
    brain_called: bool
    proposed_action: AgentAction | None
    scope_result: str | None
    authority_result: str | None
    specialized_resolution: SpecializedResolutionCategory | None
    validation_result: ValidationCategory | None
    resulting_state: ConversationState
    pipeline_outcome: TurnStageOutcome
    slice_three_outcome: SliceThreeOutcome | None
    termination_status: ConversationTerminationStatus
    fallback_used: bool = False
    selected_service_id: str | None = None
    contact_channel: str | None = None
    persistence_intent_created: bool = False
    response_plan: ResponsePlan | None = None
    rendered_response: RenderedResponse | None = None
    expectation_met: bool | None = None


@dataclass(frozen=True)
class SimulationResult:
    """Deterministic aggregate result for a completed scenario run."""

    scenario_id: str
    turns: tuple[TurnTrace, ...]
    final_context: ConversationContext
    final_state: ConversationState
    terminal: bool
    expectations_met: bool
    actor_role: ScenarioActorRole
