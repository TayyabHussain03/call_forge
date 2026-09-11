"""BrainOrchestrator — trusted Slice 1 gates plus classification-only Slice 2.

CORE CONSTRAINT (locked): orchestrator SEQUENCES, never decides policy. Verb set:
call → evaluate → route → pass → continue/stop. Har decision deterministic
component (priority/budget/scope/authority) ya Brain ka; orchestrator sirf wire karta hai.

SLICE 1 SCOPE: trusted priority → budget, with deterministic short-circuits
(priority resolve → engine result; budget EARLY_EXIT/WIND_DOWN → pipeline stop;
CONSTRAIN_RESPONSE/PROCEED → typed continuation marker). Slice 2 builds bounded
BrainInput, calls one ReasoningProvider, then classifies scope and authority.
Specialized resolution and every execution concern remain deliberately absent.

Execution spine reuse: jab koi action deterministically execute karna ho (jaise
priority action), orchestrator engine ke ADDITIVE `execute_approved_action(...)`
ko call karta hai — engine ka `process_turn` untouched, koi second orchestration
path nahi.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from app.brain.authority.models import AuthorityPolicy, AuthorityTier
from app.brain.authority.validator import AuthorityCheckInput, AuthorityPolicyValidator
from app.brain.budget.evaluator import (
    BudgetCheckInput,
    BudgetDecision,
    BudgetOutcome,
    BudgetPolicyEvaluator,
    TrustedExitSignals,
)
from app.brain.budget.models import BudgetPolicy
from app.brain.business_intelligence import InferredSignal, ObservedSignal, UnknownSlot
from app.brain.contracts import (
    BrainInput,
    BrainProposal,
    BudgetState,
    BusinessIntelligenceSnapshot,
    CommercialRequest,
)
from app.brain.scope.models import ScopePolicy
from app.brain.scope.validator import (
    ScopeCategory,
    ScopeCheckInput,
    ScopePolicyValidator,
)
from app.contracts.contact_understanding import ContactUnderstanding
from app.contracts.conversation_context import ConversationContext
from app.conversation.engine import ConversationEngine, ConversationResult
from app.conversation.guardrails.priority import (
    TrustedPriorityOutcome,
    resolve_priority_action,
)
from app.core.constants import (
    AgentAction,
    CommercialRequestKind,
    Intent,
    Tone,
    TopicCategory,
)
from app.llm.providers.reasoning_provider import ReasoningError, ReasoningProvider


class TurnStageOutcome(str, Enum):
    """High-level outcome of pre-Brain gates or Slice 2 classification.

    PIPELINE_STOPPED means later Slice-1 stages do not run; it does not claim the
    state machine is terminal. CONTINUE_TO_BRAIN is the internal seam at which
    Slice 2 starts; successful Slice 2 returns AUTHORITY_APPROVED, never execution.
    """

    PIPELINE_STOPPED = "pipeline_stopped"    # no later Slice-1 stage runs
    CONTINUE_TO_BRAIN = "continue_to_brain"  # pre-Brain stages passed; Brain would run next
    AUTHORITY_APPROVED = "authority_approved"
    ESCALATE = "escalate"
    REDIRECT = "redirect"
    FALLBACK = "fallback"


class ConversationTerminationStatus(str, Enum):
    """Whether the authoritative state machine has reached a terminal state."""

    TERMINAL = "terminal"
    NOT_TERMINAL = "not_terminal"


class BrainProposalStatus(str, Enum):
    """Typed status of the untrusted provider boundary."""

    VALID = "valid"
    NO_ACTION = "no_action"
    MALFORMED = "malformed"
    PROVIDER_FAILED = "provider_failed"


class TerminationRoute(str, Enum):
    """How a WIND_DOWN should be routed (orchestrator resolves, not BudgetPolicy).

    Slice 1 mein ye seam hai — mapping deterministic routing hai, judgment nahi.
    Close-vs-callback ka final policy future slice mein refine hoga (B2).
    """

    CLOSE = "close"
    CALLBACK = "callback"


@dataclass(frozen=True)
class BrainSnapshotInput:
    """Trusted inputs used to build one bounded immutable ``BrainInput``."""

    current_utterance: str
    current_goal: str | None = None
    recent_turns: tuple[str, ...] = ()
    business_intelligence: BusinessIntelligenceSnapshot | None = None
    campaign_policy_summary: str | None = None
    resolved_contact_context: str | None = None


@dataclass(frozen=True)
class EffectiveAuthorityBounds:
    """Policy-derived bounds attached to a policy-bounded approval."""

    max_discount_percent: float | None = None
    pricing_disclosure_allowed: bool | None = None


@dataclass(frozen=True)
class TurnDecisionTrace:
    """Small typed per-turn decision trace (auditability; no PII bloat).

    Existing typed results reuse karta hai — koi giant object, koi transcript/
    secret/PII.

    Attributes:
        priority_action: Priority-resolved action, ya None.
        budget_outcome: BudgetDecision ka outcome, ya None (agar priority ne pehle
            terminate kiya).
        budget_exit_kind: Budget exit kind label, ya None.
        termination_route: WIND_DOWN par resolved route, ya None.
        stage_outcome: Overall Slice-1 outcome.
        response_ceiling: CONSTRAIN_RESPONSE/PROCEED se aaya ceiling, ya None.
        notes: Machine-readable short note (debug).
    """

    stage_outcome: TurnStageOutcome
    priority_action: AgentAction | None = None
    budget_outcome: BudgetOutcome | None = None
    budget_exit_kind: str | None = None
    termination_route: TerminationRoute | None = None
    response_ceiling: int | None = None
    provider_name: str | None = None
    proposal_status: BrainProposalStatus | None = None
    scope_category: ScopeCategory | None = None
    authority_tier: AuthorityTier | None = None
    action_confidence: float | None = None
    notes: str | None = None


@dataclass(frozen=True)
class TurnResult:
    """Result of trusted pre-Brain gating or Slice 2 classification.

    Attributes:
        outcome: Pipeline stop or deterministic Slice 2 routing classification.
        conversation_status: Independent authoritative machine terminality.
        trace: The decision trace.
        execution: ConversationResult jab koi action execute hua (priority), ya
            None (budget-terminated / continue-marker mein koi execution nahi).
        updated_context: Context to carry forward (execution ke baad naya, warna
            same).
    """

    outcome: TurnStageOutcome
    conversation_status: ConversationTerminationStatus
    trace: TurnDecisionTrace
    execution: ConversationResult | None
    updated_context: ConversationContext
    action: AgentAction | None = None
    effective_bounds: EffectiveAuthorityBounds | None = None
    scope_policy_id: str | None = None
    authority_policy_id: str | None = None


# Termination-routing seam (B2 open): deterministic mapping, no judgment. Default
# CLOSE; agar trusted callback context ho to CALLBACK. Ye orchestrator ka pure
# routing hai — BudgetPolicy purity intact.
TerminationRouter = Callable[[ConversationContext], TerminationRoute]


def default_termination_router(context: ConversationContext) -> TerminationRoute:
    """Deterministically route a WIND_DOWN to CLOSE or CALLBACK.

    Pure routing (no intelligence): agar trusted callback context maujood hai to
    CALLBACK, warna CLOSE. Final policy B2 mein refine hoga.

    Args:
        context: Current session context.

    Returns:
        TerminationRoute: CALLBACK if a callback context exists, else CLOSE.
    """
    if context.callback:
        return TerminationRoute.CALLBACK
    return TerminationRoute.CLOSE


class BrainOrchestrator:
    """Coordinate trusted priority/budget gates and classification-only Slice 2.

    Orchestrator OWNS sequencing only. Priority/budget decisions delegate hote
    hain; only trusted pre-Brain priority actions reach the engine. Brain proposals
    are untrusted and can only reach a policy classification in this slice.

    Attributes:
        engine: The ConversationEngine (for execute_approved_action + terminal).
        budget_evaluator: Deterministic budget/early-exit evaluator.
    """

    def __init__(
        self,
        engine: ConversationEngine,
        budget_evaluator: BudgetPolicyEvaluator,
        termination_router: TerminationRouter = default_termination_router,
        reasoning_provider: ReasoningProvider | None = None,
        scope_validator: ScopePolicyValidator | None = None,
        authority_validator: AuthorityPolicyValidator | None = None,
    ) -> None:
        """Initialize the Slice 1 and Slice 2 orchestrator dependencies.

        Args:
            engine: The per-session ConversationEngine.
            budget_evaluator: Budget/early-exit evaluator.
            termination_router: WIND_DOWN → route resolver (seam; default provided).
            reasoning_provider: One-shot proposal source for Slice 2.
            scope_validator: Deterministic scope validator.
            authority_validator: Deterministic authority validator.
        """
        self._engine = engine
        self._budget = budget_evaluator
        self._route = termination_router
        self._reasoning = reasoning_provider
        self._scope = scope_validator or ScopePolicyValidator()
        self._authority = authority_validator or AuthorityPolicyValidator()

    def process_turn(
        self,
        context: ConversationContext,
        trusted_priority: TrustedPriorityOutcome,
        budget_state: "BudgetState | None",
        budget_policy: BudgetPolicy | None,
        exit_signals: TrustedExitSignals,
        brain_snapshot: BrainSnapshotInput | None = None,
        scope_policy: ScopePolicy | None = None,
        authority_policy: AuthorityPolicy | None = None,
    ) -> TurnResult:
        """Run trusted priority → budget → Brain → scope → authority.

        Priority is provided as an already validated trusted outcome. Sequencing:

            1. Priority: resolve_priority_action(trusted_priority). Mila → execute
               via engine additive entry-point → pipeline stop (Brain never runs).
            2. Budget: evaluate. EARLY_EXIT → stop; WIND_DOWN →
               termination-routing → stop; CONSTRAIN_RESPONSE/PROCEED →
               internal CONTINUE_TO_BRAIN seam, then Slice 2 classification.

        Args:
            context: Trusted session context.
            trusted_priority: Priority outcome established by a trusted,
                deterministic validation boundary.
            budget_policy: Resolved budget policy, ya None (fail-closed).
            exit_signals: Trusted early-exit signals.
            brain_snapshot: Trusted bounded-context inputs for BrainInput.
            scope_policy: Resolved trusted scope policy.
            authority_policy: Resolved trusted authority policy.

        Returns:
            TurnResult: outcome + trace + optional execution + carried context.
        """
        # 1. PRIORITY (deterministic; DNC/NOT_INTERESTED authority). Brain se pehle.
        priority_action = resolve_priority_action(
            trusted_priority,
            self._engine.machine.current_state,
            context.to_validator_context(),
        )

        if priority_action is not None:
            execution = self._engine.execute_approved_action(context, priority_action)
            trace = TurnDecisionTrace(
                stage_outcome=TurnStageOutcome.PIPELINE_STOPPED,
                priority_action=priority_action,
                notes="priority resolved; executed via engine spine",
            )
            return TurnResult(
                outcome=TurnStageOutcome.PIPELINE_STOPPED,
                conversation_status=(
                    ConversationTerminationStatus.TERMINAL
                    if execution.is_terminal
                    else ConversationTerminationStatus.NOT_TERMINAL
                ),
                trace=trace,
                execution=execution,
                updated_context=execution.updated_context,
            )

        # 2. BUDGET / EARLY-EXIT (deterministic).
        decision: BudgetDecision = self._budget.evaluate(
            BudgetCheckInput(
                budget_state=budget_state,
                exit_signals=exit_signals,
                policy=budget_policy,
            )
        )

        if decision.outcome == BudgetOutcome.EARLY_EXIT:
            trace = TurnDecisionTrace(
                stage_outcome=TurnStageOutcome.PIPELINE_STOPPED,
                budget_outcome=decision.outcome,
                budget_exit_kind=decision.exit_kind.value if decision.exit_kind else None,
                notes="budget early-exit (trusted signal)",
            )
            return TurnResult(
                TurnStageOutcome.PIPELINE_STOPPED,
                ConversationTerminationStatus.NOT_TERMINAL,
                trace,
                None,
                context,
            )

        if decision.outcome == BudgetOutcome.WIND_DOWN:
            route = self._route(context)
            trace = TurnDecisionTrace(
                stage_outcome=TurnStageOutcome.PIPELINE_STOPPED,
                budget_outcome=decision.outcome,
                budget_exit_kind=decision.exit_kind.value if decision.exit_kind else None,
                termination_route=route,
                notes="budget wind-down; orchestrator resolved route",
            )
            return TurnResult(
                TurnStageOutcome.PIPELINE_STOPPED,
                ConversationTerminationStatus.NOT_TERMINAL,
                trace,
                None,
                context,
            )

        # CONSTRAIN_RESPONSE / PROCEED → continue into Slice 2.
        trace = TurnDecisionTrace(
            stage_outcome=TurnStageOutcome.CONTINUE_TO_BRAIN,
            budget_outcome=decision.outcome,
            response_ceiling=decision.response_ceiling,
            notes="pre-Brain stages passed",
        )
        return self._run_slice_two(
            context=context,
            budget_state=budget_state,
            pre_brain_trace=trace,
            snapshot=brain_snapshot,
            scope_policy=scope_policy,
            authority_policy=authority_policy,
        )

    def _run_slice_two(
        self,
        context: ConversationContext,
        budget_state: BudgetState | None,
        pre_brain_trace: TurnDecisionTrace,
        snapshot: BrainSnapshotInput | None,
        scope_policy: ScopePolicy | None,
        authority_policy: AuthorityPolicy | None,
    ) -> TurnResult:
        """Reason once, then classify scope and authority without side effects."""
        if (
            self._reasoning is None
            or snapshot is None
            or scope_policy is None
            or authority_policy is None
        ):
            return self._brain_result(
                TurnStageOutcome.FALLBACK,
                context,
                pre_brain_trace,
                notes="missing Slice-2 dependency or trusted input",
            )

        brain_input = BrainInput(
            current_utterance=snapshot.current_utterance[:8000],
            current_state=self._engine.machine.current_state,
            current_goal=_bounded_optional(snapshot.current_goal, 200),
            recent_turns=tuple(turn[:1000] for turn in snapshot.recent_turns[-6:]),
            business_intelligence=snapshot.business_intelligence,
            known_signals=frozenset(sorted(context.known_signals)[:32]),
            eligible_service_ids=context.eligible_alternative_service_ids[:32],
            offered_service_ids=context.offered_service_ids[:32],
            campaign_policy_summary=_bounded_optional(
                snapshot.campaign_policy_summary, 1000
            ),
            budget=budget_state,
            resolved_contact_context=_bounded_optional(
                snapshot.resolved_contact_context, 500
            ),
        )

        try:
            proposal = self._reasoning.reason(brain_input)
        except ReasoningError:
            return self._brain_result(
                TurnStageOutcome.FALLBACK,
                context,
                pre_brain_trace,
                provider_name=self._reasoning.name,
                proposal_status=BrainProposalStatus.PROVIDER_FAILED,
                notes="reasoning provider failed",
            )

        if not _well_formed_proposal(proposal):
            return self._brain_result(
                TurnStageOutcome.FALLBACK,
                context,
                pre_brain_trace,
                provider_name=self._reasoning.name,
                proposal_status=BrainProposalStatus.MALFORMED,
                notes="malformed BrainProposal",
            )

        if proposal.proposed_action is None:
            return self._brain_result(
                TurnStageOutcome.FALLBACK,
                context,
                pre_brain_trace,
                provider_name=self._reasoning.name,
                proposal_status=BrainProposalStatus.NO_ACTION,
                action_confidence=proposal.action_confidence,
                notes="no_action",
            )

        scope = self._scope.validate(
            ScopeCheckInput(
                proposal_action=proposal.proposed_action,
                proposal_topic=proposal.topic_category,
                policy=scope_policy,
            )
        )
        if scope.category != ScopeCategory.IN_SCOPE:
            outcome = (
                TurnStageOutcome.REDIRECT
                if scope.category == ScopeCategory.OUT_OF_SCOPE
                else TurnStageOutcome.FALLBACK
            )
            return self._brain_result(
                outcome,
                context,
                pre_brain_trace,
                provider_name=self._reasoning.name,
                proposal_status=BrainProposalStatus.VALID,
                scope_category=scope.category,
                action_confidence=proposal.action_confidence,
                scope_policy_id=scope.policy_id,
                notes=scope.rejected_reason,
            )

        authority = self._authority.classify(
            AuthorityCheckInput(
                proposal_action=proposal.proposed_action,
                commercial_request=proposal.commercial_request,
                policy=authority_policy,
            )
        )
        common = dict(
            provider_name=self._reasoning.name,
            proposal_status=BrainProposalStatus.VALID,
            scope_category=scope.category,
            authority_tier=authority.tier,
            action_confidence=proposal.action_confidence,
            scope_policy_id=scope.policy_id,
            authority_policy_id=authority.policy_id,
        )
        if authority.tier == AuthorityTier.HUMAN_APPROVAL_REQUIRED:
            return self._brain_result(
                TurnStageOutcome.ESCALATE, context, pre_brain_trace, **common
            )
        if authority.tier == AuthorityTier.DENIED:
            return self._brain_result(
                TurnStageOutcome.REDIRECT, context, pre_brain_trace, **common
            )

        bounds = (
            _effective_bounds(authority_policy, proposal.commercial_request)
            if authority.tier == AuthorityTier.POLICY_BOUNDED
            else None
        )
        return self._brain_result(
            TurnStageOutcome.AUTHORITY_APPROVED,
            context,
            pre_brain_trace,
            action=proposal.proposed_action,
            effective_bounds=bounds,
            **common,
        )

    @staticmethod
    def _brain_result(
        outcome: TurnStageOutcome,
        context: ConversationContext,
        pre_brain_trace: TurnDecisionTrace,
        *,
        provider_name: str | None = None,
        proposal_status: BrainProposalStatus | None = None,
        scope_category: ScopeCategory | None = None,
        authority_tier: AuthorityTier | None = None,
        action_confidence: float | None = None,
        notes: str | None = None,
        action: AgentAction | None = None,
        effective_bounds: EffectiveAuthorityBounds | None = None,
        scope_policy_id: str | None = None,
        authority_policy_id: str | None = None,
    ) -> TurnResult:
        trace = TurnDecisionTrace(
            stage_outcome=outcome,
            budget_outcome=pre_brain_trace.budget_outcome,
            response_ceiling=pre_brain_trace.response_ceiling,
            provider_name=provider_name,
            proposal_status=proposal_status,
            scope_category=scope_category,
            authority_tier=authority_tier,
            action_confidence=action_confidence,
            notes=notes,
        )
        return TurnResult(
            outcome=outcome,
            conversation_status=ConversationTerminationStatus.NOT_TERMINAL,
            trace=trace,
            execution=None,
            updated_context=context,
            action=action,
            effective_bounds=effective_bounds,
            scope_policy_id=scope_policy_id,
            authority_policy_id=authority_policy_id,
        )


def _bounded_optional(value: str | None, limit: int) -> str | None:
    """Return a deterministic bounded copy of an optional string."""
    return value[:limit] if value is not None else None


def _well_formed_proposal(proposal: object) -> bool:
    """Validate all fields of an untrusted dataclass result before policy use."""
    if not isinstance(proposal, BrainProposal):
        return False
    if not isinstance(proposal.detected_intent, Intent):
        return False
    if not isinstance(proposal.tone, Tone):
        return False
    if not isinstance(proposal.topic_category, TopicCategory):
        return False
    if proposal.proposed_action is not None and not isinstance(
        proposal.proposed_action, AgentAction
    ):
        return False
    if not isinstance(proposal.needs_service_decision, bool):
        return False
    if not isinstance(proposal.involves_contact, bool):
        return False
    if proposal.commercial_request is not None:
        request = proposal.commercial_request
        if not isinstance(request, CommercialRequest):
            return False
        if not isinstance(request.kind, CommercialRequestKind):
            return False
        percent = request.requested_discount_percent
        if percent is not None and (
            not isinstance(percent, (int, float))
            or isinstance(percent, bool)
            or not math.isfinite(percent)
            or percent < 0
        ):
            return False
    if proposal.contact_understanding is not None and not isinstance(
        proposal.contact_understanding, ContactUnderstanding
    ):
        return False
    if not isinstance(proposal.intelligence_updates, tuple) or not all(
        isinstance(item, (ObservedSignal, InferredSignal, UnknownSlot))
        for item in proposal.intelligence_updates
    ):
        return False
    if proposal.proposed_goal_update is not None and not isinstance(
        proposal.proposed_goal_update, str
    ):
        return False
    if proposal.reasoning is not None and not isinstance(proposal.reasoning, str):
        return False
    confidence = proposal.action_confidence
    return (
        isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and math.isfinite(confidence)
        and 0.0 <= confidence <= 1.0
    )


def _effective_bounds(
    policy: AuthorityPolicy,
    request: CommercialRequest | None,
) -> EffectiveAuthorityBounds:
    """Derive approval bounds exclusively from trusted authority policy."""
    if request is not None and request.kind == CommercialRequestKind.DISCOUNT:
        maximum = policy.discount.max_autonomous_percent if policy.discount else None
        return EffectiveAuthorityBounds(max_discount_percent=maximum)
    if request is not None and request.kind == CommercialRequestKind.PRICING_DISCLOSURE:
        allowed = policy.pricing_disclosure.allowed if policy.pricing_disclosure else False
        return EffectiveAuthorityBounds(pricing_disclosure_allowed=allowed)
    return EffectiveAuthorityBounds()
