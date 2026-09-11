"""Behavioral tests for BrainOrchestrator Slice 2."""

from __future__ import annotations

from dataclasses import replace

from app.brain.authority.models import (
    AuthorityPolicy,
    AuthorityTier,
    DiscountRule,
    PricingDisclosureRule,
)
from app.brain.authority.validator import AuthorityPolicyValidator
from app.brain.budget.evaluator import BudgetPolicyEvaluator, TrustedExitSignals
from app.brain.budget.models import BudgetLimits, BudgetPolicy
from app.brain.business_intelligence import BusinessIntelligence, Provenance, SourceKind
from app.brain.contracts import (
    BrainProposal,
    BudgetState,
    BusinessIntelligenceSnapshot,
    CommercialRequest,
)
from app.brain.orchestrator.orchestrator import (
    BrainOrchestrator,
    BrainProposalStatus,
    BrainSnapshotInput,
    ConversationTerminationStatus,
    TurnStageOutcome,
)
from app.brain.scope.models import ScopePolicy
from app.brain.scope.validator import ScopePolicyValidator
from app.config.settings import get_settings
from app.contracts.conversation_context import ConversationContext
from app.conversation.engine import ConversationEngine
from app.conversation.guardrails.action_validator import ActionValidator
from app.conversation.guardrails.priority import TrustedPriorityOutcome
from app.conversation.state_machine.machine import ConversationStateMachine
from app.conversation.state_machine.states import load_config
from app.core.constants import (
    AgentAction,
    CommercialRequestKind,
    Intent,
    TopicCategory,
)
from app.llm.providers.reasoning_provider import MockReasoningProvider, ReasoningProvider


class CountingScopeValidator(ScopePolicyValidator):
    def __init__(self) -> None:
        self.calls = 0

    def validate(self, check):  # type: ignore[no-untyped-def]
        self.calls += 1
        return super().validate(check)


class CountingAuthorityValidator(AuthorityPolicyValidator):
    def __init__(self) -> None:
        self.calls = 0

    def classify(self, check):  # type: ignore[no-untyped-def]
        self.calls += 1
        return super().classify(check)


class MalformedProvider(ReasoningProvider):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "malformed"

    def reason(self, brain_input):  # type: ignore[no-untyped-def]
        self.calls += 1
        return object()


def _engine() -> ConversationEngine:
    config = load_config(get_settings().conversation_config_path)
    return ConversationEngine(
        ConversationStateMachine(config),
        ActionValidator(config),
    )


def _scope(*, allow_topic: TopicCategory = TopicCategory.SERVICE_DISCUSSION) -> ScopePolicy:
    return ScopePolicy(
        policy_id="scope",
        allowed_topics=frozenset({allow_topic}),
        allowed_actions=frozenset(AgentAction),
        restricted_topics=frozenset({TopicCategory.OFF_TOPIC}),
    )


def _authority(tier: AuthorityTier = AuthorityTier.AUTO_EXECUTE) -> AuthorityPolicy:
    return AuthorityPolicy(
        policy_id="authority",
        default_tier=AuthorityTier.HUMAN_APPROVAL_REQUIRED,
        action_tiers={AgentAction.GREET: tier},
        discount=DiscountRule(
            max_autonomous_percent=10,
            within_tier=AuthorityTier.POLICY_BOUNDED,
            over_tier=AuthorityTier.HUMAN_APPROVAL_REQUIRED,
        ),
        pricing_disclosure=PricingDisclosureRule(
            allowed=True,
            tier=AuthorityTier.POLICY_BOUNDED,
        ),
    )


def _proposal(
    *,
    action: AgentAction | None = AgentAction.GREET,
    topic: TopicCategory = TopicCategory.SERVICE_DISCUSSION,
    commercial: CommercialRequest | None = None,
    goal: str | None = None,
) -> BrainProposal:
    return BrainProposal(
        detected_intent=Intent.INTERESTED,
        topic_category=topic,
        proposed_action=action,
        commercial_request=commercial,
        proposed_goal_update=goal,
        action_confidence=0.01,
    )


def _run(
    provider: ReasoningProvider,
    *,
    proposal_scope: ScopePolicy | None = None,
    authority: AuthorityPolicy | None = None,
    scope_validator: ScopePolicyValidator | None = None,
    authority_validator: AuthorityPolicyValidator | None = None,
    context: ConversationContext | None = None,
    snapshot: BrainSnapshotInput | None = None,
):
    orchestrator = BrainOrchestrator(
        _engine(),
        BudgetPolicyEvaluator(),
        reasoning_provider=provider,
        scope_validator=scope_validator,
        authority_validator=authority_validator,
    )
    return orchestrator, orchestrator.process_turn(
        context or ConversationContext("call"),
        TrustedPriorityOutcome.NONE,
        BudgetState(5, 120, 5, 2),
        BudgetPolicy("budget", BudgetLimits(5, 10, 180, 2)),
        TrustedExitSignals(),
        snapshot or BrainSnapshotInput("hello"),
        proposal_scope or _scope(),
        authority or _authority(),
    )


def test_auto_execute_is_authority_approved_without_execution() -> None:
    provider = MockReasoningProvider(default=_proposal())
    orchestrator, result = _run(provider)

    assert result.outcome == TurnStageOutcome.AUTHORITY_APPROVED
    assert result.action == AgentAction.GREET
    assert result.effective_bounds is None
    assert result.scope_policy_id == "scope"
    assert result.authority_policy_id == "authority"
    assert result.execution is None
    assert orchestrator._engine.machine.current_state.value == "new_call"


def test_policy_bounded_uses_only_policy_derived_effective_bounds() -> None:
    request = CommercialRequest(CommercialRequestKind.DISCOUNT, 5)
    provider = MockReasoningProvider(default=_proposal(commercial=request))
    _, result = _run(provider)

    assert result.outcome == TurnStageOutcome.AUTHORITY_APPROVED
    assert result.effective_bounds is not None
    assert result.effective_bounds.max_discount_percent == 10
    assert result.effective_bounds.max_discount_percent != request.requested_discount_percent


def test_human_approval_required_escalates_without_execution() -> None:
    _, result = _run(
        MockReasoningProvider(default=_proposal()),
        authority=_authority(AuthorityTier.HUMAN_APPROVAL_REQUIRED),
    )

    assert result.outcome == TurnStageOutcome.ESCALATE
    assert result.execution is None
    assert result.action is None


def test_denied_redirects_without_execution() -> None:
    _, result = _run(
        MockReasoningProvider(default=_proposal()),
        authority=_authority(AuthorityTier.DENIED),
    )

    assert result.outcome == TurnStageOutcome.REDIRECT
    assert result.execution is None
    assert result.action is None


def test_provider_failure_skips_scope_and_authority() -> None:
    provider = MockReasoningProvider(should_fail=True)
    scope = CountingScopeValidator()
    authority = CountingAuthorityValidator()
    _, result = _run(provider, scope_validator=scope, authority_validator=authority)

    assert result.outcome == TurnStageOutcome.FALLBACK
    assert provider.call_count == 1
    assert scope.calls == 0
    assert authority.calls == 0


def test_malformed_proposal_skips_scope_and_authority() -> None:
    provider = MalformedProvider()
    scope = CountingScopeValidator()
    authority = CountingAuthorityValidator()
    _, result = _run(provider, scope_validator=scope, authority_validator=authority)

    assert result.outcome == TurnStageOutcome.FALLBACK
    assert provider.calls == 1
    assert scope.calls == 0
    assert authority.calls == 0


def test_no_action_is_safe_fallback_and_skips_policies() -> None:
    provider = MockReasoningProvider(default=_proposal(action=None))
    scope = CountingScopeValidator()
    authority = CountingAuthorityValidator()
    _, result = _run(provider, scope_validator=scope, authority_validator=authority)

    assert result.outcome == TurnStageOutcome.FALLBACK
    assert result.trace.proposal_status == BrainProposalStatus.NO_ACTION
    assert scope.calls == 0
    assert authority.calls == 0


def test_out_of_scope_redirects_and_skips_authority() -> None:
    provider = MockReasoningProvider(default=_proposal(topic=TopicCategory.OFF_TOPIC))
    authority = CountingAuthorityValidator()
    _, result = _run(provider, authority_validator=authority)

    assert result.outcome == TurnStageOutcome.REDIRECT
    assert authority.calls == 0


def test_unknown_scope_falls_back_and_skips_authority() -> None:
    provider = MockReasoningProvider(default=_proposal(topic=TopicCategory.UNKNOWN))
    authority = CountingAuthorityValidator()
    _, result = _run(provider, authority_validator=authority)

    assert result.outcome == TurnStageOutcome.FALLBACK
    assert authority.calls == 0


def test_normal_path_calls_provider_scope_and_authority_exactly_once() -> None:
    provider = MockReasoningProvider(default=_proposal())
    scope = CountingScopeValidator()
    authority = CountingAuthorityValidator()
    _, result = _run(provider, scope_validator=scope, authority_validator=authority)

    assert result.outcome == TurnStageOutcome.AUTHORITY_APPROVED
    assert provider.call_count == 1
    assert scope.calls == 1
    assert authority.calls == 1


def test_slice_two_does_not_mutate_context_state_or_business_intelligence() -> None:
    intelligence = BusinessIntelligence().record_observed(
        "has_site", "yes", Provenance(SourceKind.CLIENT_STATED, source_turn=1)
    )
    snapshot_bi = BusinessIntelligenceSnapshot(observed=intelligence.observed)
    snapshot = BrainSnapshotInput(
        "hello",
        business_intelligence=snapshot_bi,
        recent_turns=("a", "b"),
    )
    context = ConversationContext(
        "call",
        known_signals=frozenset({"has_site"}),
        offered_service_ids=("service-a",),
    )
    provider = MockReasoningProvider(default=_proposal(goal="changed-goal"))
    orchestrator, result = _run(provider, context=context, snapshot=snapshot)

    assert result.updated_context is context
    assert orchestrator._engine.machine.current_state.value == "new_call"
    assert snapshot.business_intelligence == snapshot_bi
    assert intelligence.observed == snapshot_bi.observed
    assert not hasattr(result.updated_context, "current_goal")


def test_action_confidence_is_observational_only() -> None:
    proposal = replace(_proposal(), action_confidence=0.0)
    _, result = _run(MockReasoningProvider(default=proposal))

    assert result.outcome == TurnStageOutcome.AUTHORITY_APPROVED
    assert result.trace.action_confidence == 0.0


def test_deterministic_replay_produces_equal_classification() -> None:
    first = _run(MockReasoningProvider(default=_proposal()))[1]
    second = _run(MockReasoningProvider(default=_proposal()))[1]

    assert first == second


def test_brain_input_is_bounded_and_immutable_snapshot() -> None:
    captured = []

    class CaptureProvider(MockReasoningProvider):
        def reason(self, brain_input):  # type: ignore[no-untyped-def]
            captured.append(brain_input)
            return super().reason(brain_input)

    snapshot = BrainSnapshotInput(
        current_utterance="u" * 9000,
        current_goal="g" * 300,
        recent_turns=tuple(str(i) * 1100 for i in range(8)),
    )
    _run(CaptureProvider(default=_proposal()), snapshot=snapshot)

    brain_input = captured[0]
    assert len(brain_input.current_utterance) == 8000
    assert len(brain_input.current_goal) == 200
    assert len(brain_input.recent_turns) == 6
    assert all(len(turn) <= 1000 for turn in brain_input.recent_turns)
    assert brain_input.current_state.value == "new_call"
    assert brain_input.recent_turns[0].startswith("2")


def test_missing_slice_two_inputs_fail_closed_before_provider() -> None:
    provider = MockReasoningProvider(default=_proposal())
    orchestrator = BrainOrchestrator(
        _engine(), BudgetPolicyEvaluator(), reasoning_provider=provider
    )
    result = orchestrator.process_turn(
        ConversationContext("call"),
        TrustedPriorityOutcome.NONE,
        BudgetState(5, 120, 5, 2),
        BudgetPolicy("budget", BudgetLimits(5, 10, 180, 2)),
        TrustedExitSignals(),
    )

    assert result.outcome == TurnStageOutcome.FALLBACK
    assert provider.call_count == 0
    assert result.conversation_status == ConversationTerminationStatus.NOT_TERMINAL
