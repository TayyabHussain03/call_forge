"""Slice 15 policy-to-planner-to-renderer integration scenarios."""

from __future__ import annotations

from app.brain.authority.models import AuthorityTier
from app.brain.scope.validator import ScopeCategory
from app.config.settings import get_settings
from app.contracts.conversation_context import ConversationContext
from app.conversation.context.builder import LeanContextBuildInput, LeanContextBuilder
from app.conversation.context.contracts import LeanTurnContext
from app.conversation.escalation.contracts import (
    AvailableEscalationCapabilities,
    EscalationRequest,
    GracefulEscalationInput,
    KnowledgeRequestKind,
    RecoveryMode,
)
from app.conversation.escalation.policy import GracefulEscalationPolicy
from app.conversation.prospect_intelligence.contracts import (
    ObservedProspectEvidence,
    ProspectEvidence,
    ProspectIntelligenceSnapshot,
)
from app.conversation.prospect_intelligence.updater import ProspectIntelligenceUpdater
from app.conversation.response_planning.contracts import (
    AuthoritativeResultKind,
    InterruptionCategory,
    InterruptionContext,
    PendingConversationIntent,
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
from app.conversation.strategy.contracts import (
    ConversationMode,
    ConversationStrategy,
    SalesStage,
    StrategyType,
)
from app.core.constants import ConversationState


def _context(
    message: str,
    *,
    prospect: ProspectIntelligenceSnapshot | None = None,
    pending: PendingConversationIntent | None = None,
) -> LeanTurnContext:
    strategy = ConversationStrategy(
        SalesStage.DISCOVERY,
        ConversationMode.QUESTION_DETOUR,
        "handle the detour without losing the discovery goal",
        StrategyType.HANDLE_QUESTION,
        resume_previous_goal=pending is not None,
    )
    return LeanContextBuilder().build(
        LeanContextBuildInput(
            call_id="call",
            current_turn_id="turn-2",
            current_turn_sequence=2,
            current_user_message=message,
            current_state=ConversationState.LISTEN,
            conversation_context=ConversationContext("call"),
            strategy=strategy,
            prospect_intelligence=prospect,
            pending_intent=pending,
            interruption=InterruptionContext(
                pending is not None,
                InterruptionCategory.QUESTION,
                pending,
            ),
            conversation_category=InterruptionCategory.QUESTION,
        )
    )


def _run(
    context,  # type: ignore[no-untyped-def]
    request: EscalationRequest,
    *,
    capabilities: AvailableEscalationCapabilities = (
        AvailableEscalationCapabilities()
    ),
    authority: AuthorityTier = AuthorityTier.AUTO_EXECUTE,
    result: AuthoritativeResultKind = AuthoritativeResultKind.EXECUTED,
):  # type: ignore[no-untyped-def]
    decision = GracefulEscalationPolicy().evaluate(
        GracefulEscalationInput(
            context,
            request,
            capabilities,
            ScopeCategory.IN_SCOPE,
            authority,
        )
    )
    plan = ResponsePlanner().plan(
        ResponsePlanningInput(
            ConversationState.LISTEN,
            result,
            context.untrusted_user_input.message,
            conversation_category=InterruptionCategory.QUESTION,
            interruption=context.interruption,
            escalation_decision=decision,
        )
    )
    rendered = DeterministicResponseRenderer().render(
        ResponseRenderInput(plan, result, ResponseRenderingBudget(4))
    )
    return decision, plan, rendered


def test_scenario_a_technical_question_without_evidence_is_honestly_unknown() -> None:
    context = _context("Does it integrate with SAP ECC 6.0?")
    decision, _, rendered = _run(
        context,
        EscalationRequest(
            KnowledgeRequestKind.TECHNICAL_DETAIL,
            fact_key="sap_ecc_6_integration",
        ),
    )
    assert decision.recovery_mode == RecoveryMode.ACKNOWLEDGE_UNKNOWN
    assert "confirmed detail" in rendered.text
    assert "SAP ECC" not in rendered.text
    assert "engineering" not in rendered.text.lower()


def test_scenario_b_technical_follow_up_is_offered_but_not_confirmed() -> None:
    context = _context("Does it integrate with SAP ECC 6.0?")
    decision, _, rendered = _run(
        context,
        EscalationRequest(
            KnowledgeRequestKind.TECHNICAL_DETAIL,
            fact_key="sap_ecc_6_integration",
        ),
        capabilities=AvailableEscalationCapabilities(
            technical_followup_available=True
        ),
    )
    assert decision.recovery_mode == RecoveryMode.OFFER_INFORMATION_FOLLOW_UP
    assert "may be requested" in rendered.text
    assert "not confirmed" in rendered.text


def test_scenario_c_human_request_without_handoff_never_claims_transfer() -> None:
    prospect = ProspectIntelligenceUpdater().update(
        ProspectIntelligenceSnapshot(),
        ProspectEvidence(
            observed=ObservedProspectEvidence("turn-2", human_request=True)
        ),
    )
    context = _context("I want to talk to a real person.", prospect=prospect)
    decision, _, rendered = _run(context, EscalationRequest())
    assert decision.recovery_mode == RecoveryMode.ACKNOWLEDGE_UNKNOWN
    assert "can't transfer" in rendered.text
    assert "transfer you now" not in rendered.text


def test_scenario_d_commercial_request_neither_commits_nor_moves_fsm() -> None:
    machine = ConversationStateMachine(
        load_config(get_settings().conversation_config_path),
        ConversationState.LISTEN,
    )
    before = machine.current_state
    context = _context("Give me 25% discount and I'll sign today.")
    decision, _, rendered = _run(
        context,
        EscalationRequest(KnowledgeRequestKind.COMMERCIAL),
        authority=AuthorityTier.HUMAN_APPROVAL_REQUIRED,
        result=AuthoritativeResultKind.ESCALATE,
    )
    assert "can't authorize or commit" in rendered.text
    assert "25%" not in rendered.text
    assert machine.current_state == before
    assert not decision.answerable


def test_scenario_e_unknown_detour_resumes_only_relevant_discovery_goal() -> None:
    pending = PendingConversationIntent(
        "discover workflow",
        "how the current intake workflow operates.",
    )
    context = _context(
        "Does it support a proprietary protocol?",
        pending=pending,
    )
    decision, plan, rendered = _run(
        context,
        EscalationRequest(
            KnowledgeRequestKind.TECHNICAL_DETAIL,
            fact_key="proprietary_protocol",
        ),
    )
    assert decision.resume_previous_goal
    assert plan.resume_previous_point
    assert "return to how the current intake workflow operates" in rendered.text
