"""Focused tests for evidence-aware graceful escalation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace

import pytest

from app.brain.authority.models import AuthorityTier
from app.brain.scope.validator import ScopeCategory
from app.contracts.conversation_context import ConversationContext
from app.conversation.context.builder import LeanContextBuildInput, LeanContextBuilder
from app.conversation.context.contracts import (
    ApprovedEvidenceItem,
    ApprovedEvidenceSourceKind,
    EvidenceScope,
    EvidenceScopeKind,
    EvidenceType,
    LeanTurnContext,
)
from app.conversation.escalation.contracts import (
    AvailableEscalationCapabilities,
    DisclosureStatus,
    EscalationCapability,
    EscalationDecision,
    EscalationReason,
    EscalationRequest,
    GracefulEscalationInput,
    KnowledgeRequestKind,
    RecoveryMode,
    RequestedFollowUp,
)
from app.conversation.escalation.evidence import ApprovedEvidenceMatcher
from app.conversation.escalation.policy import GracefulEscalationPolicy
from app.conversation.prospect_intelligence.contracts import (
    InferenceCandidate,
    InferredProspectEvidence,
    ObservedProspectEvidence,
    PreferredNextStep,
    ProspectEvidence,
    ProspectIntelligenceSnapshot,
    ProspectRole,
)
from app.conversation.prospect_intelligence.updater import ProspectIntelligenceUpdater
from app.conversation.response_planning.contracts import (
    AddresseeStatus,
    AuthoritativeResultKind,
    ConversationMove,
    InterruptionCategory,
    InterruptionContext,
    PendingConversationIntent,
    ResponsePlanningInput,
)
from app.conversation.response_planning.planner import ResponsePlanner
from app.conversation.response_rendering.contracts import (
    ResponseRenderInput,
    ResponseRenderingBudget,
    TrustedRenderingContext,
)
from app.conversation.response_rendering.renderer import (
    DeterministicResponseRenderer,
    RenderValidationError,
    validate_rendered_text,
)
from app.conversation.strategy.contracts import (
    ConversationMode,
    ConversationStrategy,
    SalesStage,
    StrategyType,
)
from app.conversation.supervisor.contracts import SupervisorInsight
from app.core.constants import ConversationState


def _strategy() -> ConversationStrategy:
    return ConversationStrategy(
        SalesStage.DISCOVERY,
        ConversationMode.QUESTION_DETOUR,
        "answer the current question safely",
        StrategyType.HANDLE_QUESTION,
    )


def _evidence(
    *,
    evidence_id: str = "evidence-1",
    fact_key: str = "appointment_reminders",
    statement: str = "The service supports automated appointment reminders.",
    evidence_type: EvidenceType = EvidenceType.TECHNICAL_FACT,
    scope: EvidenceScope | None = None,
) -> ApprovedEvidenceItem:
    return ApprovedEvidenceItem(
        evidence_id,
        fact_key,
        evidence_type,
        statement,
        ApprovedEvidenceSourceKind.CURATED_SERVICE,
        scope or EvidenceScope(EvidenceScopeKind.SERVICE, "service-a"),
        "approved-source-1",
    )


def _context(
    *,
    evidence: tuple[ApprovedEvidenceItem, ...] = (),
    prospect: ProspectIntelligenceSnapshot | None = None,
    contact_candidate: str | None = None,
    contact_confirmed: bool = False,
    pending: PendingConversationIntent | None = None,
    strategy: ConversationStrategy | None = None,
) -> LeanTurnContext:
    return LeanContextBuilder().build(
        LeanContextBuildInput(
            call_id="call",
            current_turn_id="turn-2",
            current_turn_sequence=2,
            current_user_message="Does that work with our system?",
            current_state=ConversationState.LISTEN,
            conversation_context=ConversationContext(
                "call",
                campaign_id="campaign-a",
                contact_candidate=contact_candidate,
                contact_candidate_channel="email" if contact_candidate else None,
                contact_confirmed=contact_confirmed,
                eligible_alternative_service_ids=("service-a", "service-b"),
            ),
            strategy=strategy or _strategy(),
            prospect_intelligence=prospect,
            pending_intent=pending,
            approved_evidence=evidence,
            interruption=InterruptionContext(
                was_interrupted=pending is not None,
                category=InterruptionCategory.QUESTION,
                previous_intent=pending,
            ),
            conversation_category=InterruptionCategory.QUESTION,
            addressee_status=AddresseeStatus.ADDRESSED_TO_AGENT,
        )
    )


def _request(
    kind: KnowledgeRequestKind = KnowledgeRequestKind.TECHNICAL_DETAIL,
    **changes: object,
) -> EscalationRequest:
    return EscalationRequest(kind=kind, **changes)  # type: ignore[arg-type]


def _input(
    *,
    context: LeanTurnContext | None = None,
    request: EscalationRequest | None = None,
    capabilities: AvailableEscalationCapabilities | None = None,
    scope: ScopeCategory | None = ScopeCategory.IN_SCOPE,
    authority: AuthorityTier | None = AuthorityTier.AUTO_EXECUTE,
) -> GracefulEscalationInput:
    return GracefulEscalationInput(
        context or _context(),
        request or _request(fact_key="missing"),
        capabilities or AvailableEscalationCapabilities(),
        scope,
        authority,
    )


def _decide(**changes: object) -> EscalationDecision:
    return GracefulEscalationPolicy().evaluate(_input(**changes))  # type: ignore[arg-type]


def _render(
    decision: EscalationDecision,
    context: LeanTurnContext | None = None,
    result: AuthoritativeResultKind = AuthoritativeResultKind.EXECUTED,
) -> str:
    context = context or _context()
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
    return DeterministicResponseRenderer().render(
        ResponseRenderInput(
            plan,
            result,
            ResponseRenderingBudget(4),
            TrustedRenderingContext(approved_evidence=context.approved_evidence),
        )
    ).text


def _human_prospect(value: bool = True) -> ProspectIntelligenceSnapshot:
    return ProspectIntelligenceUpdater().update(
        ProspectIntelligenceSnapshot(),
        ProspectEvidence(
            observed=ObservedProspectEvidence("turn-1", human_request=value)
        ),
    )


def test_known_fact_with_approved_evidence_can_be_answered() -> None:
    item = _evidence()
    decision = _decide(
        context=_context(evidence=(item,)),
        request=_request(
            fact_key=item.fact_key,
            evidence_type=item.evidence_type,
            service_id="service-a",
        ),
    )
    assert decision.recovery_mode == RecoveryMode.ANSWER_WITH_EVIDENCE
    assert decision.evidence_ids == (item.evidence_id,)


def test_missing_evidence_yields_unknown() -> None:
    decision = _decide()
    assert decision.reason == EscalationReason.UNKNOWN_TECHNICAL_DETAIL
    assert decision.recovery_mode == RecoveryMode.ACKNOWLEDGE_UNKNOWN


def test_unknown_technical_detail_does_not_hallucinate() -> None:
    text = _render(_decide())
    assert "don't have enough confirmed detail" in text
    assert "supports" not in text


def test_unknown_product_feature_does_not_hallucinate() -> None:
    decision = _decide(request=_request(KnowledgeRequestKind.PRODUCT_CAPABILITY))
    assert decision.reason == EscalationReason.UNKNOWN_PRODUCT_CAPABILITY
    assert not decision.answerable


def test_technical_follow_up_unavailable_prevents_promise() -> None:
    decision = _decide()
    assert decision.capability_required is None
    assert "follow-up" not in _render(decision)


def test_technical_follow_up_available_allows_offer_only() -> None:
    decision = _decide(
        capabilities=AvailableEscalationCapabilities(
            technical_followup_available=True
        )
    )
    assert decision.recovery_mode == RecoveryMode.OFFER_INFORMATION_FOLLOW_UP
    assert decision.capability_required == EscalationCapability.TECHNICAL_REVIEW
    assert "not confirmed" in _render(decision)


def test_human_requested_without_handoff_does_not_promise_transfer() -> None:
    decision = _decide(
        context=_context(prospect=_human_prospect()),
        request=EscalationRequest(),
    )
    text = _render(decision, _context(prospect=_human_prospect()))
    assert decision.reason == EscalationReason.HUMAN_REQUESTED
    assert "can't transfer" in text
    assert "transfer you now" not in text


def test_human_requested_with_handoff_allows_offer_only() -> None:
    context = _context(prospect=_human_prospect())
    decision = _decide(
        context=context,
        request=EscalationRequest(),
        capabilities=AvailableEscalationCapabilities(human_handoff_available=True),
    )
    assert decision.recovery_mode == RecoveryMode.OFFER_HUMAN_FOLLOW_UP
    assert "not confirmed" in _render(decision, context)


def test_callback_unavailable_prevents_callback_promise() -> None:
    decision = _decide(
        request=_request(requested_follow_up=RequestedFollowUp.CALLBACK)
    )
    assert decision.reason == EscalationReason.CAPABILITY_UNAVAILABLE
    assert "isn't available" in _render(decision)


def test_callback_available_allows_unconfirmed_offer() -> None:
    decision = _decide(
        request=_request(requested_follow_up=RequestedFollowUp.CALLBACK),
        capabilities=AvailableEscalationCapabilities(
            callback_workflow_available=True
        ),
    )
    text = _render(decision)
    assert decision.recovery_mode == RecoveryMode.OFFER_CALLBACK
    assert "not scheduled" in text


def test_email_unavailable_prevents_email_promise() -> None:
    decision = _decide(
        request=_request(requested_follow_up=RequestedFollowUp.EMAIL)
    )
    assert "Email follow-up isn't available" in _render(decision)


def test_email_available_without_confirmed_contact_does_not_promise_send() -> None:
    context = _context(contact_candidate="candidate@example.com")
    decision = _decide(
        context=context,
        request=_request(requested_follow_up=RequestedFollowUp.EMAIL),
        capabilities=AvailableEscalationCapabilities(email_followup_available=True),
    )
    text = _render(decision, context)
    assert decision.recovery_mode == RecoveryMode.ASK_CLARIFYING_QUESTION
    assert "confirmed contact method is needed" in text
    assert "will send" not in text


def test_approved_case_study_metric_remains_scoped() -> None:
    item = _evidence(
        fact_key="case_metric",
        statement="One approved case study reported a 12% improvement.",
        evidence_type=EvidenceType.CASE_STUDY,
    )
    context = _context(evidence=(item,))
    decision = _decide(
        context=context,
        request=_request(
            KnowledgeRequestKind.FACT,
            fact_key="case_metric",
            evidence_type=EvidenceType.CASE_STUDY,
            service_id="service-a",
        ),
    )
    assert "12%" in _render(decision, context)


def test_case_study_is_not_generalized_into_guarantee() -> None:
    item = _evidence(
        fact_key="case_metric",
        statement="One approved case study reported a 12% improvement.",
        evidence_type=EvidenceType.CASE_STUDY,
    )
    context = _context(evidence=(item,))
    decision = _decide(
        context=context,
        request=_request(
            KnowledgeRequestKind.FACT,
            fact_key="case_metric",
            evidence_type=EvidenceType.CASE_STUDY,
            service_id="service-a",
        ),
    )
    text = _render(decision, context)
    assert "you will" not in text.lower() and "guarantee" not in text.lower()


def test_conflicting_evidence_yields_safe_unknown() -> None:
    first = _evidence(evidence_id="a", statement="Supported.")
    second = _evidence(evidence_id="b", statement="Not supported.")
    context = replace(_context(), approved_evidence=(first, second))
    decision = _decide(
        context=context,
        request=_request(fact_key=first.fact_key, service_id="service-a"),
    )
    assert decision.reason == EscalationReason.EVIDENCE_CONFLICT
    assert decision.recovery_mode == RecoveryMode.ACKNOWLEDGE_UNKNOWN


def test_service_a_evidence_cannot_answer_service_b_fact() -> None:
    item = _evidence()
    decision = _decide(
        context=_context(evidence=(item,)),
        request=_request(fact_key=item.fact_key, service_id="service-b"),
    )
    assert not decision.answerable


def test_policy_restricted_fact_is_not_disclosed() -> None:
    item = _evidence()
    decision = _decide(
        context=_context(evidence=(item,)),
        request=_request(
            fact_key=item.fact_key,
            service_id="service-a",
            disclosure_status=DisclosureStatus.RESTRICTED,
        ),
    )
    assert decision.reason == EscalationReason.POLICY_RESTRICTED
    assert item.statement not in _render(decision, _context(evidence=(item,)))


def test_authority_denial_remains_authoritative() -> None:
    decision = _decide(authority=AuthorityTier.DENIED)
    assert decision.reason == EscalationReason.POLICY_RESTRICTED
    assert decision.recovery_mode == RecoveryMode.SAFE_REDIRECT


def test_discount_requiring_human_approval_remains_non_committal() -> None:
    decision = _decide(
        request=_request(KnowledgeRequestKind.COMMERCIAL),
        authority=AuthorityTier.HUMAN_APPROVAL_REQUIRED,
    )
    text = _render(decision, result=AuthoritativeResultKind.ESCALATE)
    assert decision.reason == EscalationReason.COMMERCIAL_AUTHORITY_REQUIRED
    assert "can't authorize or commit" in text
    assert "%" not in text


def test_out_of_scope_uses_safe_redirect() -> None:
    decision = _decide(scope=ScopeCategory.OUT_OF_SCOPE)
    assert decision.reason == EscalationReason.OUT_OF_SCOPE
    assert decision.recovery_mode == RecoveryMode.SAFE_REDIRECT


def test_unknown_scope_clarifies() -> None:
    decision = _decide(scope=ScopeCategory.UNKNOWN_SCOPE)
    assert decision.recovery_mode == RecoveryMode.ASK_CLARIFYING_QUESTION


def test_ambiguous_question_clarifies_before_technical_escalation() -> None:
    decision = _decide(request=_request(KnowledgeRequestKind.AMBIGUOUS))
    assert decision.reason == EscalationReason.AMBIGUOUS_REQUEST
    assert decision.recovery_mode == RecoveryMode.ASK_CLARIFYING_QUESTION


def test_supervisor_inference_alone_cannot_trigger_handoff() -> None:
    insight = SupervisorInsight(
        "call",
        "turn-1",
        1,
        ProspectEvidence(
            inferred=InferredProspectEvidence(
                "turn-1", likely_role=InferenceCandidate(ProspectRole.OWNER, 1.0)
            )
        ),
    )
    source = LeanContextBuildInput(
        call_id="call",
        current_turn_id="turn-2",
        current_turn_sequence=2,
        current_user_message="question",
        current_state=ConversationState.LISTEN,
        conversation_context=ConversationContext("call"),
        supervisor_insight=insight,
    )
    context = LeanContextBuilder().build(source)
    decision = _decide(context=context, request=EscalationRequest())
    assert decision.reason == EscalationReason.NONE


def test_high_confidence_inference_cannot_create_capability() -> None:
    context = _context(prospect=_human_prospect(False))
    decision = _decide(context=context)
    assert decision.capability_available is None


def test_contact_candidate_is_not_confirmed_follow_up_channel() -> None:
    context = _context(contact_candidate="candidate@example.com")
    decision = _decide(
        context=context,
        request=_request(requested_follow_up=RequestedFollowUp.EMAIL),
        capabilities=AvailableEscalationCapabilities(email_followup_available=True),
    )
    assert decision.recovery_mode == RecoveryMode.ASK_CLARIFYING_QUESTION


def test_callback_recommendation_does_not_confirm_callback() -> None:
    decision = _decide(
        request=_request(requested_follow_up=RequestedFollowUp.CALLBACK),
        capabilities=AvailableEscalationCapabilities(
            callback_workflow_available=True
        ),
    )
    names = {item.name for item in fields(decision)}
    assert "callback_confirmed" not in names and "scheduled" not in names


def test_escalation_decision_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        _decide().reason = EscalationReason.NONE  # type: ignore[misc]


def test_same_input_produces_same_decision() -> None:
    policy_input = _input()
    policy = GracefulEscalationPolicy()
    assert policy.evaluate(policy_input) == policy.evaluate(policy_input)


def test_policy_failure_path_creates_no_supported_claim() -> None:
    class BrokenMatcher(ApprovedEvidenceMatcher):
        def match(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("unexpected")

    decision = GracefulEscalationPolicy(BrokenMatcher()).evaluate(_input())
    assert decision.recovery_mode == RecoveryMode.ACKNOWLEDGE_UNKNOWN
    assert decision.evidence_ids == ()


def test_renderer_preserves_uncertainty() -> None:
    text = _render(_decide())
    assert "confirmed detail" in text and "definitely" not in text


@pytest.mark.parametrize(
    "follow_up",
    (
        RequestedFollowUp.HUMAN_HANDOFF,
        RequestedFollowUp.CALLBACK,
        RequestedFollowUp.EMAIL,
        RequestedFollowUp.MESSAGE,
        RequestedFollowUp.TECHNICAL_REVIEW,
    ),
)
def test_renderer_never_promises_unavailable_action(
    follow_up: RequestedFollowUp,
) -> None:
    text = _render(_decide(request=_request(requested_follow_up=follow_up))).lower()
    forbidden = (
        "engineering will confirm",
        "team will email",
        "i'll send",
        "i'll transfer",
        "i'll book",
    )
    assert all(value not in text for value in forbidden)


@pytest.mark.parametrize(
    "candidate",
    (
        "I'll transfer you now.",
        "Our team will email you.",
        "I'll send that shortly.",
        "I'll book that for you.",
        "Engineering will confirm it.",
        "The callback is scheduled.",
    ),
)
def test_render_validation_rejects_confirmed_escalation_promises(
    candidate: str,
) -> None:
    decision = _decide()
    plan = ResponsePlanner().plan(
        ResponsePlanningInput(
            ConversationState.LISTEN,
            AuthoritativeResultKind.EXECUTED,
            "question",
            escalation_decision=decision,
        )
    )
    with pytest.raises(RenderValidationError):
        validate_rendered_text(
            candidate,
            ResponseRenderInput(plan, AuthoritativeResultKind.EXECUTED, ResponseRenderingBudget(4)),
        )


def test_planner_resumes_only_relevant_previous_goal() -> None:
    pending = PendingConversationIntent("discover", "the current workflow")
    context = _context(pending=pending)
    decision = _decide(context=context)
    plan = ResponsePlanner().plan(
        ResponsePlanningInput(
            ConversationState.LISTEN,
            AuthoritativeResultKind.EXECUTED,
            "question",
            interruption=context.interruption,
            escalation_decision=decision,
        )
    )
    assert plan.resume_previous_point and plan.pending_intent is pending


def test_stale_previous_goal_is_not_resumed() -> None:
    pending = PendingConversationIntent(
        "discover", "old point", still_relevant=False
    )
    context = _context(pending=pending)
    decision = _decide(context=context)
    assert not decision.resume_previous_goal


def test_brain_answer_proposal_cannot_override_missing_evidence() -> None:
    decision = _decide()
    assert decision.recovery_mode == RecoveryMode.ACKNOWLEDGE_UNKNOWN
    assert not decision.answerable


def test_strategy_cannot_override_missing_evidence() -> None:
    strategy = replace(_strategy(), communication_goal="answer confidently")
    decision = _decide(context=_context(strategy=strategy))
    assert decision.recovery_mode == RecoveryMode.ACKNOWLEDGE_UNKNOWN


def test_operational_provider_pricing_remains_absent() -> None:
    names = {item.name for item in fields(GracefulEscalationInput)}
    assert names.isdisjoint({"provider_pricing", "token_cost", "telephony_cost"})


def test_raw_secrets_remain_absent() -> None:
    names = {item.name for item in fields(GracefulEscalationInput)}
    assert names.isdisjoint({"api_key", "credentials", "environment", "secrets"})


def test_approved_evidence_provenance_is_preserved() -> None:
    item = _evidence()
    context = _context(evidence=(item,))
    decision = _decide(
        context=context,
        request=_request(fact_key=item.fact_key, service_id="service-a"),
    )
    matched = next(
        value
        for value in context.approved_evidence
        if value.evidence_id in decision.evidence_ids
    )
    assert matched.source_reference == "approved-source-1"


def test_decision_contains_no_authority_or_execution_fields() -> None:
    names = {item.name for item in fields(EscalationDecision)}
    assert names.isdisjoint(
        {
            "next_state",
            "execute_action",
            "confirmed_followup",
            "confirmed_transfer",
            "pricing",
            "discount",
        }
    )


def test_explicit_wrap_up_preference_is_communication_only() -> None:
    prospect = ProspectIntelligenceUpdater().update(
        ProspectIntelligenceSnapshot(),
        ProspectEvidence(
            observed=ObservedProspectEvidence(
                "turn-1", explicit_next_step=PreferredNextStep.END_CONVERSATION
            )
        ),
    )
    decision = _decide(context=_context(prospect=prospect), request=EscalationRequest())
    assert decision.recovery_mode == RecoveryMode.POLITE_WRAP_UP
    assert "next_state" not in {item.name for item in fields(decision)}
