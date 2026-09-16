"""Focused tests for progressive consultative decisions and communication."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from app.conversation.consultative.contracts import (
    AcknowledgementIntent,
    ConsultativeConversationDecision,
    ConsultativeDecisionInput,
    ConsultativeMove,
    ConsultativeObjective,
    ConsultativeTurnSignals,
    ProblemCategory,
    ProblemEvidence,
    ProblemEvidenceBasis,
    ProblemField,
    ProspectProblem,
    QuestionPolicy,
    ServiceAnswerContext,
    ServiceFitDecision,
    ServiceFitStatus,
)
from app.conversation.consultative.engine import ConsultativeDecisionEngine
from app.conversation.consultative.problem import ProblemModelUpdater
from app.conversation.context.contracts import (
    ApprovedEvidenceItem,
    ApprovedEvidenceSourceKind,
    EvidenceScope,
    EvidenceScopeKind,
    EvidenceType,
)
from app.conversation.prospect_intelligence.contracts import (
    CurrentSolutionContext,
    DecisionAuthority,
    EvidenceProvenance,
    EvidenceSourceKind,
    ObjectionType,
    ObservedValue,
    PreferredNextStep,
    ProspectIntelligenceSummary,
    ProspectRole,
    SolutionSatisfaction,
)
from app.conversation.response_planning.contracts import (
    AddresseeStatus,
    AuthoritativeResultKind,
    ConversationMove,
    InterruptionCategory,
    InterruptionContext,
    PendingConversationIntent,
    QuestionStrategy,
    ResponseLength,
    ResponsePlanningInput,
)
from app.conversation.response_planning.planner import ResponsePlanner
from app.conversation.response_rendering.contracts import (
    ResponseRenderInput,
    ResponseRenderingBudget,
)
from app.conversation.response_rendering.renderer import DeterministicResponseRenderer
from app.conversation.strategy.contracts import (
    ConversationMode,
    ConversationStrategy,
    SalesStage,
    StrategyType,
)
from app.conversation.understanding.contracts import (
    ConversationalRegister,
    LanguageProfile,
    LanguageScript,
)
from app.core.constants import ConversationState


def _problem(**changes):  # type: ignore[no-untyped-def]
    values = {
        "category": ProblemCategory.FOLLOW_UP,
        "explicit_description": "the team follows up late",
        "evidence_basis": ProblemEvidenceBasis.EXPLICIT,
        "source_turn_id": "t1",
    }
    values.update(changes)
    return ProspectProblem(**values)


def _fit(
    status: ServiceFitStatus = ServiceFitStatus.INSUFFICIENT_CONTEXT,
    *,
    missing: tuple[ProblemField, ...] = (),
) -> ServiceFitDecision:
    if status == ServiceFitStatus.SUPPORTED_FIT:
        return ServiceFitDecision(
            status,
            "automation",
            ("follow-up gap",),
            evidence_ids=("e1",),
            explanation_allowed=True,
        )
    return ServiceFitDecision(status, missing_information=missing)


def _provenance() -> EvidenceProvenance:
    return EvidenceProvenance("t1", EvidenceSourceKind.EXPLICIT_STATEMENT)


def _prospect(
    *,
    role: ProspectRole | None = None,
    authority: DecisionAuthority | None = None,
    busy: bool | None = None,
    interested: bool | None = None,
    objection: ObjectionType | None = None,
    next_step: PreferredNextStep | None = None,
    current_solution: bool = False,
) -> ProspectIntelligenceSummary:
    provenance = _provenance()
    return ProspectIntelligenceSummary(
        explicit_role=ObservedValue(role, provenance) if role is not None else None,
        explicit_decision_authority=(
            ObservedValue(authority, provenance) if authority is not None else None
        ),
        busy=ObservedValue(busy, provenance) if busy is not None else None,
        interested=(
            ObservedValue(interested, provenance) if interested is not None else None
        ),
        explicit_objection=(
            ObservedValue(objection, provenance) if objection is not None else None
        ),
        explicit_next_step=(
            ObservedValue(next_step, provenance) if next_step is not None else None
        ),
        current_solution=(
            CurrentSolutionContext(
                "another provider",
                None,
                SolutionSatisfaction.UNKNOWN,
                provenance,
            )
            if current_solution
            else None
        ),
    )


def _strategy(stage: SalesStage = SalesStage.DISCOVERY) -> ConversationStrategy:
    return ConversationStrategy(
        stage,
        ConversationMode.NORMAL,
        "understand the current problem",
        StrategyType.DISCOVER_NEED,
    )


def _input(
    *,
    problem: ProspectProblem | None = None,
    fit: ServiceFitDecision | None = None,
    prospect: ProspectIntelligenceSummary | None = None,
    signals: ConsultativeTurnSignals = ConsultativeTurnSignals(),
    strategy: ConversationStrategy | None = None,
    category: InterruptionCategory = InterruptionCategory.OTHER,
    addressee: AddresseeStatus = AddresseeStatus.ADDRESSED_TO_AGENT,
    pending: PendingConversationIntent | None = None,
    language: LanguageProfile | None = None,
) -> ConsultativeDecisionInput:
    return ConsultativeDecisionInput(
        problem or ProspectProblem(),
        fit or _fit(),
        prospect,
        strategy,
        language,
        signals,
        category,
        addressee,
        pending,
    )


def _decide(**values):  # type: ignore[no-untyped-def]
    return ConsultativeDecisionEngine().decide(_input(**values))


def _evidence(
    evidence_type: EvidenceType = EvidenceType.APPROVED_CLAIM,
) -> ApprovedEvidenceItem:
    return ApprovedEvidenceItem(
        "e1",
        "workflow_automation",
        evidence_type,
        "The service supports new-lead workflow automation.",
        ApprovedEvidenceSourceKind.CURATED_SERVICE,
        EvidenceScope(EvidenceScopeKind.SERVICE, "automation"),
    )


def _answer(
    evidence_type: EvidenceType = EvidenceType.APPROVED_CLAIM,
) -> ServiceAnswerContext:
    return ServiceAnswerContext(
        "automation",
        "AI Automation",
        None,
        ("workflow_automation",),
        (_evidence(evidence_type),),
        "the team follows up late",
        "staff checks email manually",
    )


def _plan_and_render(
    decision: ConsultativeConversationDecision,
    *,
    answer: ServiceAnswerContext | None = None,
    language: LanguageProfile | None = None,
    category: InterruptionCategory = InterruptionCategory.OTHER,
    interruption: InterruptionContext = InterruptionContext(),
) -> tuple[object, str]:
    plan = ResponsePlanner().plan(
        ResponsePlanningInput(
            ConversationState.LISTEN,
            AuthoritativeResultKind.EXECUTED,
            "current prospect statement",
            conversation_category=category,
            interruption=interruption,
            language_profile=language,
            consultative_decision=decision,
            service_answer_context=answer,
        )
    )
    rendered = DeterministicResponseRenderer().render(
        ResponseRenderInput(
            plan,
            AuthoritativeResultKind.EXECUTED,
            ResponseRenderingBudget(5),
        )
    )
    return plan, rendered.text


def test_consultative_decision_is_immutable() -> None:
    decision = _decide()
    with pytest.raises(FrozenInstanceError):
        decision.move = ConsultativeMove.GRACEFUL_CLOSE  # type: ignore[misc]


def test_consultative_signals_reject_enum_like_non_booleans() -> None:
    with pytest.raises(TypeError):
        ConsultativeTurnSignals(direct_question="true")  # type: ignore[arg-type]


def test_same_input_produces_same_decision() -> None:
    value = _input(problem=_problem())
    engine = ConsultativeDecisionEngine()
    assert engine.decide(value) == engine.decide(value)


def test_decision_has_no_authority_or_execution_fields() -> None:
    names = {item.name for item in fields(ConsultativeConversationDecision)}
    assert names.isdisjoint(
        {
            "next_state",
            "execute_action",
            "confirmed_callback",
            "confirmed_demo",
            "approved_discount",
            "service_authorization",
            "persistence_instruction",
            "dnc",
            "trusted_not_interested",
        }
    )


def test_one_primary_question_is_default_for_discovery() -> None:
    decision = _decide()
    assert decision.question_policy == QuestionPolicy.ONE_PRIMARY
    assert decision.primary_information_gap == ProblemField.UNDERLYING_PROBLEM


def test_discovery_selects_only_one_gap() -> None:
    decision = _decide(problem=_problem(current_process=None, impact=None))
    assert decision.primary_information_gap == ProblemField.CURRENT_PROCESS
    assert not isinstance(decision.primary_information_gap, tuple)


def test_two_way_ambiguity_uses_short_contrast() -> None:
    decision = _decide(
        signals=ConsultativeTurnSignals(two_way_ambiguity=True)
    )
    assert decision.question_policy == QuestionPolicy.SHORT_CONTRAST


def test_requested_chatbot_does_not_prove_fit() -> None:
    decision = _decide(
        problem=ProspectProblem(requested_solution="chatbot"),
        fit=_fit(ServiceFitStatus.NO_AUTHORIZED_FIT),
    )
    assert decision.move == ConsultativeMove.DISCOVER_PROBLEM
    assert decision.primary_information_gap == ProblemField.UNDERLYING_PROBLEM


def test_requested_solution_remains_separate_from_problem() -> None:
    problem = ProspectProblem(requested_solution="chatbot")
    assert problem.requested_solution == "chatbot"
    assert not problem.meaningful


def test_known_problem_moves_to_current_process() -> None:
    decision = _decide(problem=_problem())
    assert decision.move == ConsultativeMove.UNDERSTAND_CURRENT_PROCESS


def test_known_current_process_is_not_asked_again() -> None:
    decision = _decide(
        problem=_problem(current_process="staff checks email manually")
    )
    assert decision.move == ConsultativeMove.UNDERSTAND_IMPACT
    assert decision.primary_information_gap == ProblemField.IMPACT


def test_known_source_does_not_become_a_repeated_source_question() -> None:
    decision = _decide(
        problem=_problem(
            source_or_channel="website",
            current_process="email notification",
        )
    )
    assert decision.primary_information_gap != ProblemField.SOURCE_OR_CHANNEL


def test_unstated_impact_remains_unknown() -> None:
    problem = _problem(current_process="manual email")
    decision = _decide(problem=problem)
    assert problem.impact is None
    assert decision.primary_information_gap == ProblemField.IMPACT


def test_irrelevant_budget_gap_is_never_created() -> None:
    assert "budget" not in {item.value for item in ProblemField}


def test_opening_is_truthful_permission_oriented_not_a_pitch() -> None:
    decision = _decide(strategy=_strategy(SalesStage.OPENING))
    assert decision.objective == ConsultativeObjective.OPEN_TRUTHFULLY
    assert decision.move == ConsultativeMove.OPEN_CONVERSATION
    assert decision.question_policy == QuestionPolicy.ONE_PRIMARY


def test_direct_question_precedes_pending_sales_agenda() -> None:
    pending = PendingConversationIntent("discover", "ask about current workflow")
    decision = _decide(
        signals=ConsultativeTurnSignals(direct_question=True),
        pending=pending,
    )
    assert decision.move == ConsultativeMove.ANSWER_DIRECT_QUESTION
    assert decision.resume_previous_goal


def test_commercial_purpose_question_requires_truthful_answer() -> None:
    decision = _decide(
        signals=ConsultativeTurnSignals(commercial_purpose_question=True)
    )
    assert decision.commercial_transparency_required


def test_commercial_purpose_truth_wins_over_simultaneous_misunderstanding() -> None:
    decision = _decide(
        signals=ConsultativeTurnSignals(
            commercial_purpose_question=True,
            misunderstanding=True,
        )
    )
    assert decision.move == ConsultativeMove.ANSWER_DIRECT_QUESTION
    assert decision.commercial_transparency_required
    assert decision.move == ConsultativeMove.ANSWER_DIRECT_QUESTION


def test_busy_and_interested_choose_low_pressure_callback() -> None:
    decision = _decide(prospect=_prospect(busy=True, interested=True))
    assert decision.move == ConsultativeMove.LOW_PRESSURE_CALLBACK
    assert decision.question_policy == QuestionPolicy.NONE


def test_busy_does_not_equal_uninterested() -> None:
    prospect = _prospect(busy=True, interested=True)
    assert prospect.busy is not None and prospect.busy.value
    assert prospect.interested is not None and prospect.interested.value


@pytest.mark.parametrize("role", [ProspectRole.RECEPTIONIST, ProspectRole.GATEKEEPER])
def test_routing_roles_do_not_receive_full_discovery(role: ProspectRole) -> None:
    decision = _decide(prospect=_prospect(role=role), problem=_problem())
    assert decision.move == ConsultativeMove.ROUTE_TO_DECISION_MAKER
    assert decision.primary_information_gap == ProblemField.ROLE_ROUTING


def test_manager_role_does_not_imply_authority() -> None:
    prospect = _prospect(role=ProspectRole.MANAGER)
    assert prospect.explicit_decision_authority is None


def test_explicit_authority_remains_separate_from_role() -> None:
    prospect = _prospect(
        role=ProspectRole.MANAGER,
        authority=DecisionAuthority.LOW,
    )
    assert prospect.explicit_role is not None
    assert prospect.explicit_decision_authority is not None
    assert prospect.explicit_decision_authority.value == DecisionAuthority.LOW


def test_existing_provider_prompts_satisfaction_context_not_attack() -> None:
    decision = _decide(prospect=_prospect(current_solution=True))
    assert decision.move == ConsultativeMove.UNDERSTAND_EXISTING_SOLUTION
    assert decision.primary_information_gap == ProblemField.PROVIDER_SATISFACTION


def test_existing_provider_does_not_imply_dissatisfaction() -> None:
    prospect = _prospect(current_solution=True)
    assert prospect.current_solution is not None
    assert prospect.current_solution.satisfaction == SolutionSatisfaction.UNKNOWN


def test_objection_without_reason_clarifies_before_rebuttal() -> None:
    decision = _decide(
        prospect=_prospect(objection=ObjectionType.TRUST),
        problem=_problem(friction=None),
    )
    assert decision.move == ConsultativeMove.HANDLE_OBJECTION
    assert decision.primary_information_gap == ProblemField.OBJECTION_REASON


def test_explicit_no_problem_gracefully_closes() -> None:
    decision = _decide(
        signals=ConsultativeTurnSignals(explicit_no_problem=True),
        fit=_fit(ServiceFitStatus.NO_AUTHORIZED_FIT),
    )
    assert decision.move == ConsultativeMove.GRACEFUL_CLOSE


def test_explicit_no_problem_is_not_overridden_by_existing_provider_context() -> None:
    decision = _decide(
        prospect=_prospect(current_solution=True),
        signals=ConsultativeTurnSignals(explicit_no_problem=True),
    )
    assert decision.move == ConsultativeMove.GRACEFUL_CLOSE


def test_supported_fit_explains_relevance() -> None:
    decision = _decide(problem=_problem(), fit=_fit(ServiceFitStatus.SUPPORTED_FIT))
    assert decision.move == ConsultativeMove.EXPLAIN_RELEVANT_FIT
    assert decision.explanation_depth == ResponseLength.MODERATE


def test_possible_fit_does_not_become_explanation() -> None:
    fit = ServiceFitDecision(
        ServiceFitStatus.POSSIBLE_FIT,
        "automation",
        missing_information=(ProblemField.CURRENT_PROCESS,),
    )
    decision = _decide(problem=_problem(), fit=fit)
    assert decision.move == ConsultativeMove.UNDERSTAND_CURRENT_PROCESS
    assert not fit.explanation_allowed


def test_buying_signal_accelerates_without_confirming_demo() -> None:
    decision = _decide(
        problem=_problem(),
        fit=_fit(ServiceFitStatus.SUPPORTED_FIT),
        signals=ConsultativeTurnSignals(buying_signal=True),
        prospect=_prospect(next_step=PreferredNextStep.DEMO),
    )
    assert decision.move == ConsultativeMove.ASK_MICRO_COMMITMENT
    assert not hasattr(decision, "confirmed_demo")


def test_service_information_with_context_explains_one_fit() -> None:
    decision = _decide(
        problem=_problem(),
        fit=_fit(ServiceFitStatus.SUPPORTED_FIT),
        signals=ConsultativeTurnSignals(service_information_request=True),
    )
    assert decision.move == ConsultativeMove.EXPLAIN_RELEVANT_FIT


def test_service_information_without_context_clarifies() -> None:
    decision = _decide(
        signals=ConsultativeTurnSignals(service_information_request=True)
    )
    assert decision.move == ConsultativeMove.DISCOVER_PROBLEM


def test_can_you_solve_this_requires_supported_fit() -> None:
    possible = ServiceFitDecision(
        ServiceFitStatus.POSSIBLE_FIT,
        "automation",
        missing_information=(ProblemField.APPROVED_EVIDENCE,),
    )
    decision = _decide(
        problem=_problem(),
        fit=possible,
        signals=ConsultativeTurnSignals(service_information_request=True),
    )
    assert decision.move == ConsultativeMove.DISCOVER_PROBLEM


def test_explanation_request_adapts_depth() -> None:
    decision = _decide(
        problem=_problem(),
        fit=_fit(ServiceFitStatus.SUPPORTED_FIT),
        signals=ConsultativeTurnSignals(explanation_requested=True),
    )
    assert decision.explanation_depth == ResponseLength.DETAILED


def test_simple_direct_question_stays_short() -> None:
    decision = _decide(signals=ConsultativeTurnSignals(direct_question=True))
    assert decision.explanation_depth == ResponseLength.SHORT


def test_misunderstanding_requires_simplification() -> None:
    decision = _decide(
        problem=_problem(),
        signals=ConsultativeTurnSignals(misunderstanding=True),
    )
    assert decision.move == ConsultativeMove.SIMPLIFY_EXPLANATION
    assert decision.simplification_required


def test_language_recovery_retains_relevant_pending_goal() -> None:
    pending = PendingConversationIntent("explain", "the workflow fit")
    decision = _decide(
        signals=ConsultativeTurnSignals(misunderstanding=True),
        pending=pending,
    )
    assert decision.resume_previous_goal


def test_language_profile_has_no_locale_inference() -> None:
    profile = LanguageProfile("ur", script=LanguageScript.LATIN)
    assert not hasattr(profile, "locale")
    assert not hasattr(profile, "country")


def test_roman_urdu_language_recovery_is_conversational() -> None:
    profile = LanguageProfile(
        "ur",
        script=LanguageScript.LATIN,
        conversational_register=ConversationalRegister.CASUAL,
        preferred_response_language="ur",
        preferred_script=LanguageScript.LATIN,
    )
    decision = _decide(
        signals=ConsultativeTurnSignals(misunderstanding=True),
        language=profile,
    )
    _, text = _plan_and_render(decision, language=profile)
    assert "simple words" in text
    assert "therefore" not in text.lower()


def test_hindi_profile_does_not_require_formal_register() -> None:
    profile = LanguageProfile(
        "hi",
        script=LanguageScript.LATIN,
        conversational_register=ConversationalRegister.CASUAL,
    )
    assert profile.conversational_register == ConversationalRegister.CASUAL


def test_problem_evidence_and_model_are_immutable() -> None:
    evidence = ProblemEvidence("t1", explicit_description="slow follow-up")
    model = ProblemModelUpdater().update(ProspectProblem(), evidence)
    with pytest.raises(FrozenInstanceError):
        evidence.correction = True  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        model.impact = "lost revenue"  # type: ignore[misc]


def test_problem_updater_does_not_invent_impact() -> None:
    updated = ProblemModelUpdater().update(
        ProspectProblem(),
        ProblemEvidence("t1", explicit_description="slow follow-up"),
    )
    assert updated.impact is None


def test_progressive_problem_update_reuses_prior_answer() -> None:
    updater = ProblemModelUpdater()
    first = updater.update(
        ProspectProblem(),
        ProblemEvidence(
            "t1",
            ProblemCategory.FOLLOW_UP,
            "follow-up is slow",
            source_or_channel="website",
        ),
    )
    second = updater.update(
        first,
        ProblemEvidence("t2", current_process="staff checks email manually"),
    )
    assert second.source_or_channel == "website"
    assert second.current_process == "staff checks email manually"


def test_correction_discards_stale_problem_hypothesis() -> None:
    updater = ProblemModelUpdater()
    previous = _problem(
        category=ProblemCategory.LEAD_FLOW,
        explicit_description="not enough leads",
    )
    corrected = updater.update(
        previous,
        ProblemEvidence(
            "t2",
            ProblemCategory.FOLLOW_UP,
            "leads exist but follow-up does not happen",
            friction="staff forget follow-up",
            correction=True,
        ),
    )
    assert corrected.category == ProblemCategory.FOLLOW_UP
    assert "not enough" not in (corrected.explicit_description or "")


def test_explicit_problem_is_not_overwritten_by_later_inference() -> None:
    updater = ProblemModelUpdater()
    previous = _problem()
    inferred = ProblemEvidence(
        "t2",
        ProblemCategory.BRANDING,
        "maybe branding",
        evidence_basis=ProblemEvidenceBasis.INFERRED,
    )
    assert updater.update(previous, inferred) == previous


def test_correction_signal_without_revised_problem_clarifies() -> None:
    decision = _decide(
        category=InterruptionCategory.CORRECTION,
        problem=ProspectProblem(),
    )
    assert decision.move == ConsultativeMove.CLARIFY


def test_consultative_planner_emits_one_question() -> None:
    decision = _decide(problem=_problem())
    plan, text = _plan_and_render(decision)
    assert plan.question_strategy == QuestionStrategy.CLARIFY_CURRENT_INPUT
    assert text.count("?") == 1


def test_renderer_does_not_create_giant_option_menu() -> None:
    decision = _decide(
        signals=ConsultativeTurnSignals(two_way_ambiguity=True)
    )
    _, text = _plan_and_render(decision)
    assert text.count("?") == 1
    assert text.count(" or ") == 1
    assert text.count(",") <= 1


def test_renderer_uses_known_problem_for_grounded_service_fit() -> None:
    decision = _decide(problem=_problem(), fit=_fit(ServiceFitStatus.SUPPORTED_FIT))
    _, text = _plan_and_render(decision, answer=_answer())
    assert "team follows up late" in text
    assert "new-lead workflow automation" in text


def test_no_service_answer_context_means_no_factual_claim() -> None:
    decision = _decide(problem=_problem(), fit=_fit(ServiceFitStatus.SUPPORTED_FIT))
    _, text = _plan_and_render(decision)
    assert "don't have enough confirmed detail" in text


def test_case_study_renderer_explicitly_avoids_guarantee() -> None:
    decision = _decide(problem=_problem(), fit=_fit(ServiceFitStatus.SUPPORTED_FIT))
    _, text = _plan_and_render(decision, answer=_answer(EvidenceType.CASE_STUDY))
    assert "not a guarantee" in text


@pytest.mark.parametrize(
    "forbidden",
    ["increase conversions", "save 40%", "never miss a lead", "guaranteed"],
)
def test_grounded_renderer_does_not_invent_roi_or_guarantees(forbidden: str) -> None:
    decision = _decide(problem=_problem(), fit=_fit(ServiceFitStatus.SUPPORTED_FIT))
    _, text = _plan_and_render(decision, answer=_answer())
    assert forbidden not in text.lower()


def test_truthful_commercial_answer_does_not_hide_sales_purpose() -> None:
    decision = _decide(
        signals=ConsultativeTurnSignals(commercial_purpose_question=True)
    )
    plan, text = _plan_and_render(decision)
    assert plan.communicative_goal == ConversationMove.TRUTHFUL_COMMERCIAL_ANSWER
    assert "business call" in text
    assert "not selling" not in text.lower()
    assert "referral" not in text.lower()


def test_low_pressure_callback_is_not_confirmation() -> None:
    decision = _decide(prospect=_prospect(busy=True, interested=True))
    _, text = _plan_and_render(decision)
    assert "only discuss a callback" in text
    assert "scheduled" not in text
    assert "confirmed" not in text


def test_no_fit_renderer_does_not_force_service() -> None:
    decision = _decide(
        signals=ConsultativeTurnSignals(explicit_no_problem=True),
        fit=_fit(ServiceFitStatus.NO_AUTHORIZED_FIT),
    )
    _, text = _plan_and_render(decision)
    assert "won't force it" in text
    assert "service" not in text.lower()


def test_micro_commitment_renderer_does_not_confirm_next_step() -> None:
    decision = _decide(
        problem=_problem(),
        fit=_fit(ServiceFitStatus.SUPPORTED_FIT),
        signals=ConsultativeTurnSignals(buying_signal=True),
    )
    _, text = _plan_and_render(decision, answer=_answer())
    assert "without treating it as confirmed" in text


def test_addressee_uncertainty_still_wins_in_planner() -> None:
    decision = _decide()
    plan = ResponsePlanner().plan(
        ResponsePlanningInput(
            ConversationState.LISTEN,
            AuthoritativeResultKind.EXECUTED,
            "background speech",
            addressee_status=AddresseeStatus.ADDRESSEE_UNCERTAIN,
            consultative_decision=decision,
        )
    )
    assert plan.communicative_goal == ConversationMove.CLARIFY_ADDRESSEE


def test_interruption_direct_question_drops_stale_agenda_from_question() -> None:
    pending = PendingConversationIntent("pitch", "an older service pitch")
    decision = _decide(
        signals=ConsultativeTurnSignals(direct_question=True),
        pending=pending,
    )
    plan, _ = _plan_and_render(
        decision,
        category=InterruptionCategory.QUESTION,
        interruption=InterruptionContext(True, previous_intent=pending),
    )
    assert plan.communicative_goal == ConversationMove.ANSWER_CURRENT_QUESTION


def test_safety_result_precedes_consultative_decision() -> None:
    decision = _decide(problem=_problem())
    plan = ResponsePlanner().plan(
        ResponsePlanningInput(
            ConversationState.LISTEN,
            AuthoritativeResultKind.ESCALATE,
            "question",
            consultative_decision=decision,
        )
    )
    assert plan.communicative_goal == ConversationMove.ACKNOWLEDGE_ESCALATION


def test_response_plan_gains_no_execution_authority() -> None:
    decision = _decide(problem=_problem())
    plan, _ = _plan_and_render(decision)
    for forbidden in ("next_state", "execute_action", "persist_contact"):
        assert not hasattr(plan, forbidden)


def test_problem_and_strategy_cannot_create_approved_evidence() -> None:
    assert not hasattr(_problem(), "approved_evidence")
    assert not hasattr(_decide(problem=_problem()), "approved_evidence")


@pytest.mark.parametrize(
    "field",
    [
        "operational_capability",
        "dnc",
        "trusted_not_interested",
        "approved_pricing",
        "approved_discount",
        "persist_contact",
        "next_state",
    ],
)
def test_consultative_decision_cannot_own_forbidden_domain_fields(field: str) -> None:
    assert not hasattr(_decide(problem=_problem()), field)


def test_no_fixed_role_script_or_campaign_keyword_tree_exists() -> None:
    decision = _decide(prospect=_prospect(role=ProspectRole.RECEPTIONIST))
    assert decision.primary_information_gap == ProblemField.ROLE_ROUTING
    assert not hasattr(decision, "script")
    assert not hasattr(decision, "keyword")


def test_problem_model_contains_no_full_transcript_or_secrets() -> None:
    names = {item.name for item in fields(ProspectProblem)}
    assert names.isdisjoint(
        {"full_transcript", "api_key", "credentials", "provider_pricing"}
    )
