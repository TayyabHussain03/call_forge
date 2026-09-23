"""Focused tests for deterministic advisory sales playbook intelligence."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace

import pytest

from app.conversation.consultative.contracts import (
    ProblemCategory,
    ProblemEvidenceBasis,
    ProspectProblem,
)
from app.conversation.prospect_intelligence.contracts import (
    CurrentSolutionContext,
    EvidenceProvenance,
    EvidenceSourceKind,
    ObjectionType,
    ObservedValue,
    ProspectIntelligenceSummary,
    SolutionSatisfaction,
)
from app.conversation.response_planning.contracts import (
    AddresseeStatus,
    AcknowledgementKind,
    AuthoritativeResultKind,
    InterruptionHandling,
    QuestionStrategy,
    ResponseLength,
    ResponsePlanningInput,
)
from app.conversation.response_planning.planner import ResponsePlanner
from app.conversation.sales_cognition.contracts import (
    BuyingReadinessGuidance,
    ConversationEnergy,
    ConversationMomentum,
    CuriosityFocus,
    DiscoveryReadiness,
    EmotionalPosture,
    InterestStrength,
    ObjectionUnderstanding,
    PressureState,
    QuestionPriority,
    RelationshipState,
    RoleConversationStyle,
    SalesConversationGuidance,
    TrustState,
)
from app.conversation.sales_playbook.contracts import (
    AcknowledgementStrategy,
    BenefitCategory,
    BusinessContext,
    BusinessSituation,
    ClarificationStrategy,
    ConsultantStep,
    DiscoveryTopic,
    EducationStrategy,
    ObjectionGuidance,
    PitchReadiness,
    PlaybookRestriction,
    SalesPlaybookInput,
    ServicePlaybook,
)
from app.conversation.sales_playbook.engine import SalesPlaybookIntelligenceEngine
from app.conversation.sales_playbook.validator import (
    PlaybookGuidanceValidationError,
    validate_playbook_guidance,
)
from app.core.constants import ConversationState, Tone


def _provenance() -> EvidenceProvenance:
    return EvidenceProvenance("turn-1", EvidenceSourceKind.EXPLICIT_STATEMENT)


def _prospect(
    *,
    busy: bool | None = None,
    interested: bool | None = None,
    objection: ObjectionType | None = None,
    provider: bool = False,
) -> ProspectIntelligenceSummary:
    provenance = _provenance()
    return ProspectIntelligenceSummary(
        busy=ObservedValue(busy, provenance) if busy is not None else None,
        interested=(
            ObservedValue(interested, provenance) if interested is not None else None
        ),
        explicit_objection=(
            ObservedValue(objection, provenance) if objection is not None else None
        ),
        current_solution=(
            CurrentSolutionContext(
                "existing provider", None, SolutionSatisfaction.UNKNOWN, provenance
            )
            if provider
            else None
        ),
    )


def _problem(*, impact: bool = False) -> ProspectProblem:
    return ProspectProblem(
        ProblemCategory.WORKFLOW,
        "manual work slows the team",
        current_process="manual spreadsheet",
        impact="follow-up is delayed" if impact else None,
        evidence_basis=ProblemEvidenceBasis.EXPLICIT,
        source_turn_id="turn-1",
    )


def _playbook(
    service_id: str,
    situations: frozenset[BusinessSituation],
    topics: tuple[DiscoveryTopic, ...],
    *,
    priority: int = 100,
    future: tuple[str, ...] = (),
) -> ServicePlaybook:
    return ServicePlaybook(
        service_id,
        service_id.replace("_", " ").title(),
        "business_service",
        "Solves a structured business problem without making promises.",
        frozenset({"restaurant", "business"}),
        situations,
        topics,
        objections=(
            ObjectionGuidance(
                ObjectionType.PRICE,
                AcknowledgementStrategy.ACKNOWLEDGE_CONSTRAINT,
                ClarificationStrategy.CLARIFY_BUDGET_CONTEXT,
                EducationStrategy.EXPLAIN_RELEVANT_OUTCOME,
            ),
        ),
        benefits=frozenset({BenefitCategory.OPERATIONAL}),
        future_service_ids=future,
        restrictions=frozenset(
            {
                PlaybookRestriction.NO_GUARANTEED_REVENUE,
                PlaybookRestriction.NO_GUARANTEED_ROI,
                PlaybookRestriction.NO_FAKE_TIMELINES,
            }
        ),
        priority=priority,
    )


def _playbooks() -> tuple[ServicePlaybook, ...]:
    return (
        _playbook(
            "website",
            frozenset({BusinessSituation.NO_ONLINE_PRESENCE}),
            (
                DiscoveryTopic.CURRENT_WEBSITE,
                DiscoveryTopic.CUSTOMER_JOURNEY,
            ),
            priority=10,
            future=("seo", "branding"),
        ),
        _playbook(
            "seo",
            frozenset(
                {
                    BusinessSituation.LOW_VISIBILITY,
                    BusinessSituation.NO_TRAFFIC,
                    BusinessSituation.POOR_RANKING,
                }
            ),
            (DiscoveryTopic.GOOGLE_PRESENCE, DiscoveryTopic.BUSINESS_IMPACT),
            priority=20,
        ),
        _playbook(
            "automation",
            frozenset(
                {
                    BusinessSituation.MANUAL_WORKFLOW,
                    BusinessSituation.REPEATED_TASKS,
                    BusinessSituation.LOST_TIME,
                }
            ),
            (DiscoveryTopic.CURRENT_WORKFLOW, DiscoveryTopic.BUSINESS_IMPACT),
            priority=5,
            future=("crm",),
        ),
        _playbook(
            "crm",
            frozenset({BusinessSituation.CRM_UNDERPERFORMING}),
            (DiscoveryTopic.CURRENT_SOFTWARE, DiscoveryTopic.BUSINESS_IMPACT),
            priority=30,
        ),
    )


def _guidance(interest: InterestStrength) -> SalesConversationGuidance:
    return SalesConversationGuidance(
        ConversationMomentum.IMPROVING,
        TrustState.BUILDING,
        interest,
        DiscoveryReadiness.READY_FOR_NEXT_STEP,
        ObjectionUnderstanding.NONE,
        ConversationEnergy.NORMAL,
        QuestionPriority.NONE,
        None,
        RelationshipState.DEVELOPING,
        PressureState.READY_TO_CONTINUE,
        CuriosityFocus.NONE,
        RoleConversationStyle.GENERAL_CONSULTATIVE,
        EmotionalPosture.INTERESTED,
        BuyingReadinessGuidance.CONTINUE_DISCOVERY,
        ResponseLength.SHORT,
    )


def _input(
    situations: frozenset[BusinessSituation],
    *,
    industry: str = "business",
    known: frozenset[DiscoveryTopic] = frozenset(),
    prospect: ProspectIntelligenceSummary | None = None,
    eligible: frozenset[str] = frozenset({"website", "seo", "automation", "crm"}),
    existing: frozenset[str] = frozenset(),
    no_fit: bool = False,
    ambiguity: bool = False,
    discussed: frozenset[str] = frozenset(),
    problem: ProspectProblem | None = None,
    guidance: SalesConversationGuidance | None = None,
) -> SalesPlaybookInput:
    return SalesPlaybookInput(
        BusinessContext(
            industry,
            situations,
            known,
            existing_service_ids=existing,
            no_fit=no_fit,
            true_ambiguity=ambiguity,
        ),
        _playbooks(),
        eligible,
        prospect or _prospect(),
        None,
        problem or _problem(),
        None,
        guidance,
        discussed,
    )


def _evaluate(*args, **kwargs):  # type: ignore[no-untyped-def]
    return SalesPlaybookIntelligenceEngine().evaluate(_input(*args, **kwargs))


def test_restaurant_without_website_creates_website_opportunity() -> None:
    result = _evaluate(
        frozenset({BusinessSituation.NO_ONLINE_PRESENCE}), industry="restaurant"
    )
    assert result.selected_service_id == "website"
    assert result.primary_question == DiscoveryTopic.CURRENT_WEBSITE


def test_restaurant_with_website_prioritizes_visibility_not_replacement() -> None:
    result = _evaluate(
        frozenset(
            {BusinessSituation.WEBSITE_EXISTS, BusinessSituation.LOW_VISIBILITY}
        ),
        industry="restaurant",
        existing=frozenset({"website"}),
    )
    assert result.selected_service_id == "seo"
    assert not result.existing_customer_optimization


def test_manual_workflow_creates_automation_opportunity() -> None:
    result = _evaluate(frozenset({BusinessSituation.MANUAL_WORKFLOW}))
    assert result.selected_service_id == "automation"
    assert result.primary_question == DiscoveryTopic.CURRENT_WORKFLOW


def test_existing_crm_is_treated_as_optimization() -> None:
    result = _evaluate(
        frozenset(
            {BusinessSituation.CRM_EXISTS, BusinessSituation.CRM_UNDERPERFORMING}
        ),
        existing=frozenset({"crm"}),
    )
    assert result.selected_service_id == "crm"
    assert result.existing_customer_optimization


def test_existing_provider_is_acknowledged_and_clarified_without_attack() -> None:
    result = _evaluate(
        frozenset({BusinessSituation.LOW_VISIBILITY}),
        prospect=_prospect(
            objection=ObjectionType.ALREADY_HAVE_PROVIDER, provider=True
        ),
    )
    assert result.objection_guidance is not None
    assert result.objection_guidance.acknowledgement == (
        AcknowledgementStrategy.ACKNOWLEDGE_EXISTING_CHOICE
    )
    assert result.primary_question == DiscoveryTopic.PROVIDER_SATISFACTION
    assert result.pitch_readiness == PitchReadiness.EDUCATION


def test_busy_prospect_suppresses_questions_and_pitching() -> None:
    result = _evaluate(
        frozenset({BusinessSituation.MANUAL_WORKFLOW}),
        prospect=_prospect(busy=True),
    )
    assert result.primary_question is None
    assert result.secondary_question is None
    assert result.pitch_readiness == PitchReadiness.DISCOVERY_REQUIRED
    assert result.consultant_step == ConsultantStep.CLARIFY


def test_interested_prospect_with_complete_discovery_allows_micro_commitment() -> None:
    known = frozenset(
        {DiscoveryTopic.CURRENT_WORKFLOW, DiscoveryTopic.BUSINESS_IMPACT}
    )
    result = _evaluate(
        frozenset({BusinessSituation.MANUAL_WORKFLOW}),
        known=known,
        prospect=_prospect(interested=True),
        problem=_problem(impact=True),
        guidance=_guidance(InterestStrength.STRONG),
    )
    assert result.pitch_readiness == PitchReadiness.READY_FOR_MICRO_COMMITMENT
    assert result.consultant_step == ConsultantStep.MICRO_COMMITMENT


@pytest.mark.parametrize(
    "changes",
    [
        {"situations": frozenset()},
        {"situations": frozenset({BusinessSituation.NO_IDENTIFIED_NEED})},
        {
            "situations": frozenset({BusinessSituation.MANUAL_WORKFLOW}),
            "no_fit": True,
        },
    ],
)
def test_no_fit_business_ends_gracefully(changes) -> None:  # type: ignore[no-untyped-def]
    situations = changes["situations"]
    result = _evaluate(
        situations, **{key: value for key, value in changes.items() if key != "situations"}
    )
    assert result.selected_service_id is None
    assert result.pitch_readiness == PitchReadiness.NO_FIT
    assert result.consultant_step == ConsultantStep.GRACEFUL_END


def test_multiple_services_choose_one_highest_value_opportunity() -> None:
    result = _evaluate(
        frozenset(
            {
                BusinessSituation.NO_ONLINE_PRESENCE,
                BusinessSituation.MANUAL_WORKFLOW,
                BusinessSituation.LOST_TIME,
            }
        )
    )
    assert result.selected_service_id == "automation"
    assert result.future_opportunity_ids == ("website", "crm")
    assert not hasattr(result, "recommended_services")


def test_only_one_question_is_selected_without_true_ambiguity() -> None:
    result = _evaluate(frozenset({BusinessSituation.NO_ONLINE_PRESENCE}))
    assert result.primary_question is not None
    assert result.secondary_question is None


def test_second_question_requires_explicit_true_ambiguity() -> None:
    result = _evaluate(
        frozenset({BusinessSituation.NO_ONLINE_PRESENCE}), ambiguity=True
    )
    assert result.primary_question == DiscoveryTopic.CURRENT_WEBSITE
    assert result.secondary_question == DiscoveryTopic.CUSTOMER_JOURNEY


def test_price_objection_uses_strategy_categories_not_rebuttal_script() -> None:
    result = _evaluate(
        frozenset({BusinessSituation.NO_ONLINE_PRESENCE}),
        prospect=_prospect(objection=ObjectionType.PRICE),
    )
    assert result.objection_guidance is not None
    assert result.objection_guidance.clarification == (
        ClarificationStrategy.CLARIFY_BUDGET_CONTEXT
    )
    assert not hasattr(result.objection_guidance, "script")


def test_unauthorized_playbook_cannot_be_selected_or_cross_sold() -> None:
    result = _evaluate(
        frozenset(
            {BusinessSituation.NO_ONLINE_PRESENCE, BusinessSituation.LOW_VISIBILITY}
        ),
        eligible=frozenset({"seo"}),
    )
    assert result.selected_service_id == "seo"
    assert "website" not in result.future_opportunity_ids


def test_cross_sell_is_suppressed_when_existing_or_already_discussed() -> None:
    result = _evaluate(
        frozenset(
            {BusinessSituation.NO_ONLINE_PRESENCE, BusinessSituation.LOW_VISIBILITY}
        ),
        existing=frozenset({"seo"}),
        discussed=frozenset({"branding"}),
    )
    assert result.selected_service_id == "website"
    assert result.future_opportunity_ids == ()


def test_restrictions_are_enforced_on_every_selected_opportunity() -> None:
    result = _evaluate(frozenset({BusinessSituation.NO_ONLINE_PRESENCE}))
    assert PlaybookRestriction.NO_GUARANTEED_REVENUE in result.restrictions
    assert PlaybookRestriction.NO_GUARANTEED_ROI in result.restrictions
    assert PlaybookRestriction.NO_PRESSURE_LANGUAGE in result.restrictions


def test_validator_rejects_eligibility_ambiguity_and_restriction_escapes() -> None:
    value = _input(frozenset({BusinessSituation.NO_ONLINE_PRESENCE}))
    guidance = SalesPlaybookIntelligenceEngine().evaluate(value)
    unsafe_values = (
        replace(guidance, selected_service_id="unauthorized"),
        replace(guidance, secondary_question=DiscoveryTopic.CUSTOMER_JOURNEY),
        replace(guidance, restrictions=frozenset()),
    )
    for unsafe in unsafe_values:
        with pytest.raises(PlaybookGuidanceValidationError):
            validate_playbook_guidance(unsafe, value)


def test_deterministic_replay_and_immutable_output() -> None:
    value = _input(frozenset({BusinessSituation.MANUAL_WORKFLOW}))
    first = SalesPlaybookIntelligenceEngine().evaluate(value)
    second = SalesPlaybookIntelligenceEngine().evaluate(value)
    assert first == second
    with pytest.raises(FrozenInstanceError):
        first.selected_service_id = "website"  # type: ignore[misc]


def test_guidance_has_no_execution_authority_evidence_or_wording() -> None:
    guidance = _evaluate(frozenset({BusinessSituation.MANUAL_WORKFLOW}))
    names = {item.name for item in fields(guidance)}
    assert names.isdisjoint(
        {
            "action",
            "next_state",
            "approved_evidence",
            "execution",
            "response_text",
            "price",
            "discount",
            "booking",
        }
    )


def test_response_planning_carries_guidance_without_changing_plan_decision() -> None:
    guidance = _evaluate(frozenset({BusinessSituation.MANUAL_WORKFLOW}))
    planning_input = ResponsePlanningInput(
        ConversationState.LISTEN,
        AuthoritativeResultKind.EXECUTED,
        "We use spreadsheets.",
        tone=Tone.NEUTRAL,
        addressee_status=AddresseeStatus.ADDRESSED_TO_AGENT,
        playbook_guidance=guidance,
    )
    plan = ResponsePlanner().plan(planning_input)
    assert plan.playbook_guidance is guidance
    assert plan.question_strategy == QuestionStrategy.NONE
    assert plan.acknowledgement == AcknowledgementKind.NONE
    assert plan.interruption_handling == InterruptionHandling.NONE
