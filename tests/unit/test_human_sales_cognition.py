"""Focused invariants for the advisory human-sales cognition layer."""

from dataclasses import FrozenInstanceError, fields, replace

import pytest

from app.conversation.consultative.contracts import (
    ConsultativeConversationDecision,
    ConsultativeMove,
    ConsultativeObjective,
    ProblemCategory,
    ProblemEvidenceBasis,
    ProblemField,
    ProspectProblem,
    QuestionPolicy,
    ServiceFitDecision,
    ServiceFitStatus,
)
from app.conversation.prospect_intelligence.contracts import (
    CurrentSolutionContext,
    EvidenceProvenance,
    EvidenceSourceKind,
    ObjectionType,
    ObservedValue,
    PreferredNextStep,
    ProspectIntelligenceSummary,
    ProspectRole,
    SolutionSatisfaction,
)
from app.conversation.response_planning.contracts import ResponseLength
from app.conversation.sales_cognition.contracts import (
    BuyingReadinessGuidance,
    CognitionSignals,
    ConversationEnergy,
    ConversationMomentum,
    DiscoveryReadiness,
    EmotionalPosture,
    InterestStrength,
    ObjectionUnderstanding,
    PressureState,
    QuestionPriority,
    RelationshipState,
    RoleConversationStyle,
    SalesCognitionInput,
    TrustState,
)
from app.conversation.sales_cognition.engine import HumanSalesCognitionEngine
from app.conversation.sales_cognition.validator import (
    SalesGuidanceValidationError,
    validate_sales_guidance,
)


def _observed(value):  # type: ignore[no-untyped-def]
    return ObservedValue(
        value, EvidenceProvenance("turn-1", EvidenceSourceKind.EXPLICIT_STATEMENT)
    )


def _prospect(**values):  # type: ignore[no-untyped-def]
    mapped = {}
    for key, value in values.items():
        mapped[key] = _observed(value)
    return ProspectIntelligenceSummary(**mapped)


def _problem(**changes):  # type: ignore[no-untyped-def]
    values = {
        "category": ProblemCategory.FOLLOW_UP,
        "explicit_description": "follow-up is delayed",
        "evidence_basis": ProblemEvidenceBasis.EXPLICIT,
        "source_turn_id": "turn-1",
    }
    values.update(changes)
    return ProspectProblem(**values)


def _fit(supported: bool = False) -> ServiceFitDecision:
    if supported:
        return ServiceFitDecision(
            ServiceFitStatus.SUPPORTED_FIT,
            "automation",
            ("follow-up",),
            evidence_ids=("e1",),
            explanation_allowed=True,
        )
    return ServiceFitDecision(ServiceFitStatus.INSUFFICIENT_CONTEXT)


def _input(**changes):  # type: ignore[no-untyped-def]
    fit = changes.pop("service_fit", _fit())
    decision = changes.pop(
        "consultative_decision",
        ConsultativeConversationDecision(
            ConsultativeObjective.UNDERSTAND,
            ConsultativeMove.UNDERSTAND_CURRENT_PROCESS,
            ProblemField.CURRENT_PROCESS,
            service_fit=fit,
            question_policy=QuestionPolicy.ONE_PRIMARY,
        ),
    )
    values = {
        "prospect": ProspectIntelligenceSummary(),
        "problem": _problem(),
        "service_fit": fit,
        "strategy": None,
        "consultative_decision": decision,
    }
    values.update(changes)
    return SalesCognitionInput(**values)


def _evaluate(**changes):  # type: ignore[no-untyped-def]
    return HumanSalesCognitionEngine().evaluate(_input(**changes))


def test_guidance_is_immutable_deterministic_and_advisory_only() -> None:
    cognition_input = _input()
    first = HumanSalesCognitionEngine().evaluate(cognition_input)
    assert first == HumanSalesCognitionEngine().evaluate(cognition_input)
    with pytest.raises(FrozenInstanceError):
        first.interest = InterestStrength.STRONG  # type: ignore[misc]
    forbidden = {
        "action", "next_state", "service_id", "approved_evidence", "price",
        "discount", "authority", "persistence",
    }
    assert forbidden.isdisjoint(field.name for field in fields(first))
    assert first.max_primary_questions == 1


@pytest.mark.parametrize(
    ("role", "style"),
    [
        (ProspectRole.RECEPTIONIST, RoleConversationStyle.BRIEF_ROUTING),
        (ProspectRole.GATEKEEPER, RoleConversationStyle.BRIEF_ROUTING),
        (ProspectRole.OWNER, RoleConversationStyle.OWNER_VALUE),
        (ProspectRole.FOUNDER, RoleConversationStyle.OWNER_VALUE),
        (ProspectRole.OPERATIONS_MANAGER, RoleConversationStyle.OPERATIONS_WORKFLOW),
        (ProspectRole.TECHNICAL_MANAGER, RoleConversationStyle.TECHNICAL_PRECISION),
        (ProspectRole.SALES_MANAGER, RoleConversationStyle.SALES_PROCESS),
        (ProspectRole.FINANCE_CONTACT, RoleConversationStyle.FINANCIAL_RESTRAINT),
        (ProspectRole.ASSISTANT, RoleConversationStyle.SUPPORTIVE),
        (ProspectRole.UNKNOWN, RoleConversationStyle.GENERAL_CONSULTATIVE),
    ],
)
def test_role_changes_style_only(role, style) -> None:  # type: ignore[no-untyped-def]
    result = _evaluate(prospect=_prospect(explicit_role=role))
    assert result.role_style == style
    assert not hasattr(result, "authority")


@pytest.mark.parametrize(
    ("objection", "understanding"),
    [
        (ObjectionType.PRICE, ObjectionUnderstanding.BUDGET),
        (ObjectionType.NO_TIME, ObjectionUnderstanding.NO_TIME),
        (
            ObjectionType.ALREADY_HAVE_PROVIDER,
            ObjectionUnderstanding.EXISTING_PROVIDER,
        ),
        (ObjectionType.INTERNAL_TEAM, ObjectionUnderstanding.INTERNAL_TEAM),
        (ObjectionType.CONTRACT_LOCK, ObjectionUnderstanding.CONTRACT_LOCK),
        (ObjectionType.SEND_INFORMATION, ObjectionUnderstanding.RESEARCH_ONLY),
        (ObjectionType.UNKNOWN, ObjectionUnderstanding.UNKNOWN),
    ],
)
def test_structured_objections_are_preserved(  # type: ignore[no-untyped-def]
    objection, understanding
) -> None:
    result = _evaluate(prospect=_prospect(explicit_objection=objection))
    assert result.objection == understanding


def test_current_provider_does_not_assume_dissatisfaction() -> None:
    provenance = EvidenceProvenance(
        "turn-1", EvidenceSourceKind.EXPLICIT_STATEMENT
    )
    prospect = ProspectIntelligenceSummary(
        current_solution=CurrentSolutionContext(
            "incumbent", None, SolutionSatisfaction.UNKNOWN, provenance
        )
    )
    result = _evaluate(prospect=prospect)
    assert result.question_focus == ProblemField.PROVIDER_SATISFACTION
    assert result.objection == ObjectionUnderstanding.UNKNOWN


def test_recent_question_is_not_repeated_and_only_one_is_selected() -> None:
    result = _evaluate(recent_question_concepts=(ProblemField.CURRENT_PROCESS,))
    assert result.question_focus == ProblemField.IMPACT
    assert result.question_priority == QuestionPriority.IMPACT
    assert result.max_primary_questions == 1


@pytest.mark.parametrize("signal", ["confused", "correction"])
def test_repair_signals_produce_fragile_recovering_guidance(signal: str) -> None:
    result = _evaluate(signals=CognitionSignals(**{signal: True}))
    assert result.trust == TrustState.FRAGILE
    assert result.momentum == ConversationMomentum.RECOVERING
    assert result.pressure == PressureState.NEEDS_CLARIFICATION


@pytest.mark.parametrize(
    ("prospect", "signals", "expected"),
    [
        (ProspectIntelligenceSummary(), CognitionSignals(), InterestStrength.WEAK),
        (_prospect(interested=False), CognitionSignals(), InterestStrength.NONE),
        (_prospect(interested=True), CognitionSignals(), InterestStrength.MODERATE),
        (
            _prospect(explicit_next_step=PreferredNextStep.DEMO),
            CognitionSignals(),
            InterestStrength.BUYING_SIGNAL,
        ),
        (
            ProspectIntelligenceSummary(),
            CognitionSignals(research_only=True),
            InterestStrength.RESEARCH_MODE,
        ),
    ],
)
def test_interest_uses_only_structured_evidence(  # type: ignore[no-untyped-def]
    prospect, signals, expected
) -> None:
    assert _evaluate(prospect=prospect, signals=signals).interest == expected


def test_discovery_readiness_progresses_from_missing_information() -> None:
    assert (
        _evaluate(problem=ProspectProblem()).discovery_readiness
        == DiscoveryReadiness.NEED_PROBLEM
    )
    assert _evaluate().discovery_readiness == DiscoveryReadiness.NEED_WORKFLOW
    problem = _problem(current_process="manual", impact="missed leads")
    assert (
        _evaluate(problem=problem).discovery_readiness
        == DiscoveryReadiness.NEED_DECISION_CONTEXT
    )
    assert (
        _evaluate(problem=problem, service_fit=_fit(True)).discovery_readiness
        == DiscoveryReadiness.READY_FOR_FIT
    )


@pytest.mark.parametrize(
    ("prospect", "signals", "depth", "energy"),
    [
        (
            _prospect(busy=True),
            CognitionSignals(),
            ResponseLength.SHORT,
            ConversationEnergy.FAST,
        ),
        (
            ProspectIntelligenceSummary(),
            CognitionSignals(confused=True),
            ResponseLength.SHORT,
            ConversationEnergy.SLOW,
        ),
        (
            ProspectIntelligenceSummary(),
            CognitionSignals(fatigued=True),
            ResponseLength.SHORT,
            ConversationEnergy.FATIGUED,
        ),
        (
            ProspectIntelligenceSummary(),
            CognitionSignals(curious=True),
            ResponseLength.MODERATE,
            ConversationEnergy.NORMAL,
        ),
        (
            ProspectIntelligenceSummary(),
            CognitionSignals(detailed_explanation_requested=True),
            ResponseLength.DETAILED,
            ConversationEnergy.NORMAL,
        ),
    ],
)
def test_response_depth_and_energy_are_bounded(  # type: ignore[no-untyped-def]
    prospect, signals, depth, energy
) -> None:
    result = _evaluate(prospect=prospect, signals=signals)
    assert result.recommended_response_depth == depth
    assert result.energy == energy


def test_explicit_no_need_produces_graceful_close() -> None:
    result = _evaluate(signals=CognitionSignals(explicit_no_need=True))
    assert result.buying_guidance == BuyingReadinessGuidance.GRACEFUL_CLOSE
    assert result.momentum == ConversationMomentum.DECLINING


def test_supported_buying_signal_allows_only_advisory_micro_commitment() -> None:
    result = _evaluate(
        service_fit=_fit(True),
        prospect=_prospect(explicit_next_step=PreferredNextStep.DEMO),
    )
    assert result.buying_guidance == BuyingReadinessGuidance.SUGGEST_MICRO_COMMITMENT
    assert result.discovery_readiness == DiscoveryReadiness.READY_FOR_NEXT_STEP
    assert not hasattr(result, "confirmed_demo")


def test_supported_fit_explanation_is_policy_grounded() -> None:
    result = _evaluate(service_fit=_fit(True))
    assert result.buying_guidance == BuyingReadinessGuidance.EXPLAIN_FIT
    forged_input = _input()
    with pytest.raises(SalesGuidanceValidationError):
        validate_sales_guidance(
            replace(result, buying_guidance=BuyingReadinessGuidance.EXPLAIN_FIT),
            forged_input,
        )


def test_relationship_and_trust_use_bounded_history() -> None:
    first = _evaluate()
    second = _evaluate(
        prior_guidance=first,
        returning_discussion=True,
        prospect=_prospect(interested=True),
    )
    assert second.relationship == RelationshipState.RETURNING_DISCUSSION
    assert second.trust == TrustState.ESTABLISHED
    assert second.emotional_posture == EmotionalPosture.INTERESTED
