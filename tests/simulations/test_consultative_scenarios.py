"""Deterministic multi-turn simulations for Slice 17 consultative behavior."""

from __future__ import annotations

from app.catalog.loader import load_catalog
from app.catalog.scoped_catalog import ScopedCatalog
from app.conversation.consultative.contracts import (
    ConsultativeDecisionInput,
    ConsultativeMove,
    ConsultativeTurnSignals,
    ProblemCategory,
    ProblemEvidence,
    ProblemEvidenceBasis,
    ProblemField,
    ProspectProblem,
    ServiceFitDecision,
    ServiceFitStatus,
)
from app.conversation.consultative.engine import ConsultativeDecisionEngine
from app.conversation.consultative.problem import ProblemModelUpdater
from app.conversation.consultative.service_relevance import (
    ServiceRelevanceResolver,
    ServiceRelevanceRule,
)
from app.conversation.context.contracts import (
    ApprovedEvidenceItem,
    ApprovedEvidenceSourceKind,
    EvidenceScope,
    EvidenceScopeKind,
    EvidenceType,
)
from app.conversation.prospect_intelligence.contracts import (
    CurrentSolutionContext,
    EvidenceProvenance,
    EvidenceSourceKind,
    ObservedValue,
    ProspectIntelligenceSummary,
    SolutionSatisfaction,
)
from app.conversation.response_planning.contracts import (
    AuthoritativeResultKind,
    InterruptionCategory,
    PendingConversationIntent,
    ResponsePlanningInput,
)
from app.conversation.response_planning.planner import ResponsePlanner
from app.conversation.response_rendering.contracts import (
    ResponseRenderInput,
    ResponseRenderingBudget,
)
from app.conversation.response_rendering.renderer import DeterministicResponseRenderer
from app.conversation.understanding.contracts import LanguageProfile, LanguageScript
from app.core.constants import ConversationState


def _fit(status: ServiceFitStatus = ServiceFitStatus.INSUFFICIENT_CONTEXT):
    return ServiceFitDecision(status)


def _decide(
    problem: ProspectProblem,
    *,
    fit: ServiceFitDecision | None = None,
    signals: ConsultativeTurnSignals = ConsultativeTurnSignals(),
    prospect: ProspectIntelligenceSummary | None = None,
    language: LanguageProfile | None = None,
    pending: PendingConversationIntent | None = None,
):
    return ConsultativeDecisionEngine().decide(
        ConsultativeDecisionInput(
            problem,
            fit or _fit(),
            prospect=prospect,
            language_profile=language,
            signals=signals,
            pending_intent=pending,
        )
    )


def _render(decision, language=None) -> str:  # type: ignore[no-untyped-def]
    plan = ResponsePlanner().plan(
        ResponsePlanningInput(
            ConversationState.LISTEN,
            AuthoritativeResultKind.EXECUTED,
            "bounded current turn",
            language_profile=language,
            consultative_decision=decision,
        )
    )
    return DeterministicResponseRenderer().render(
        ResponseRenderInput(
            plan,
            AuthoritativeResultKind.EXECUTED,
            ResponseRenderingBudget(5),
        )
    ).text


def _update(previous: ProspectProblem, turn: str, **values) -> ProspectProblem:
    return ProblemModelUpdater().update(
        previous,
        ProblemEvidence(
            source_turn_id=turn,
            evidence_basis=ProblemEvidenceBasis.EXPLICIT,
            **values,
        ),
    )


def test_requested_chatbot_then_problem_discovery_does_not_prescribe_early() -> None:
    first = _update(
        ProspectProblem(),
        "t1",
        requested_solution="chatbot",
    )
    first_decision = _decide(first)
    second = _update(
        first,
        "t2",
        category=ProblemCategory.CUSTOMER_QUESTIONS,
        explicit_description="staff repeatedly answers the same questions",
    )
    second_decision = _decide(second)

    assert first_decision.move == ConsultativeMove.DISCOVER_PROBLEM
    assert second.requested_solution == "chatbot"
    assert second_decision.move == ConsultativeMove.UNDERSTAND_CURRENT_PROCESS


def test_progressive_follow_up_discovery_reaches_only_grounded_fit() -> None:
    problem = _update(
        ProspectProblem(),
        "t1",
        category=ProblemCategory.FOLLOW_UP,
        explicit_description="new leads are followed up late",
    )
    assert _decide(problem).move == ConsultativeMove.UNDERSTAND_CURRENT_PROCESS
    problem = _update(
        problem,
        "t2",
        current_process="staff checks a shared inbox manually",
        impact="some inquiries go cold",
    )
    evidence = ApprovedEvidenceItem(
        "e1",
        "workflow_automation",
        EvidenceType.APPROVED_CLAIM,
        "AI Automation supports new-lead workflow automation.",
        ApprovedEvidenceSourceKind.CURATED_SERVICE,
        EvidenceScope(EvidenceScopeKind.SERVICE, "ai_automation"),
    )
    resolver = ServiceRelevanceResolver(
        ScopedCatalog(load_catalog("app/config/service_config.yaml"), "campaign_full"),
        (
            ServiceRelevanceRule(
                "ai_automation",
                frozenset({ProblemCategory.FOLLOW_UP}),
                ("manual follow-up gap",),
                frozenset({ProblemField.CURRENT_PROCESS}),
                frozenset({"workflow_automation"}),
            ),
        ),
    )
    fit = resolver.resolve(
        problem,
        eligible_service_ids=("ai_automation",),
        approved_evidence=(evidence,),
    )

    assert fit.status == ServiceFitStatus.SUPPORTED_FIT
    assert _decide(problem, fit=fit).move == ConsultativeMove.EXPLAIN_RELEVANT_FIT


def test_later_correction_replaces_stale_problem_and_next_question() -> None:
    first = _update(
        ProspectProblem(),
        "t1",
        category=ProblemCategory.WEBSITE,
        explicit_description="the website needs redesign",
    )
    corrected = ProblemModelUpdater().update(
        first,
        ProblemEvidence(
            "t2",
            ProblemCategory.FOLLOW_UP,
            "the website is fine; follow-up is slow",
            evidence_basis=ProblemEvidenceBasis.EXPLICIT,
            correction=True,
        ),
    )

    assert corrected.category == ProblemCategory.FOLLOW_UP
    assert corrected.current_process is None
    assert _decide(corrected).move == ConsultativeMove.UNDERSTAND_CURRENT_PROCESS


def test_existing_provider_is_explored_without_assuming_dissatisfaction() -> None:
    provenance = EvidenceProvenance("t1", EvidenceSourceKind.EXPLICIT_STATEMENT)
    prospect = ProspectIntelligenceSummary(
        current_solution=CurrentSolutionContext(
            "existing provider", None, SolutionSatisfaction.UNKNOWN, provenance
        )
    )
    first = _decide(ProspectProblem(), prospect=prospect)
    second_problem = _update(
        ProspectProblem(),
        "t2",
        category=ProblemCategory.VISIBILITY,
        explicit_description="visibility is still inconsistent",
    )

    assert first.move == ConsultativeMove.UNDERSTAND_EXISTING_SOLUTION
    assert (
        _decide(second_problem, prospect=prospect).move
        == ConsultativeMove.UNDERSTAND_EXISTING_SOLUTION
    )


def test_busy_but_interested_stays_low_pressure_across_turns() -> None:
    provenance = EvidenceProvenance("t1", EvidenceSourceKind.EXPLICIT_STATEMENT)
    prospect = ProspectIntelligenceSummary(
        busy=ObservedValue(True, provenance),
        interested=ObservedValue(True, provenance),
    )
    first = _decide(ProspectProblem(), prospect=prospect)
    second = _decide(
        ProspectProblem(),
        prospect=prospect,
        pending=PendingConversationIntent("discover", "understand current process"),
    )

    assert first.move == second.move == ConsultativeMove.LOW_PRESSURE_CALLBACK
    assert "not scheduled" not in _render(second).lower()


def test_multilingual_misunderstanding_simplifies_then_resumes_discovery() -> None:
    language = LanguageProfile(
        primary_language="ur",
        script=LanguageScript.LATIN,
        preferred_response_language="ur",
        preferred_script=LanguageScript.LATIN,
    )
    pending = PendingConversationIntent("discover", "understand current process")
    recovery = _decide(
        ProspectProblem(),
        signals=ConsultativeTurnSignals(misunderstanding=True),
        language=language,
        pending=pending,
    )
    resumed = _decide(
        _update(
            ProspectProblem(),
            "t2",
            category=ProblemCategory.WORKFLOW,
            explicit_description="manual workflow is slow",
        ),
        language=language,
    )

    assert recovery.move == ConsultativeMove.SIMPLIFY_EXPLANATION
    assert "simple words" in _render(recovery, language)
    assert resumed.move == ConsultativeMove.UNDERSTAND_CURRENT_PROCESS


def test_direct_question_temporarily_wins_then_discovery_can_continue() -> None:
    problem = _update(
        ProspectProblem(),
        "t1",
        category=ProblemCategory.LEAD_FLOW,
        explicit_description="lead volume is uneven",
    )
    direct = _decide(
        problem,
        signals=ConsultativeTurnSignals(direct_question=True),
        pending=PendingConversationIntent("discover", "understand lead flow"),
    )
    continued = _decide(problem)

    assert direct.move == ConsultativeMove.ANSWER_DIRECT_QUESTION
    assert continued.move == ConsultativeMove.UNDERSTAND_CURRENT_PROCESS


def test_explicit_no_problem_closes_without_forcing_a_service_next_turn() -> None:
    first = _decide(
        ProspectProblem(),
        signals=ConsultativeTurnSignals(explicit_no_problem=True),
    )
    second = _decide(
        ProspectProblem(),
        fit=ServiceFitDecision(ServiceFitStatus.NO_AUTHORIZED_FIT),
        signals=ConsultativeTurnSignals(explicit_no_problem=True),
    )

    assert first.move == second.move == ConsultativeMove.GRACEFUL_CLOSE
    assert "won't force it" in _render(second)
