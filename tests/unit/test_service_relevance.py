"""Focused deterministic service-relevance and grounding tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from app.catalog.models import Campaign, Service, ServiceCatalog
from app.catalog.scoped_catalog import ScopedCatalog
from app.conversation.consultative.contracts import (
    ProblemCategory,
    ProblemField,
    ProspectProblem,
    ServiceAnswerContext,
    ServiceFitDecision,
    ServiceFitStatus,
)
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


def _catalog(*service_ids: str) -> ScopedCatalog:
    services = {
        service_id: Service(
            service_id,
            service_id.replace("_", " ").title(),
            capabilities=(f"{service_id}_capability",),
        )
        for service_id in service_ids
    }
    return ScopedCatalog(
        ServiceCatalog(
            services,
            {"campaign": Campaign("campaign", "Campaign", tuple(service_ids))},
        ),
        "campaign",
    )


def _rule(
    service_id: str = "automation",
    category: ProblemCategory = ProblemCategory.FOLLOW_UP,
    *,
    required: frozenset[ProblemField] = frozenset(),
    fact_key: str = "workflow_automation",
) -> ServiceRelevanceRule:
    return ServiceRelevanceRule(
        service_id,
        frozenset({category}),
        ("manual follow-up gap",),
        required,
        frozenset({fact_key}),
    )


def _problem(**changes):  # type: ignore[no-untyped-def]
    values = {
        "category": ProblemCategory.FOLLOW_UP,
        "explicit_description": "follow-up is delayed",
    }
    values.update(changes)
    return ProspectProblem(**values)


def _evidence(
    *,
    evidence_id: str = "e1",
    fact_key: str = "workflow_automation",
    statement: str = "The service supports new-lead workflow automation.",
    service_id: str = "automation",
    evidence_type: EvidenceType = EvidenceType.APPROVED_CLAIM,
) -> ApprovedEvidenceItem:
    return ApprovedEvidenceItem(
        evidence_id,
        fact_key,
        evidence_type,
        statement,
        ApprovedEvidenceSourceKind.CURATED_SERVICE,
        EvidenceScope(EvidenceScopeKind.SERVICE, service_id),
    )


def _resolver(
    *rules: ServiceRelevanceRule,
    services: tuple[str, ...] = ("automation",),
) -> ServiceRelevanceResolver:
    return ServiceRelevanceResolver(_catalog(*services), rules or (_rule(),))


def test_service_fit_decision_is_immutable() -> None:
    decision = ServiceFitDecision(ServiceFitStatus.NO_AUTHORIZED_FIT)
    with pytest.raises(FrozenInstanceError):
        decision.status = ServiceFitStatus.POSSIBLE_FIT  # type: ignore[misc]


def test_relevance_rule_cannot_reference_unauthorized_service() -> None:
    with pytest.raises(ValueError):
        ServiceRelevanceResolver(_catalog("seo"), (_rule("automation"),))


def test_no_problem_returns_insufficient_context() -> None:
    result = _resolver().resolve(
        ProspectProblem(), eligible_service_ids=("automation",)
    )
    assert result.status == ServiceFitStatus.INSUFFICIENT_CONTEXT
    assert result.missing_information == (ProblemField.UNDERLYING_PROBLEM,)


def test_no_eligible_service_returns_no_authorized_fit() -> None:
    result = _resolver().resolve(_problem(), eligible_service_ids=())
    assert result.status == ServiceFitStatus.NO_AUTHORIZED_FIT
    assert result.service_id is None


def test_resolver_cannot_expand_eligibility() -> None:
    result = _resolver().resolve(_problem(), eligible_service_ids=("unknown",))
    assert result.status == ServiceFitStatus.NO_AUTHORIZED_FIT


def test_category_mismatch_does_not_create_fit_from_weak_overlap() -> None:
    problem = _problem(category=ProblemCategory.BRANDING)
    result = _resolver().resolve(
        problem,
        eligible_service_ids=("automation",),
        approved_evidence=(_evidence(statement="Mentions automation."),),
    )
    assert result.status == ServiceFitStatus.NO_AUTHORIZED_FIT


def test_required_current_process_missing_keeps_fit_possible() -> None:
    resolver = _resolver(
        _rule(required=frozenset({ProblemField.CURRENT_PROCESS}))
    )
    result = resolver.resolve(
        _problem(),
        eligible_service_ids=("automation",),
        approved_evidence=(_evidence(),),
    )
    assert result.status == ServiceFitStatus.POSSIBLE_FIT
    assert result.missing_information == (ProblemField.CURRENT_PROCESS,)
    assert not result.explanation_allowed


def test_known_current_process_satisfies_relevance_requirement() -> None:
    resolver = _resolver(
        _rule(required=frozenset({ProblemField.CURRENT_PROCESS}))
    )
    result = resolver.resolve(
        _problem(current_process="staff checks email manually"),
        eligible_service_ids=("automation",),
        approved_evidence=(_evidence(),),
    )
    assert result.status == ServiceFitStatus.SUPPORTED_FIT


def test_supported_fit_requires_approved_evidence() -> None:
    result = _resolver().resolve(
        _problem(), eligible_service_ids=("automation",)
    )
    assert result.status == ServiceFitStatus.POSSIBLE_FIT
    assert result.missing_information == (ProblemField.APPROVED_EVIDENCE,)


def test_matching_evidence_enables_supported_fit() -> None:
    result = _resolver().resolve(
        _problem(),
        eligible_service_ids=("automation",),
        approved_evidence=(_evidence(),),
    )
    assert result == ServiceFitDecision(
        ServiceFitStatus.SUPPORTED_FIT,
        "automation",
        ("manual follow-up gap",),
        evidence_ids=("e1",),
        explanation_allowed=True,
    )


def test_wrong_fact_key_cannot_ground_fit() -> None:
    result = _resolver().resolve(
        _problem(),
        eligible_service_ids=("automation",),
        approved_evidence=(_evidence(fact_key="unrelated"),),
    )
    assert result.status == ServiceFitStatus.POSSIBLE_FIT


def test_other_service_evidence_cannot_ground_fit() -> None:
    result = _resolver().resolve(
        _problem(),
        eligible_service_ids=("automation",),
        approved_evidence=(_evidence(service_id="seo"),),
    )
    assert result.status == ServiceFitStatus.POSSIBLE_FIT


def test_conflicting_evidence_fails_closed() -> None:
    result = _resolver().resolve(
        _problem(),
        eligible_service_ids=("automation",),
        approved_evidence=(
            _evidence(evidence_id="e1", statement="Capability A."),
            _evidence(evidence_id="e2", statement="Capability B."),
        ),
    )
    assert result.status == ServiceFitStatus.POSSIBLE_FIT
    assert not result.explanation_allowed


def test_policy_restriction_wins_over_known_evidence() -> None:
    result = _resolver().resolve(
        _problem(),
        eligible_service_ids=("automation",),
        approved_evidence=(_evidence(),),
        disclosure_allowed=False,
    )
    assert result.status == ServiceFitStatus.POSSIBLE_FIT
    assert result.missing_information == (ProblemField.POLICY_DISCLOSURE,)


def test_already_offered_service_is_not_repeated() -> None:
    result = _resolver().resolve(
        _problem(),
        eligible_service_ids=("automation",),
        offered_service_ids=("automation",),
        approved_evidence=(_evidence(),),
    )
    assert result.status == ServiceFitStatus.INSUFFICIENT_CONTEXT
    assert result.missing_information == (ProblemField.ALREADY_OFFERED,)


def test_multiple_plausible_services_clarify_instead_of_selecting() -> None:
    resolver = _resolver(
        _rule("automation"),
        _rule("crm_workflow"),
        services=("automation", "crm_workflow"),
    )
    result = resolver.resolve(
        _problem(),
        eligible_service_ids=("automation", "crm_workflow"),
        approved_evidence=(
            _evidence(service_id="automation"),
            _evidence(
                evidence_id="e2",
                service_id="crm_workflow",
            ),
        ),
    )
    assert result.status == ServiceFitStatus.INSUFFICIENT_CONTEXT
    assert result.service_id is None


def test_multiple_fit_resolution_is_deterministic() -> None:
    resolver = _resolver(
        _rule("automation"),
        _rule("crm_workflow"),
        services=("automation", "crm_workflow"),
    )
    inputs = {"eligible_service_ids": ("crm_workflow", "automation")}
    assert resolver.resolve(_problem(), **inputs) == resolver.resolve(
        _problem(), **inputs
    )


def test_supported_fit_never_contains_execution_or_authority() -> None:
    result = _resolver().resolve(
        _problem(),
        eligible_service_ids=("automation",),
        approved_evidence=(_evidence(),),
    )
    for forbidden in (
        "next_state",
        "execute_action",
        "approved_discount",
        "service_authorized",
    ):
        assert not hasattr(result, forbidden)


def test_answer_context_requires_supported_fit() -> None:
    answer = _resolver().build_answer_context(
        ServiceFitDecision(ServiceFitStatus.POSSIBLE_FIT, "automation"),
        _problem(),
        (_evidence(),),
    )
    assert answer is None


def test_answer_context_contains_only_selected_service_evidence() -> None:
    resolver = _resolver()
    fit = resolver.resolve(
        _problem(),
        eligible_service_ids=("automation",),
        approved_evidence=(_evidence(),),
    )
    unrelated = _evidence(
        evidence_id="other",
        fact_key="other",
        service_id="seo",
    )
    answer = resolver.build_answer_context(
        fit, _problem(), (_evidence(), unrelated)
    )
    assert answer is not None
    assert answer.service_id == "automation"
    assert tuple(item.evidence_id for item in answer.approved_evidence) == ("e1",)


def test_answer_context_revalidates_evidence_instead_of_trusting_reused_id() -> None:
    resolver = _resolver()
    fit = resolver.resolve(
        _problem(),
        eligible_service_ids=("automation",),
        approved_evidence=(_evidence(),),
    )
    substituted = _evidence(
        fact_key="unapproved_fact",
        statement="A substituted statement must not pass by id alone.",
    )

    assert resolver.build_answer_context(fit, _problem(), (substituted,)) is None


def test_answer_context_preserves_problem_and_current_process() -> None:
    problem = _problem(current_process="staff checks email manually")
    resolver = _resolver()
    fit = resolver.resolve(
        problem,
        eligible_service_ids=("automation",),
        approved_evidence=(_evidence(),),
    )
    answer = resolver.build_answer_context(fit, problem, (_evidence(),))
    assert answer is not None
    assert answer.problem_summary == "follow-up is delayed"
    assert answer.current_process_summary == "staff checks email manually"


def test_service_description_is_selected_only_from_approved_evidence() -> None:
    description = _evidence(
        statement="Approved service description.",
        evidence_type=EvidenceType.SERVICE_DESCRIPTION,
    )
    resolver = _resolver()
    fit = resolver.resolve(
        _problem(),
        eligible_service_ids=("automation",),
        approved_evidence=(description,),
    )
    answer = resolver.build_answer_context(fit, _problem(), (description,))
    assert answer is not None
    assert answer.approved_description == "Approved service description."


def test_answer_context_rejects_empty_evidence() -> None:
    with pytest.raises(ValueError):
        ServiceAnswerContext(
            "automation",
            "Automation",
            None,
            (),
            (),
            "follow-up is delayed",
        )


def test_case_study_fit_does_not_encode_a_guarantee() -> None:
    case_study = _evidence(
        evidence_type=EvidenceType.CASE_STUDY,
        statement="One client reduced manual handling time.",
    )
    resolver = _resolver()
    fit = resolver.resolve(
        _problem(),
        eligible_service_ids=("automation",),
        approved_evidence=(case_study,),
    )
    answer = resolver.build_answer_context(fit, _problem(), (case_study,))
    assert answer is not None
    assert not hasattr(answer, "guaranteed_outcome")
