"""Deterministic relevance constrained by catalog authorization and eligibility."""

from __future__ import annotations

from dataclasses import dataclass

from app.catalog.scoped_catalog import ScopedCatalog
from app.conversation.consultative.contracts import (
    ProblemCategory,
    ProblemField,
    ProspectProblem,
    ServiceAnswerContext,
    ServiceFitDecision,
    ServiceFitStatus,
)
from app.conversation.context.contracts import (
    ApprovedEvidenceItem,
    EvidenceScopeKind,
    EvidenceType,
)


@dataclass(frozen=True)
class ServiceRelevanceRule:
    """Configured relevance only; it grants neither eligibility nor claims."""

    service_id: str
    problem_categories: frozenset[ProblemCategory]
    matched_problem_aspects: tuple[str, ...]
    required_fields: frozenset[ProblemField] = frozenset()
    evidence_fact_keys: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.service_id.strip() or len(self.service_id) > 100:
            raise ValueError("rule service id must be bounded")
        if not self.problem_categories or any(
            not isinstance(item, ProblemCategory) for item in self.problem_categories
        ):
            raise ValueError("relevance rule requires typed problem categories")
        if len(self.matched_problem_aspects) > 4 or any(
            not item.strip() or len(item) > 100
            for item in self.matched_problem_aspects
        ):
            raise ValueError("matched problem aspects must be bounded")
        if any(not isinstance(item, ProblemField) for item in self.required_fields):
            raise TypeError("required fields must contain ProblemField values")
        if len(self.evidence_fact_keys) > 8 or any(
            not item.strip() or len(item) > 100 for item in self.evidence_fact_keys
        ):
            raise ValueError("evidence fact keys must be bounded")


class ServiceRelevanceResolver:
    """Resolve one useful service only inside existing deterministic authority."""

    def __init__(
        self,
        scoped_catalog: ScopedCatalog,
        rules: tuple[ServiceRelevanceRule, ...],
    ) -> None:
        self._catalog = scoped_catalog
        self._rules = tuple(rules)
        if len({rule.service_id for rule in self._rules}) != len(self._rules):
            raise ValueError("service relevance rules must have unique service ids")
        if any(
            not scoped_catalog.is_service_authorized(rule.service_id)
            for rule in self._rules
        ):
            raise ValueError("relevance rules may reference only authorized services")

    def resolve(
        self,
        problem: ProspectProblem,
        *,
        eligible_service_ids: tuple[str, ...],
        offered_service_ids: tuple[str, ...] = (),
        approved_evidence: tuple[ApprovedEvidenceItem, ...] = (),
        disclosure_allowed: bool = True,
    ) -> ServiceFitDecision:
        if not problem.meaningful:
            return ServiceFitDecision(
                ServiceFitStatus.INSUFFICIENT_CONTEXT,
                missing_information=(ProblemField.UNDERLYING_PROBLEM,),
            )
        authorized = self._catalog.authorized_service_ids()
        eligible = authorized.intersection(eligible_service_ids)
        offered = frozenset(offered_service_ids)
        matching = tuple(
            rule
            for rule in self._rules
            if rule.service_id in eligible
            and rule.service_id not in offered
            and problem.category in rule.problem_categories
        )
        offered_matches = tuple(
            rule
            for rule in self._rules
            if rule.service_id in eligible
            and rule.service_id in offered
            and problem.category in rule.problem_categories
        )
        if not matching:
            if offered_matches:
                return ServiceFitDecision(
                    ServiceFitStatus.INSUFFICIENT_CONTEXT,
                    missing_information=(ProblemField.ALREADY_OFFERED,),
                )
            return ServiceFitDecision(ServiceFitStatus.NO_AUTHORIZED_FIT)
        if len(matching) > 1:
            return ServiceFitDecision(
                ServiceFitStatus.INSUFFICIENT_CONTEXT,
                missing_information=(_highest_value_gap(problem),),
            )
        rule = matching[0]
        missing = tuple(
            field for field in rule.required_fields if not _known(problem, field)
        )
        if missing:
            return ServiceFitDecision(
                ServiceFitStatus.POSSIBLE_FIT,
                rule.service_id,
                rule.matched_problem_aspects,
                tuple(sorted(missing, key=lambda item: item.value))[:3],
            )
        if not disclosure_allowed:
            return ServiceFitDecision(
                ServiceFitStatus.POSSIBLE_FIT,
                rule.service_id,
                rule.matched_problem_aspects,
                (ProblemField.POLICY_DISCLOSURE,),
            )
        evidence = _matching_evidence(
            approved_evidence,
            rule,
            self._catalog.campaign_id,
        )
        if not evidence:
            return ServiceFitDecision(
                ServiceFitStatus.POSSIBLE_FIT,
                rule.service_id,
                rule.matched_problem_aspects,
                (ProblemField.APPROVED_EVIDENCE,),
            )
        return ServiceFitDecision(
            ServiceFitStatus.SUPPORTED_FIT,
            rule.service_id,
            rule.matched_problem_aspects,
            evidence_ids=tuple(item.evidence_id for item in evidence),
            explanation_allowed=True,
        )

    def build_answer_context(
        self,
        fit: ServiceFitDecision,
        problem: ProspectProblem,
        approved_evidence: tuple[ApprovedEvidenceItem, ...],
        *,
        disclosure_allowed: bool = True,
        unresolved_technical_questions: tuple[str, ...] = (),
    ) -> ServiceAnswerContext | None:
        if (
            fit.status != ServiceFitStatus.SUPPORTED_FIT
            or not fit.explanation_allowed
            or fit.service_id is None
            or not disclosure_allowed
        ):
            return None
        service = self._catalog.get_service(fit.service_id)
        if service is None:
            return None
        rule = next(
            (item for item in self._rules if item.service_id == fit.service_id),
            None,
        )
        if rule is None:
            return None
        evidence = _matching_evidence(
            approved_evidence,
            rule,
            self._catalog.campaign_id,
        )
        if tuple(item.evidence_id for item in evidence) != fit.evidence_ids:
            return None
        description = next(
            (
                item.statement
                for item in evidence
                if item.evidence_type == EvidenceType.SERVICE_DESCRIPTION
            ),
            None,
        )
        return ServiceAnswerContext(
            service.id,
            service.name,
            description,
            tuple(service.capabilities[:4]),
            evidence,
            problem.explicit_description or problem.friction or "current problem",
            problem.current_process,
            True,
            unresolved_technical_questions,
        )


def _matching_evidence(
    items: tuple[ApprovedEvidenceItem, ...],
    rule: ServiceRelevanceRule,
    campaign_id: str,
) -> tuple[ApprovedEvidenceItem, ...]:
    matches = tuple(
        sorted(
            (
                item
                for item in items
                if item.fact_key in rule.evidence_fact_keys
                and _scope_matches(item, rule.service_id, campaign_id)
            ),
            key=lambda item: item.evidence_id,
        )
    )
    statements_by_fact: dict[str, set[str]] = {}
    for item in matches:
        statements_by_fact.setdefault(item.fact_key, set()).add(item.statement)
    if any(len(statements) > 1 for statements in statements_by_fact.values()):
        return ()
    return matches[:8]


def _scope_matches(
    item: ApprovedEvidenceItem, service_id: str, campaign_id: str
) -> bool:
    if item.scope.kind == EvidenceScopeKind.GLOBAL:
        return True
    if item.scope.kind == EvidenceScopeKind.CAMPAIGN:
        return item.scope.scope_id == campaign_id
    return item.scope.scope_id == service_id


def _known(problem: ProspectProblem, field: ProblemField) -> bool:
    return {
        ProblemField.UNDERLYING_PROBLEM: problem.meaningful,
        ProblemField.CURRENT_PROCESS: problem.current_process is not None,
        ProblemField.FRICTION: problem.friction is not None,
        ProblemField.IMPACT: problem.impact is not None,
        ProblemField.DESIRED_OUTCOME: problem.desired_outcome is not None,
        ProblemField.SOURCE_OR_CHANNEL: problem.source_or_channel is not None,
    }.get(field, False)


def _highest_value_gap(problem: ProspectProblem) -> ProblemField:
    for field in (
        ProblemField.CURRENT_PROCESS,
        ProblemField.FRICTION,
        ProblemField.IMPACT,
        ProblemField.DESIRED_OUTCOME,
    ):
        if not _known(problem, field):
            return field
    return ProblemField.UNDERLYING_PROBLEM
