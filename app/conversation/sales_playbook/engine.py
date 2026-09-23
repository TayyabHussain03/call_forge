"""Pure deterministic opportunity and consultative sequencing reasoning."""

from __future__ import annotations

from app.conversation.prospect_intelligence.contracts import ObjectionType
from app.conversation.sales_cognition.contracts import InterestStrength
from app.conversation.sales_playbook.contracts import (
    AcknowledgementStrategy,
    BenefitCategory,
    BusinessSituation,
    ClarificationStrategy,
    ConsultantStep,
    DiscoveryTopic,
    EducationStrategy,
    ObjectionGuidance,
    OpportunityGuidance,
    PitchReadiness,
    PlaybookRestriction,
    SalesPlaybookInput,
    ServicePlaybook,
)
from app.conversation.sales_playbook.validator import validate_playbook_guidance


class SalesPlaybookIntelligenceEngine:
    """Rank eligible structured playbooks and emit one advisory opportunity."""

    def evaluate(self, value: SalesPlaybookInput) -> OpportunityGuidance:
        if value.context.no_fit or _observed(value.prospect.interested) is False:
            return validate_playbook_guidance(_no_fit(), value)
        candidates = _candidates(value)
        if not candidates:
            return validate_playbook_guidance(_no_fit(), value)
        selected, matched = candidates[0]
        objection = _observed(value.prospect.explicit_objection)
        objection_guidance = _objection_guidance(selected, objection)
        missing = tuple(
            topic
            for topic in selected.discovery_topics
            if topic not in value.context.known_topics
        )
        primary, secondary = _questions(value, missing, objection)
        readiness, step = _readiness(value, selected, primary, objection_guidance)
        future = _future_opportunities(value, selected, candidates)
        existing = selected.service_id in value.context.existing_service_ids
        guidance = OpportunityGuidance(
            selected.service_id,
            matched,
            readiness,
            step,
            primary,
            secondary,
            objection_guidance,
            tuple(sorted(selected.benefits, key=lambda item: item.value)),
            future,
            selected.restrictions
            | frozenset({PlaybookRestriction.NO_PRESSURE_LANGUAGE}),
            existing,
        )
        return validate_playbook_guidance(guidance, value)


def _candidates(
    value: SalesPlaybookInput,
) -> tuple[tuple[ServicePlaybook, tuple[BusinessSituation, ...]], ...]:
    candidates = []
    for playbook in value.playbooks:
        if playbook.service_id not in value.eligible_service_ids:
            continue
        if playbook.industries and value.context.industry not in playbook.industries:
            continue
        matched = tuple(
            sorted(
                playbook.solved_situations & value.context.situations,
                key=lambda item: item.value,
            )
        )
        if not matched:
            continue
        candidates.append((playbook, matched))
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                -_opportunity_score(item[0], item[1], value),
                item[0].priority,
                item[0].service_id,
            ),
        )
    )


def _opportunity_score(
    playbook: ServicePlaybook,
    matched: tuple[BusinessSituation, ...],
    value: SalesPlaybookInput,
) -> int:
    return (
        len(matched) * 100
        + len(playbook.qualification_signals & value.context.qualification_signals) * 10
        + len(playbook.buying_signals & value.context.buying_signals) * 5
    )


def _questions(
    value: SalesPlaybookInput,
    missing: tuple[DiscoveryTopic, ...],
    objection: ObjectionType | None,
) -> tuple[DiscoveryTopic | None, DiscoveryTopic | None]:
    if _observed(value.prospect.busy) is True:
        return None, None
    preferred = []
    if objection == ObjectionType.ALREADY_HAVE_PROVIDER:
        preferred.append(DiscoveryTopic.PROVIDER_SATISFACTION)
    preferred.extend(missing)
    unique = tuple(dict.fromkeys(preferred))
    primary = unique[0] if unique else None
    secondary = (
        unique[1]
        if value.context.true_ambiguity and len(unique) > 1
        else None
    )
    return primary, secondary


def _readiness(
    value: SalesPlaybookInput,
    playbook: ServicePlaybook,
    question: DiscoveryTopic | None,
    objection: ObjectionGuidance | None,
) -> tuple[PitchReadiness, ConsultantStep]:
    if _observed(value.prospect.busy) is True:
        return PitchReadiness.DISCOVERY_REQUIRED, ConsultantStep.CLARIFY
    if objection is not None:
        return PitchReadiness.EDUCATION, ConsultantStep.CLARIFY
    if question is not None:
        return PitchReadiness.DISCOVERY_REQUIRED, ConsultantStep.UNDERSTAND
    interest = value.sales_guidance.interest if value.sales_guidance is not None else None
    playbook_buying_signal = bool(
        playbook.buying_signals & value.context.buying_signals
    )
    if (
        (
            playbook_buying_signal
            or interest
            in {
                InterestStrength.STRONG,
                InterestStrength.BUYING_SIGNAL,
            }
        )
        and value.problem.impact is not None
    ):
        return (
            PitchReadiness.READY_FOR_MICRO_COMMITMENT,
            ConsultantStep.MICRO_COMMITMENT,
        )
    if value.problem.impact is not None:
        return PitchReadiness.VALUE_DISCUSSION, ConsultantStep.RELATE
    if value.problem.meaningful:
        return PitchReadiness.EDUCATION, ConsultantStep.EDUCATE
    return PitchReadiness.DISCOVERY_COMPLETE, ConsultantStep.RECOMMEND


def _objection_guidance(
    playbook: ServicePlaybook, objection: ObjectionType | None
) -> ObjectionGuidance | None:
    if objection is None or objection == ObjectionType.NONE:
        return None
    configured = next(
        (item for item in playbook.objections if item.objection == objection), None
    )
    if configured is not None:
        return configured
    if objection == ObjectionType.ALREADY_HAVE_PROVIDER:
        return ObjectionGuidance(
            objection,
            AcknowledgementStrategy.ACKNOWLEDGE_EXISTING_CHOICE,
            ClarificationStrategy.CLARIFY_CURRENT_SATISFACTION,
            EducationStrategy.EXPLAIN_OPTIMIZATION_PATH,
        )
    if objection in {ObjectionType.NO_TIME, ObjectionType.BAD_TIMING}:
        return ObjectionGuidance(
            objection,
            AcknowledgementStrategy.ACKNOWLEDGE_CONSTRAINT,
            ClarificationStrategy.CLARIFY_TIMING,
            EducationStrategy.DEFER_EDUCATION,
        )
    if objection == ObjectionType.PRICE:
        return ObjectionGuidance(
            objection,
            AcknowledgementStrategy.ACKNOWLEDGE_CONSTRAINT,
            ClarificationStrategy.CLARIFY_BUDGET_CONTEXT,
            EducationStrategy.EXPLAIN_RELEVANT_OUTCOME,
        )
    return ObjectionGuidance(
        objection,
        AcknowledgementStrategy.ACKNOWLEDGE_UNCERTAINTY,
        ClarificationStrategy.CLARIFY_CONCERN,
        EducationStrategy.EXPLAIN_WITH_APPROVED_EVIDENCE,
    )


def _future_opportunities(
    value: SalesPlaybookInput,
    selected: ServicePlaybook,
    candidates: tuple[tuple[ServicePlaybook, tuple[BusinessSituation, ...]], ...],
) -> tuple[str, ...]:
    possible = (
        *(item.service_id for item, _ in candidates[1:]),
        *selected.future_service_ids,
    )
    excluded = (
        value.already_discussed_service_ids
        | value.context.existing_service_ids
        | {selected.service_id}
    )
    return tuple(
        item
        for item in dict.fromkeys(possible)
        if item in value.eligible_service_ids and item not in excluded
    )[:3]


def _no_fit() -> OpportunityGuidance:
    return OpportunityGuidance(
        None,
        (),
        PitchReadiness.NO_FIT,
        ConsultantStep.GRACEFUL_END,
        None,
        None,
        None,
        (),
        (),
        frozenset(
            {
                PlaybookRestriction.NO_GUARANTEED_REVENUE,
                PlaybookRestriction.NO_GUARANTEED_ROI,
                PlaybookRestriction.NO_PRESSURE_LANGUAGE,
            }
        ),
    )


def _observed(value):  # type: ignore[no-untyped-def]
    return value.value if value is not None else None
