"""Deterministic Post-Conversation Outcome Intelligence engine.

Converts trusted structured state from the completed conversation into
a bounded LeadOutcome. No model call. No raw text parsing.
Every field is derived from authoritative pipeline state or explicitly
marked UNKNOWN.
"""

from __future__ import annotations

from app.conversation.business_conversation.contracts import (
    BusinessConversationSnapshot,
    ObservedBusinessProblem,
)
from app.conversation.business_diagnostic.contracts import BusinessDiagnosticSnapshot
from app.conversation.prospect_intelligence.contracts import (
    DecisionAuthority,
    InferredProspectState,
    ObservedProspectFacts,
    ObjectionType,
    PreferredNextStep,
)
from app.conversation.qualification.contracts import (
    QualificationCompleteness,
    QualificationSnapshot,
)
from app.conversation.sales_playbook.contracts import OpportunityGuidance
from app.runtime.chat.outcome.contracts import (
    DNCStatus,
    DecisionMakerStatus,
    ExecutionStatus,
    FieldConfidence,
    FollowUpNeed,
    IdentifiedProblem,
    InterestLevel,
    LeadOutcome,
    OutcomeIntelligenceInput,
    OutcomeStatus,
    OutcomeSummaryLine,
    RecordedObjection,
    RelevantService,
    TerminationReason,
)


class OutcomeIntelligenceEngine:
    """Deterministic outcome resolver. No side effects, no authority."""

    def resolve(self, inp: OutcomeIntelligenceInput) -> LeadOutcome:
        dnc_status = _dnc_status(inp)
        execution_status = _execution_status(inp)
        outcome_status = _outcome_status(inp, dnc_status)
        interest_level = _interest_level(inp)
        problems = _problems(inp)
        services = _services(inp)
        objections = _objections(inp)
        decision_maker = _decision_maker_status(inp)
        qualification_completeness = _qualification_completeness(inp)
        next_step = _next_step(inp)
        follow_up = _follow_up(inp, outcome_status, next_step)
        questions = _outstanding_questions(inp)
        summary = _summary(
            outcome_status, interest_level, dnc_status, problems,
            services, objections, next_step, decision_maker, questions,
        )

        return LeadOutcome(
            lead_id=inp.lead_id,
            campaign_id=inp.campaign_id,
            session_id=inp.session_id,
            tenant_id=inp.tenant_id,
            outcome_status=outcome_status,
            interest_level=interest_level,
            dnc_status=dnc_status,
            execution_status=execution_status,
            termination_reason=inp.termination_reason,
            identified_problems=problems,
            relevant_services=services,
            objections=objections,
            decision_maker_status=decision_maker,
            qualification_completeness=qualification_completeness,
            next_step=next_step,
            follow_up_needed=follow_up,
            outstanding_questions=questions,
            summary_lines=summary,
        )


# ---------------------------------------------------------------------------
# Resolvers — each derives one field from trusted state
# ---------------------------------------------------------------------------


def _dnc_status(inp: OutcomeIntelligenceInput) -> DNCStatus:
    if inp.is_dnc:
        return DNCStatus.DNC_CONFIRMED
    return DNCStatus.NOT_DNC


def _execution_status(inp: OutcomeIntelligenceInput) -> ExecutionStatus:
    if inp.session_error:
        return ExecutionStatus.FAILED
    return ExecutionStatus.COMPLETED


def _outcome_status(inp: OutcomeIntelligenceInput, dnc: DNCStatus) -> OutcomeStatus:
    if dnc == DNCStatus.DNC_CONFIRMED:
        return OutcomeStatus.DNC

    if inp.is_not_interested:
        return OutcomeStatus.NOT_INTERESTED

    observed = _get_observed(inp)
    inferred = _get_inferred(inp)

    has_interest = (
        observed is not None
        and observed.explicit_interest_signal is not None
        and observed.explicit_interest_signal.value is True
    )
    has_next_step = _next_step(inp) is not None

    if has_interest and has_next_step:
        return OutcomeStatus.FOLLOW_UP

    if has_interest:
        return OutcomeStatus.INTERESTED

    playbook = _get_playbook(inp)
    if playbook is not None and playbook.selected_service_id is not None:
        if has_next_step:
            return OutcomeStatus.FOLLOW_UP
        return OutcomeStatus.NEEDS_MORE_INFORMATION

    if not inp.conversation_terminal:
        return OutcomeStatus.UNKNOWN

    return OutcomeStatus.COMPLETED_NO_COMMITMENT


def _interest_level(inp: OutcomeIntelligenceInput) -> InterestLevel:
    if inp.is_dnc or inp.is_not_interested:
        return InterestLevel.NO_INTEREST

    observed = _get_observed(inp)
    if observed is not None and observed.explicit_interest_signal is not None:
        if observed.explicit_interest_signal.value is True:
            return InterestLevel.EXPLICIT_INTEREST
        return InterestLevel.NO_INTEREST

    inferred = _get_inferred(inp)
    if inferred is not None and inferred.openness is not None:
        from app.conversation.prospect_intelligence.contracts import InformationLevel
        if inferred.openness.value in (InformationLevel.HIGH, InformationLevel.MEDIUM):
            return InterestLevel.POSSIBLE_INTEREST

    return InterestLevel.UNKNOWN


def _problems(inp: OutcomeIntelligenceInput) -> tuple[IdentifiedProblem, ...]:
    result: list[IdentifiedProblem] = []

    observed = _get_observed(inp)
    if observed is not None:
        for pain in observed.explicit_pain_points:
            result.append(IdentifiedProblem(
                description=pain.summary[:200],
                confidence=FieldConfidence.OBSERVED,
            ))

    bcs = _get_business_conversation(inp)
    if bcs is not None:
        for problem in bcs.problems:
            desc = problem.detail[:200] if hasattr(problem, "detail") else str(problem)[:200]
            if not any(p.description == desc for p in result):
                result.append(IdentifiedProblem(
                    description=desc,
                    confidence=FieldConfidence.OBSERVED,
                ))

    return tuple(result[:10])


def _services(inp: OutcomeIntelligenceInput) -> tuple[RelevantService, ...]:
    result: list[RelevantService] = []

    playbook = _get_playbook(inp)
    if playbook is not None and playbook.selected_service_id is not None:
        situations = ", ".join(
            s.value if hasattr(s, "value") else str(s)
            for s in playbook.matched_situations
        ) if playbook.matched_situations else "playbook match"
        result.append(RelevantService(
            service_id=playbook.selected_service_id,
            match_reason=situations[:200],
            confidence=FieldConfidence.OBSERVED,
        ))
        for future_id in playbook.future_opportunity_ids:
            if future_id != playbook.selected_service_id:
                result.append(RelevantService(
                    service_id=future_id,
                    match_reason="future opportunity from playbook",
                    confidence=FieldConfidence.INFERRED,
                ))

    return tuple(result[:10])


def _objections(inp: OutcomeIntelligenceInput) -> tuple[RecordedObjection, ...]:
    result: list[RecordedObjection] = []

    observed = _get_observed(inp)
    if observed is not None and observed.explicit_objection is not None:
        if observed.explicit_objection.value != ObjectionType.NONE:
            result.append(RecordedObjection(
                objection_type=observed.explicit_objection.value.value,
                confidence=FieldConfidence.OBSERVED,
            ))

    inferred = _get_inferred(inp)
    if inferred is not None and inferred.objection_type is not None:
        if inferred.objection_type.value != ObjectionType.NONE:
            inferred_type = inferred.objection_type.value.value
            if not any(o.objection_type == inferred_type for o in result):
                result.append(RecordedObjection(
                    objection_type=inferred_type,
                    confidence=FieldConfidence.INFERRED,
                ))

    return tuple(result[:10])


def _decision_maker_status(inp: OutcomeIntelligenceInput) -> DecisionMakerStatus:
    observed = _get_observed(inp)
    if observed is not None and observed.explicit_decision_authority_statement is not None:
        auth = observed.explicit_decision_authority_statement.value
        if auth in (DecisionAuthority.FINAL, DecisionAuthority.HIGH):
            return DecisionMakerStatus.CONFIRMED
        if auth == DecisionAuthority.LOW:
            return DecisionMakerStatus.NOT_DECISION_MAKER

    inferred = _get_inferred(inp)
    if inferred is not None and inferred.decision_authority is not None:
        auth = inferred.decision_authority.value
        if auth in (DecisionAuthority.FINAL, DecisionAuthority.HIGH):
            return DecisionMakerStatus.CONFIRMED
        if auth == DecisionAuthority.LOW:
            return DecisionMakerStatus.NOT_DECISION_MAKER

    return DecisionMakerStatus.UNKNOWN


def _qualification_completeness(inp: OutcomeIntelligenceInput) -> str | None:
    qual = _get_qualification(inp)
    if qual is None:
        return None
    return qual.completeness.value


def _next_step(inp: OutcomeIntelligenceInput) -> str | None:
    observed = _get_observed(inp)
    if observed is not None and observed.explicit_next_step_request is not None:
        step = observed.explicit_next_step_request.value
        if step != PreferredNextStep.UNKNOWN:
            return step.value

    inferred = _get_inferred(inp)
    if inferred is not None and inferred.preferred_next_step is not None:
        step = inferred.preferred_next_step.value
        if step != PreferredNextStep.UNKNOWN:
            return step.value

    return None


def _follow_up(
    inp: OutcomeIntelligenceInput,
    outcome: OutcomeStatus,
    next_step: str | None,
) -> FollowUpNeed:
    if inp.is_dnc:
        return FollowUpNeed.NOT_NEEDED
    if inp.is_not_interested:
        return FollowUpNeed.NOT_NEEDED
    if inp.session_error:
        return FollowUpNeed.NEEDED
    if outcome == OutcomeStatus.FOLLOW_UP:
        return FollowUpNeed.NEEDED
    if next_step is not None:
        return FollowUpNeed.NEEDED
    if outcome == OutcomeStatus.INTERESTED:
        return FollowUpNeed.NEEDED
    if outcome == OutcomeStatus.NEEDS_MORE_INFORMATION:
        return FollowUpNeed.NEEDED
    return FollowUpNeed.UNKNOWN


def _outstanding_questions(inp: OutcomeIntelligenceInput) -> tuple[str, ...]:
    result: list[str] = []

    bcs = _get_business_conversation(inp)
    if bcs is not None:
        for area in bcs.unknown_areas:
            result.append(area.value if hasattr(area, "value") else str(area))

    qual = _get_qualification(inp)
    if qual is not None:
        for field_state in qual.fields:
            if hasattr(field_state, "observed_value") and field_state.observed_value is None:
                if hasattr(field_state, "field") and hasattr(field_state.field, "display_label"):
                    result.append(field_state.field.display_label)

    return tuple(result[:10])


def _summary(
    outcome: OutcomeStatus,
    interest: InterestLevel,
    dnc: DNCStatus,
    problems: tuple[IdentifiedProblem, ...],
    services: tuple[RelevantService, ...],
    objections: tuple[RecordedObjection, ...],
    next_step: str | None,
    decision_maker: DecisionMakerStatus,
    questions: tuple[str, ...],
) -> tuple[OutcomeSummaryLine, ...]:
    lines: list[OutcomeSummaryLine] = []

    lines.append(OutcomeSummaryLine("Outcome", outcome.value))
    lines.append(OutcomeSummaryLine("Interest", interest.value))

    if dnc == DNCStatus.DNC_CONFIRMED:
        lines.append(OutcomeSummaryLine("DNC", "Confirmed"))

    if problems:
        lines.append(OutcomeSummaryLine(
            "Problem", problems[0].description,
        ))

    if services:
        lines.append(OutcomeSummaryLine(
            "Relevant service", services[0].service_id,
        ))

    if objections:
        lines.append(OutcomeSummaryLine(
            "Objection", objections[0].objection_type,
        ))

    if decision_maker != DecisionMakerStatus.UNKNOWN:
        lines.append(OutcomeSummaryLine(
            "Decision maker", decision_maker.value,
        ))

    if next_step is not None:
        lines.append(OutcomeSummaryLine("Next step", next_step))

    if questions:
        lines.append(OutcomeSummaryLine(
            "Outstanding question", questions[0],
        ))

    return tuple(lines)


# ---------------------------------------------------------------------------
# Typed accessors — safe extraction from generic input
# ---------------------------------------------------------------------------


def _get_observed(inp: OutcomeIntelligenceInput) -> ObservedProspectFacts | None:
    if isinstance(inp.prospect_observed, ObservedProspectFacts):
        return inp.prospect_observed
    return None


def _get_inferred(inp: OutcomeIntelligenceInput) -> InferredProspectState | None:
    if isinstance(inp.prospect_inferred, InferredProspectState):
        return inp.prospect_inferred
    return None


def _get_business_conversation(inp: OutcomeIntelligenceInput) -> BusinessConversationSnapshot | None:
    if isinstance(inp.business_conversation, BusinessConversationSnapshot):
        return inp.business_conversation
    return None


def _get_diagnostic(inp: OutcomeIntelligenceInput) -> BusinessDiagnosticSnapshot | None:
    if isinstance(inp.business_diagnostic, BusinessDiagnosticSnapshot):
        return inp.business_diagnostic
    return None


def _get_qualification(inp: OutcomeIntelligenceInput) -> QualificationSnapshot | None:
    if isinstance(inp.qualification, QualificationSnapshot):
        return inp.qualification
    return None


def _get_playbook(inp: OutcomeIntelligenceInput) -> OpportunityGuidance | None:
    if isinstance(inp.playbook_guidance, OpportunityGuidance):
        return inp.playbook_guidance
    return None
