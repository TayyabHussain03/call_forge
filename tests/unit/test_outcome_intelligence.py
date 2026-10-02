"""Comprehensive tests for Post-Conversation Outcome Intelligence."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from app.conversation.business_conversation.contracts import (
    BusinessConversationSnapshot,
    BusinessProblemKind,
    ObservedBusinessProblem,
    UnknownArea,
)
from app.conversation.prospect_intelligence.contracts import (
    DecisionAuthority,
    EvidenceProvenance,
    EvidenceSourceKind,
    InferredProspectState,
    InferredValue,
    InformationLevel,
    ObjectionType,
    ObservedProspectFacts,
    ObservedValue,
    PainCategory,
    PreferredNextStep,
    ProspectPain,
)
from app.conversation.sales_playbook.contracts import (
    BusinessSituation,
    ConsultantStep,
    OpportunityGuidance,
    PitchReadiness,
)
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
from app.runtime.chat.outcome.engine import OutcomeIntelligenceEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ENGINE = OutcomeIntelligenceEngine()

_PROV = EvidenceProvenance("t1", EvidenceSourceKind.EXPLICIT_STATEMENT)
_INF_PROV = EvidenceProvenance("t1", EvidenceSourceKind.INFERENCE)


def _input(**overrides) -> OutcomeIntelligenceInput:
    values = dict(
        lead_id="lead_1",
        campaign_id="campaign_1",
        session_id="sess_1",
        tenant_id="tenant_1",
        conversation_terminal=True,
    )
    values.update(overrides)
    return OutcomeIntelligenceInput(**values)


def _observed(**overrides) -> ObservedProspectFacts:
    return ObservedProspectFacts(**overrides)


def _inferred(**overrides) -> InferredProspectState:
    return InferredProspectState(**overrides)


def _playbook(
    service_id: str = "website",
    situations: tuple[BusinessSituation, ...] = (BusinessSituation.NO_ONLINE_PRESENCE,),
    future_ids: tuple[str, ...] = (),
) -> OpportunityGuidance:
    return OpportunityGuidance(
        selected_service_id=service_id,
        matched_situations=situations,
        pitch_readiness=PitchReadiness.DISCOVERY_COMPLETE,
        consultant_step=ConsultantStep.EDUCATE,
        primary_question=None,
        secondary_question=None,
        objection_guidance=None,
        benefit_categories=(),
        future_opportunity_ids=future_ids,
        restrictions=frozenset(),
    )


# ===========================================================================
# Contract validation
# ===========================================================================


class TestLeadOutcomeContract:
    def test_valid_outcome(self):
        o = LeadOutcome(
            lead_id="l1", campaign_id="c1", session_id="s1", tenant_id="t1",
            outcome_status=OutcomeStatus.INTERESTED,
            interest_level=InterestLevel.EXPLICIT_INTEREST,
            dnc_status=DNCStatus.NOT_DNC,
            execution_status=ExecutionStatus.COMPLETED,
            termination_reason=TerminationReason.NATURAL_COMPLETION,
        )
        assert o.outcome_status == OutcomeStatus.INTERESTED

    def test_empty_lead_id_rejected(self):
        with pytest.raises(ValueError, match="lead_id"):
            LeadOutcome(
                lead_id="", campaign_id="c1", session_id="s1", tenant_id="t1",
                outcome_status=OutcomeStatus.UNKNOWN,
                interest_level=InterestLevel.UNKNOWN,
                dnc_status=DNCStatus.NOT_DNC,
                execution_status=ExecutionStatus.COMPLETED,
                termination_reason=TerminationReason.UNKNOWN,
            )

    def test_dnc_confirmed_requires_dnc_outcome(self):
        with pytest.raises(ValueError, match="DNC_CONFIRMED"):
            LeadOutcome(
                lead_id="l1", campaign_id="c1", session_id="s1", tenant_id="t1",
                outcome_status=OutcomeStatus.INTERESTED,
                interest_level=InterestLevel.UNKNOWN,
                dnc_status=DNCStatus.DNC_CONFIRMED,
                execution_status=ExecutionStatus.COMPLETED,
                termination_reason=TerminationReason.UNKNOWN,
            )

    def test_frozen(self):
        o = LeadOutcome(
            lead_id="l1", campaign_id="c1", session_id="s1", tenant_id="t1",
            outcome_status=OutcomeStatus.UNKNOWN,
            interest_level=InterestLevel.UNKNOWN,
            dnc_status=DNCStatus.NOT_DNC,
            execution_status=ExecutionStatus.COMPLETED,
            termination_reason=TerminationReason.UNKNOWN,
        )
        with pytest.raises(FrozenInstanceError):
            o.outcome_status = OutcomeStatus.DNC


# ===========================================================================
# Interested lead
# ===========================================================================


class TestInterestedLead:
    def test_explicit_interest(self):
        result = ENGINE.resolve(_input(
            prospect_observed=_observed(
                explicit_interest_signal=ObservedValue(True, _PROV),
            ),
        ))
        assert result.outcome_status == OutcomeStatus.INTERESTED
        assert result.interest_level == InterestLevel.EXPLICIT_INTEREST

    def test_interested_with_next_step_becomes_follow_up(self):
        result = ENGINE.resolve(_input(
            prospect_observed=_observed(
                explicit_interest_signal=ObservedValue(True, _PROV),
                explicit_next_step_request=ObservedValue(PreferredNextStep.CALLBACK, _PROV),
            ),
        ))
        assert result.outcome_status == OutcomeStatus.FOLLOW_UP
        assert result.follow_up_needed == FollowUpNeed.NEEDED
        assert result.next_step == "callback"


# ===========================================================================
# Not interested
# ===========================================================================


class TestNotInterested:
    def test_not_interested(self):
        result = ENGINE.resolve(_input(is_not_interested=True))
        assert result.outcome_status == OutcomeStatus.NOT_INTERESTED
        assert result.interest_level == InterestLevel.NO_INTEREST
        assert result.follow_up_needed == FollowUpNeed.NOT_NEEDED

    def test_not_interested_separate_from_dnc(self):
        result = ENGINE.resolve(_input(is_not_interested=True))
        assert result.dnc_status == DNCStatus.NOT_DNC
        assert result.outcome_status != OutcomeStatus.DNC


# ===========================================================================
# DNC
# ===========================================================================


class TestDNC:
    def test_dnc_highest_priority(self):
        result = ENGINE.resolve(_input(
            is_dnc=True,
            is_not_interested=True,
            prospect_observed=_observed(
                explicit_interest_signal=ObservedValue(True, _PROV),
            ),
        ))
        assert result.outcome_status == OutcomeStatus.DNC
        assert result.dnc_status == DNCStatus.DNC_CONFIRMED
        assert result.follow_up_needed == FollowUpNeed.NOT_NEEDED

    def test_dnc_in_summary(self):
        result = ENGINE.resolve(_input(is_dnc=True))
        dnc_lines = [l for l in result.summary_lines if l.label == "DNC"]
        assert len(dnc_lines) == 1
        assert dnc_lines[0].value == "Confirmed"


# ===========================================================================
# Busy + interested
# ===========================================================================


class TestBusyInterested:
    def test_busy_but_interested_with_callback(self):
        result = ENGINE.resolve(_input(
            prospect_observed=_observed(
                explicit_interest_signal=ObservedValue(True, _PROV),
                explicit_busy_signal=ObservedValue(True, _PROV),
                explicit_next_step_request=ObservedValue(PreferredNextStep.CALLBACK, _PROV),
            ),
        ))
        assert result.outcome_status == OutcomeStatus.FOLLOW_UP
        assert result.interest_level == InterestLevel.EXPLICIT_INTEREST
        assert result.next_step == "callback"


# ===========================================================================
# No fit
# ===========================================================================


class TestNoFit:
    def test_no_interest_no_service(self):
        result = ENGINE.resolve(_input(
            prospect_observed=_observed(
                explicit_interest_signal=ObservedValue(False, _PROV),
            ),
        ))
        assert result.interest_level == InterestLevel.NO_INTEREST


# ===========================================================================
# Unknown outcome
# ===========================================================================


class TestUnknownOutcome:
    def test_no_signals(self):
        result = ENGINE.resolve(_input(conversation_terminal=False))
        assert result.outcome_status == OutcomeStatus.UNKNOWN
        assert result.interest_level == InterestLevel.UNKNOWN


# ===========================================================================
# Provider/runtime failure
# ===========================================================================


class TestSessionFailure:
    def test_session_error_separates_from_customer_outcome(self):
        result = ENGINE.resolve(_input(
            session_error=True,
            termination_reason=TerminationReason.RUNTIME_ERROR,
        ))
        assert result.execution_status == ExecutionStatus.FAILED
        assert result.outcome_status != OutcomeStatus.NOT_INTERESTED
        assert result.interest_level != InterestLevel.NO_INTEREST
        assert result.follow_up_needed == FollowUpNeed.NEEDED

    def test_interested_plus_runtime_failure_preserves_interest(self):
        result = ENGINE.resolve(_input(
            session_error=True,
            prospect_observed=_observed(
                explicit_interest_signal=ObservedValue(True, _PROV),
            ),
            termination_reason=TerminationReason.RUNTIME_ERROR,
        ))
        assert result.outcome_status == OutcomeStatus.INTERESTED
        assert result.interest_level == InterestLevel.EXPLICIT_INTEREST
        assert result.execution_status == ExecutionStatus.FAILED

    def test_follow_up_plus_runtime_failure_preserves_follow_up(self):
        result = ENGINE.resolve(_input(
            session_error=True,
            prospect_observed=_observed(
                explicit_interest_signal=ObservedValue(True, _PROV),
                explicit_next_step_request=ObservedValue(PreferredNextStep.CALLBACK, _PROV),
            ),
            termination_reason=TerminationReason.RUNTIME_ERROR,
        ))
        assert result.outcome_status == OutcomeStatus.FOLLOW_UP
        assert result.next_step == "callback"
        assert result.execution_status == ExecutionStatus.FAILED

    def test_dnc_plus_runtime_failure_remains_dnc(self):
        result = ENGINE.resolve(_input(
            session_error=True,
            is_dnc=True,
            termination_reason=TerminationReason.RUNTIME_ERROR,
        ))
        assert result.outcome_status == OutcomeStatus.DNC
        assert result.dnc_status == DNCStatus.DNC_CONFIRMED
        assert result.execution_status == ExecutionStatus.FAILED

    def test_no_signals_plus_runtime_failure_outcome_unknown(self):
        result = ENGINE.resolve(_input(
            session_error=True,
            conversation_terminal=False,
            termination_reason=TerminationReason.RUNTIME_ERROR,
        ))
        assert result.outcome_status == OutcomeStatus.UNKNOWN
        assert result.execution_status == ExecutionStatus.FAILED

    def test_not_interested_plus_runtime_failure_preserves_not_interested(self):
        result = ENGINE.resolve(_input(
            session_error=True,
            is_not_interested=True,
            termination_reason=TerminationReason.RUNTIME_ERROR,
        ))
        assert result.outcome_status == OutcomeStatus.NOT_INTERESTED
        assert result.execution_status == ExecutionStatus.FAILED

    def test_clean_completion_marks_execution_completed(self):
        result = ENGINE.resolve(_input(session_error=False))
        assert result.execution_status == ExecutionStatus.COMPLETED

    def test_session_failed_removed_from_outcome_enum(self):
        assert not hasattr(OutcomeStatus, "SESSION_FAILED")


# ===========================================================================
# Website opportunity
# ===========================================================================


class TestWebsiteOpportunity:
    def test_confirmed_website_opportunity(self):
        result = ENGINE.resolve(_input(
            playbook_guidance=_playbook(
                service_id="website",
                situations=(BusinessSituation.NO_ONLINE_PRESENCE,),
            ),
        ))
        assert len(result.relevant_services) == 1
        assert result.relevant_services[0].service_id == "website"
        assert result.relevant_services[0].confidence == FieldConfidence.OBSERVED


# ===========================================================================
# Automation opportunity
# ===========================================================================


class TestAutomationOpportunity:
    def test_manual_process_detected(self):
        result = ENGINE.resolve(_input(
            playbook_guidance=_playbook(
                service_id="ai_automation",
                situations=(BusinessSituation.MANUAL_WORKFLOW,),
            ),
        ))
        assert len(result.relevant_services) == 1
        assert result.relevant_services[0].service_id == "ai_automation"
        assert "manual_workflow" in result.relevant_services[0].match_reason


# ===========================================================================
# Objection preservation
# ===========================================================================


class TestObjectionPreservation:
    def test_observed_objection(self):
        result = ENGINE.resolve(_input(
            prospect_observed=_observed(
                explicit_objection=ObservedValue(ObjectionType.PRICE, _PROV),
            ),
        ))
        assert len(result.objections) == 1
        assert result.objections[0].objection_type == "price"
        assert result.objections[0].confidence == FieldConfidence.OBSERVED

    def test_inferred_objection(self):
        result = ENGINE.resolve(_input(
            prospect_inferred=_inferred(
                objection_type=InferredValue(ObjectionType.NO_TIME, 0.7, _INF_PROV),
            ),
        ))
        assert len(result.objections) == 1
        assert result.objections[0].objection_type == "no_time"
        assert result.objections[0].confidence == FieldConfidence.INFERRED

    def test_no_duplicate_objections(self):
        result = ENGINE.resolve(_input(
            prospect_observed=_observed(
                explicit_objection=ObservedValue(ObjectionType.PRICE, _PROV),
            ),
            prospect_inferred=_inferred(
                objection_type=InferredValue(ObjectionType.PRICE, 0.8, _INF_PROV),
            ),
        ))
        assert len(result.objections) == 1


# ===========================================================================
# Decision maker
# ===========================================================================


class TestDecisionMaker:
    def test_decision_maker_confirmed(self):
        result = ENGINE.resolve(_input(
            prospect_observed=_observed(
                explicit_decision_authority_statement=ObservedValue(
                    DecisionAuthority.FINAL, _PROV,
                ),
            ),
        ))
        assert result.decision_maker_status == DecisionMakerStatus.CONFIRMED

    def test_decision_maker_unknown(self):
        result = ENGINE.resolve(_input())
        assert result.decision_maker_status == DecisionMakerStatus.UNKNOWN

    def test_not_decision_maker(self):
        result = ENGINE.resolve(_input(
            prospect_observed=_observed(
                explicit_decision_authority_statement=ObservedValue(
                    DecisionAuthority.LOW, _PROV,
                ),
            ),
        ))
        assert result.decision_maker_status == DecisionMakerStatus.NOT_DECISION_MAKER


# ===========================================================================
# Next step
# ===========================================================================


class TestNextStep:
    def test_next_step_known(self):
        result = ENGINE.resolve(_input(
            prospect_observed=_observed(
                explicit_next_step_request=ObservedValue(PreferredNextStep.DEMO, _PROV),
            ),
        ))
        assert result.next_step == "demo"

    def test_next_step_unknown(self):
        result = ENGINE.resolve(_input())
        assert result.next_step is None


# ===========================================================================
# Missing fields stay UNKNOWN
# ===========================================================================


class TestMissingFieldsUnknown:
    def test_empty_input_all_unknown(self):
        result = ENGINE.resolve(_input(conversation_terminal=False))
        assert result.interest_level == InterestLevel.UNKNOWN
        assert result.decision_maker_status == DecisionMakerStatus.UNKNOWN
        assert result.follow_up_needed == FollowUpNeed.UNKNOWN
        assert result.next_step is None
        assert result.qualification_completeness is None
        assert len(result.identified_problems) == 0
        assert len(result.relevant_services) == 0
        assert len(result.objections) == 0


# ===========================================================================
# No service hallucination
# ===========================================================================


class TestNoServiceHallucination:
    def test_no_services_without_playbook(self):
        result = ENGINE.resolve(_input(
            prospect_observed=_observed(
                explicit_interest_signal=ObservedValue(True, _PROV),
            ),
        ))
        assert len(result.relevant_services) == 0

    def test_no_services_with_empty_playbook(self):
        result = ENGINE.resolve(_input(
            playbook_guidance=OpportunityGuidance(
                selected_service_id=None,
                matched_situations=(),
                pitch_readiness=PitchReadiness.DISCOVERY_REQUIRED,
                consultant_step=ConsultantStep.UNDERSTAND,
                primary_question=None,
                secondary_question=None,
                objection_guidance=None,
                benefit_categories=(),
                future_opportunity_ids=(),
                restrictions=frozenset(),
            ),
        ))
        assert len(result.relevant_services) == 0


# ===========================================================================
# No transcript keyword inference
# ===========================================================================


class TestNoTranscriptInference:
    def test_no_transcript_parsing_in_engine(self):
        from pathlib import Path
        import app.runtime.chat.outcome.engine as m

        source = Path(m.__file__).read_text()
        assert "import re" not in source
        assert "re.search" not in source
        assert "re.match" not in source
        assert ".transcript" not in source
        assert "split(" not in source


# ===========================================================================
# Deterministic replay
# ===========================================================================


class TestDeterministicReplay:
    def test_same_input_same_output(self):
        inp = _input(
            prospect_observed=_observed(
                explicit_interest_signal=ObservedValue(True, _PROV),
                explicit_objection=ObservedValue(ObjectionType.PRICE, _PROV),
            ),
            playbook_guidance=_playbook(),
        )
        r1 = ENGINE.resolve(inp)
        r2 = ENGINE.resolve(inp)
        assert r1 == r2

    def test_no_random_in_engine(self):
        from pathlib import Path
        import app.runtime.chat.outcome.engine as m

        source = Path(m.__file__).read_text()
        assert "uuid" not in source
        assert "random" not in source
        assert "datetime.now" not in source


# ===========================================================================
# Tenant / campaign isolation
# ===========================================================================


class TestIsolation:
    def test_outcome_carries_tenant_and_campaign(self):
        result = ENGINE.resolve(_input(
            tenant_id="t_abc", campaign_id="c_xyz",
        ))
        assert result.tenant_id == "t_abc"
        assert result.campaign_id == "c_xyz"

    def test_different_tenants_independent(self):
        r1 = ENGINE.resolve(_input(tenant_id="t1", lead_id="l1"))
        r2 = ENGINE.resolve(_input(tenant_id="t2", lead_id="l2"))
        assert r1.tenant_id != r2.tenant_id
        assert r1.lead_id != r2.lead_id


# ===========================================================================
# No execution side effects
# ===========================================================================


class TestNoSideEffects:
    def test_no_execution_imports(self):
        from pathlib import Path
        import app.runtime.chat.outcome.engine as m

        source = Path(m.__file__).read_text()
        assert "send_email" not in source
        assert "schedule" not in source
        assert "write_crm" not in source
        assert "place_call" not in source
        assert "twilio" not in source.lower()
        assert "smtp" not in source.lower()
        assert "openpyxl" not in source.lower()

    def test_no_llm_imports(self):
        from pathlib import Path
        import app.runtime.chat.outcome.engine as m

        source = Path(m.__file__).read_text()
        assert "openai" not in source.lower()
        assert "gemini" not in source.lower()
        assert "anthropic" not in source.lower()
        assert "LLM" not in source
        assert "llm" not in source


# ===========================================================================
# Problem identification from BCI
# ===========================================================================


class TestProblemIdentification:
    def test_observed_pain_points(self):
        result = ENGINE.resolve(_input(
            prospect_observed=_observed(
                explicit_pain_points=(
                    ProspectPain(PainCategory.TIME, "Spending hours on manual data entry", _PROV),
                ),
            ),
        ))
        assert len(result.identified_problems) == 1
        assert result.identified_problems[0].confidence == FieldConfidence.OBSERVED

    def test_business_conversation_problems(self):
        bcs = BusinessConversationSnapshot(
            problems=(
                ObservedBusinessProblem(
                    BusinessProblemKind.OPERATIONAL,
                    "No website for customer acquisition",
                    "t1",
                ),
            ),
        )
        result = ENGINE.resolve(_input(business_conversation=bcs))
        assert len(result.identified_problems) == 1
        assert "No website" in result.identified_problems[0].description


# ===========================================================================
# Outstanding questions from unknown areas
# ===========================================================================


class TestOutstandingQuestions:
    def test_unknown_areas_become_questions(self):
        bcs = BusinessConversationSnapshot(
            unknown_areas=frozenset({UnknownArea.CURRENT_SOFTWARE, UnknownArea.DECISION_MAKER}),
        )
        result = ENGINE.resolve(_input(business_conversation=bcs))
        assert len(result.outstanding_questions) == 2


# ===========================================================================
# Summary structure
# ===========================================================================


class TestSummaryStructure:
    def test_summary_has_outcome_and_interest(self):
        result = ENGINE.resolve(_input())
        labels = [l.label for l in result.summary_lines]
        assert "Outcome" in labels
        assert "Interest" in labels

    def test_summary_includes_problem(self):
        result = ENGINE.resolve(_input(
            prospect_observed=_observed(
                explicit_pain_points=(
                    ProspectPain(PainCategory.COST, "High overhead costs", _PROV),
                ),
            ),
        ))
        labels = [l.label for l in result.summary_lines]
        assert "Problem" in labels

    def test_summary_not_prose(self):
        result = ENGINE.resolve(_input(
            is_dnc=True,
            termination_reason=TerminationReason.DNC_REQUEST,
        ))
        for line in result.summary_lines:
            assert len(line.value) <= 300
            assert "\n" not in line.value
