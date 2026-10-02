"""Comprehensive deterministic tests for the Sales Knowledge Intelligence Engine."""

from dataclasses import replace

import pytest

from app.conversation.business_conversation.contracts import (
    BusinessConversationSnapshot,
    BusinessFactKind,
    BusinessGoalKind,
    ConversationTopic,
    ObservedBusinessFact,
)
from app.conversation.business_diagnostic.contracts import BusinessDiagnosticSnapshot
from app.conversation.consultative.contracts import (
    ProblemCategory,
    ProblemEvidenceBasis,
    ProspectProblem,
    ServiceFitDecision,
    ServiceFitStatus,
)
from app.conversation.conversation_steering.contracts import ConversationPrioritySnapshot
from app.conversation.prospect_intelligence.contracts import (
    EvidenceProvenance,
    EvidenceSourceKind,
    InferredValue,
    ObservedValue,
    ProspectIntelligenceSummary,
    ProspectRole,
)
from app.conversation.response_planning.contracts import ResponseLength
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
    ConsultantStep,
    OpportunityGuidance,
    PitchReadiness,
    PlaybookRestriction,
)
from app.conversation.understanding.contracts import (
    ConversationalRegister,
    LanguageProfile,
    LanguageScript,
)
from app.knowledge.evidence_validation.contracts import (
    ApprovedEvidence,
    ApprovedEvidenceSet,
    ApprovalReason,
    EvidenceGroup,
    EvidenceQuality,
)
from app.knowledge.sales_intelligence.contracts import (
    AnalogyStrategy,
    BusinessValueFocus,
    CognitiveLoad,
    ConversationObjective,
    DisclosureLevel,
    ExplanationDepth,
    ExplanationPlan,
    FollowUpStyle,
    KnowledgeConfidence,
    KnowledgeConversationContext,
    KnowledgeIntent,
    KnowledgePriority,
    MomentumSignal,
    ProgressiveDisclosurePlan,
    ResponseComplexity,
    SelectedEvidence,
    SuppressedEvidence,
    SuppressionReason,
)
from app.knowledge.sales_intelligence.engine import SalesKnowledgeIntelligenceEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHECK = "a" * 64


def _evidence(
    approval_id: str = "ae_1",
    group: EvidenceGroup = EvidenceGroup.BENEFIT,
    quality: EvidenceQuality = EvidenceQuality.DIRECT,
    section: str = "Features",
    text: str = "Test evidence text.",
) -> ApprovedEvidence:
    return ApprovedEvidence(
        approval_id=approval_id,
        claim_id="claim-1",
        document_id="doc",
        version_id="v1",
        document_version=1,
        chunk_id="c1",
        section=section,
        approved_text=text,
        group=group,
        quality=quality,
        approval_reasons=(ApprovalReason.VALID_SCOPE,),
        source_snapshot_id="snap",
        checksum=CHECK,
        validator_version="evae-v1",
    )


def _guidance(**changes) -> SalesConversationGuidance:
    values = dict(
        momentum=ConversationMomentum.STABLE,
        trust=TrustState.ESTABLISHED,
        interest=InterestStrength.MODERATE,
        discovery_readiness=DiscoveryReadiness.READY_FOR_FIT,
        objection=ObjectionUnderstanding.NONE,
        energy=ConversationEnergy.NORMAL,
        question_priority=QuestionPriority.NONE,
        question_focus=None,
        relationship=RelationshipState.DEVELOPING,
        pressure=PressureState.COMFORTABLE,
        curiosity_focus=CuriosityFocus.NONE,
        role_style=RoleConversationStyle.GENERAL_CONSULTATIVE,
        emotional_posture=EmotionalPosture.NEUTRAL,
        buying_guidance=BuyingReadinessGuidance.CONTINUE_DISCOVERY,
        recommended_response_depth=ResponseLength.MODERATE,
    )
    values.update(changes)
    return SalesConversationGuidance(**values)


def _language(**changes) -> LanguageProfile:
    values = dict(primary_language="en")
    values.update(changes)
    return LanguageProfile(**values)


def _problem(**changes) -> ProspectProblem:
    return ProspectProblem(**changes)


def _prospect(**changes) -> ProspectIntelligenceSummary:
    return ProspectIntelligenceSummary(**changes)


def _service_fit(**changes) -> ServiceFitDecision:
    values = dict(status=ServiceFitStatus.INSUFFICIENT_CONTEXT)
    values.update(changes)
    return ServiceFitDecision(**values)


def _conversation(**changes) -> BusinessConversationSnapshot:
    return BusinessConversationSnapshot(**changes)


def _playbook(**changes) -> OpportunityGuidance:
    values = dict(
        selected_service_id="website",
        matched_situations=(),
        pitch_readiness=PitchReadiness.DISCOVERY_COMPLETE,
        consultant_step=ConsultantStep.EDUCATE,
        primary_question=None,
        secondary_question=None,
        objection_guidance=None,
        benefit_categories=(),
        future_opportunity_ids=(),
        restrictions=frozenset({PlaybookRestriction.NO_PRESSURE_LANGUAGE}),
    )
    values.update(changes)
    return OpportunityGuidance(**values)


def _input(
    evidence_items: tuple[ApprovedEvidence, ...] = (),
    guidance: SalesConversationGuidance | None = None,
    problem: ProspectProblem | None = None,
    prospect: ProspectIntelligenceSummary | None = None,
    conversation: BusinessConversationSnapshot | None = None,
    language: LanguageProfile | None = None,
    playbook: OpportunityGuidance | None = None,
    already_discussed: frozenset[str] = frozenset(),
) -> "SalesKnowledgeInput":
    from app.knowledge.sales_intelligence.contracts import SalesKnowledgeInput

    return SalesKnowledgeInput(
        approved_evidence=ApprovedEvidenceSet(evidence_items),
        prospect=prospect or _prospect(),
        problem=problem or _problem(),
        service_fit=_service_fit(),
        sales_guidance=guidance or _guidance(),
        playbook_guidance=playbook,
        language_profile=language or _language(),
        conversation=conversation or _conversation(),
        already_discussed_evidence_ids=already_discussed,
    )


ENGINE = SalesKnowledgeIntelligenceEngine()


# ---------------------------------------------------------------------------
# Beginner / simple explanation
# ---------------------------------------------------------------------------


class TestBeginnerExplanation:
    def test_simple_depth_for_short_response(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                guidance=_guidance(recommended_response_depth=ResponseLength.SHORT),
            )
        )
        assert result.explanation.depth == ExplanationDepth.SIMPLE

    def test_simple_complexity_for_simple_depth(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                guidance=_guidance(recommended_response_depth=ResponseLength.SHORT),
            )
        )
        assert result.explanation.response_complexity == ResponseComplexity.SIMPLE

    def test_gatekeeper_gets_simple_depth(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                prospect=_prospect(
                    explicit_role=ObservedValue(value=ProspectRole.RECEPTIONIST, provenance=EvidenceProvenance(source_turn_id="t1", source_kind=EvidenceSourceKind.EXPLICIT_STATEMENT)),
                ),
            )
        )
        assert result.explanation.depth == ExplanationDepth.SIMPLE


# ---------------------------------------------------------------------------
# Technical prospect
# ---------------------------------------------------------------------------


class TestTechnicalProspect:
    def test_technical_manager_gets_detailed_depth(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                prospect=_prospect(
                    explicit_role=ObservedValue(value=ProspectRole.TECHNICAL_MANAGER, provenance=EvidenceProvenance(source_turn_id="t1", source_kind=EvidenceSourceKind.EXPLICIT_STATEMENT)),
                ),
            )
        )
        assert result.explanation.depth == ExplanationDepth.DETAILED

    def test_technical_facts_produce_technical_complexity(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                conversation=_conversation(
                    facts=(
                        ObservedBusinessFact(kind=BusinessFactKind.SOFTWARE, value="Laravel", source_turn_id="t1"),
                    ),
                ),
                guidance=_guidance(recommended_response_depth=ResponseLength.DETAILED),
            )
        )
        assert result.explanation.response_complexity == ResponseComplexity.TECHNICAL


# ---------------------------------------------------------------------------
# Executive / owner
# ---------------------------------------------------------------------------


class TestExecutiveExplanation:
    def test_owner_gets_normal_depth_by_default(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                prospect=_prospect(
                    explicit_role=ObservedValue(value=ProspectRole.OWNER, provenance=EvidenceProvenance(source_turn_id="t1", source_kind=EvidenceSourceKind.EXPLICIT_STATEMENT)),
                ),
            )
        )
        assert result.explanation.depth == ExplanationDepth.NORMAL


# ---------------------------------------------------------------------------
# Educational intent
# ---------------------------------------------------------------------------


class TestEducationalIntent:
    def test_educate_intent_from_playbook_step(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                playbook=_playbook(consultant_step=ConsultantStep.EDUCATE),
            )
        )
        assert result.explanation.knowledge_intent == KnowledgeIntent.EDUCATE

    def test_answer_question_intent_when_solution_requested(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                problem=_problem(requested_solution="website design"),
            )
        )
        assert result.explanation.knowledge_intent == KnowledgeIntent.ANSWER_QUESTION

    def test_build_awareness_for_unknown_problem(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                problem=_problem(category=ProblemCategory.UNKNOWN),
                guidance=_guidance(interest=InterestStrength.WEAK),
            )
        )
        assert result.explanation.knowledge_intent == KnowledgeIntent.BUILD_AWARENESS

    def test_illustrate_benefit_for_strong_interest(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                problem=_problem(category=ProblemCategory.WEBSITE, explicit_description="need website"),
                guidance=_guidance(interest=InterestStrength.STRONG),
            )
        )
        assert result.explanation.knowledge_intent == KnowledgeIntent.ILLUSTRATE_BENEFIT


# ---------------------------------------------------------------------------
# Suppression
# ---------------------------------------------------------------------------


class TestSuppression:
    def test_off_topic_evidence_suppressed(self):
        website_evidence = _evidence(approval_id="ae_web", group=EvidenceGroup.BENEFIT)
        impl_evidence = _evidence(approval_id="ae_impl", group=EvidenceGroup.IMPLEMENTATION)
        result = ENGINE.design(
            _input(
                evidence_items=(website_evidence, impl_evidence),
                conversation=_conversation(current_focus=ConversationTopic.WEBSITE),
                guidance=_guidance(recommended_response_depth=ResponseLength.SHORT),
            )
        )
        suppressed_ids = {item.evidence.approval_id for item in result.suppressed}
        assert "ae_impl" in suppressed_ids

    def test_already_discussed_evidence_suppressed(self):
        e1 = _evidence(approval_id="ae_1")
        e2 = _evidence(approval_id="ae_2", group=EvidenceGroup.FEATURE)
        result = ENGINE.design(
            _input(
                evidence_items=(e1, e2),
                already_discussed=frozenset({"ae_1"}),
            )
        )
        suppressed_ids = {item.evidence.approval_id for item in result.suppressed}
        assert "ae_1" in suppressed_ids
        assert any(
            item.reason == SuppressionReason.ALREADY_DISCUSSED for item in result.suppressed
        )

    def test_commercial_policy_suppressed_when_trust_building(self):
        commercial = _evidence(approval_id="ae_comm", group=EvidenceGroup.COMMERCIAL_POLICY)
        benefit = _evidence(approval_id="ae_ben", group=EvidenceGroup.BENEFIT)
        result = ENGINE.design(
            _input(
                evidence_items=(benefit, commercial),
                guidance=_guidance(trust=TrustState.BUILDING),
            )
        )
        suppressed_ids = {item.evidence.approval_id for item in result.suppressed}
        assert "ae_comm" in suppressed_ids

    def test_case_study_suppressed_when_no_interest(self):
        case = _evidence(approval_id="ae_case", group=EvidenceGroup.CASE_STUDY)
        benefit = _evidence(approval_id="ae_ben", group=EvidenceGroup.BENEFIT)
        result = ENGINE.design(
            _input(
                evidence_items=(benefit, case),
                guidance=_guidance(interest=InterestStrength.NONE),
            )
        )
        suppressed_ids = {item.evidence.approval_id for item in result.suppressed}
        assert "ae_case" in suppressed_ids


# ---------------------------------------------------------------------------
# Progressive disclosure
# ---------------------------------------------------------------------------


class TestProgressiveDisclosure:
    def test_minimal_disclosure_for_low_interest(self):
        e1 = _evidence(approval_id="ae_1")
        e2 = _evidence(approval_id="ae_2", group=EvidenceGroup.FEATURE)
        e3 = _evidence(approval_id="ae_3", group=EvidenceGroup.FAQ)
        result = ENGINE.design(
            _input(
                evidence_items=(e1, e2, e3),
                guidance=_guidance(interest=InterestStrength.WEAK),
            )
        )
        assert result.disclosure.disclosure_level == DisclosureLevel.MINIMAL
        assert len(result.selected) <= 1

    def test_full_disclosure_for_buying_signal(self):
        items = tuple(
            _evidence(
                approval_id=f"ae_{i}",
                group=[EvidenceGroup.BENEFIT, EvidenceGroup.FEATURE, EvidenceGroup.FAQ, EvidenceGroup.PROCESS][i],
            )
            for i in range(4)
        )
        result = ENGINE.design(
            _input(
                evidence_items=items,
                guidance=_guidance(interest=InterestStrength.BUYING_SIGNAL),
            )
        )
        assert result.disclosure.disclosure_level == DisclosureLevel.FULL
        assert len(result.selected) <= 4

    def test_moderate_disclosure_for_moderate_interest(self):
        items = tuple(
            _evidence(approval_id=f"ae_{i}", group=EvidenceGroup.BENEFIT)
            for i in range(4)
        )
        result = ENGINE.design(
            _input(
                evidence_items=items,
                guidance=_guidance(interest=InterestStrength.MODERATE),
            )
        )
        assert result.disclosure.disclosure_level == DisclosureLevel.MODERATE
        assert len(result.selected) <= 2


# ---------------------------------------------------------------------------
# Evidence compression (many approved → few selected)
# ---------------------------------------------------------------------------


class TestEvidenceCompression:
    def test_five_approved_compresses_to_max_two_at_moderate(self):
        items = tuple(
            _evidence(approval_id=f"ae_{i}", group=EvidenceGroup.BENEFIT)
            for i in range(5)
        )
        result = ENGINE.design(
            _input(
                evidence_items=items,
                guidance=_guidance(interest=InterestStrength.MODERATE),
            )
        )
        assert len(result.selected) <= 2
        assert len(result.suppressed) >= 3


# ---------------------------------------------------------------------------
# One-question flow
# ---------------------------------------------------------------------------


class TestOneQuestionFlow:
    def test_single_evidence_fully_selected(self):
        result = ENGINE.design(
            _input(evidence_items=(_evidence(),))
        )
        assert len(result.selected) == 1
        assert result.selected[0].priority == KnowledgePriority.PRIMARY


# ---------------------------------------------------------------------------
# Industry adaptation via analogy
# ---------------------------------------------------------------------------


class TestIndustryAdaptation:
    def test_industry_example_analogy_for_simple_with_industry_fact(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                conversation=_conversation(
                    facts=(
                        ObservedBusinessFact(kind=BusinessFactKind.INDUSTRY, value="restaurant", source_turn_id="t1"),
                    ),
                ),
                guidance=_guidance(recommended_response_depth=ResponseLength.SHORT),
            )
        )
        assert result.explanation.analogy == AnalogyStrategy.INDUSTRY_EXAMPLE

    def test_workflow_parallel_when_current_process_known(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                problem=_problem(current_process="manual phone orders"),
            )
        )
        assert result.explanation.analogy == AnalogyStrategy.WORKFLOW_PARALLEL

    def test_no_analogy_for_detailed_depth(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                prospect=_prospect(
                    explicit_role=ObservedValue(value=ProspectRole.TECHNICAL_MANAGER, provenance=EvidenceProvenance(source_turn_id="t1", source_kind=EvidenceSourceKind.EXPLICIT_STATEMENT)),
                ),
            )
        )
        assert result.explanation.analogy == AnalogyStrategy.NONE


# ---------------------------------------------------------------------------
# Replay determinism
# ---------------------------------------------------------------------------


class TestReplayDeterminism:
    def test_same_input_produces_same_output(self):
        inp = _input(
            evidence_items=(
                _evidence(approval_id="ae_1"),
                _evidence(approval_id="ae_2", group=EvidenceGroup.FEATURE),
            ),
        )
        r1 = ENGINE.design(inp)
        r2 = ENGINE.design(inp)
        assert r1 == r2

    def test_determinism_across_evidence_ordering(self):
        e1 = _evidence(approval_id="ae_1", group=EvidenceGroup.BENEFIT)
        e2 = _evidence(approval_id="ae_2", group=EvidenceGroup.FEATURE)
        r1 = ENGINE.design(_input(evidence_items=(e1, e2)))
        r2 = ENGINE.design(_input(evidence_items=(e2, e1)))
        assert r1 == r2


# ---------------------------------------------------------------------------
# No authority leakage
# ---------------------------------------------------------------------------


class TestNoAuthorityLeakage:
    def test_output_contains_no_service_authorization(self):
        result = ENGINE.design(_input(evidence_items=(_evidence(),)))
        context = result
        for item in context.selected:
            assert not hasattr(item, "service_authorized")
            assert not hasattr(item, "execution_allowed")
        assert not hasattr(context, "transition")
        assert not hasattr(context, "pricing")
        assert not hasattr(context, "discount")
        assert not hasattr(context, "schedule")

    def test_output_is_frozen(self):
        from dataclasses import FrozenInstanceError

        result = ENGINE.design(_input(evidence_items=(_evidence(),)))
        with pytest.raises(FrozenInstanceError):
            result.selected = ()


# ---------------------------------------------------------------------------
# No Brain mutation
# ---------------------------------------------------------------------------


class TestNoBrainMutation:
    def test_input_evidence_unchanged_after_design(self):
        evidence = _evidence()
        evidence_set = ApprovedEvidenceSet((evidence,))
        inp = _input(evidence_items=(evidence,))
        ENGINE.design(inp)
        assert inp.approved_evidence == evidence_set
        assert inp.approved_evidence.items[0] is evidence


# ---------------------------------------------------------------------------
# Empty evidence
# ---------------------------------------------------------------------------


class TestEmptyEvidence:
    def test_empty_evidence_produces_empty_selection(self):
        result = ENGINE.design(_input(evidence_items=()))
        assert len(result.selected) == 0
        assert len(result.suppressed) == 0


# ---------------------------------------------------------------------------
# Follow-up style
# ---------------------------------------------------------------------------


class TestFollowUpStyle:
    def test_offer_next_topic_when_interested_with_deferred(self):
        items = tuple(
            _evidence(approval_id=f"ae_{i}", group=EvidenceGroup.BENEFIT)
            for i in range(4)
        )
        result = ENGINE.design(
            _input(
                evidence_items=items,
                guidance=_guidance(interest=InterestStrength.STRONG),
            )
        )
        if result.suppressed:
            assert result.disclosure.follow_up in {
                FollowUpStyle.OFFER_NEXT_TOPIC,
                FollowUpStyle.CURIOSITY_HOOK,
            }

    def test_ask_permission_when_trust_building(self):
        e1 = _evidence(approval_id="ae_1", group=EvidenceGroup.BENEFIT)
        e2 = _evidence(approval_id="ae_2", group=EvidenceGroup.FEATURE)
        e3 = _evidence(approval_id="ae_3", group=EvidenceGroup.FAQ)
        result = ENGINE.design(
            _input(
                evidence_items=(e1, e2, e3),
                guidance=_guidance(
                    trust=TrustState.BUILDING,
                    interest=InterestStrength.MODERATE,
                ),
            )
        )
        if any(item.reason == SuppressionReason.OVERLOAD for item in result.suppressed):
            assert result.disclosure.follow_up == FollowUpStyle.ASK_PERMISSION

    def test_no_follow_up_when_all_selected(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                guidance=_guidance(interest=InterestStrength.MODERATE),
            )
        )
        if not any(
            item.reason in {SuppressionReason.OVERLOAD, SuppressionReason.PREMATURE}
            for item in result.suppressed
        ):
            assert result.disclosure.follow_up == FollowUpStyle.NONE


# ---------------------------------------------------------------------------
# Business value focus
# ---------------------------------------------------------------------------


class TestBusinessValueFocus:
    def test_revenue_focus_for_lead_flow_problem(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                problem=_problem(category=ProblemCategory.LEAD_FLOW, explicit_description="no leads"),
            )
        )
        assert result.explanation.business_value_focus == BusinessValueFocus.REVENUE

    def test_goal_overrides_problem_category(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                problem=_problem(category=ProblemCategory.WEBSITE, explicit_description="need site"),
                conversation=_conversation(
                    goals=frozenset({BusinessGoalKind.REDUCE_COST}),
                ),
            )
        )
        assert result.explanation.business_value_focus == BusinessValueFocus.COST_SAVINGS


# ---------------------------------------------------------------------------
# Multilingual simplicity
# ---------------------------------------------------------------------------


class TestMultilingualSimplicity:
    def test_formal_register_gets_normal_depth(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                language=_language(
                    primary_language="ur",
                    conversational_register=ConversationalRegister.FORMAL,
                    script=LanguageScript.URDU_ARABIC,
                    preferred_response_language="ur",
                    preferred_script=LanguageScript.URDU_ARABIC,
                ),
            )
        )
        assert result.explanation.depth == ExplanationDepth.NORMAL


# ---------------------------------------------------------------------------
# Sequence ordering
# ---------------------------------------------------------------------------


class TestSequenceOrdering:
    def test_selected_evidence_has_ascending_positions(self):
        items = tuple(
            _evidence(approval_id=f"ae_{i}", group=EvidenceGroup.BENEFIT)
            for i in range(3)
        )
        result = ENGINE.design(
            _input(
                evidence_items=items,
                guidance=_guidance(interest=InterestStrength.MODERATE),
            )
        )
        positions = [item.sequence_position for item in result.selected]
        assert positions == sorted(positions)
        assert len(set(positions)) == len(positions)

    def test_primary_is_always_first(self):
        items = tuple(
            _evidence(approval_id=f"ae_{i}", group=EvidenceGroup.BENEFIT)
            for i in range(3)
        )
        result = ENGINE.design(
            _input(
                evidence_items=items,
                guidance=_guidance(interest=InterestStrength.MODERATE),
            )
        )
        if result.selected:
            assert result.selected[0].priority == KnowledgePriority.PRIMARY
            assert result.selected[0].sequence_position == 0


# ---------------------------------------------------------------------------
# Comparison scenario
# ---------------------------------------------------------------------------


class TestComparison:
    def test_comparison_intent_when_software_fact_and_solution_requested(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                conversation=_conversation(
                    facts=(
                        ObservedBusinessFact(kind=BusinessFactKind.SOFTWARE, value="WordPress", source_turn_id="t1"),
                    ),
                ),
                problem=_problem(requested_solution="custom vs wordpress"),
            )
        )
        assert result.explanation.knowledge_intent == KnowledgeIntent.ANSWER_QUESTION


# ---------------------------------------------------------------------------
# Wrong depth suppression
# ---------------------------------------------------------------------------


class TestWrongDepthSuppression:
    def test_implementation_suppressed_for_short_responses(self):
        impl = _evidence(approval_id="ae_impl", group=EvidenceGroup.IMPLEMENTATION)
        benefit = _evidence(approval_id="ae_ben", group=EvidenceGroup.BENEFIT)
        result = ENGINE.design(
            _input(
                evidence_items=(benefit, impl),
                guidance=_guidance(recommended_response_depth=ResponseLength.SHORT),
            )
        )
        suppressed_reasons = {
            (item.evidence.approval_id, item.reason) for item in result.suppressed
        }
        assert ("ae_impl", SuppressionReason.WRONG_DEPTH) in suppressed_reasons


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------


class TestContractValidation:
    def test_selected_cannot_exceed_four(self):
        with pytest.raises(ValueError, match="selected evidence must not exceed four"):
            KnowledgeConversationContext(
                selected=tuple(
                    SelectedEvidence(
                        evidence=_evidence(approval_id=f"ae_{i}"),
                        priority=KnowledgePriority.PRIMARY if i == 0 else KnowledgePriority.SUPPORTING,
                        business_value_focus=BusinessValueFocus.NONE,
                        sequence_position=i,
                    )
                    for i in range(5)
                ),
                suppressed=(),
                explanation=ExplanationPlan(
                    depth=ExplanationDepth.NORMAL,
                    knowledge_intent=KnowledgeIntent.EDUCATE,
                    analogy=AnalogyStrategy.NONE,
                    business_value_focus=BusinessValueFocus.NONE,
                    response_complexity=ResponseComplexity.MODERATE,
                    confidence=KnowledgeConfidence.HIGH,
                ),
                disclosure=ProgressiveDisclosurePlan(
                    disclosure_level=DisclosureLevel.FULL,
                    max_evidence_this_turn=4,
                    follow_up=FollowUpStyle.NONE,
                ),
                conversation_objective=ConversationObjective.EDUCATE,
                cognitive_load=CognitiveLoad.NORMAL,
                momentum=MomentumSignal.CONTINUE,
                knowledge_saturated=False,
            )

    def test_same_evidence_cannot_be_selected_and_suppressed(self):
        evidence = _evidence()
        with pytest.raises(ValueError, match="cannot be both selected and suppressed"):
            KnowledgeConversationContext(
                selected=(
                    SelectedEvidence(
                        evidence=evidence,
                        priority=KnowledgePriority.PRIMARY,
                        business_value_focus=BusinessValueFocus.NONE,
                        sequence_position=0,
                    ),
                ),
                suppressed=(
                    SuppressedEvidence(evidence=evidence, reason=SuppressionReason.OFF_TOPIC),
                ),
                explanation=ExplanationPlan(
                    depth=ExplanationDepth.NORMAL,
                    knowledge_intent=KnowledgeIntent.EDUCATE,
                    analogy=AnalogyStrategy.NONE,
                    business_value_focus=BusinessValueFocus.NONE,
                    response_complexity=ResponseComplexity.MODERATE,
                    confidence=KnowledgeConfidence.HIGH,
                ),
                disclosure=ProgressiveDisclosurePlan(
                    disclosure_level=DisclosureLevel.MODERATE,
                    max_evidence_this_turn=2,
                    follow_up=FollowUpStyle.NONE,
                ),
                conversation_objective=ConversationObjective.EDUCATE,
                cognitive_load=CognitiveLoad.NORMAL,
                momentum=MomentumSignal.CONTINUE,
                knowledge_saturated=False,
            )


# ---------------------------------------------------------------------------
# Conversation objective
# ---------------------------------------------------------------------------


class TestConversationObjective:
    def test_handle_objection_when_objection_present(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                guidance=_guidance(objection=ObjectionUnderstanding.BUDGET),
            )
        )
        assert result.conversation_objective == ConversationObjective.HANDLE_OBJECTION

    def test_build_trust_when_trust_building(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                guidance=_guidance(trust=TrustState.BUILDING, interest=InterestStrength.MODERATE),
            )
        )
        assert result.conversation_objective == ConversationObjective.BUILD_TRUST

    def test_clarify_when_answer_question_intent(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                problem=_problem(requested_solution="website design"),
            )
        )
        assert result.conversation_objective == ConversationObjective.CLARIFY

    def test_educate_for_educate_intent(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                playbook=_playbook(consultant_step=ConsultantStep.EDUCATE),
            )
        )
        assert result.conversation_objective == ConversationObjective.EDUCATE

    def test_close_gracefully_for_buying_signal(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                problem=_problem(category=ProblemCategory.WEBSITE, explicit_description="need site"),
                guidance=_guidance(interest=InterestStrength.BUYING_SIGNAL),
            )
        )
        assert result.conversation_objective == ConversationObjective.CLOSE_GRACEFULLY


# ---------------------------------------------------------------------------
# Knowledge confidence
# ---------------------------------------------------------------------------


class TestKnowledgeConfidence:
    def test_high_confidence_all_direct(self):
        result = ENGINE.design(
            _input(
                evidence_items=(
                    _evidence(approval_id="ae_1", quality=EvidenceQuality.DIRECT),
                ),
            )
        )
        assert result.explanation.confidence == KnowledgeConfidence.HIGH

    def test_minimal_confidence_with_limited_evidence(self):
        result = ENGINE.design(
            _input(
                evidence_items=(
                    _evidence(approval_id="ae_1", quality=EvidenceQuality.LIMITED),
                ),
            )
        )
        assert result.explanation.confidence == KnowledgeConfidence.MINIMAL

    def test_medium_confidence_with_supported_evidence(self):
        result = ENGINE.design(
            _input(
                evidence_items=(
                    _evidence(approval_id="ae_1", quality=EvidenceQuality.SUPPORTED),
                ),
            )
        )
        assert result.explanation.confidence == KnowledgeConfidence.MEDIUM

    def test_minimal_confidence_empty_evidence(self):
        result = ENGINE.design(_input(evidence_items=()))
        assert result.explanation.confidence == KnowledgeConfidence.MINIMAL


# ---------------------------------------------------------------------------
# Cognitive load
# ---------------------------------------------------------------------------


class TestCognitiveLoad:
    def test_low_load_for_simple_depth(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                guidance=_guidance(recommended_response_depth=ResponseLength.SHORT),
            )
        )
        assert result.cognitive_load == CognitiveLoad.LOW

    def test_high_load_for_technical_manager(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                prospect=_prospect(
                    explicit_role=ObservedValue(value=ProspectRole.TECHNICAL_MANAGER, provenance=EvidenceProvenance(source_turn_id="t1", source_kind=EvidenceSourceKind.EXPLICIT_STATEMENT)),
                ),
            )
        )
        assert result.cognitive_load == CognitiveLoad.HIGH

    def test_normal_load_default(self):
        result = ENGINE.design(
            _input(evidence_items=(_evidence(),))
        )
        assert result.cognitive_load == CognitiveLoad.NORMAL


# ---------------------------------------------------------------------------
# Momentum signal
# ---------------------------------------------------------------------------


class TestMomentumSignal:
    def test_pivot_when_pending_topic_differs(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                conversation=_conversation(
                    current_focus=ConversationTopic.WEBSITE,
                    pending_topic=ConversationTopic.CRM,
                ),
            )
        )
        assert result.momentum == MomentumSignal.PIVOT

    def test_stay_on_topic_when_declining(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                guidance=_guidance(momentum=ConversationMomentum.DECLINING),
            )
        )
        assert result.momentum == MomentumSignal.STAY_ON_TOPIC

    def test_stay_on_topic_when_objection(self):
        result = ENGINE.design(
            _input(
                evidence_items=(_evidence(),),
                guidance=_guidance(objection=ObjectionUnderstanding.NO_TIME),
            )
        )
        assert result.momentum == MomentumSignal.STAY_ON_TOPIC

    def test_continue_by_default(self):
        result = ENGINE.design(
            _input(evidence_items=(_evidence(),))
        )
        assert result.momentum == MomentumSignal.CONTINUE


# ---------------------------------------------------------------------------
# Knowledge saturation
# ---------------------------------------------------------------------------


class TestKnowledgeSaturation:
    def test_not_saturated_when_fresh(self):
        result = ENGINE.design(
            _input(evidence_items=(_evidence(),))
        )
        assert result.knowledge_saturated is False

    def test_saturated_when_all_discussed(self):
        e1 = _evidence(approval_id="ae_1")
        e2 = _evidence(approval_id="ae_2", group=EvidenceGroup.FEATURE)
        result = ENGINE.design(
            _input(
                evidence_items=(e1, e2),
                already_discussed=frozenset({"ae_1", "ae_2"}),
            )
        )
        assert result.knowledge_saturated is True

    def test_not_saturated_when_partially_discussed(self):
        e1 = _evidence(approval_id="ae_1")
        e2 = _evidence(approval_id="ae_2", group=EvidenceGroup.FEATURE)
        result = ENGINE.design(
            _input(
                evidence_items=(e1, e2),
                already_discussed=frozenset({"ae_1"}),
            )
        )
        assert result.knowledge_saturated is False

    def test_not_saturated_with_empty_evidence_and_no_history(self):
        result = ENGINE.design(_input(evidence_items=()))
        assert result.knowledge_saturated is False
