"""Deterministic tests for the Grounded Response Composer."""

import pytest

from app.conversation.business_conversation.contracts import (
    BusinessConversationSnapshot,
    BusinessFactKind,
    BusinessGoalKind,
    ConversationTopic,
    ObservedBusinessFact,
)
from app.conversation.consultative.contracts import (
    ProblemCategory,
    ProspectProblem,
    ServiceFitDecision,
    ServiceFitStatus,
)
from app.conversation.prospect_intelligence.contracts import (
    EvidenceProvenance,
    EvidenceSourceKind,
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
from app.knowledge.grounded_response.contracts import (
    EvidenceFraming,
    ExplanationTone,
    GroundedConversationPlan,
    LanguageAlignment,
    ResponseNaturalness,
    TransitionStyle,
    ValueIntegrationStyle,
)
from app.knowledge.grounded_response.engine import GroundedResponseComposer
from app.knowledge.sales_intelligence.contracts import SalesKnowledgeInput
from app.knowledge.sales_intelligence.engine import SalesKnowledgeIntelligenceEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHECK = "a" * 64
SKIE = SalesKnowledgeIntelligenceEngine()
GRC = GroundedResponseComposer()


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


def _compose(
    evidence_items=(_evidence(),),
    guidance=None,
    problem=None,
    prospect=None,
    conversation=None,
    language=None,
    playbook=None,
    already_discussed=frozenset(),
):
    from app.knowledge.grounded_response.contracts import GroundedResponseInput

    g = guidance or _guidance()
    p = problem or _problem()
    pr = prospect or _prospect()
    c = conversation or _conversation()
    lp = language or _language()

    skie_input = SalesKnowledgeInput(
        approved_evidence=ApprovedEvidenceSet(evidence_items),
        prospect=pr,
        problem=p,
        service_fit=ServiceFitDecision(status=ServiceFitStatus.INSUFFICIENT_CONTEXT),
        sales_guidance=g,
        playbook_guidance=playbook,
        language_profile=lp,
        conversation=c,
        already_discussed_evidence_ids=already_discussed,
    )
    kc = SKIE.design(skie_input)

    grc_input = GroundedResponseInput(
        knowledge_context=kc,
        prospect=pr,
        problem=p,
        sales_guidance=g,
        playbook_guidance=playbook,
        language_profile=lp,
        conversation=c,
    )
    return GRC.compose(grc_input)


# ---------------------------------------------------------------------------
# Explanation tone
# ---------------------------------------------------------------------------


class TestExplanationTone:
    def test_reassuring_tone_for_objection(self):
        result = _compose(
            guidance=_guidance(objection=ObjectionUnderstanding.BUDGET),
        )
        assert result.explanation_tone == ExplanationTone.REASSURING

    def test_conversational_tone_for_trust_building(self):
        result = _compose(
            guidance=_guidance(trust=TrustState.BUILDING, interest=InterestStrength.MODERATE),
        )
        assert result.explanation_tone == ExplanationTone.CONVERSATIONAL

    def test_direct_tone_for_answer_question(self):
        result = _compose(
            problem=_problem(requested_solution="website design"),
        )
        assert result.explanation_tone == ExplanationTone.DIRECT

    def test_educational_tone_for_educate(self):
        result = _compose(
            playbook=_playbook(consultant_step=ConsultantStep.EDUCATE),
        )
        assert result.explanation_tone == ExplanationTone.EDUCATIONAL

    def test_consultative_for_buying_signal(self):
        result = _compose(
            problem=_problem(category=ProblemCategory.WEBSITE, explicit_description="need site"),
            guidance=_guidance(interest=InterestStrength.BUYING_SIGNAL),
        )
        assert result.explanation_tone == ExplanationTone.CONSULTATIVE


# ---------------------------------------------------------------------------
# Language alignment
# ---------------------------------------------------------------------------


class TestLanguageAlignment:
    def test_simplify_for_low_cognitive_load(self):
        result = _compose(
            guidance=_guidance(recommended_response_depth=ResponseLength.SHORT),
        )
        assert result.language_alignment == LanguageAlignment.SIMPLIFY

    def test_elevate_for_formal_register(self):
        result = _compose(
            language=_language(
                primary_language="en",
                conversational_register=ConversationalRegister.FORMAL,
            ),
        )
        assert result.language_alignment == LanguageAlignment.ELEVATE

    def test_mirror_for_casual_register(self):
        result = _compose(
            language=_language(
                primary_language="en",
                conversational_register=ConversationalRegister.CASUAL,
            ),
        )
        assert result.language_alignment == LanguageAlignment.MIRROR

    def test_neutral_default(self):
        result = _compose()
        assert result.language_alignment == LanguageAlignment.NEUTRAL


# ---------------------------------------------------------------------------
# Response guardrail
# ---------------------------------------------------------------------------


class TestResponseGuardrail:
    def test_one_concept_for_low_load(self):
        result = _compose(
            guidance=_guidance(recommended_response_depth=ResponseLength.SHORT),
        )
        assert result.guardrail.max_concepts_per_sentence == 1

    def test_three_concepts_for_high_load(self):
        result = _compose(
            prospect=_prospect(
                explicit_role=ObservedValue(
                    value=ProspectRole.TECHNICAL_MANAGER,
                    provenance=EvidenceProvenance(source_turn_id="t1", source_kind=EvidenceSourceKind.EXPLICIT_STATEMENT),
                ),
            ),
        )
        assert result.guardrail.max_concepts_per_sentence == 3

    def test_avoid_jargon_for_simple(self):
        result = _compose(
            guidance=_guidance(recommended_response_depth=ResponseLength.SHORT),
        )
        assert result.guardrail.avoid_jargon is True

    def test_avoid_feature_listing_for_non_detailed(self):
        result = _compose()
        assert result.guardrail.avoid_feature_listing is True

    def test_cautious_naturalness_when_trust_building(self):
        result = _compose(
            guidance=_guidance(trust=TrustState.BUILDING, interest=InterestStrength.MODERATE),
        )
        assert result.guardrail.naturalness == ResponseNaturalness.CAUTIOUS

    def test_minimal_naturalness_when_saturated(self):
        e1 = _evidence(approval_id="ae_1")
        e2 = _evidence(approval_id="ae_2", group=EvidenceGroup.FEATURE)
        result = _compose(
            evidence_items=(e1, e2),
            already_discussed=frozenset({"ae_1", "ae_2"}),
        )
        assert result.guardrail.naturalness == ResponseNaturalness.MINIMAL


# ---------------------------------------------------------------------------
# Evidence framing
# ---------------------------------------------------------------------------


class TestEvidenceFraming:
    def test_question_answer_for_answer_question(self):
        result = _compose(
            problem=_problem(requested_solution="website design"),
        )
        if result.grounded_items:
            assert result.grounded_items[0].framing == EvidenceFraming.QUESTION_ANSWER

    def test_example_framing_for_industry_analogy(self):
        result = _compose(
            conversation=_conversation(
                facts=(
                    ObservedBusinessFact(kind=BusinessFactKind.INDUSTRY, value="restaurant", source_turn_id="t1"),
                ),
            ),
            guidance=_guidance(recommended_response_depth=ResponseLength.SHORT),
        )
        if result.grounded_items:
            assert result.grounded_items[0].framing == EvidenceFraming.EXAMPLE

    def test_story_framing_for_workflow_parallel(self):
        result = _compose(
            problem=_problem(current_process="manual phone orders"),
        )
        if result.grounded_items:
            assert result.grounded_items[0].framing == EvidenceFraming.STORY


# ---------------------------------------------------------------------------
# Transition style
# ---------------------------------------------------------------------------


class TestTransitionStyle:
    def test_no_transition_for_single_item(self):
        result = _compose(evidence_items=(_evidence(),))
        if result.grounded_items:
            assert result.grounded_items[0].transition == TransitionStyle.NONE

    def test_building_on_for_second_item(self):
        items = tuple(
            _evidence(approval_id=f"ae_{i}", group=EvidenceGroup.BENEFIT)
            for i in range(3)
        )
        result = _compose(
            evidence_items=items,
            guidance=_guidance(interest=InterestStrength.MODERATE),
        )
        if len(result.grounded_items) > 1:
            assert result.grounded_items[1].transition == TransitionStyle.BUILDING_ON


# ---------------------------------------------------------------------------
# Value integration
# ---------------------------------------------------------------------------


class TestValueIntegration:
    def test_leading_for_close_gracefully(self):
        result = _compose(
            problem=_problem(category=ProblemCategory.WEBSITE, explicit_description="need site"),
            guidance=_guidance(interest=InterestStrength.BUYING_SIGNAL),
        )
        if result.grounded_items:
            assert result.grounded_items[0].value_integration == ValueIntegrationStyle.LEADING

    def test_closing_hook_for_objection(self):
        result = _compose(
            guidance=_guidance(objection=ObjectionUnderstanding.BUDGET),
        )
        if result.grounded_items:
            assert result.grounded_items[0].value_integration == ValueIntegrationStyle.CLOSING_HOOK

    def test_embedded_for_primary(self):
        result = _compose()
        if result.grounded_items:
            assert result.grounded_items[0].value_integration == ValueIntegrationStyle.EMBEDDED


# ---------------------------------------------------------------------------
# Grounded items match SKIE selection
# ---------------------------------------------------------------------------


class TestGroundedItemsMatchSKIE:
    def test_grounded_count_matches_selected(self):
        items = tuple(
            _evidence(approval_id=f"ae_{i}", group=EvidenceGroup.BENEFIT)
            for i in range(3)
        )
        result = _compose(
            evidence_items=items,
            guidance=_guidance(interest=InterestStrength.MODERATE),
        )
        skie_input = SalesKnowledgeInput(
            approved_evidence=ApprovedEvidenceSet(items),
            prospect=_prospect(),
            problem=_problem(),
            service_fit=ServiceFitDecision(status=ServiceFitStatus.INSUFFICIENT_CONTEXT),
            sales_guidance=_guidance(interest=InterestStrength.MODERATE),
            playbook_guidance=None,
            language_profile=_language(),
            conversation=_conversation(),
        )
        kc = SKIE.design(skie_input)
        assert len(result.grounded_items) == len(kc.selected)

    def test_empty_evidence_produces_empty_grounded(self):
        result = _compose(evidence_items=())
        assert len(result.grounded_items) == 0


# ---------------------------------------------------------------------------
# Replay determinism
# ---------------------------------------------------------------------------


class TestGRCReplayDeterminism:
    def test_same_input_same_output(self):
        r1 = _compose()
        r2 = _compose()
        assert r1 == r2


# ---------------------------------------------------------------------------
# No authority leakage
# ---------------------------------------------------------------------------


class TestGRCNoAuthorityLeakage:
    def test_output_is_frozen(self):
        from dataclasses import FrozenInstanceError

        result = _compose()
        with pytest.raises(FrozenInstanceError):
            result.grounded_items = ()

    def test_no_execution_fields(self):
        result = _compose()
        assert not hasattr(result, "transition")
        assert not hasattr(result, "pricing")
        assert not hasattr(result, "discount")
        assert not hasattr(result, "service_authorized")


# ---------------------------------------------------------------------------
# Knowledge context passthrough
# ---------------------------------------------------------------------------


class TestKnowledgeContextPassthrough:
    def test_knowledge_context_preserved(self):
        result = _compose()
        assert result.knowledge_context is not None
        assert result.knowledge_context.selected is not None
