"""Slice 48 — Participant continuity + discussion coverage + policy tests."""

from __future__ import annotations

import pytest

from app.application.sales_runtime.serializers import (
    DiscussionCoverageSerializer,
    ParticipantStateSerializer,
    ProspectIntelligenceSerializer,
)
from app.conversation.participant.contracts import (
    DiscussionCoverageEntry,
    DiscussionCoverageState,
    ExplanationDepth,
    ParticipantContinuityDecision,
    ParticipantContinuityEvidence,
    ParticipantContinuityState,
    ParticipantConversationState,
    ParticipantEvidenceKind,
    ParticipantReference,
    ParticipantRoleHint,
    ReExplanationStrategy,
)
from app.conversation.participant.policy import (
    ParticipantContinuityPolicy,
    ReExplanationIntent,
    ReExplanationPolicy,
    deterministic_participant_factory,
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
    ProspectIntelligenceSnapshot,
    ProspectRole,
)
from app.persistence.errors import InvalidPersistedPayloadError


# ---------------------------------------------------------------------------
# Participant contract validation
# ---------------------------------------------------------------------------


class TestParticipantContract:
    def test_participant_reference_requires_id(self) -> None:
        with pytest.raises(ValueError):
            ParticipantReference(reference_id="")

    def test_new_confirmed_requires_distinct_archived(self) -> None:
        with pytest.raises(ValueError):
            ParticipantContinuityDecision(
                state=ParticipantContinuityState.NEW_CONFIRMED,
                active_participant=ParticipantReference("p1"),
                evidence_kind=ParticipantEvidenceKind.EXPLICIT_HANDOFF,
                archived_prior=ParticipantReference("p1"),
            )


# ---------------------------------------------------------------------------
# Continuity policy — deterministic mapping from evidence
# ---------------------------------------------------------------------------


class TestContinuityPolicy:
    def _factory(self, start: int = 1):
        return deterministic_participant_factory("sess-x", start=start)

    def test_explicit_handoff_marks_new_confirmed(self) -> None:
        p = ParticipantContinuityPolicy()
        active = ParticipantReference("sess-x:participant_1", role_hint=ParticipantRoleHint.EMPLOYEE)
        ev = ParticipantContinuityEvidence(
            kind=ParticipantEvidenceKind.EXPLICIT_HANDOFF,
            source_turn_id="t5",
            claimed_role_hint=ParticipantRoleHint.OWNER,
        )
        decision = p.decide(active, ev, self._factory())
        assert decision.state == ParticipantContinuityState.NEW_CONFIRMED
        assert decision.active_participant.role_hint == ParticipantRoleHint.OWNER
        assert decision.archived_prior == active

    def test_explicit_not_part_of_prior_marks_new_confirmed(self) -> None:
        p = ParticipantContinuityPolicy()
        active = ParticipantReference("sess-x:participant_1")
        ev = ParticipantContinuityEvidence(
            kind=ParticipantEvidenceKind.EXPLICIT_NOT_PART_OF_PRIOR_DISCUSSION,
            source_turn_id="t5",
            claimed_role_hint=ParticipantRoleHint.OWNER,
        )
        decision = p.decide(active, ev, self._factory())
        assert decision.state == ParticipantContinuityState.NEW_CONFIRMED

    def test_inferred_evidence_cannot_confirm_new(self) -> None:
        """INFERRED only → LIKELY_NEW. Never NEW_CONFIRMED. Fail-closed."""
        p = ParticipantContinuityPolicy()
        active = ParticipantReference("sess-x:participant_1")
        ev = ParticipantContinuityEvidence(
            kind=ParticipantEvidenceKind.INFERRED,
            source_turn_id="t3",
        )
        decision = p.decide(active, ev, self._factory())
        assert decision.state == ParticipantContinuityState.LIKELY_NEW

    def test_no_evidence_preserves_active(self) -> None:
        """Ambiguous "What was this about?" must not create a new participant."""
        p = ParticipantContinuityPolicy()
        active = ParticipantReference("sess-x:participant_1")
        ev = ParticipantContinuityEvidence(
            kind=ParticipantEvidenceKind.NONE,
            source_turn_id="t3",
        )
        decision = p.decide(active, ev, self._factory())
        assert decision.state == ParticipantContinuityState.SAME_CONFIRMED
        assert decision.active_participant == active

    def test_no_active_no_evidence_is_unknown(self) -> None:
        p = ParticipantContinuityPolicy()
        ev = ParticipantContinuityEvidence(
            kind=ParticipantEvidenceKind.NONE,
            source_turn_id="t1",
        )
        decision = p.decide(None, ev, self._factory())
        assert decision.state == ParticipantContinuityState.UNKNOWN

    def test_self_identification_same_participant(self) -> None:
        p = ParticipantContinuityPolicy()
        active = ParticipantReference(
            "sess-x:participant_1", role_hint=ParticipantRoleHint.UNKNOWN,
        )
        ev = ParticipantContinuityEvidence(
            kind=ParticipantEvidenceKind.EXPLICIT_SELF_IDENTIFICATION,
            source_turn_id="t2",
            claimed_role_hint=ParticipantRoleHint.OWNER,
        )
        decision = p.decide(active, ev, self._factory())
        assert decision.state == ParticipantContinuityState.SAME_CONFIRMED
        assert decision.active_participant.role_hint == ParticipantRoleHint.OWNER


# ---------------------------------------------------------------------------
# Re-explanation policy — participant-aware repetition
# ---------------------------------------------------------------------------


class TestReExplanationPolicy:
    def test_new_participant_asking_direct_service_question_re_explains(self) -> None:
        p = ReExplanationPolicy()
        coverage = DiscussionCoverageState(
            entries=(
                DiscussionCoverageEntry(
                    "sess:participant_1", "svc.automation",
                    ExplanationDepth.EXPLAINED, "t3",
                ),
            ),
        )
        decision = p.decide(
            ParticipantContinuityState.NEW_CONFIRMED,
            "sess:participant_2", "svc.automation", coverage,
            intent=ReExplanationIntent.DIRECT_SERVICE_QUESTION,
        )
        assert decision.strategy == ReExplanationStrategy.FULL_RE_EXPLAIN_FOR_NEW_PARTICIPANT

    def test_same_participant_already_explained_suppress(self) -> None:
        p = ReExplanationPolicy()
        coverage = DiscussionCoverageState(
            entries=(
                DiscussionCoverageEntry(
                    "sess:participant_1", "svc.automation",
                    ExplanationDepth.EXPLAINED, "t3",
                ),
            ),
        )
        decision = p.decide(
            ParticipantContinuityState.SAME_CONFIRMED,
            "sess:participant_1", "svc.automation", coverage,
            intent=ReExplanationIntent.NONE,
        )
        assert decision.strategy == ReExplanationStrategy.SUPPRESS

    def test_same_participant_recap_request(self) -> None:
        p = ReExplanationPolicy()
        coverage = DiscussionCoverageState(
            entries=(
                DiscussionCoverageEntry(
                    "sess:participant_1", "svc.automation",
                    ExplanationDepth.EXPLAINED, "t3",
                ),
            ),
        )
        decision = p.decide(
            ParticipantContinuityState.SAME_CONFIRMED,
            "sess:participant_1", "svc.automation", coverage,
            intent=ReExplanationIntent.RECAP_REQUEST,
        )
        assert decision.strategy == ReExplanationStrategy.CONCISE_RECAP

    def test_simplify_request(self) -> None:
        p = ReExplanationPolicy()
        decision = p.decide(
            ParticipantContinuityState.SAME_CONFIRMED,
            "sess:participant_1", "svc.automation",
            DiscussionCoverageState(),
            intent=ReExplanationIntent.SIMPLIFY_REQUEST,
        )
        assert decision.strategy == ReExplanationStrategy.SIMPLIFY

    def test_deepen_request(self) -> None:
        p = ReExplanationPolicy()
        decision = p.decide(
            ParticipantContinuityState.SAME_CONFIRMED,
            "sess:participant_1", "svc.automation",
            DiscussionCoverageState(),
            intent=ReExplanationIntent.DEEPEN_REQUEST,
        )
        assert decision.strategy == ReExplanationStrategy.DEEPEN

    def test_no_prior_coverage(self) -> None:
        p = ReExplanationPolicy()
        decision = p.decide(
            ParticipantContinuityState.SAME_CONFIRMED,
            "sess:participant_1", "svc.automation",
            DiscussionCoverageState(),
        )
        assert decision.strategy == ReExplanationStrategy.NO_PRIOR_COVERAGE


# ---------------------------------------------------------------------------
# Discussion coverage queries
# ---------------------------------------------------------------------------


class TestDiscussionCoverage:
    def test_heard_by_participant_is_participant_scoped(self) -> None:
        state = DiscussionCoverageState(
            entries=(
                DiscussionCoverageEntry(
                    "p1", "svc.a", ExplanationDepth.EXPLAINED, "t1",
                ),
                DiscussionCoverageEntry(
                    "p2", "svc.b", ExplanationDepth.BRIEF, "t3",
                ),
            ),
        )
        assert state.heard_by("p1") == frozenset({"svc.a"})
        assert state.heard_by("p2") == frozenset({"svc.b"})

    def test_depth_takes_highest_recorded(self) -> None:
        state = DiscussionCoverageState(
            entries=(
                DiscussionCoverageEntry(
                    "p1", "svc.a", ExplanationDepth.BRIEF, "t1",
                ),
                DiscussionCoverageEntry(
                    "p1", "svc.a", ExplanationDepth.EXPLAINED, "t2",
                ),
            ),
        )
        assert state.depth_for("p1", "svc.a") == ExplanationDepth.EXPLAINED


# ---------------------------------------------------------------------------
# Serializer round-trips
# ---------------------------------------------------------------------------


class TestParticipantStateSerializer:
    def test_roundtrip_minimal(self) -> None:
        ser = ParticipantStateSerializer()
        env = ser.serialize(None, ())
        active, archived = ser.deserialize(env)
        assert active is None
        assert archived == ()

    def test_roundtrip_full(self) -> None:
        ser = ParticipantStateSerializer()
        active = ParticipantReference(
            "sess:p2", role_hint=ParticipantRoleHint.OWNER, display_label="owner",
        )
        archived = (
            ParticipantConversationState(
                participant=ParticipantReference(
                    "sess:p1", role_hint=ParticipantRoleHint.EMPLOYEE,
                ),
                heard_explanation_concepts=frozenset({"svc.automation"}),
                role_hint=ParticipantRoleHint.EMPLOYEE,
                interest_observed=True,
                objection_concepts=frozenset({"price"}),
                language_preference="en",
                busy_observed=False,
            ),
        )
        env = ser.serialize(active, archived)
        back_active, back_archived = ser.deserialize(env)
        assert back_active == active
        assert back_archived == archived


class TestDiscussionCoverageSerializer:
    def test_roundtrip_empty(self) -> None:
        ser = DiscussionCoverageSerializer()
        back = ser.deserialize(ser.serialize(DiscussionCoverageState()))
        assert back == DiscussionCoverageState()

    def test_roundtrip_full(self) -> None:
        ser = DiscussionCoverageSerializer()
        state = DiscussionCoverageState(
            entries=(
                DiscussionCoverageEntry(
                    "p1", "svc.a", ExplanationDepth.EXPLAINED, "t3",
                ),
                DiscussionCoverageEntry(
                    "p1", "svc.b", ExplanationDepth.BRIEF, "t2",
                ),
                DiscussionCoverageEntry(
                    "p2", "svc.a", ExplanationDepth.MENTIONED, "t5",
                ),
            ),
            asked_question_concepts=(("p1", "q.workflow"),),
            handled_objection_concepts=(("p1", "obj.price"),),
            discussed_evidence_ids=frozenset({"ev-1", "ev-2"}),
        )
        back = ser.deserialize(ser.serialize(state))
        assert back == state

    def test_deterministic_output(self) -> None:
        ser = DiscussionCoverageSerializer()
        s = DiscussionCoverageState(
            entries=(
                DiscussionCoverageEntry(
                    "p2", "svc.b", ExplanationDepth.BRIEF, "t2",
                ),
                DiscussionCoverageEntry(
                    "p1", "svc.a", ExplanationDepth.EXPLAINED, "t3",
                ),
            ),
            discussed_evidence_ids=frozenset({"ev-z", "ev-a"}),
        )
        assert ser.serialize(s) == ser.serialize(s)


class TestProspectIntelligenceSerializer:
    def test_roundtrip_empty(self) -> None:
        ser = ProspectIntelligenceSerializer()
        back = ser.deserialize(ser.serialize(ProspectIntelligenceSnapshot()))
        assert back == ProspectIntelligenceSnapshot()

    def test_roundtrip_with_observed_role(self) -> None:
        ser = ProspectIntelligenceSerializer()
        prov = EvidenceProvenance("t1", EvidenceSourceKind.EXPLICIT_STATEMENT)
        snap = ProspectIntelligenceSnapshot(
            observed=ObservedProspectFacts(
                explicit_role=ObservedValue(
                    value=ProspectRole.OWNER, provenance=prov,
                ),
                explicit_interest_signal=ObservedValue(
                    value=True, provenance=prov,
                ),
                explicit_busy_signal=ObservedValue(
                    value=False, provenance=prov,
                ),
            ),
        )
        back = ser.deserialize(ser.serialize(snap))
        assert back.observed.explicit_role is not None
        assert back.observed.explicit_role.value == ProspectRole.OWNER
        assert back.observed.explicit_interest_signal.value is True
        assert back.observed.explicit_busy_signal.value is False

    def test_roundtrip_with_inferred_fields(self) -> None:
        ser = ProspectIntelligenceSerializer()
        prov = EvidenceProvenance("t2", EvidenceSourceKind.INFERENCE)
        snap = ProspectIntelligenceSnapshot(
            inferred=InferredProspectState(
                decision_authority=InferredValue(
                    value=DecisionAuthority.HIGH,
                    confidence=0.6, provenance=prov,
                ),
                objection_type=InferredValue(
                    value=ObjectionType.ALREADY_HAVE_PROVIDER,
                    confidence=0.4, provenance=prov,
                ),
            ),
        )
        back = ser.deserialize(ser.serialize(snap))
        assert back.inferred.decision_authority is not None
        assert back.inferred.decision_authority.value == DecisionAuthority.HIGH
        assert back.inferred.objection_type.value == ObjectionType.ALREADY_HAVE_PROVIDER

    def test_observed_never_becomes_inferred(self) -> None:
        """Invariant: round-trip never promotes INFERRED → OBSERVED or vice versa."""
        ser = ProspectIntelligenceSerializer()
        prov = EvidenceProvenance("t1", EvidenceSourceKind.EXPLICIT_STATEMENT)
        snap = ProspectIntelligenceSnapshot(
            observed=ObservedProspectFacts(
                explicit_role=ObservedValue(ProspectRole.OWNER, prov),
            ),
        )
        back = ser.deserialize(ser.serialize(snap))
        # Observed stays in observed; inferred stays empty
        assert back.observed.explicit_role is not None
        assert back.inferred.likely_role is None

    def test_unknown_enum_rejected(self) -> None:
        ser = ProspectIntelligenceSerializer()
        bad = ('{"contract":"prospect_intelligence","version":1,'
               '"payload":{"observed":{"explicit_role":'
               '{"value":"not_a_role","provenance":'
               '{"source_turn_id":"t1","source_kind":"explicit_statement"}}},'
               '"inferred":{}}}')
        with pytest.raises(InvalidPersistedPayloadError):
            ser.deserialize(bad)


# ---------------------------------------------------------------------------
# Business-vs-participant-vs-conversation scoping tests
# ---------------------------------------------------------------------------


class TestScopingInvariants:
    """Participant-scoped state must NOT transfer between participants."""

    def test_participant_a_interest_not_attributed_to_b(self) -> None:
        state_a = ParticipantConversationState(
            participant=ParticipantReference("p1"),
            interest_observed=True,
        )
        state_b = ParticipantConversationState(
            participant=ParticipantReference("p2"),
        )
        # Fresh participant B has no inherited interest
        assert state_b.interest_observed is None
        assert state_a.interest_observed is True

    def test_participant_objections_isolated(self) -> None:
        state_a = ParticipantConversationState(
            participant=ParticipantReference("p1"),
            objection_concepts=frozenset({"price"}),
        )
        state_b = ParticipantConversationState(
            participant=ParticipantReference("p2"),
        )
        assert "price" in state_a.objection_concepts
        assert state_b.objection_concepts == frozenset()

    def test_participant_language_preference_isolated(self) -> None:
        state_a = ParticipantConversationState(
            participant=ParticipantReference("p1"),
            language_preference="ur",
        )
        state_b = ParticipantConversationState(
            participant=ParticipantReference("p2"),
        )
        assert state_a.language_preference == "ur"
        assert state_b.language_preference is None


# ---------------------------------------------------------------------------
# No reasoning imports in participant layer
# ---------------------------------------------------------------------------


class TestNoReasoningInParticipantLayer:
    def test_no_forbidden_imports(self) -> None:
        import inspect
        from app.conversation.participant import contracts, policy
        for module in (contracts, policy):
            source = inspect.getsource(module)
            for forbidden in (
                "BrainOrchestrator", "StateMachine", "ActionValidator",
                "ReasoningProvider", "HybridRetrievalEngine",
                "ServiceRelevanceResolver", "ResponseRenderer",
            ):
                assert forbidden not in source
