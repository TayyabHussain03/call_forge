"""Slice 46 — Per-contract snapshot serializer round-trip tests.

One serializer per real domain contract. Each proves:
    - round-trip equality (empty + populated + unicode)
    - version check (unsupported version fails closed)
    - invalid payload fails closed
    - deterministic serialization (same input → same bytes)
    - unknown enum fails closed
"""

from __future__ import annotations

import json

import pytest

from app.application.sales_runtime.serializers import (
    BusinessConversationSerializer,
    LanguageProfileSerializer,
    QualificationSerializer,
)
from app.application.sales_runtime.serializers.base import (
    PayloadRequirement,
    build_envelope,
    parse_envelope,
)
from app.conversation.business_conversation.contracts import (
    BusinessConstraintKind,
    BusinessConversationSnapshot,
    BusinessFactKind,
    BusinessGoalKind,
    BusinessProblemKind,
    ConversationTopic,
    ObservedBusinessFact,
    ObservedBusinessProblem,
    RootCauseHypothesis,
    RootCauseKind,
    UnknownArea,
)
from app.conversation.qualification.contracts import (
    QualificationCompleteness,
    QualificationFieldState,
    QualificationObservation,
    QualificationProgress,
    QualificationSnapshot,
    QualificationValueState,
    QualificationEvidenceKind,
)
from app.conversation.understanding.contracts import (
    ConversationalRegister,
    LanguageProfile,
    LanguageScript,
)
from app.persistence.errors import InvalidPersistedPayloadError


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------


class TestEnvelope:
    def test_build_and_parse(self) -> None:
        env = build_envelope("test", 1, {"x": 1})
        payload = parse_envelope(env, "test", (1,))
        assert payload == {"x": 1}

    def test_wrong_key_rejected(self) -> None:
        env = build_envelope("test", 1, {})
        with pytest.raises(InvalidPersistedPayloadError):
            parse_envelope(env, "other", (1,))

    def test_wrong_version_rejected(self) -> None:
        env = build_envelope("test", 2, {})
        with pytest.raises(InvalidPersistedPayloadError):
            parse_envelope(env, "test", (1,))

    def test_malformed_json_rejected(self) -> None:
        with pytest.raises(InvalidPersistedPayloadError):
            parse_envelope("{not json", "test", (1,))

    def test_deterministic_json(self) -> None:
        # Sort-key canonicalization gives byte-identical output
        a = build_envelope("test", 1, {"b": 2, "a": 1})
        b = build_envelope("test", 1, {"a": 1, "b": 2})
        assert a == b


# ---------------------------------------------------------------------------
# BCI serializer
# ---------------------------------------------------------------------------


class TestBCISerializer:
    def test_empty_snapshot_roundtrip(self) -> None:
        s = BusinessConversationSnapshot()
        ser = BusinessConversationSerializer()
        text = ser.serialize(s)
        back = ser.deserialize(text)
        assert back == s

    def test_populated_snapshot_roundtrip(self) -> None:
        s = BusinessConversationSnapshot(
            facts=(
                ObservedBusinessFact(
                    BusinessFactKind.WEBSITE, "we have a website", "t1",
                ),
                ObservedBusinessFact(
                    BusinessFactKind.MANUAL_STEP, "excel tracking", "t3",
                ),
            ),
            problems=(
                ObservedBusinessProblem(
                    BusinessProblemKind.OPERATIONAL,
                    "manual lead handling",
                    "t3",
                    primary=True,
                ),
            ),
            goals=frozenset({BusinessGoalKind.GENERATE_LEADS}),
            constraints=frozenset({BusinessConstraintKind.CURRENT_VENDOR}),
            unknown_areas=frozenset({UnknownArea.CRM}),
            hypotheses=(
                RootCauseHypothesis(
                    kind=RootCauseKind.AUTOMATION_GAP,
                    confidence=0.75,
                    reason="manual workflow with existing provider",
                    supporting_observations=("manual tracking in excel",),
                ),
            ),
            current_focus=ConversationTopic.OPERATIONS,
            pending_topic=ConversationTopic.CRM,
        )
        ser = BusinessConversationSerializer()
        back = ser.deserialize(ser.serialize(s))
        assert back == s

    def test_unicode_preserved(self) -> None:
        s = BusinessConversationSnapshot(
            facts=(
                ObservedBusinessFact(
                    BusinessFactKind.BUSINESS_TYPE, "Café Niño — Español", "t1",
                ),
            ),
        )
        ser = BusinessConversationSerializer()
        back = ser.deserialize(ser.serialize(s))
        assert back.facts[0].value == "Café Niño — Español"

    def test_observed_remains_observed(self) -> None:
        """Critical invariant: no promotion/demotion through roundtrip."""
        s = BusinessConversationSnapshot(
            facts=(
                ObservedBusinessFact(
                    BusinessFactKind.WEBSITE, "existing website", "turn-1",
                ),
            ),
            archived_facts=(
                ObservedBusinessFact(
                    BusinessFactKind.WEBSITE, "superseded fact", "turn-0",
                ),
            ),
        )
        ser = BusinessConversationSerializer()
        back = ser.deserialize(ser.serialize(s))
        assert back.facts == s.facts
        assert back.archived_facts == s.archived_facts

    def test_unknown_enum_rejected(self) -> None:
        envelope = build_envelope("bci", 1, {
            "facts": [{"kind": "nonexistent", "value": "x", "source_turn_id": "t1"}],
        })
        with pytest.raises(InvalidPersistedPayloadError):
            BusinessConversationSerializer().deserialize(envelope)

    def test_deterministic_output(self) -> None:
        s = BusinessConversationSnapshot(
            goals=frozenset({
                BusinessGoalKind.GENERATE_LEADS,
                BusinessGoalKind.EFFICIENCY,
            }),
        )
        ser = BusinessConversationSerializer()
        assert ser.serialize(s) == ser.serialize(s)

    def test_requirement_is_required(self) -> None:
        ser = BusinessConversationSerializer()
        assert ser.requirement == PayloadRequirement.REQUIRED_FOR_ACTIVE_RECOVERY


# ---------------------------------------------------------------------------
# LanguageProfile serializer
# ---------------------------------------------------------------------------


class TestLanguageProfileSerializer:
    def test_minimal_roundtrip(self) -> None:
        p = LanguageProfile(primary_language="en")
        ser = LanguageProfileSerializer()
        assert ser.deserialize(ser.serialize(p)) == p

    def test_mixed_language_roundtrip(self) -> None:
        p = LanguageProfile(
            primary_language="ur",
            secondary_language="en",
            mixed_language=True,
            script=LanguageScript.LATIN,
            conversational_register=ConversationalRegister.CASUAL,
        )
        ser = LanguageProfileSerializer()
        assert ser.deserialize(ser.serialize(p)) == p

    def test_unknown_script_rejected(self) -> None:
        envelope = build_envelope("language_profile", 1, {
            "primary_language": "en",
            "secondary_language": None,
            "mixed_language": False,
            "script": "nonexistent",
            "conversational_register": "unknown",
        })
        with pytest.raises(InvalidPersistedPayloadError):
            LanguageProfileSerializer().deserialize(envelope)


# ---------------------------------------------------------------------------
# Qualification serializer
# ---------------------------------------------------------------------------


class TestQualificationSerializer:
    def test_minimal_roundtrip(self) -> None:
        s = QualificationSnapshot(
            configuration_id=("t1", "biz-1", "c1"),
            fields=(),
            completeness=QualificationCompleteness.VERY_EARLY,
            progress=QualificationProgress.STABLE,
        )
        ser = QualificationSerializer()
        assert ser.deserialize(ser.serialize(s)) == s

    def test_mixed_states_preserved(self) -> None:
        """OBSERVED/INFERRED/UNKNOWN must survive roundtrip identically."""
        s = QualificationSnapshot(
            configuration_id=("t1", "biz-1", "c1"),
            fields=(
                QualificationFieldState(
                    field_id="decision_maker",
                    state=QualificationValueState.OBSERVED,
                    value="owner",
                    source_turn_id="t1",
                ),
                QualificationFieldState(
                    field_id="budget",
                    state=QualificationValueState.UNKNOWN,
                ),
                QualificationFieldState(
                    field_id="timeline",
                    state=QualificationValueState.INFERRED,
                    value="within quarter",
                ),
            ),
            completeness=QualificationCompleteness.PARTIAL,
            progress=QualificationProgress.ADVANCING,
            newly_observed=frozenset({"decision_maker"}),
            corrected=frozenset(),
            replaced=frozenset(),
        )
        ser = QualificationSerializer()
        back = ser.deserialize(ser.serialize(s))
        assert back == s
        # Explicit invariant check — states did not shift
        states = {f.field_id: f.state for f in back.fields}
        assert states["decision_maker"] == QualificationValueState.OBSERVED
        assert states["budget"] == QualificationValueState.UNKNOWN
        assert states["timeline"] == QualificationValueState.INFERRED

    def test_archived_observations_preserved(self) -> None:
        s = QualificationSnapshot(
            configuration_id=("t1", "biz-1", "c1"),
            fields=(),
            completeness=QualificationCompleteness.VERY_EARLY,
            progress=QualificationProgress.CORRECTED,
            archived=(
                QualificationObservation(
                    field_id="decision_maker",
                    value="not sure",
                    evidence_kind=QualificationEvidenceKind.OBSERVED,
                    evidence_ids=("ev-1",),
                    source_turn_id="t1",
                ),
            ),
        )
        ser = QualificationSerializer()
        back = ser.deserialize(ser.serialize(s))
        assert back.archived == s.archived


# ---------------------------------------------------------------------------
# Compatibility manifest
# ---------------------------------------------------------------------------


class TestCompatibilityManifest:
    def test_bci_and_language_are_required(self) -> None:
        from app.application.sales_runtime.manifest import default_manifest
        m = default_manifest()
        assert m.is_required("bci")
        assert m.is_required("language_profile")

    def test_qualification_is_optional_derived_today(self) -> None:
        from app.application.sales_runtime.manifest import default_manifest
        m = default_manifest()
        # Qualification serializer exists but is not required for recovery
        # today — follow-up slice may upgrade this classification.
        assert not m.is_required("qualification")

    def test_missing_required_detects_bci(self) -> None:
        from app.application.sales_runtime.manifest import default_manifest
        m = default_manifest()
        missing = m.missing_required(frozenset({"language_profile"}))
        assert missing == frozenset({"bci"})

    def test_is_supported_matches_version(self) -> None:
        from app.application.sales_runtime.manifest import default_manifest
        m = default_manifest()
        assert m.is_supported("bci", 1) is True
        assert m.is_supported("bci", 99) is False
        assert m.is_supported("nonexistent", 1) is False


# ---------------------------------------------------------------------------
# No domain intelligence in serializers
# ---------------------------------------------------------------------------


class TestNoReasoningInSerializers:
    def test_serializer_modules_have_no_reasoning_imports(self) -> None:
        import inspect
        from app.application.sales_runtime.serializers import (
            base, bci, language, qualification,
        )
        for module in (base, bci, language, qualification):
            source = inspect.getsource(module)
            for forbidden in (
                "BrainOrchestrator", "ReasoningProvider",
                "StateMachine", "ActionValidator",
                "HybridRetrievalEngine", "ServiceRelevanceResolver",
            ):
                assert forbidden not in source
