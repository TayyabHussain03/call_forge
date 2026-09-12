"""Focused tests for the optional supervisor and versioned strategy buffer."""

from __future__ import annotations

from dataclasses import fields

import pytest

from app.config.settings import get_settings
from app.conversation.prospect_intelligence.contracts import (
    DecisionAuthority,
    InferenceCandidate,
    InferredProspectEvidence,
    ObjectionType,
    ObservedProspectEvidence,
    ProspectEvidence,
    ProspectIntelligenceSnapshot,
    ProspectRole,
)
from app.conversation.prospect_intelligence.updater import ProspectIntelligenceUpdater
from app.conversation.state_machine.machine import ConversationStateMachine
from app.conversation.state_machine.states import load_config
from app.conversation.strategy.contracts import (
    ConversationStrategyHint,
    ConversationStrategyInput,
    SalesStage,
    StrategyType,
)
from app.conversation.strategy.engine import ConversationStrategyEngine
from app.conversation.supervisor.buffer import BufferWriteOutcome, StrategyBuffer
from app.conversation.supervisor.contracts import SupervisorInput, SupervisorInsight
from app.conversation.supervisor.coordinator import (
    SupervisorCoordinator,
    SupervisorRunOutcome,
)
from app.conversation.supervisor.provider import (
    MockSupervisorProvider,
    SupervisorProvider,
)
from app.core.constants import ConversationState


def _strategy_input(
    snapshot: ProspectIntelligenceSnapshot | None = None,
    hint: ConversationStrategyHint | None = None,
):  # type: ignore[no-untyped-def]
    return ConversationStrategyInput(
        current_stage=SalesStage.DISCOVERY,
        current_state=ConversationState.LISTEN,
        prospect_intelligence=snapshot or ProspectIntelligenceSnapshot(),
        strategy_hint=hint,
    )


def _strategy():  # type: ignore[no-untyped-def]
    return ConversationStrategyEngine().recommend(_strategy_input())


def _input(sequence: int = 1, turn_id: str = "turn-1") -> SupervisorInput:
    return SupervisorInput(
        call_id="call",
        turn_id=turn_id,
        source_turn_sequence=sequence,
        current_turn_excerpt="bounded completed-turn excerpt",
        current_state=ConversationState.LISTEN,
        prospect_intelligence=ProspectIntelligenceSnapshot(),
        conversation_strategy=_strategy(),
    )


def _inference(
    turn_id: str,
    *,
    role: ProspectRole | None = None,
    authority: DecisionAuthority | None = None,
    objection: ObjectionType | None = None,
    confidence: float = 0.8,
) -> ProspectEvidence:
    return ProspectEvidence(
        inferred=InferredProspectEvidence(
            turn_id,
            likely_role=(InferenceCandidate(role, confidence) if role else None),
            decision_authority=(
                InferenceCandidate(authority, confidence) if authority else None
            ),
            objection_type=(
                InferenceCandidate(objection, confidence) if objection else None
            ),
        )
    )


def _insight(
    sequence: int = 1,
    turn_id: str = "turn-1",
    *,
    call_id: str = "call",
    role: ProspectRole = ProspectRole.MANAGER,
    confidence: float = 0.8,
    hint: ConversationStrategyHint | None = None,
) -> SupervisorInsight:
    return SupervisorInsight(
        call_id,
        turn_id,
        sequence,
        _inference(turn_id, role=role, confidence=confidence),
        hint,
    )


class MalformedSupervisorProvider(SupervisorProvider):
    """Return a deliberately malformed value for boundary testing."""

    @property
    def name(self) -> str:
        return "malformed"

    def analyze(self, supervisor_input):  # type: ignore[no-untyped-def]
        return object()


def test_supervisor_result_is_tied_to_source_turn_and_sequence() -> None:
    insight = _insight(7, "turn-7")

    assert insight.source_turn_id == "turn-7"
    assert insight.source_turn_sequence == 7
    assert insight.prospect_evidence is not None
    assert insight.prospect_evidence.inferred is not None
    assert insight.prospect_evidence.inferred.source_turn_id == "turn-7"


def test_latest_result_is_accepted() -> None:
    buffer = StrategyBuffer("call")

    assert buffer.record(_insight()) == BufferWriteOutcome.ACCEPTED
    assert buffer.latest() == _insight()


def test_stale_result_is_rejected() -> None:
    buffer = StrategyBuffer("call")
    buffer.record(_insight(6, "turn-6"))

    assert buffer.record(_insight(5, "turn-5")) == BufferWriteOutcome.STALE_REJECTED
    assert buffer.latest() == _insight(6, "turn-6")


def test_same_turn_duplicate_is_ignored_idempotently() -> None:
    buffer = StrategyBuffer("call")
    insight = _insight()
    buffer.record(insight)

    assert buffer.record(insight) == BufferWriteOutcome.DUPLICATE_IGNORED
    assert buffer.latest() == insight


def test_later_turn_result_replaces_earlier_result() -> None:
    buffer = StrategyBuffer("call")
    buffer.record(_insight(1, "turn-1"))
    later = _insight(2, "turn-2", role=ProspectRole.OWNER)

    assert buffer.record(later) == BufferWriteOutcome.ACCEPTED
    assert buffer.latest() == later


def test_insight_cannot_apply_backward_or_to_its_source_turn() -> None:
    buffer = StrategyBuffer("call")
    buffer.record(_insight(12, "turn-12"))

    assert buffer.latest_applicable_for(11) is None
    assert buffer.latest_applicable_for(12) is None
    assert buffer.latest_applicable_for(13) is not None


def test_separate_call_buffers_are_isolated() -> None:
    first = StrategyBuffer("first")
    second = StrategyBuffer("second")
    first.record(_insight(call_id="first"))

    assert first.latest() is not None
    assert second.latest() is None


def test_supervisor_failure_leaves_buffer_unchanged() -> None:
    buffer = StrategyBuffer("call")
    buffer.record(_insight(0, "turn-0"))
    before = buffer.snapshot()
    provider = MockSupervisorProvider(should_fail=True)
    result = SupervisorCoordinator(provider, buffer).analyze(_input())

    assert result.outcome == SupervisorRunOutcome.FAILED
    assert result.buffer == before == buffer.snapshot()
    assert provider.call_count == 1


def test_invalid_output_leaves_buffer_unchanged() -> None:
    buffer = StrategyBuffer("call")
    buffer.record(_insight(0, "turn-0"))
    before = buffer.snapshot()

    result = SupervisorCoordinator(MalformedSupervisorProvider(), buffer).analyze(
        _input()
    )

    assert result.outcome == SupervisorRunOutcome.INVALID_OUTPUT
    assert buffer.snapshot() == before


def test_supervisor_inference_remains_inferred_after_consumption() -> None:
    snapshot = ProspectIntelligenceUpdater().update(
        ProspectIntelligenceSnapshot(),
        _insight().prospect_evidence,  # type: ignore[arg-type]
    )

    assert snapshot.inferred.likely_role is not None
    assert snapshot.observed.explicit_role is None


def test_confidence_one_does_not_promote_supervisor_inference() -> None:
    evidence = _inference("turn-1", role=ProspectRole.OWNER, confidence=1.0)
    snapshot = ProspectIntelligenceUpdater().update(
        ProspectIntelligenceSnapshot(), evidence
    )

    assert snapshot.inferred.likely_role is not None
    assert snapshot.inferred.likely_role.confidence == 1.0
    assert snapshot.observed.explicit_role is None


def test_observed_role_overrides_supervisor_role_inference() -> None:
    updater = ProspectIntelligenceUpdater()
    inferred = updater.update(
        ProspectIntelligenceSnapshot(),
        _inference("turn-1", role=ProspectRole.OWNER),
    )
    observed = updater.update(
        inferred,
        ProspectEvidence(
            observed=ObservedProspectEvidence(
                "turn-2", explicit_role=ProspectRole.RECEPTIONIST
            )
        ),
    )

    assert observed.observed.explicit_role is not None
    assert observed.observed.explicit_role.value == ProspectRole.RECEPTIONIST
    assert observed.inferred.likely_role is None


def test_current_explicit_evidence_outranks_same_cycle_supervisor_inference() -> None:
    updater = ProspectIntelligenceUpdater()
    with_supervisor = updater.update(
        ProspectIntelligenceSnapshot(),
        _inference("turn-1", role=ProspectRole.OWNER),
    )
    current = updater.update(
        with_supervisor,
        ProspectEvidence(
            observed=ObservedProspectEvidence(
                "turn-2", explicit_role=ProspectRole.GATEKEEPER
            )
        ),
    )

    assert current.inferred.likely_role is None
    assert current.observed.explicit_role is not None
    assert current.observed.explicit_role.value == ProspectRole.GATEKEEPER


def test_supervisor_objection_remains_advisory() -> None:
    evidence = _inference("turn-1", objection=ObjectionType.PRICE)
    snapshot = ProspectIntelligenceUpdater().update(
        ProspectIntelligenceSnapshot(), evidence
    )

    assert snapshot.inferred.objection_type is not None
    assert "action" not in {item.name for item in fields(snapshot.inferred)}


def test_supervisor_cannot_create_trusted_priority() -> None:
    names = {item.name for item in fields(SupervisorInsight)}

    assert names.isdisjoint({"dnc", "not_interested", "trusted_priority"})


def test_supervisor_cannot_create_fsm_transition() -> None:
    names = {item.name for item in fields(SupervisorInsight)}

    assert names.isdisjoint({"next_state", "transition", "execute_action"})


@pytest.mark.parametrize("forbidden", ("pricing", "pricing_authority"))
def test_supervisor_cannot_authorize_pricing(forbidden: str) -> None:
    assert forbidden not in {item.name for item in fields(SupervisorInsight)}


@pytest.mark.parametrize("forbidden", ("discount", "discount_authority"))
def test_supervisor_cannot_authorize_discount(forbidden: str) -> None:
    assert forbidden not in {item.name for item in fields(SupervisorInsight)}


def test_supervisor_cannot_confirm_callback_or_demo() -> None:
    names = {item.name for item in fields(SupervisorInsight)}

    assert names.isdisjoint({"callback_confirmed", "demo_confirmed", "booking"})


def test_supervisor_cannot_persist_contact() -> None:
    names = {item.name for item in fields(SupervisorInsight)}

    assert names.isdisjoint({"persist_contact", "confirmed_contact", "contact_write"})


def test_strategy_hint_is_advisory_and_has_no_action() -> None:
    hint = ConversationStrategyHint(
        StrategyType.DISCOVER_BUSINESS_IMPACT,
        0.8,
    )

    assert "action" not in {item.name for item in fields(hint)}
    assert "next_state" not in {item.name for item in fields(hint)}


def test_strategy_engine_rejects_hint_conflicting_with_observed_role() -> None:
    snapshot = ProspectIntelligenceUpdater().update(
        ProspectIntelligenceSnapshot(),
        ProspectEvidence(
            observed=ObservedProspectEvidence(
                "turn-1", explicit_role=ProspectRole.RECEPTIONIST
            )
        ),
    )
    hint = ConversationStrategyHint(
        StrategyType.DISCOVER_BUSINESS_IMPACT,
        1.0,
    )

    result = ConversationStrategyEngine().recommend(_strategy_input(snapshot, hint))
    assert result.strategy_type == StrategyType.ROUTE_TO_DECISION_MAKER
    assert "appropriate person" in result.communication_goal


def test_valid_consistent_hint_can_influence_strategy_goal() -> None:
    snapshot = ProspectIntelligenceUpdater().update(
        ProspectIntelligenceSnapshot(),
        _inference("turn-1", role=ProspectRole.MANAGER),
    )
    hint = ConversationStrategyHint(
        StrategyType.DISCOVER_BUSINESS_IMPACT,
        0.7,
    )

    result = ConversationStrategyEngine().recommend(_strategy_input(snapshot, hint))
    assert result.strategy_type == hint.strategy_type
    assert "operational business impact" in result.communication_goal


def test_fast_path_has_no_required_supervisor_result() -> None:
    assert StrategyBuffer("call").latest_applicable_for(1) is None


def test_fast_path_can_continue_after_supervisor_failure() -> None:
    buffer = StrategyBuffer("call")
    SupervisorCoordinator(MockSupervisorProvider(should_fail=True), buffer).analyze(
        _input()
    )

    assert buffer.consume_for(2) is None
    assert ConversationStrategyEngine().recommend(_strategy_input()).strategy_type


def test_fake_supervisor_provider_replays_deterministically() -> None:
    insight = _insight()
    provider = MockSupervisorProvider(default=insight)

    assert provider.analyze(_input()) == insight
    assert provider.analyze(_input()) == insight
    assert provider.call_count == 2


def test_supervisor_input_is_bounded_and_excludes_full_transcript() -> None:
    names = {item.name for item in fields(SupervisorInput)}

    assert "full_transcript" not in names
    with pytest.raises(ValueError):
        SupervisorInput(
            call_id="call",
            turn_id="turn-1",
            source_turn_sequence=1,
            current_turn_excerpt="x" * 501,
            current_state=ConversationState.LISTEN,
            prospect_intelligence=ProspectIntelligenceSnapshot(),
            conversation_strategy=_strategy(),
        )


def test_buffer_snapshot_excludes_raw_reasoning() -> None:
    snapshot = StrategyBuffer("call").snapshot()

    assert "reasoning" not in {item.name for item in fields(snapshot)}
    assert "chain_of_thought" not in {item.name for item in fields(snapshot)}


def test_supervisor_prospect_update_is_immutable() -> None:
    previous = ProspectIntelligenceSnapshot()
    updated = ProspectIntelligenceUpdater().update(
        previous, _inference("turn-1", role=ProspectRole.MANAGER)
    )

    assert updated is not previous
    assert previous == ProspectIntelligenceSnapshot()


def test_repeated_consumption_does_not_duplicate_evidence() -> None:
    buffer = StrategyBuffer("call")
    buffer.record(_insight())

    assert buffer.consume_for(2) is not None
    assert buffer.consume_for(2) is None
    assert buffer.consume_for(3) is None


def test_recording_wrong_call_insight_is_rejected() -> None:
    buffer = StrategyBuffer("call")

    assert buffer.record(_insight(call_id="other")) == (
        BufferWriteOutcome.INVALID_REJECTED
    )
    assert buffer.latest() is None


def test_supervisor_contract_cannot_emit_observed_facts() -> None:
    with pytest.raises(ValueError):
        SupervisorInsight(
            "call",
            "turn-1",
            1,
            ProspectEvidence(
                observed=ObservedProspectEvidence(
                    "turn-1", explicit_role=ProspectRole.OWNER
                )
            ),
        )


def test_supervisor_analysis_never_mutates_state_machine() -> None:
    machine = ConversationStateMachine(
        load_config(get_settings().conversation_config_path)
    )
    before = machine.current_state
    SupervisorCoordinator(
        MockSupervisorProvider(default=_insight()), StrategyBuffer("call")
    ).analyze(_input())

    assert machine.current_state == before
