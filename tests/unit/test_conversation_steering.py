"""Focused deterministic tests for priority evolution and curiosity protection."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace

import pytest

from app.conversation.business_conversation.contracts import BusinessConversationSnapshot, BusinessFactKind, ObservedBusinessFact, UnknownArea
from app.conversation.business_diagnostic.contracts import BusinessDiagnosticInput
from app.conversation.business_diagnostic.engine import BusinessDiagnosticEngine
from app.conversation.conversation_steering.contracts import (
    ConversationSteeringInput, CuriosityBudgetState, DiagnosticPriority, DiscussionTiming,
    CuriosityMemory, PriorityEvolution, QuestionDecision,
)
from app.conversation.conversation_steering.engine import ConversationSteeringEngine
from app.core.constants import ConversationState
from app.conversation.strategy.contracts import ConversationMode, ConversationStrategy, SalesStage, StrategyType


def _conversation(*facts: ObservedBusinessFact, unknowns=frozenset()):
    return BusinessConversationSnapshot(facts=facts, unknown_areas=unknowns)


def _fact(kind: BusinessFactKind, value: str) -> ObservedBusinessFact:
    return ObservedBusinessFact(kind, value, "turn-1")


def _steer(conversation, prior=None):
    diagnostic = BusinessDiagnosticEngine().diagnose(BusinessDiagnosticInput(conversation))
    return ConversationSteeringEngine().steer(ConversationSteeringInput(conversation, diagnostic, None, None, ConversationState.LISTEN, prior))


def test_first_priority_is_raised_and_enforces_exactly_one_question() -> None:
    snapshot = _steer(_conversation(_fact(BusinessFactKind.MANUAL_STEP, "manual invoices")))
    assert snapshot.current_priority == DiagnosticPriority.WORKFLOW
    assert snapshot.evolution == PriorityEvolution.RAISED
    assert snapshot.question_decision == QuestionDecision.ASK_ONE
    assert snapshot.discussion_timing == DiscussionTiming.PREMATURE


def test_same_priority_becomes_complete_and_suppresses_repeat_question() -> None:
    first = _steer(_conversation(_fact(BusinessFactKind.MANUAL_STEP, "manual invoices")))
    second = _steer(_conversation(_fact(BusinessFactKind.MANUAL_STEP, "manual invoices")), first)
    assert second.evolution == PriorityEvolution.LOWERED
    assert second.question_decision == QuestionDecision.ASK_NONE
    assert second.curiosity_budget == CuriosityBudgetState.EXHAUSTED


def test_new_customer_focus_replaces_priority_and_preserves_pending_priority() -> None:
    workflow = _steer(_conversation(_fact(BusinessFactKind.MANUAL_STEP, "manual invoices")))
    website = _steer(_conversation(_fact(BusinessFactKind.WEBSITE, "exists")), workflow)
    assert website.current_priority == DiagnosticPriority.WEBSITE_PERFORMANCE
    assert website.evolution == PriorityEvolution.REPLACED
    assert website.pending_priority == DiagnosticPriority.WORKFLOW


def test_unknowns_do_not_become_known_or_trigger_service_discussion() -> None:
    snapshot = _steer(_conversation(unknowns=frozenset({UnknownArea.CRM})))
    assert snapshot.current_priority == DiagnosticPriority.BUSINESS_CONTEXT
    assert snapshot.discussion_timing == DiscussionTiming.PREMATURE
    assert DiagnosticPriority.BUSINESS_CONTEXT in snapshot.memory.still_unknown


def test_correction_reopens_previously_explored_priority() -> None:
    first = _steer(_conversation(_fact(BusinessFactKind.WEBSITE, "no website")))
    correction = BusinessConversationSnapshot(
        facts=(_fact(BusinessFactKind.WEBSITE, "website exists"),), archived_facts=(_fact(BusinessFactKind.WEBSITE, "no website"),)
    )
    second = _steer(correction, first)
    assert DiagnosticPriority.WEBSITE_PERFORMANCE in second.memory.corrected
    assert second.question_decision == QuestionDecision.ASK_ONE


def test_deferred_and_refused_priority_are_preserved_without_questions() -> None:
    first = _steer(_conversation(_fact(BusinessFactKind.WEBSITE, "exists")))
    deferred = _steer(_conversation(_fact(BusinessFactKind.WEBSITE, "exists")), replace(first, memory=CuriosityMemory(deferred=frozenset({DiagnosticPriority.WEBSITE_PERFORMANCE}))))
    blocked = _steer(_conversation(_fact(BusinessFactKind.WEBSITE, "exists")), replace(first, memory=CuriosityMemory(refused=frozenset({DiagnosticPriority.WEBSITE_PERFORMANCE}))))
    assert deferred.evolution == PriorityEvolution.DEFERRED
    assert deferred.question_decision == QuestionDecision.DEFER
    assert blocked.evolution == PriorityEvolution.BLOCKED
    assert blocked.question_decision == QuestionDecision.ASK_NONE


def test_busy_conversation_limits_curiosity_without_changing_priority() -> None:
    conversation = _conversation(_fact(BusinessFactKind.MANUAL_STEP, "manual invoices"))
    diagnostic = BusinessDiagnosticEngine().diagnose(BusinessDiagnosticInput(conversation))
    strategy = ConversationStrategy(SalesStage.DISCOVERY, ConversationMode.BUSY, "understand", StrategyType.DISCOVER_NEED)
    snapshot = ConversationSteeringEngine().steer(ConversationSteeringInput(conversation, diagnostic, None, strategy, ConversationState.LISTEN))
    assert snapshot.current_priority == DiagnosticPriority.WORKFLOW
    assert snapshot.curiosity_budget == CuriosityBudgetState.LIMITED
    assert snapshot.question_decision == QuestionDecision.WAIT_FOR_CUSTOMER


def test_replay_immutability_and_no_authority_or_service_fields() -> None:
    conversation = _conversation(_fact(BusinessFactKind.CURRENT_WORKFLOW, "spreadsheet"))
    assert _steer(conversation) == _steer(conversation)
    snapshot = _steer(conversation)
    with pytest.raises(FrozenInstanceError):
        snapshot.current_priority = DiagnosticPriority.WORKFLOW  # type: ignore[misc]
    forbidden = {"service", "product", "action", "authority", "state", "price", "execution", "recommend"}
    assert not any(any(word in item.name for word in forbidden) for item in fields(type(snapshot)))
