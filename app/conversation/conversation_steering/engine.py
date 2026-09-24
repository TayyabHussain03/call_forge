"""Pure deterministic prioritization and one-question curiosity protection."""

from __future__ import annotations

from app.conversation.business_diagnostic.contracts import DiagnosticFocus
from app.conversation.conversation_steering.contracts import *


class ConversationSteeringEngine:
    def steer(self, value: ConversationSteeringInput) -> ConversationPrioritySnapshot:
        priority = _priority(value.diagnostic)
        prior = value.prior
        memory = _memory(value, priority)
        fatigue = _fatigue(value)
        blocked = priority in memory.refused
        deferred = priority in memory.deferred
        complete = priority in memory.explored and priority not in memory.corrected
        budget = (CuriosityBudgetState.EXHAUSTED if complete else CuriosityBudgetState.LIMITED if fatigue in {CustomerFatigue.QUESTION_FATIGUE, CustomerFatigue.TOPIC_FATIGUE} else CuriosityBudgetState.OPEN)
        decision, reason = _question(blocked, deferred, complete, fatigue, value)
        status = (DiscoveryStatus.DISCOVERY_BLOCKED if blocked else DiscoveryStatus.DISCOVERY_DEFERRED if deferred else DiscoveryStatus.DISCOVERY_COMPLETE if complete else DiscoveryStatus.DISCOVERY_IN_PROGRESS if prior else DiscoveryStatus.DISCOVERY_REQUIRED)
        timing = DiscussionTiming.BLOCKED if blocked else DiscussionTiming.PREMATURE if status != DiscoveryStatus.DISCOVERY_COMPLETE else DiscussionTiming.NATURAL
        evolution = _evolution(prior, priority, blocked, deferred, complete)
        pending = prior.current_priority if prior and prior.current_priority != priority else None
        readiness = (ConversationReadiness.NO_ACTION if blocked else ConversationReadiness.UNDERSTAND_MORE if status != DiscoveryStatus.DISCOVERY_COMPLETE else ConversationReadiness.READY_FOR_PLAYBOOK)
        return ConversationPrioritySnapshot(priority, evolution, prior.current_priority if prior else None, pending, status, readiness, budget, decision, reason, timing, fatigue, memory)


def _priority(diagnostic):
    mapping = {DiagnosticFocus.WEBSITE_PERFORMANCE: DiagnosticPriority.WEBSITE_PERFORMANCE, DiagnosticFocus.SALES_PROCESS: DiagnosticPriority.SALES_PROCESS,
               DiagnosticFocus.LEAD_MANAGEMENT: DiagnosticPriority.LEAD_HANDLING, DiagnosticFocus.WORKFLOW: DiagnosticPriority.WORKFLOW,
               DiagnosticFocus.REPORTING: DiagnosticPriority.REPORTING, DiagnosticFocus.BUSINESS_CONTEXT: DiagnosticPriority.BUSINESS_CONTEXT}
    return mapping[diagnostic.diagnostic_focus]


def _memory(value, priority):
    prior = value.prior.memory if value.prior else CuriosityMemory()
    corrected = set(prior.corrected)
    if value.conversation.archived_facts or value.conversation.archived_problems:
        corrected.add(priority)
    explored = set(prior.explored)
    if value.prior and value.prior.current_priority == priority and value.prior.question_decision == QuestionDecision.ASK_ONE:
        explored.add(priority)
    return CuriosityMemory(frozenset(explored), frozenset({priority} if priority not in explored else ()), prior.refused, prior.deferred, frozenset(corrected))


def _fatigue(value):
    if value.strategy is not None and value.strategy.conversation_mode.value in {"busy", "objection"}:
        return CustomerFatigue.QUESTION_FATIGUE
    return CustomerFatigue.NEUTRAL


def _question(blocked, deferred, complete, fatigue, value):
    if blocked: return QuestionDecision.ASK_NONE, QuestionReason.CONVERSATION_FATIGUE
    if deferred: return QuestionDecision.DEFER, QuestionReason.NATURAL_INTERRUPTION
    if complete: return QuestionDecision.ASK_NONE, QuestionReason.CUSTOMER_ALREADY_ANSWERED
    if fatigue in {CustomerFatigue.QUESTION_FATIGUE, CustomerFatigue.TOPIC_FATIGUE}: return QuestionDecision.WAIT_FOR_CUSTOMER, QuestionReason.CONVERSATION_FATIGUE
    if value.strategy is not None and value.strategy.conversation_mode.value == "question_detour": return QuestionDecision.WAIT_FOR_CUSTOMER, QuestionReason.DIRECT_QUESTION
    return QuestionDecision.ASK_ONE, QuestionReason.MISSING_WORKFLOW


def _evolution(prior, priority, blocked, deferred, complete):
    if blocked: return PriorityEvolution.BLOCKED
    if deferred: return PriorityEvolution.DEFERRED
    if prior is None: return PriorityEvolution.RAISED
    if prior.current_priority == priority: return PriorityEvolution.LOWERED if complete else PriorityEvolution.UNCHANGED
    return PriorityEvolution.REPLACED
