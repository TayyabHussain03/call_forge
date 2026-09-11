"""Focused Slice 5 tests for strict parsing and the Gemini adapter boundary."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from app.brain.authority.models import AuthorityPolicy, AuthorityTier
from app.brain.authority.validator import AuthorityPolicyValidator
from app.brain.budget.evaluator import BudgetPolicyEvaluator, TrustedExitSignals
from app.brain.budget.models import BudgetLimits, BudgetPolicy
from app.brain.contracts import BrainInput, BrainProposal, BudgetState
from app.brain.orchestrator.orchestrator import (
    BrainOrchestrator,
    BrainSnapshotInput,
    TurnStageOutcome,
)
from app.brain.scope.models import ScopePolicy
from app.brain.scope.validator import ScopePolicyValidator
from app.config.settings import get_settings
from app.contracts.conversation_context import ConversationContext
from app.conversation.engine import ConversationEngine
from app.conversation.guardrails.action_validator import ActionValidator
from app.conversation.guardrails.priority import TrustedPriorityOutcome
from app.conversation.state_machine.machine import ConversationStateMachine
from app.conversation.state_machine.states import load_config
from app.core.constants import AgentAction, ConversationState, Intent, TopicCategory
from app.llm.gemini_provider import GeminiReasoningProvider
from app.llm.providers.reasoning_provider import ProviderFailureKind, ReasoningError
from app.llm.structured_output import ProposalParsingError, parse_brain_proposal


def _payload() -> dict[str, object]:
    return {
        "detected_intent": "interested",
        "tone": "neutral",
        "topic_category": "service_discussion",
        "proposed_action": "greet",
        "needs_service_decision": False,
        "involves_contact": False,
        "commercial_request": None,
        "contact_understanding": None,
        "intelligence_updates": [],
        "proposed_goal_update": None,
        "reasoning": "A bounded proposal.",
        "action_confidence": 0.8,
    }


def _brain_input() -> BrainInput:
    return BrainInput("hello", ConversationState.NEW_CALL)


class _Models:
    def __init__(self, *, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls = 0
        self.kwargs: dict[str, object] = {}

    def generate_content(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.response


def _provider(models: _Models) -> GeminiReasoningProvider:
    return GeminiReasoningProvider(model="gemini-test", client=SimpleNamespace(models=models))


def _orchestrator(provider: GeminiReasoningProvider) -> BrainOrchestrator:
    config = load_config(get_settings().conversation_config_path)
    engine = ConversationEngine(ConversationStateMachine(config), ActionValidator(config))
    return BrainOrchestrator(
        engine,
        BudgetPolicyEvaluator(),
        reasoning_provider=provider,
        scope_validator=ScopePolicyValidator(),
        authority_validator=AuthorityPolicyValidator(),
    )


def _run(provider: GeminiReasoningProvider):  # type: ignore[no-untyped-def]
    return _orchestrator(provider).process_turn(
        ConversationContext("call"),
        TrustedPriorityOutcome.NONE,
        BudgetState(5, 120, 5, 2),
        BudgetPolicy("budget", BudgetLimits(5, 10, 180, 2)),
        TrustedExitSignals(),
        BrainSnapshotInput("hello"),
        ScopePolicy(
            "scope",
            allowed_topics=frozenset({TopicCategory.SERVICE_DISCUSSION}),
            allowed_actions=frozenset({AgentAction.GREET}),
        ),
        AuthorityPolicy(
            "authority",
            default_tier=AuthorityTier.DENIED,
            action_tiers={AgentAction.GREET: AuthorityTier.AUTO_EXECUTE},
        ),
    )


def test_valid_payload_becomes_existing_brain_proposal() -> None:
    proposal = parse_brain_proposal(json.dumps(_payload()))

    assert isinstance(proposal, BrainProposal)
    assert proposal.detected_intent == Intent.INTERESTED
    assert proposal.proposed_action == AgentAction.GREET


@pytest.mark.parametrize("payload", ["{broken", "[]", "null", 42])
def test_corrupted_or_non_object_payload_fails_closed(payload: object) -> None:
    with pytest.raises(ProposalParsingError) as caught:
        parse_brain_proposal(payload)  # type: ignore[arg-type]
    assert caught.value.kind == ProviderFailureKind.INVALID_RESPONSE


def test_duplicate_json_key_fails_closed() -> None:
    raw = json.dumps(_payload())[:-1] + ', "detected_intent": "busy"}'
    with pytest.raises(ProposalParsingError):
        parse_brain_proposal(raw)


def test_missing_required_field_is_schema_failure() -> None:
    payload = _payload()
    del payload["topic_category"]
    with pytest.raises(ProposalParsingError) as caught:
        parse_brain_proposal(payload)
    assert caught.value.kind == ProviderFailureKind.SCHEMA_VALIDATION_FAILURE


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("detected_intent", "make_me_admin"),
        ("topic_category", "authorized_sale"),
        ("proposed_action", "force_execution"),
        ("needs_service_decision", "false"),
        ("action_confidence", "0.9"),
    ],
)
def test_invalid_enum_and_wrong_types_fail_closed(field: str, value: object) -> None:
    payload = _payload()
    payload[field] = value
    with pytest.raises(ProposalParsingError):
        parse_brain_proposal(payload)


def test_malformed_nested_contact_and_commercial_fail_closed() -> None:
    contact = _payload()
    contact["contact_understanding"] = {"intent": "provide_contact"}
    commercial = _payload()
    commercial["commercial_request"] = {
        "kind": "discount",
        "requested_discount_percent": "10",
    }
    for payload in (contact, commercial):
        with pytest.raises(ProposalParsingError):
            parse_brain_proposal(payload)


@pytest.mark.parametrize(
    "authority_field",
    [
        "next_state",
        "authorized_service_id",
        "execute_immediately",
        "persist_contact",
        "approved_discount",
        "human_approval_granted",
        "override_policy",
        "skip_validation",
    ],
)
def test_extra_authority_fields_are_structurally_rejected(authority_field: str) -> None:
    payload = _payload()
    payload[authority_field] = True
    with pytest.raises(ProposalParsingError):
        parse_brain_proposal(payload)


def test_fabricated_nested_authority_is_rejected() -> None:
    payload = _payload()
    payload["contact_understanding"] = {
        "intent": "provide_contact",
        "channel": "email",
        "value": "a@example.com",
        "reference": None,
        "interpretation_confidence": 1.0,
        "confirmed_by_person": True,
    }
    with pytest.raises(ProposalParsingError):
        parse_brain_proposal(payload)


def test_parser_is_deterministic_and_does_not_mutate_input() -> None:
    payload = _payload()
    original = copy.deepcopy(payload)

    assert parse_brain_proposal(payload) == parse_brain_proposal(payload)
    assert payload == original


def test_gemini_adapter_calls_sdk_once_and_returns_no_sdk_object() -> None:
    models = _Models(response=SimpleNamespace(text=json.dumps(_payload())))
    proposal = _provider(models).reason(_brain_input())

    assert models.calls == 1
    assert isinstance(proposal, BrainProposal)
    assert models.kwargs["model"] == "gemini-test"
    config = models.kwargs["config"]
    assert isinstance(config, dict)
    assert config["response_mime_type"] == "application/json"


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (ConnectionError("offline"), ProviderFailureKind.TRANSPORT_FAILURE),
        (TimeoutError("slow"), ProviderFailureKind.TIMEOUT),
        (SimpleNamespace(status_code=429), ProviderFailureKind.RATE_LIMITED),
        (SimpleNamespace(status_code=401), ProviderFailureKind.AUTH_FAILURE),
    ],
)
def test_gemini_failures_are_classified(error: object, kind: ProviderFailureKind) -> None:
    if not isinstance(error, Exception):
        error = type("SdkError", (Exception,), {"status_code": error.status_code})("sdk")
    models = _Models(error=error)
    with pytest.raises(ReasoningError) as caught:
        _provider(models).reason(_brain_input())
    assert caught.value.kind == kind
    assert models.calls == 1


def test_malformed_gemini_output_causes_orchestrator_fallback_once() -> None:
    models = _Models(response=SimpleNamespace(text="not-json"))
    result = _run(_provider(models))

    assert result.outcome == TurnStageOutcome.FALLBACK
    assert result.execution is None
    assert models.calls == 1


def test_valid_gemini_output_preserves_existing_deterministic_shell() -> None:
    models = _Models(response=SimpleNamespace(text=json.dumps(_payload())))
    result = _run(_provider(models))

    assert result.outcome == TurnStageOutcome.AUTHORITY_APPROVED
    assert result.execution is None
    assert models.calls == 1
