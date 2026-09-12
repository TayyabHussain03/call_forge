"""Minimal direct safety regression for the authoritative state machine."""

from __future__ import annotations

from dataclasses import fields

import pytest

from app.brain.contracts import BrainProposal
from app.config.settings import get_settings
from app.conversation.state_machine.machine import ConversationStateMachine
from app.conversation.state_machine.states import load_config
from app.core.constants import AgentAction, ConversationState, Intent
from app.core.exceptions import StateTransitionError


def _machine() -> ConversationStateMachine:
    return ConversationStateMachine(
        load_config(get_settings().conversation_config_path)
    )


def test_approved_valid_action_transitions_through_real_state_machine() -> None:
    machine = _machine()

    transition = machine.apply_transition(AgentAction.GREET)

    assert transition.from_state == ConversationState.NEW_CALL
    assert transition.to_state == ConversationState.GREETING
    assert machine.current_state == ConversationState.GREETING


def test_invalid_transition_is_rejected_without_state_change() -> None:
    machine = _machine()

    with pytest.raises(StateTransitionError):
        machine.apply_transition(AgentAction.ASK_EMAIL)

    assert machine.current_state == ConversationState.NEW_CALL


def test_terminal_state_remains_terminal() -> None:
    machine = _machine()
    machine.apply_transition(AgentAction.MARK_DNC)

    with pytest.raises(StateTransitionError):
        machine.apply_transition(AgentAction.GREET)

    assert machine.current_state == ConversationState.END_CALL
    assert machine.is_terminal()


def test_brain_proposal_cannot_set_arbitrary_next_state() -> None:
    machine = _machine()
    proposal = BrainProposal(
        detected_intent=Intent.INTERESTED,
        proposed_action=AgentAction.GREET,
    )

    assert "next_state" not in {field.name for field in fields(proposal)}
    assert machine.current_state == ConversationState.NEW_CALL
