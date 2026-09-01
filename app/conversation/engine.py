"""Conversation engine / orchestrator.

STEP A scope: sirf `ConversationResult` contract (immutable, invariant-enforced).
Orchestration flow (process_turn) Step B+ mein aayega.

ConversationResult ek frozen, self-consistent result hai jo ek turn ka natija
batata hai. Iske invariants enforce hote hain construction par — contradictory
result (jaise is_terminal=True + continues=True) ban hi nahi sakta.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.contracts.conversation_context import ConversationContext
from app.contracts.validation import ValidationResult
from app.core.constants import AgentAction, ConversationState
from app.core.exceptions import ValidationError

if TYPE_CHECKING:
    from app.conversation.guardrails.clarification import ClarificationDecision


@dataclass(frozen=True)
class ConversationResult:
    """Immutable, self-consistent result of processing one conversation turn.

    Invariants (construction par enforce):
        - is_terminal True ho to continues False hona chahiye.
        - is_terminal False ho to next_state terminal nahi hona chahiye.
    Contradictory combinations ValidationError raise karte hain.

    Attributes:
        current_state: Turn se pehle ki authoritative state.
        approved_action: Jo action approve/apply hua, ya None (jaise pure terminal
            guard par).
        next_state: Turn ke baad ki authoritative state (machine se).
        outcome: Explicit terminal outcome label (do_not_call/qualified/...), ya
            None agar terminal nahi.
        is_terminal: Kya conversation ab terminal hai.
        continues: Kya conversation aage badhti hai (hamesha not is_terminal).
        execution_required: Kya caller ko koi action execute karna hai (response
            bolna/tool chalana).
        response_key: Deterministic response identifier (fallback/clarification
            se), ya None.
        validation: Us action ka ValidationResult agar relevant, ya None.
        fallback_used: Kya ye result fallback path se aaya.
        clarification: ClarificationDecision agar clarification chali, ya None.
        updated_context: Naya context jo caller agle turn ke liye rakhe.
        path: Debug label — kaunsa path liya ("terminal"/"priority"/"clarification"/
            "normal"/"fallback").
    """

    current_state: ConversationState
    next_state: ConversationState
    is_terminal: bool
    continues: bool
    execution_required: bool
    updated_context: ConversationContext
    path: str
    approved_action: AgentAction | None = None
    outcome: str | None = None
    response_key: str | None = None
    validation: ValidationResult | None = None
    fallback_used: bool = False
    clarification: "ClarificationDecision | None" = None
    terminal_states: frozenset[ConversationState] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Enforce result invariants; reject contradictory combinations.

        Raises:
            ValidationError: Agar is_terminal aur continues dono True hon, ya
                terminal/continues next_state ke saath inconsistent ho.
        """
        if self.is_terminal and self.continues:
            raise ValidationError(
                "ConversationResult invalid: is_terminal=True but continues=True."
            )
        if not self.is_terminal and not self.continues:
            raise ValidationError(
                "ConversationResult invalid: is_terminal=False but continues=False."
            )
        # Agar terminal_states diya gaya, to next_state ke terminal-ness ke saath
        # is_terminal consistent hona chahiye.
        if self.terminal_states:
            next_is_terminal = self.next_state in self.terminal_states
            if next_is_terminal != self.is_terminal:
                raise ValidationError(
                    f"ConversationResult invalid: next_state={self.next_state.value!r} "
                    f"terminal-ness ({next_is_terminal}) != is_terminal "
                    f"({self.is_terminal})."
                )