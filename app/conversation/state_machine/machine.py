"""The conversation state machine — authoritative control of conversation state.

CORE PRINCIPLE (guide): LLM proposes, State Machine decides. Ye machine state ki
FINAL AUTHORITY hai, lekin business/context validation ki jagah NAHI — woh
Sitting 2 ka `action_validator.py` karega, jo yahan ek pluggable hook se judega
bina is file ko rewrite kiye.

SCOPE (Sitting 1): structural authority only —
    - konse actions is state mein eligible hain (whitelist)
    - ek action structurally transition kar sakta hai ya nahi
    - terminal states se koi wapasi nahi
    - detour states (HANDLE_QUESTION) previous state restore karte hain
    - har transition apna explicit outcome carry karti hai

BOUNDARIES:
    - Per-session instance: koi global mutable state nahi. Har call apni machine.
    - Memory machine ke BAHAR: machine context consume karti hai, store nahi.
    - Koi Vapi/STT/TTS/LLM/DB/network dependency nahi.
    - LLM ka proposed action/state kabhi automatically authoritative nahi —
      caller propose karta hai, machine `apply_transition` se decide karti hai.
"""

from __future__ import annotations

from collections.abc import Callable

from app.conversation.state_machine.states import (
    PREVIOUS_STATE_SENTINEL,
    MachineConfig,
    TransitionRule,
)
from app.conversation.state_machine.transitions import ApprovedTransition
from app.core.constants import AgentAction, ConversationState
from app.core.exceptions import StateTransitionError

# Sitting 2 plug-in point: ek optional validator callable. Ye (rule, context) le
# kar True/False deta hai. Sitting 1 mein None — sirf structural eligibility.
# Sitting 2 mein action_validator.py isko inject karega bina machine rewrite ke.
TransitionValidator = Callable[[TransitionRule, "dict[str, object]"], bool]


class ConversationStateMachine:
    """Per-session authoritative conversation state controller.

    Ek instance ek call/session ke liye. State sirf `apply_transition` ke through
    badalti hai — koi direct mutation method nahi. Detour states ke liye machine
    ek chhota previous-state stack rakhti hai taake question ke baad wapas aaya
    ja sake.

    Attributes:
        config: Validated machine configuration.
        current_state: The authoritative current state.
    """

    def __init__(
        self,
        config: MachineConfig,
        initial_state: ConversationState | None = None,
    ) -> None:
        """Initialize a fresh session machine.

        Args:
            config: Loaded, validated MachineConfig.
            initial_state: Optional override for the starting state (e.g. a
                follow-up call may start elsewhere). Defaults to config's
                initial_state. Must be a defined, non-terminal state.

        Raises:
            StateTransitionError: Agar diya gaya initial_state invalid ho.
        """
        self._config = config
        start = initial_state or config.initial_state
        if start not in config.states:
            raise StateTransitionError(f"Unknown initial state: {start.value!r}")
        if start in config.terminal_states:
            raise StateTransitionError(
                f"Cannot start in terminal state: {start.value!r}"
            )
        self._current_state: ConversationState = start
        # Detour restore ke liye stack — HANDLE_QUESTION jaise states yahan se
        # previous meaningful state wapas laate hain.
        self._state_stack: list[ConversationState] = []

    @property
    def config(self) -> MachineConfig:
        """Return the machine configuration.

        Returns:
            MachineConfig: The validated config this machine runs on.
        """
        return self._config

    @property
    def current_state(self) -> ConversationState:
        """Return the authoritative current state.

        Returns:
            ConversationState: The current state.
        """
        return self._current_state

    def is_terminal(self) -> bool:
        """Whether the machine is in a terminal state.

        Returns:
            bool: True if the current state has no outgoing transitions.
        """
        return self._current_state in self._config.terminal_states

    def available_actions(self) -> frozenset[AgentAction]:
        """Return the actions structurally allowed in the current state.

        NOTE: ye sirf structural whitelist hai. Context-sensitive filtering
        (email candidate hai ya nahi) Sitting 2 ka validator karega.

        Returns:
            frozenset[AgentAction]: Allowed actions (empty in terminal states).
        """
        return self._config.states[self._current_state].allowed_actions

    def get_state_requirements(
        self, state: ConversationState | None = None
    ) -> tuple[str, ...]:
        """Return the context preconditions required to enter a state.

        Ye Sitting 2 ke validator ke liye hai — machine khud inhe enforce nahi
        karti (kyunki context machine ke bahar hai).

        Args:
            state: State to inspect; defaults to current state.

        Returns:
            tuple[str, ...]: Declared context requirement keys.
        """
        target = state or self._current_state
        return self._config.states[target].requires

    def is_transition_eligible(self, action: AgentAction) -> bool:
        """Whether an action can STRUCTURALLY transition from the current state.

        Ye pure structural check hai: action current state ke transitions mein
        defined hai ya nahi. Business/context validation (Sitting 2) alag hai.

        Args:
            action: The proposed action.

        Returns:
            bool: True if a transition rule exists for this action here.
        """
        state_def = self._config.states[self._current_state]
        return action in state_def.transitions

    def _resolve_target(self, rule: TransitionRule) -> ConversationState:
        """Resolve a transition rule's target into a concrete state.

        Detour restore (PREVIOUS_STATE_SENTINEL) ko stack se pop karke actual
        previous state mein badalta hai.

        Args:
            rule: The transition rule being applied.

        Returns:
            ConversationState: The concrete destination state.

        Raises:
            StateTransitionError: Agar restore ke liye koi previous state na ho.
        """
        if rule.target == PREVIOUS_STATE_SENTINEL:
            if not self._state_stack:
                raise StateTransitionError(
                    "Detour return requested but no previous state on stack."
                )
            return self._state_stack[-1]
        return ConversationState(rule.target)

    def apply_transition(
        self,
        action: AgentAction,
        context: dict[str, object] | None = None,
        validator: TransitionValidator | None = None,
    ) -> ApprovedTransition:
        """Validate and apply an action, mutating state only if authorized.

        Ye machine ka SIRF ek state-mutation entry point hai. Flow:
            1. Structural eligibility check (action is defined here).
            2. Optional Sitting-2 validator hook (context/business rules).
            3. Resolve target (detour restore included).
            4. Mutate current state, manage detour stack.
            5. Return an ApprovedTransition carrying the explicit outcome.

        LLM ka proposed action yahan sirf ek proposal hai — jab tak ye dono
        checks pass na kare, state nahi badalti.

        Args:
            action: The proposed (e.g. LLM-suggested) action.
            context: Conversation/person/lead context for the validator. Machine
                ise store nahi karti — sirf validator ko pass karti hai.
            validator: Optional Sitting-2 hook. None => structural check only.

        Returns:
            ApprovedTransition: The authorized transition result.

        Raises:
            StateTransitionError: Agar action structurally ineligible ho, ya
                validator use reject kare. State un-mutated rehti hai.
        """
        ctx = context or {}

        if self.is_terminal():
            raise StateTransitionError(
                f"Cannot transition from terminal state {self._current_state.value!r}."
            )

        state_def = self._config.states[self._current_state]
        rule = state_def.transitions.get(action)
        if rule is None:
            raise StateTransitionError(
                f"Action {action.value!r} is not eligible in state "
                f"{self._current_state.value!r}."
            )

        # Sitting 2 plug-in: context/business validation. Reject => no mutation.
        if validator is not None and not validator(rule, ctx):
            raise StateTransitionError(
                f"Action {action.value!r} rejected by validator in state "
                f"{self._current_state.value!r}."
            )

        from_state = self._current_state
        is_detour_return = rule.target == PREVIOUS_STATE_SENTINEL
        target = self._resolve_target(rule)

        # ── manage detour stack ──
        entering_detour = self._config.states.get(target)
        if is_detour_return:
            # wapas aa gaye — us previous state ko stack se hata do.
            self._state_stack.pop()
        elif entering_detour is not None and entering_detour.is_detour:
            # detour mein ja rahe hain — current (meaningful) state ko yaad rakho.
            self._state_stack.append(from_state)

        self._current_state = target

        return ApprovedTransition(
            action=action,
            from_state=from_state,
            to_state=target,
            outcome=rule.outcome,
            preserved_interest=rule.preserve_interest,
            was_detour_return=is_detour_return,
        )