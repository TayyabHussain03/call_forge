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

from app.contracts.conversation import ProposedConversationDecision
from app.contracts.conversation_context import ConversationContext
from app.contracts.validation import ValidationResult
from app.conversation.guardrails.action_validator import ActionValidator
from app.conversation.guardrails.fallbacks import FallbackEngine
from app.conversation.guardrails.priority import resolve_priority_action
from app.conversation.state_machine.machine import ConversationStateMachine
from app.core.constants import AgentAction, ConversationState
from app.core.exceptions import StateTransitionError, ValidationError

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
        validation: Us action ka ValidationResult jo ABHI consider/execute ho
            raha hai. Normal path mein: selected action ka result. Fallback path
            mein: FALLBACK action ka validation result. (Original rejection
            `original_validation` mein alag rehta hai — koi redundant storage
            nahi.)
        original_validation: Original proposal ka rejection result — SIRF tab
            present jab fallback path enter hua. Normal/priority success par None.
        fallback_used: Kya ye result fallback path se aaya.
        fallback_failed: True jab fallback SELECT hua lekin validate fail kar gaya
            (execute NAHI hua). fallback_used=True + fallback_failed=True =
            recovery-failure. fallback_used=True + fallback_failed=False =
            fallback valid aur executed.
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
    original_validation: ValidationResult | None = None
    fallback_used: bool = False
    fallback_failed: bool = False
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


class ConversationEngine:
    """Deterministic orchestrator coordinating the conversation components.

    STEP B scope: proposal → priority → validator → state machine → result.
    FallbackEngine aur ClarificationEngine abhi wire NAHI (Step C/D).

    Ye engine COORDINATOR hai — kisi component ka logic duplicate nahi karta.
    Per-session: har call ka apna engine + machine + context. Koi global mutable
    state nahi. Koi external dependency (LLM/Vapi/DB/network) nahi.

    Attributes:
        machine: The per-session state machine (state authority).
        validator: The action validator (validation authority).
    """

    def __init__(
        self,
        machine: ConversationStateMachine,
        validator: ActionValidator,
        fallback_engine: "FallbackEngine | None" = None,
    ) -> None:
        """Initialize with a per-session machine, validator, and fallback engine.

        Args:
            machine: This session's ConversationStateMachine.
            validator: The ActionValidator (config-bound, stateless).
            fallback_engine: The FallbackEngine for deterministic recovery. None
                => fallback path disabled (Step B behaviour: reject only).
        """
        self._machine = machine
        self._validator = validator
        self._fallback = fallback_engine

    @property
    def machine(self) -> ConversationStateMachine:
        """Return the underlying state machine (read access).

        Returns:
            ConversationStateMachine: The session's machine.
        """
        return self._machine

    def _terminal_result(
        self, context: ConversationContext
    ) -> ConversationResult:
        """Build the result for an already-terminal conversation.

        Terminal guard: koi transition nahi, koi action nahi, state unchanged.

        Args:
            context: The current session context (returned unchanged).

        Returns:
            ConversationResult: A consistent terminal result.
        """
        state = self._machine.current_state
        return ConversationResult(
            current_state=state,
            next_state=state,
            is_terminal=True,
            continues=False,
            execution_required=False,
            updated_context=context,
            path="terminal",
            approved_action=None,
            outcome=None,
            terminal_states=self._machine.config.terminal_states,
        )

    def _rejected_result(
        self,
        context: ConversationContext,
        action: AgentAction,
        validation: ValidationResult,
        path: str,
    ) -> ConversationResult:
        """Build the result when a selected action was rejected by the validator.

        State unchanged rehti hai. Rejection ko END_CALL mein convert NAHI karte
        (Step B mein fallback nahi). Conversation abhi bhi active hai.

        Args:
            context: Current session context (unchanged).
            action: The action that was rejected.
            validation: The rejection ValidationResult.
            path: Path label for debugging.

        Returns:
            ConversationResult: A consistent rejected (no-transition) result.
        """
        state = self._machine.current_state
        return ConversationResult(
            current_state=state,
            next_state=state,  # unchanged
            is_terminal=False,
            continues=True,
            execution_required=False,
            updated_context=context,
            path=path,
            approved_action=None,  # nothing approved
            outcome=None,
            validation=validation,
            fallback_used=False,
            clarification=None,
            terminal_states=self._machine.config.terminal_states,
        )

    def _apply_and_build(
        self,
        context: ConversationContext,
        action: AgentAction,
        validation: ValidationResult,
        path: str,
    ) -> ConversationResult:
        """Apply an approved action through the machine and build the result.

        Actual next state SIRF machine se aata hai — LLM ka proposed_next_state
        kabhi nahi. Context deterministic-updates ke saath naya copy banta hai
        (interest_preserved, dnc_pending), original mutate nahi hota.

        Args:
            context: Current session context (read-only).
            action: The validated action to apply.
            validation: The (allowed) validation result.
            path: Path label for debugging.

        Returns:
            ConversationResult: The result after the state transition.

        Raises:
            StateTransitionError: Agar machine action ko structurally reject kare
                (validator ke baad aisa nahi hona chahiye, lekin defense).
        """
        from_state = self._machine.current_state
        transition = self._machine.apply_transition(action)
        next_state = transition.to_state
        is_terminal = self._machine.is_terminal()

        # Deterministic context updates from the transition (Step B: minimal).
        updates: dict[str, object] = {}
        if transition.preserved_interest:
            updates["interest_preserved"] = True
        if action == AgentAction.MARK_DNC:
            updates["dnc_pending"] = True
        new_context = context.with_updates(**updates) if updates else context

        return ConversationResult(
            current_state=from_state,
            next_state=next_state,
            is_terminal=is_terminal,
            continues=not is_terminal,
            execution_required=True,
            updated_context=new_context,
            path=path,
            approved_action=action,
            outcome=transition.outcome,
            validation=validation,
            fallback_used=False,
            clarification=None,
            terminal_states=self._machine.config.terminal_states,
        )

    def _fallback_path(
        self,
        context: ConversationContext,
        original_validation: ValidationResult,
        validator_ctx: dict[str, object],
    ) -> ConversationResult:
        """Attempt deterministic recovery via FallbackEngine (exactly once).

        Flow (guide ke mutabiq):
            1. FallbackEngine.select() ONCE — safe recovery action.
            2. validator.validate(fallback_action) ONCE.
            3. valid → machine.apply_transition → executed result.
               invalid → recovery-failure result: state UNCHANGED, outcome None,
               koi business outcome invent NAHI. No recursion, no re-select.

        `original_validation` original rejection carry karta hai; `validation`
        fallback ka validation hota hai (kyunki ab fallback hi current action
        hai) — koi ValidationResult do jagah nahi.

        Args:
            context: Current session context (read-only).
            original_validation: The original action's rejection result.
            validator_ctx: Flat validator context (read-only dict).

        Returns:
            ConversationResult: Executed fallback, ya recovery-failure result.
        """
        assert self._fallback is not None  # caller guarantees this
        state = self._machine.current_state

        # 1. Select fallback ONCE.
        decision = self._fallback.select(state, original_validation, validator_ctx)
        fallback_action = decision.action

        # 2. Validate fallback ONCE.
        fb_validation = self._validator.validate(state, fallback_action, validator_ctx)

        if fb_validation.allowed:
            # 3a. Apply through machine — actual outcome ONLY from real transition.
            from_state = self._machine.current_state
            try:
                transition = self._machine.apply_transition(fallback_action)
            except StateTransitionError:
                # Machine rejected structurally despite validator — treat as
                # recovery failure (no fabricated outcome).
                return self._fallback_failed_result(
                    context, original_validation, fb_validation, decision.response_key
                )
            next_state = transition.to_state
            is_terminal = self._machine.is_terminal()

            updates: dict[str, object] = {}
            if transition.preserved_interest:
                updates["interest_preserved"] = True
            if fallback_action == AgentAction.MARK_DNC:
                updates["dnc_pending"] = True
            new_context = context.with_updates(**updates) if updates else context

            return ConversationResult(
                current_state=from_state,
                next_state=next_state,
                is_terminal=is_terminal,
                continues=not is_terminal,
                execution_required=True,
                updated_context=new_context,
                path="fallback",
                approved_action=fallback_action,
                outcome=transition.outcome,  # only from real ApprovedTransition
                response_key=decision.response_key,
                validation=fb_validation,           # current (fallback) validation
                original_validation=original_validation,  # original rejection
                fallback_used=True,
                fallback_failed=False,
                terminal_states=self._machine.config.terminal_states,
            )

        # 3b. Fallback invalid → recovery-failure. NO state mutation, NO invented
        #     outcome, NO recursion.
        return self._fallback_failed_result(
            context, original_validation, fb_validation, decision.response_key
        )

    def _fallback_failed_result(
        self,
        context: ConversationContext,
        original_validation: ValidationResult,
        fb_validation: ValidationResult,
        response_key: str | None,
    ) -> ConversationResult:
        """Build a recovery-failure result: state unchanged, no invented outcome.

        Guide + note ke mutabiq: fallback fail hone par END_CALL force NAHI karte,
        koi business outcome (not_interested/failed/qualified/dnc) invent NAHI
        karte. State jaisi thi waisi. is_terminal current state se derive hota hai
        (agar current state hi terminal tha to terminal, warna active).

        Args:
            context: Current session context (returned unchanged).
            original_validation: The original action's rejection.
            fb_validation: The fallback action's (failed) validation.
            response_key: The fallback's response key (still informative).

        Returns:
            ConversationResult: A consistent recovery-failure result.
        """
        state = self._machine.current_state
        is_terminal = self._machine.is_terminal()
        return ConversationResult(
            current_state=state,
            next_state=state,  # unchanged
            is_terminal=is_terminal,
            continues=not is_terminal,
            execution_required=False,
            updated_context=context,
            path="fallback",
            approved_action=None,          # nothing executed
            outcome=None,                  # NO invented business outcome
            response_key=response_key,
            validation=fb_validation,           # current (fallback) validation
            original_validation=original_validation,
            fallback_used=True,
            fallback_failed=True,
            terminal_states=self._machine.config.terminal_states,
        )

    def process_turn(
        self,
        context: ConversationContext,
        proposal: ProposedConversationDecision,
    ) -> ConversationResult:
        """Process one conversation turn deterministically (Step B path).

        Pipeline: terminal guard → priority resolution → action selection →
        validation → state transition → result. Fallback/clarification abhi nahi.

        LLM ka `proposed_next_state` IGNORE hota hai — actual next state sirf
        state machine se. Priority action (DNC/not-interested) conflicting
        proposal ko override karta hai.

        NOTE (duplicate-turn protection): future mein ek explicit turn/event id
        yahan add hoga taake same logical turn dobara process na ho. Abhi ye
        boundary documented hai, implement nahi — caller ek turn ek baar bheje.

        Args:
            context: Session context (read-only; engine ise mutate nahi karta).
            proposal: The (untrusted) LLM proposal. detected_intent aur
                proposed_action use hote hain; proposed_next_state ignore.

        Returns:
            ConversationResult: A structured, invariant-consistent result. Caller
            `updated_context` ko agle turn ke liye rakhe.
        """
        # 1. Terminal guard — already terminal to kuch process nahi.
        if self._machine.is_terminal():
            return self._terminal_result(context)

        current_state = self._machine.current_state
        validator_ctx = context.to_validator_context()

        # 2. Priority resolution — DNC/not-interested deterministic override.
        priority_action = resolve_priority_action(
            proposal.detected_intent.intent, current_state, validator_ctx
        )

        if priority_action is not None:
            selected_action = priority_action
            path = "priority"
        else:
            # 3. Normal flow — LLM proposed action (non-authoritative).
            selected_action = proposal.proposed_action
            path = "normal"

        # 4. Validation.
        validation = self._validator.validate(
            current_state, selected_action, validator_ctx
        )
        if not validation.allowed:
            # Fallback path (Step C). Agar fallback engine nahi diya, Step B
            # behaviour: structured rejection, koi recovery nahi.
            if self._fallback is None:
                return self._rejected_result(context, selected_action, validation, path)
            return self._fallback_path(context, validation, validator_ctx)

        # 5. State transition (machine is the state authority).
        try:
            return self._apply_and_build(context, selected_action, validation, path)
        except StateTransitionError:
            # Validator passed but machine rejected structurally — should not
            # happen; surface as rejection rather than silently ending the call.
            return self._rejected_result(context, selected_action, validation, path)