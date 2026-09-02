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

from app.contracts.contact_info import ContactChannel, ExtractedContact
from app.contracts.contact_understanding import (
    ContactIntent,
    ContactUnderstanding,
    ResolutionContext,
    ResolutionOutcome,
)
from app.contracts.conversation import ProposedConversationDecision
from app.contracts.conversation_context import ConversationContext
from app.contracts.pending_contact_method import PendingContactMethod
from app.contracts.validation import ValidationResult
from app.conversation.contact.resolver import ContactResolver
from app.conversation.guardrails.action_validator import ActionValidator
from app.conversation.guardrails.clarification import (
    ClarificationEngine,
    ClarificationOutcome,
    ClarificationState,
)
from app.conversation.guardrails.fallbacks import FallbackDecision, FallbackEngine
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
        contact_to_persist: PendingContactMethod jab successful confirmation ne is
            method ko persist-eligible banaya, warna None. Engine sirf INTENT deta
            hai — actual DB service layer karta hai.
        updated_context: Naya context jo caller agle turn ke liye rakhe.
        path: Debug label.
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
    contact_to_persist: "PendingContactMethod | None" = None
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
        clarification_engine: "ClarificationEngine | None" = None,
        contact_resolver: "ContactResolver | None" = None,
    ) -> None:
        """Initialize with per-session machine, validator, fallback, clarification.

        Args:
            machine: This session's ConversationStateMachine.
            validator: The ActionValidator (config-bound, stateless).
            fallback_engine: FallbackEngine for recovery. None => Step B behaviour.
            clarification_engine: ClarificationEngine for contact turns. None =>
                clarification disabled (Step B/C behaviour).
            contact_resolver: ContactResolver for understanding→candidate. None =>
                contact-understanding disabled.
        """
        self._machine = machine
        self._validator = validator
        self._fallback = fallback_engine
        self._clarification = clarification_engine
        self._resolver = contact_resolver

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
        original_validation: ValidationResult | None,
        validator_ctx: dict[str, object],
        decision: "FallbackDecision | None" = None,
        clarification: "ClarificationDecision | None" = None,
    ) -> ConversationResult:
        """Attempt deterministic recovery via FallbackEngine (exactly once).

        Do entry modes:
            - Rejection recovery: `original_validation` diya, `decision` None →
              FallbackEngine.select() se decision banega (category-aware).
            - Exhaustion recovery: `decision` pehle se diya (select_for_state se),
              `original_validation` None → koi fake validation failure nahi.

        Dono cases mein: fallback action ONCE validate hota hai; valid → machine,
        invalid → recovery-failure (state unchanged, outcome None, no recursion).

        Args:
            context: Current session context (read-only).
            original_validation: Original rejection (rejection-recovery mein), ya
                None (exhaustion-recovery mein).
            validator_ctx: Flat validator context (read-only dict).
            decision: Pre-selected FallbackDecision (exhaustion), ya None (select
                from validation).
            clarification: ClarificationDecision to attach to the result, if any.

        Returns:
            ConversationResult: Executed fallback, ya recovery-failure result.
        """
        assert self._fallback is not None
        state = self._machine.current_state

        # 1. Select fallback ONCE (unless caller pre-selected for exhaustion).
        if decision is None:
            assert original_validation is not None
            decision = self._fallback.select(state, original_validation, validator_ctx)
        fallback_action = decision.action

        # 2. Validate fallback ONCE.
        fb_validation = self._validator.validate(state, fallback_action, validator_ctx)

        if fb_validation.allowed:
            from_state = self._machine.current_state
            try:
                transition = self._machine.apply_transition(fallback_action)
            except StateTransitionError:
                return self._fallback_failed_result(
                    context, original_validation, fb_validation,
                    decision.response_key, clarification,
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
                outcome=transition.outcome,
                response_key=decision.response_key,
                validation=fb_validation,
                original_validation=original_validation,
                fallback_used=True,
                fallback_failed=False,
                clarification=clarification,
                terminal_states=self._machine.config.terminal_states,
            )

        return self._fallback_failed_result(
            context, original_validation, fb_validation,
            decision.response_key, clarification,
        )

    def _fallback_failed_result(
        self,
        context: ConversationContext,
        original_validation: ValidationResult | None,
        fb_validation: ValidationResult,
        response_key: str | None,
        clarification: "ClarificationDecision | None" = None,
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
            clarification=clarification,
            terminal_states=self._machine.config.terminal_states,
        )

    # Contact-collection/confirmation states jahan clarification relevant hai.
    _CONTACT_STATES: frozenset[ConversationState] = frozenset(
        {ConversationState.COLLECT_EMAIL, ConversationState.CONFIRM_CONTACT}
    )

    def process_turn(
        self,
        context: ConversationContext,
        proposal: ProposedConversationDecision,
        contact_understanding: ContactUnderstanding | None = None,
    ) -> ConversationResult:
        """Process one conversation turn deterministically.

        Precedence: terminal → priority (DNC/not-interested) → contact
        understanding+resolution (contact turns) → normal action.

        LLM ka `proposed_next_state` IGNORE hota hai. Priority contact-resolution
        se pehle — DNC/not-interested resolver ko bypass karte hain.

        NOTE (duplicate-turn protection): future turn/event id yahan aayega.

        Args:
            context: Session context (read-only; engine mutate nahi karta).
            proposal: Untrusted LLM proposal (intent + action; next_state ignore).
            contact_understanding: Optional structured contact interpretation.
                Sirf contact-states mein resolver+clarification trigger karta hai.

        Returns:
            ConversationResult: Structured, invariant-consistent result. Caller
            `updated_context` ko agle turn ke liye rakhe.
        """
        # 1. Terminal guard.
        if self._machine.is_terminal():
            return self._terminal_result(context)

        current_state = self._machine.current_state
        validator_ctx = context.to_validator_context()

        # 2. Priority resolution — resolution se PEHLE (DNC/not-interested win).
        priority_action = resolve_priority_action(
            proposal.detected_intent.intent, current_state, validator_ctx
        )
        if priority_action is not None:
            return self._validate_and_apply(
                context, priority_action, validator_ctx, "priority"
            )

        # 3. Contact understanding + resolution — sirf jab relevant.
        if self._should_handle_contact(current_state, contact_understanding):
            return self._contact_path(
                context, contact_understanding, validator_ctx  # type: ignore[arg-type]
            )

        # 4. Normal flow — LLM proposed action (non-authoritative).
        return self._validate_and_apply(
            context, proposal.proposed_action, validator_ctx, "normal"
        )

    def _should_handle_contact(
        self, state: ConversationState, understanding: ContactUnderstanding | None
    ) -> bool:
        """Whether contact resolution/clarification should run this turn.

        Sirf tab True jab resolver + clarification available hon, understanding
        diya ho, aur current state contact-collection/confirmation ho.

        Args:
            state: Current conversational state.
            understanding: The contact understanding, if any.

        Returns:
            bool: True if contact handling is relevant this turn.
        """
        return (
            self._resolver is not None
            and self._clarification is not None
            and understanding is not None
            and state in self._CONTACT_STATES
        )

    def _validate_and_apply(
        self,
        context: ConversationContext,
        action: AgentAction,
        validator_ctx: dict[str, object],
        path: str,
    ) -> ConversationResult:
        """Validate an action and either apply it or enter the fallback path.

        Args:
            context: Current session context.
            action: The selected action to validate.
            validator_ctx: Flat validator context.
            path: Path label ("priority"/"normal").

        Returns:
            ConversationResult: Applied result, rejection, or fallback recovery.
        """
        state = self._machine.current_state
        validation = self._validator.validate(state, action, validator_ctx)
        if not validation.allowed:
            if self._fallback is None:
                return self._rejected_result(context, action, validation, path)
            return self._fallback_path(context, validation, validator_ctx)
        try:
            return self._apply_and_build(context, action, validation, path)
        except StateTransitionError:
            # Validator passed but machine rejected structurally — should not
            # happen; surface as rejection rather than silently ending the call.
            return self._rejected_result(context, action, validation, path)

    def _contact_path(
        self,
        context: ConversationContext,
        understanding: ContactUnderstanding,
        validator_ctx: dict[str, object],
    ) -> ConversationResult:
        """Resolve a contact understanding, then route to clarification/fallback.

        Flow (E1 — NO persistence):
            resolver.resolve(understanding, trusted ResolutionContext)
            → RESOLVED           → candidate context mein → clarification CLEAR
                                    → CONFIRM_CONTACT (candidate ready, NOT yet
                                      a confirmed ContactMethod — E2 persists).
            → NEEDS_CLARIFICATION → clarification RETRY (bounded)
            → INVALID            → clarification RETRY (bounded)
            → UNAVAILABLE_REFERENCE → clarification RETRY (bounded)
            → UNSUPPORTED_REFERENCE → FallbackEngine (deterministic).

        Resolver ko trusted values context ke fields se milte hain (call_number,
        business_phone, business_email) — jo application populate karti hai, LLM
        nahi. Koi ContactMethodRepository yahan NAHI.

        Args:
            context: Current session context (read-only).
            understanding: The (untrusted) contact interpretation.
            validator_ctx: Flat validator context.

        Returns:
            ConversationResult: Routed result.
        """
        assert self._resolver is not None
        assert self._clarification is not None
        state = self._machine.current_state

        # CONFIRM_CONTACT state + confirm/correct intent = client confirmation.
        # Ye woh point hai jahan candidate persist-eligible ban sakta hai.
        if state == ConversationState.CONFIRM_CONTACT and understanding.intent in (
            ContactIntent.CONFIRM_CONTACT,
        ):
            return self._handle_confirmation(context, validator_ctx)

        # CORRECT_CONTACT ko normal PROVIDE_CONTACT ki tarah resolve karo — naya
        # candidate banega (purana method persistence layer mein untouched rehta).
        # Resolution level par correction aur provide same hain.

        # Trusted resolution context — sirf application-populated fields se.
        res_ctx = ResolutionContext(
            call_number=context.call_number,
            business_phone=context.business_phone,
            business_email=context.business_email,
            region=None,
        )
        result = self._resolver.resolve(understanding, res_ctx)

        # UNSUPPORTED_REFERENCE → deterministic fallback (retry se theek nahi hoga).
        if result.outcome == ResolutionOutcome.UNSUPPORTED_REFERENCE:
            if self._fallback is None:
                return self._rejected_result(
                    context, AgentAction.CLARIFY_CONTACT,
                    self._validator.validate(state, AgentAction.CLARIFY_CONTACT, validator_ctx),
                    "contact",
                )
            fb_decision = self._fallback.select_for_state(state)
            return self._fallback_path(
                context, None, validator_ctx, decision=fb_decision
            )

        # RESOLVED → candidate ready (clear). NEEDS/INVALID/UNAVAILABLE → unclear.
        is_clear = result.outcome == ResolutionOutcome.RESOLVED

        # Candidate ko context mein daalo (RESOLVED par) — DB mein NAHI (E1).
        # value + channel + provenance SYNCHRONIZED.
        new_context = context
        if is_clear and result.value_normalized:
            new_context = context.with_updates(
                contact_candidate=result.value_normalized,
                contact_candidate_channel=(
                    result.channel.value if result.channel else None
                ),
                contact_candidate_provenance=(
                    result.provenance.value if result.provenance else None
                ),
            )
            validator_ctx = {
                **validator_ctx,
                "contact_candidate": result.value_normalized,
            }

        return self._run_clarification(
            new_context, understanding.channel, is_clear, validator_ctx, result
        )

    def _handle_confirmation(
        self, context: ConversationContext, validator_ctx: dict[str, object]
    ) -> ConversationResult:
        """Handle explicit client confirmation of a pending candidate.

        Client ne `CONFIRM_CONTACT` state mein confirm kiya. Agar candidate +
        channel + provenance + current_contact_id sab maujood hon, to:
            - contact_confirmed=True set → validator pass → end_call(qualified)
            - persist-INTENT (PendingContactMethod) result mein — DB engine nahi.
        Agar koi zaroori data missing ho → confirmation valid nahi, candidate
        discard (no persist), safe route.

        Args:
            context: Current session context.
            validator_ctx: Flat validator context.

        Returns:
            ConversationResult: Confirmed+persist-intent result, ya safe rejection.
        """
        state = self._machine.current_state

        # Required data for a real confirmation → persist intent.
        has_all = (
            context.contact_candidate
            and context.contact_candidate.strip()
            and context.contact_candidate_channel
            and context.contact_candidate_provenance
            and context.current_contact_id
        )
        if not has_all:
            # Confirmation without complete candidate data → cannot persist.
            # Do NOT fabricate; route back as rejection (no state mutation).
            validation = self._validator.validate(
                state, AgentAction.CONFIRM_CONTACT, validator_ctx
            )
            return self._rejected_result(
                context, AgentAction.CONFIRM_CONTACT, validation, "contact"
            )

        # Set contact_confirmed so the qualified transition's precondition passes.
        confirmed_ctx = context.with_updates(contact_confirmed=True)
        confirm_validator_ctx = {**validator_ctx, "contact_confirmed": True}

        validation = self._validator.validate(
            state, AgentAction.CONFIRM_CONTACT, confirm_validator_ctx
        )
        if not validation.allowed:
            return self._rejected_result(
                context, AgentAction.CONFIRM_CONTACT, validation, "contact"
            )

        try:
            transition = self._machine.apply_transition(AgentAction.CONFIRM_CONTACT)
        except StateTransitionError:
            return self._rejected_result(
                context, AgentAction.CONFIRM_CONTACT, validation, "contact"
            )

        # Persist INTENT (not DB). value+channel+provenance+contact_id from context.
        pending = PendingContactMethod(
            contact_id=context.current_contact_id,  # type: ignore[arg-type]
            channel=context.contact_candidate_channel,  # type: ignore[arg-type]
            value_normalized=context.contact_candidate,  # type: ignore[arg-type]
            provenance=context.contact_candidate_provenance,  # type: ignore[arg-type]
        )

        next_state = transition.to_state
        is_terminal = self._machine.is_terminal()
        return ConversationResult(
            current_state=state,
            next_state=next_state,
            is_terminal=is_terminal,
            continues=not is_terminal,
            execution_required=True,
            updated_context=confirmed_ctx,
            path="contact",
            approved_action=AgentAction.CONFIRM_CONTACT,
            outcome=transition.outcome,
            validation=validation,
            fallback_used=False,
            contact_to_persist=pending,
            terminal_states=self._machine.config.terminal_states,
        )

    def _run_clarification(
        self,
        context: ConversationContext,
        channel: ContactChannel,
        is_clear: bool,
        validator_ctx: dict[str, object],
        resolution: object,
    ) -> ConversationResult:
        """Drive the ClarificationEngine with a clear/unclear signal.

        CLEAR → CONFIRM_CONTACT action; RETRY → CLARIFY_CONTACT; EXHAUSTED →
        existing fallback path. Candidate persistence NAHI hoti (E1).

        Args:
            context: Current session context.
            channel: The contact channel.
            is_clear: Whether the resolution produced a usable candidate.
            validator_ctx: Flat validator context (candidate included if clear).
            resolution: The ResolutionResult (attached to result for traceability).

        Returns:
            ConversationResult: CLEAR/RETRY applied, ya EXHAUSTED fallback.
        """
        assert self._clarification is not None
        state = self._machine.current_state

        clar_state = context.clarification
        if clar_state is None or clar_state.channel != channel:
            clar_state = self._clarification.new_session(channel)

        decision = self._clarification.evaluate(clar_state, extraction_clear=is_clear)
        new_context = context.with_updates(clarification=decision.new_state)

        if decision.outcome == ClarificationOutcome.EXHAUSTED:
            fb_decision = (
                self._fallback.select_for_state(state) if self._fallback else None
            )
            if fb_decision is None:
                return self._rejected_result(
                    new_context, AgentAction.CLARIFY_CONTACT,
                    self._validator.validate(state, AgentAction.CLARIFY_CONTACT, validator_ctx),
                    "contact",
                )
            return self._fallback_path(
                new_context, None, validator_ctx,
                decision=fb_decision, clarification=decision,
            )

        action = decision.action
        assert action is not None
        validation = self._validator.validate(state, action, validator_ctx)
        if not validation.allowed:
            if self._fallback is None:
                return self._rejected_result(new_context, action, validation, "contact")
            return self._fallback_path(
                new_context, validation, validator_ctx, clarification=decision
            )

        try:
            transition = self._machine.apply_transition(action)
        except StateTransitionError:
            return self._rejected_result(new_context, action, validation, "contact")

        next_state = transition.to_state
        is_terminal = self._machine.is_terminal()
        return ConversationResult(
            current_state=state,
            next_state=next_state,
            is_terminal=is_terminal,
            continues=not is_terminal,
            execution_required=True,
            updated_context=new_context,
            path="contact",
            approved_action=action,
            outcome=transition.outcome,
            response_key=decision.response_key,
            validation=validation,
            fallback_used=False,
            clarification=decision,
            terminal_states=self._machine.config.terminal_states,
        )