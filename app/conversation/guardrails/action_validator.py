"""Action validator — answers "is this proposed action valid in this context?".

SCOPE (Sitting 2A): sirf do cheezein —
    1. Structural eligibility: action current state mein defined transition hai?
    2. Contextual prerequisites: us transition ka `requires` list context ke
       against poora hota hai?

Ye validator STATE MUTATE NAHI karta, transition NAHI karta, lead/contact/memory
data NAHI badalta, LLM/DB/Vapi ko NAHI jaanta. Sirf ek boolean-ish jawab
(ValidationResult) deta hai.

CONTEXT IS READ-ONLY: validator `context` dict ko sirf padhta hai, kabhi mutate
nahi karta.

NOT IN THIS STEP: DNC override, NOT_INTERESTED handling, fallbacks, clarification
retry, priority-action resolution. Ye Sitting 2B mein aayenge — is validator ke
upar, ise rewrite kiye bina.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from app.contracts.validation import ValidationCategory, ValidationResult
from app.conversation.state_machine.states import MachineConfig, TransitionRule
from app.core.constants import AgentAction, ConversationState

# Ek prerequisite predicate: read-only context leta hai, bool deta hai.
PrerequisitePredicate = Callable[[Mapping[str, Any]], bool]


def _has_nonempty(context: Mapping[str, Any], key: str) -> bool:
    """Return True if context[key] exists and is not None/empty-string.

    Semantically explicit: missing/None → False, empty string → False, warna
    truthy value → True. Ye read-only check hai.

    Args:
        context: Read-only conversation context.
        key: The context key to test.

    Returns:
        bool: True only when a meaningful (non-empty) value is present.
    """
    value = context.get(key)
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    return bool(value)


def _is_true(context: Mapping[str, Any], key: str) -> bool:
    """Return True only if context[key] is explicitly truthy.

    Args:
        context: Read-only conversation context.
        key: The context key to test.

    Returns:
        bool: bool(context.get(key)).
    """
    return bool(context.get(key))


# Known requirement registry. Har config `requires` key ka yahan ek predicate
# hona chahiye. Agar config mein koi key aaye jo yahan nahi → UNKNOWN_REQUIREMENT
# (config/design error, silently pass/fail nahi).
_PREREQUISITES: dict[str, PrerequisitePredicate] = {
    "email_candidate_exists": lambda ctx: _has_nonempty(ctx, "email_candidate"),
    "email_confirmed": lambda ctx: _is_true(ctx, "email_confirmed"),
    "callback_context_exists": lambda ctx: _has_nonempty(ctx, "callback"),
    "previous_conversation_exists": lambda ctx: _is_true(
        ctx, "previous_conversation_exists"
    ),
}


class ActionValidator:
    """Validates proposed actions against state + context prerequisites.

    Ye validator config ka `requires` declarations consume karta hai aur unhe
    read-only context ke against evaluate karta hai. Iska koi side effect nahi —
    na state, na data, na network.

    Attributes:
        config: The validated machine configuration (transitions + requires).
    """

    def __init__(self, config: MachineConfig) -> None:
        """Initialize with the machine configuration.

        Args:
            config: Loaded, validated MachineConfig.
        """
        self._config = config

    def _find_rule(
        self, state: ConversationState, action: AgentAction
    ) -> TransitionRule | None:
        """Return the transition rule for an action in a state, if any.

        Args:
            state: The current conversational state.
            action: The proposed action.

        Returns:
            TransitionRule | None: The rule, or None if action not eligible.
        """
        state_def = self._config.states.get(state)
        if state_def is None:
            return None
        return state_def.transitions.get(action)

    def validate(
        self,
        state: ConversationState,
        action: AgentAction,
        context: Mapping[str, Any] | None = None,
    ) -> ValidationResult:
        """Validate a proposed action against the state and context.

        Checks, is order mein:
            1. Structural eligibility — action is state mein transition rakhta hai?
            2. Contextual prerequisites — rule ke `requires` sab poore hain?
               (Unknown requirement key → fail clearly as UNKNOWN_REQUIREMENT.)

        Context read-only hai — ye method use mutate nahi karta.

        Args:
            state: The authoritative current state.
            action: The proposed (e.g. LLM-suggested) action.
            context: Read-only conversation context. None => empty context.

        Returns:
            ValidationResult: allowed=True (OK) or a rejection with category.
        """
        ctx: Mapping[str, Any] = context or {}

        # 0. DNC defensive safety-net. Agar context bataye ke client ne DNC maanga
        #    hai (intent do_not_call), to koi bhi non-DNC/non-terminating action
        #    defensively reject — chahe woh structurally eligible ho. Ye primary
        #    decision NAHI hai (woh resolve_priority_action karta hai); ye us
        #    decision ke bypass hone ki soorat mein ek safety net hai.
        if ctx.get("dnc_pending") and action not in (
            AgentAction.MARK_DNC,
            AgentAction.END_CALL,
        ):
            return ValidationResult.rejected(
                ValidationCategory.DNC_CONFLICT,
                reason=f"{action.value} rejected: do-not-call is pending",
            )

        state_def = self._config.states.get(state)

        # Terminating/escape actions (MARK_DNC, END_CALL) state ke entry-
        # preconditions se exempt hain — ye state se BAHAR nikalne ke liye hain,
        # wahan rehne/aage badhne ke liye nahi. DNC har active state se kaam kare
        # chahe us state ki business-preconditions poori hon ya nahi.
        _terminating = (AgentAction.MARK_DNC, AgentAction.END_CALL)

        # 1. State-level entry preconditions. Agar current state ka apna `requires`
        #    poora nahi, to yahan hona hi invalid tha — koi bhi (non-escape)
        #    action reject.
        if state_def is not None and action not in _terminating:
            for requirement in state_def.requires:
                predicate = _PREREQUISITES.get(requirement)
                if predicate is None:
                    return ValidationResult.rejected(
                        ValidationCategory.UNKNOWN_REQUIREMENT,
                        reason=f"unknown requirement: {requirement}",
                    )
                if not predicate(ctx):
                    return ValidationResult.rejected(
                        ValidationCategory.MISSING_PREREQUISITE,
                        reason=f"missing prerequisite: {requirement}",
                    )

        # 2. Structural eligibility — action is state mein defined transition hai?
        rule = self._find_rule(state, action)
        if rule is None:
            return ValidationResult.rejected(
                ValidationCategory.NOT_ELIGIBLE,
                reason=f"{action.value} not allowed in {state.value}",
            )

        # 3. Transition-level prerequisites.
        for requirement in rule.requires:
            predicate = _PREREQUISITES.get(requirement)
            if predicate is None:
                # Config/design error — silently missing/allowed nahi treat karna.
                return ValidationResult.rejected(
                    ValidationCategory.UNKNOWN_REQUIREMENT,
                    reason=f"unknown requirement: {requirement}",
                )
            if not predicate(ctx):
                return ValidationResult.rejected(
                    ValidationCategory.MISSING_PREREQUISITE,
                    reason=f"missing prerequisite: {requirement}",
                )

        return ValidationResult.ok()

    def as_machine_hook(
        self, context: Mapping[str, Any] | None = None
    ) -> Callable[[TransitionRule, "dict[str, Any]"], bool]:
        """LEGACY structural adapter — NOT the canonical validation path.

        Canonical live flow validation ke liye `validate(state, action, context)`
        use karo, jo state-level + transition-level preconditions + DNC safety-net
        sab check karta hai. Ye adapter machine ke purane `(rule, ctx) -> bool`
        hook signature ke saath compatible hai, lekin ismein current STATE
        available nahi — isliye ye SIRF transition-level `requires` enforce karta
        hai, state-level entry-preconditions ya DNC safety-net NAHI.

        Ise sirf tab use karo jab koi legacy caller machine hook chahta ho. Naya
        code ko external-validation flow follow karna chahiye:
        LLM → resolve_priority_action → validate() → machine.apply_transition().

        Args:
            context: Optional read-only context. Machine apna context pass kare to
                woh preferred; warna ye default. Function ise mutate nahi karta.

        Returns:
            Callable: A `(rule, ctx) -> bool` hook (transition-level only).
        """

        def _hook(rule: TransitionRule, ctx: dict[str, Any]) -> bool:
            effective_ctx: Mapping[str, Any] = ctx or context or {}
            for requirement in rule.requires:
                predicate = _PREREQUISITES.get(requirement)
                if predicate is None:
                    return False  # unknown requirement → reject (fail closed)
                if not predicate(effective_ctx):
                    return False
            return True

        return _hook