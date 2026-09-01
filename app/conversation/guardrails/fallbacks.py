"""Deterministic fallback engine.

Jab `ActionValidator` ek proposed action reject kare, ye engine ek SAFE recovery
action deta hai — LLM ko dobara call kiye BINA (no regeneration). Selection
config-driven aur side-effect free hai.

RESPONSIBILITY BOUNDARY (guide ke mutabiq):
    - FallbackEngine ye DECIDE nahi karta ke action valid hai ya nahi — woh
      ActionValidator ka kaam hai. Engine sirf validation RESULT consume karke
      safe recovery chunta hai.
    - Selection deterministic + side-effect free: na state, na context, na
      LLM/DB/network.

ANTI-RECURSION (guide safeguard): chuna hua fallback action bhi blindly valid
    nahi maana jaata — caller use ek baar validate karega. Agar woh bhi invalid
    ho, engine ka `terminal_safe()` deta hai — koi recursive fallback selection
    ya doosra LLM call NAHI.

RESPONSE KEYS: engine natural-language text return nahi karta — sirf stable
    machine-readable `response_key` identifiers. Actual wording baad mein LLM/
    prompt layer bharega.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app.contracts.validation import ValidationCategory, ValidationResult
from app.conversation.state_machine.states import MachineConfig
from app.core.constants import AgentAction, ConversationState

# Terminal-safe response key jab koi recovery possible/safe na ho.
_TERMINAL_SAFE_KEY = "safe_close"


@dataclass(frozen=True)
class FallbackDecision:
    """A deterministic safe-recovery decision.

    Attributes:
        action: Safe action to take next.
        response_key: Stable machine-readable identifier (NOT spoken text).
        is_terminal: True agar ye recovery conversation ko end karta hai.
        reason: Short machine-readable explanation of why this fallback (debug).
    """

    action: AgentAction
    response_key: str
    is_terminal: bool
    reason: str


class FallbackEngine:
    """Selects deterministic safe recovery actions from validation failures.

    Attributes:
        config: The validated machine configuration (holds fallback mappings).
    """

    def __init__(self, config: MachineConfig) -> None:
        """Initialize with machine configuration.

        Args:
            config: Loaded, validated MachineConfig (fallbacks already verified
                at startup).

        Raises:
            ValueError: Agar config mein fallbacks define hi nahi (startup bug).
        """
        if config.fallbacks is None:
            raise ValueError("MachineConfig has no fallbacks configured.")
        self._config = config
        self._fb = config.fallbacks

    def terminal_safe(self, reason: str = "terminal_safe") -> FallbackDecision:
        """Return the always-safe terminal fallback (ends the conversation).

        Ye ultimate safety net hai: jab koi recovery safe/possible na ho, ya jab
        chuna hua fallback khud invalid nikle. Koi recursion nahi — bas end.

        Args:
            reason: Machine-readable reason for the terminal fallback.

        Returns:
            FallbackDecision: A terminal end_call decision.
        """
        return FallbackDecision(
            action=AgentAction.END_CALL,
            response_key=_TERMINAL_SAFE_KEY,
            is_terminal=True,
            reason=reason,
        )

    def select(
        self,
        state: ConversationState,
        result: ValidationResult,
        context: Mapping[str, Any] | None = None,
    ) -> FallbackDecision:
        """Select a deterministic safe recovery for a rejected action.

        Precedence:
            1. Terminal state → koi conversational recovery nahi; terminal-safe.
            2. Category-specific fallback (dnc_conflict, unknown_requirement) —
               state-specific par override.
            3. State-specific fallback.
            4. Default safe-close.

        Selection deterministic aur side-effect free hai — na state, na context
        mutate hota hai. Context abhi read bhi nahi hota (future-use param).

        Args:
            state: The authoritative current state.
            result: The ValidationResult that caused this fallback.
            context: Read-only context (future use). Not mutated.

        Returns:
            FallbackDecision: The chosen deterministic recovery.
        """
        # 1. Terminal state: never produce a non-terminal recovery action.
        if state in self._config.terminal_states:
            return self.terminal_safe(reason="already_terminal")

        # 2. Category-specific (override). DNC conflict + unknown requirement.
        category_key = result.category.value
        if category_key in self._fb.by_category:
            mapping = self._fb.by_category[category_key]
            is_terminal = (
                mapping.action == AgentAction.END_CALL
                or mapping.action == AgentAction.MARK_DNC
            )
            # unknown_requirement must never invent recovery — it maps to a safe
            # terminal close by config; assert that intent via is_terminal.
            if result.category == ValidationCategory.UNKNOWN_REQUIREMENT:
                is_terminal = True
            return FallbackDecision(
                action=mapping.action,
                response_key=mapping.response_key,
                is_terminal=is_terminal,
                reason=f"category:{category_key}",
            )

        # 3. State-specific fallback.
        if state in self._fb.by_state:
            mapping = self._fb.by_state[state]
            return FallbackDecision(
                action=mapping.action,
                response_key=mapping.response_key,
                is_terminal=mapping.action == AgentAction.END_CALL,
                reason=f"state:{state.value}",
            )

        # 4. Default safe-close.
        return FallbackDecision(
            action=self._fb.default.action,
            response_key=self._fb.default.response_key,
            is_terminal=self._fb.default.action == AgentAction.END_CALL,
            reason="default",
        )