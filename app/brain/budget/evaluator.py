"""Deterministic budget / early-exit evaluator.

RESOURCE GUARD, not conversation intelligence. Answer deta hai: "koi trusted exit
signal hai? budget bacha hai? response constrain karna hai?" — aur kuch nahi.

LOCKED BOUNDARIES:
    - Trusted early-exit signals ONLY (DNC / NOT_INTERESTED / confirmed callback /
      confirmed contact) — trusted state se, kabhi raw words/Brain judgment se.
    - "Goal achieved" yahan NAHI — woh Brain proposal ke roop mein budget ke BAAD
      normal path (scope→authority→validator→state-machine) se aata hai.
    - response_length exhaustion → CONSTRAIN_RESPONSE (call jaari), turn/time/
      reasoning exhaustion → WIND_DOWN. Response limit ≠ call termination.
    - WIND_DOWN sirf WIND_DOWN — close-vs-callback future orchestrator decide karega.
    - FAIL-CLOSED = conservative: missing budget_state/policy → WIND_DOWN, kabhi
      unbounded PROCEED nahi.

INTERNAL PRECEDENCE (only within this layer): trusted early-exit → budget
    exhaustion → response constraint → proceed. (Brain-derived exit is NOT an
    internal check — it happens after this layer in the master flow.)

Evaluator PURE: koi mutation/LLM/network/DB/raw-text/keyword/execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.brain.budget.models import BudgetPolicy
from app.brain.contracts import BudgetState


class BudgetOutcome(str, Enum):
    """The outcome of a budget / early-exit evaluation."""

    PROCEED = "proceed"                        # budget healthy, no trigger
    CONSTRAIN_RESPONSE = "constrain_response"  # response_length low — cap THIS turn
    WIND_DOWN = "wind_down"                    # per-call ceiling hit — controlled close/callback
    EARLY_EXIT = "early_exit"                  # trusted trigger fired — terminate


class ExitKind(str, Enum):
    """Terminal cause category for EARLY_EXIT / WIND_DOWN traceability."""

    DNC = "dnc"
    NOT_INTERESTED = "not_interested"
    CALLBACK = "callback"
    CONTACT_OBTAINED = "contact_obtained"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True)
class TrustedExitSignals:
    """Trusted deterministic early-exit signals (from trusted state, not words).

    Orchestrator inhe existing trusted context / priority outcome se bharta hai —
    naye speculative fields nahi. Ye UNTRUSTED utterance/Brain se NAHI aate.

    Attributes:
        dnc: DNC established (trusted priority outcome).
        not_interested: NOT_INTERESTED established (trusted priority outcome).
        callback_scheduled: Callback confirmed/scheduled (trusted context).
        contact_confirmed: Contact confirmed (trusted context flag).
    """

    dnc: bool = False
    not_interested: bool = False
    callback_scheduled: bool = False
    contact_confirmed: bool = False


@dataclass(frozen=True)
class BudgetCheckInput:
    """Minimal, data-only input for budget / early-exit evaluation.

    Koi DB/LLM/transcript/catalog. Remaining budget numbers orchestrator (trusted
    counting) se aate hain.

    Attributes:
        budget_state: Remaining budgets (BudgetState), ya None → fail-closed.
        exit_signals: Trusted early-exit signals.
        policy: Resolved, trusted BudgetPolicy, ya None → fail-closed.
    """

    budget_state: BudgetState | None
    exit_signals: TrustedExitSignals
    policy: BudgetPolicy | None


@dataclass(frozen=True)
class BudgetDecision:
    """Deterministic budget decision (executes nothing).

    Attributes:
        outcome: PROCEED / CONSTRAIN_RESPONSE / WIND_DOWN / EARLY_EXIT.
        reason: Machine-readable reason.
        exit_kind: Terminal cause (EARLY_EXIT/WIND_DOWN), ya None.
        response_ceiling: For CONSTRAIN_RESPONSE — max sentences this turn, ya None.
        policy_id: Which policy (traceability), ya None (fail-closed no-policy).
    """

    outcome: BudgetOutcome
    reason: str
    exit_kind: ExitKind | None = None
    response_ceiling: int | None = None
    policy_id: str | None = None


class BudgetPolicyEvaluator:
    """Evaluates budget/early-exit deterministically.

    Stateless, pure. Same input + same policy → same decision.
    """

    def evaluate(self, check: BudgetCheckInput) -> BudgetDecision:
        """Classify the budget/early-exit outcome (deterministic, fail-closed).

        Order (this layer's internal precedence):
            1. Trusted early-exit signals → EARLY_EXIT (highest; correct reason
               over 'budget ran out').
            2. Fail-closed: missing policy/budget_state → WIND_DOWN (conservative).
            3. Per-call ceilings (reasoning/turns/duration) exhausted → WIND_DOWN.
            4. Per-turn response_length low → CONSTRAIN_RESPONSE (call continues).
            5. Else PROCEED.

        Args:
            check: The budget-check input.

        Returns:
            BudgetDecision: Classified outcome + traceability. Executes nothing.
        """
        sig = check.exit_signals

        # 1. Trusted early-exit — BEFORE budget (correct cause wins over cost).
        if sig.dnc:
            return BudgetDecision(
                BudgetOutcome.EARLY_EXIT, "trusted signal: dnc",
                exit_kind=ExitKind.DNC, policy_id=self._pid(check),
            )
        if sig.not_interested:
            return BudgetDecision(
                BudgetOutcome.EARLY_EXIT, "trusted signal: not_interested",
                exit_kind=ExitKind.NOT_INTERESTED, policy_id=self._pid(check),
            )
        if sig.callback_scheduled:
            return BudgetDecision(
                BudgetOutcome.EARLY_EXIT, "trusted signal: callback_scheduled",
                exit_kind=ExitKind.CALLBACK, policy_id=self._pid(check),
            )
        if sig.contact_confirmed:
            return BudgetDecision(
                BudgetOutcome.EARLY_EXIT, "trusted signal: contact_confirmed",
                exit_kind=ExitKind.CONTACT_OBTAINED, policy_id=self._pid(check),
            )

        # 2. FAIL-CLOSED: without trustworthy policy/state, wind down (never
        #    unbounded proceed).
        if check.policy is None or check.budget_state is None:
            return BudgetDecision(
                BudgetOutcome.WIND_DOWN,
                "fail-closed: missing budget policy or state",
                exit_kind=ExitKind.BUDGET_EXHAUSTED,
                policy_id=self._pid(check),
            )

        state = check.budget_state

        # 3. Per-call ceilings exhausted → WIND_DOWN. remaining <= 0 (ya None ==
        #    unknown → conservative wind-down for that dimension).
        if _exhausted(state.reasoning_calls_remaining):
            return self._wind("reasoning_calls exhausted", check)
        if _exhausted(state.turns_remaining):
            return self._wind("turns exhausted", check)
        if _exhausted(state.seconds_remaining):
            return self._wind("call_duration exhausted", check)

        # 4. Per-turn response ceiling: hamesha compute hota hai (har turn ka
        #    max_response_sentences). Agar caller-supplied response ceiling policy
        #    se BHI kam hai (response budget tight) → CONSTRAIN_RESPONSE; warna
        #    normal PROCEED with ceiling attached.
        policy_ceiling = check.policy.limits.max_response_sentences
        if (
            state.max_response_sentences is not None
            and state.max_response_sentences < policy_ceiling
        ):
            return BudgetDecision(
                BudgetOutcome.CONSTRAIN_RESPONSE,
                "response budget tight; tighter ceiling applied",
                response_ceiling=state.max_response_sentences,
                policy_id=check.policy.policy_id,
            )

        # 5. Healthy budget → PROCEED (with the standard per-turn ceiling attached).
        return BudgetDecision(
            BudgetOutcome.PROCEED,
            "within budget",
            response_ceiling=policy_ceiling,
            policy_id=check.policy.policy_id,
        )

    def _wind(self, reason: str, check: BudgetCheckInput) -> BudgetDecision:
        """Build a WIND_DOWN decision (budget-exhausted).

        Args:
            reason: Machine-readable reason.
            check: The input (for policy id).

        Returns:
            BudgetDecision: WIND_DOWN with BUDGET_EXHAUSTED.
        """
        return BudgetDecision(
            BudgetOutcome.WIND_DOWN, reason,
            exit_kind=ExitKind.BUDGET_EXHAUSTED, policy_id=self._pid(check),
        )

    @staticmethod
    def _pid(check: BudgetCheckInput) -> str | None:
        """Return the policy id if a policy is present.

        Args:
            check: The input.

        Returns:
            str | None: Policy id, or None.
        """
        return check.policy.policy_id if check.policy is not None else None


def _exhausted(remaining: int | None) -> bool:
    """Whether a per-call budget dimension is exhausted.

    Conservative: None (unknown) → treated as exhausted (fail-closed), <=0 →
    exhausted.

    Args:
        remaining: Remaining count for this dimension, or None.

    Returns:
        bool: True if exhausted/unknown.
    """
    return remaining is None or remaining <= 0