"""Budget policy model and fail-fast loader.

BudgetPolicy = TRUSTED resource-limit config. Four independent budgets. Validator
ko ek RESOLVED single policy deta hai (merging future orchestrator).

FAIL-FAST: negative/malformed limits → ConfigurationError. Actual FAIL-CLOSED
runtime behavior (missing budget_state → WIND_DOWN) evaluator mein.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.core.exceptions import ConfigurationError


@dataclass(frozen=True)
class BudgetLimits:
    """The four independent budget limits.

    Attributes:
        max_reasoning_calls: Per-call ceiling on reasoning invocations.
        max_turns: Per-call ceiling on conversation turns.
        max_call_seconds: Per-call ceiling on wall-clock duration.
        max_response_sentences: Per-TURN response constraint (not a call ceiling).
    """

    max_reasoning_calls: int
    max_turns: int
    max_call_seconds: int
    max_response_sentences: int


@dataclass(frozen=True)
class BudgetPolicy:
    """A single resolved, trusted budget policy.

    Attributes:
        policy_id: Stable identifier (traceability).
        limits: The four independent budget limits.
    """

    policy_id: str
    limits: BudgetLimits


def _parse_policy(policy_id: str, body: dict[str, Any]) -> BudgetPolicy:
    """Parse and validate one budget policy, failing fast.

    Args:
        policy_id: The policy id.
        body: Raw policy dict.

    Returns:
        BudgetPolicy: Parsed policy.

    Raises:
        ConfigurationError: On missing/negative limits.
    """
    limits_raw = body.get("limits")
    if not limits_raw:
        raise ConfigurationError(f"Budget policy {policy_id!r} missing limits.")

    required = (
        "max_reasoning_calls",
        "max_turns",
        "max_call_seconds",
        "max_response_sentences",
    )
    values: dict[str, int] = {}
    for key in required:
        if key not in limits_raw:
            raise ConfigurationError(
                f"Budget policy {policy_id!r} missing limit {key!r}."
            )
        val = limits_raw[key]
        if not isinstance(val, int) or val < 1:
            raise ConfigurationError(
                f"Budget policy {policy_id!r} limit {key!r} must be a positive int."
            )
        values[key] = val

    return BudgetPolicy(
        policy_id=body.get("policy_id", policy_id),
        limits=BudgetLimits(**values),
    )


def load_budget_policies(path: str | Path) -> dict[str, BudgetPolicy]:
    """Load and validate all budget policies, failing fast.

    Args:
        path: Path to budget_policy.yaml.

    Returns:
        dict[str, BudgetPolicy]: Policies keyed by id.

    Raises:
        ConfigurationError: On any structural problem.
    """
    policy_path = Path(path)
    if not policy_path.exists():
        raise ConfigurationError(f"Budget policy file not found: {policy_path}")

    raw: dict[str, Any] = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
    policies_raw = raw.get("policies", {})
    if not policies_raw:
        raise ConfigurationError("Budget policy config has no policies.")

    return {pid: _parse_policy(pid, body or {}) for pid, body in policies_raw.items()}