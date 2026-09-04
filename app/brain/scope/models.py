"""Scope policy model and fail-fast loader.

ScopePolicy = TRUSTED conversational-scope config. Catalog/pricing authority se
ALAG. Ye validator ko ek RESOLVED single policy deta hai (global→campaign→call
merging future orchestrator ka kaam).

FAIL-FAST: unknown topic/action category, allow/deny overlap → ConfigurationError,
boot rukega. FAIL-CLOSED semantics: jo allowed-list mein nahi, woh out-of-scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.core.constants import AgentAction, TopicCategory
from app.core.exceptions import ConfigurationError


@dataclass(frozen=True)
class ScopePolicy:
    """A single resolved, trusted conversational-scope policy.

    Attributes:
        policy_id: Stable identifier (traceability).
        allowed_topics: Topic categories permitted in this scope.
        allowed_actions: Agent actions permitted in this scope.
        restricted_topics: Explicitly disallowed topics (deny beats allow).
        escalation_topics: In-scope topics that FLAG for authority/human routing
            (flag only — authority decision AuthorityPolicy leti hai, scope nahi).
    """

    policy_id: str
    allowed_topics: frozenset[TopicCategory]
    allowed_actions: frozenset[AgentAction]
    restricted_topics: frozenset[TopicCategory] = field(default_factory=frozenset)
    escalation_topics: frozenset[TopicCategory] = field(default_factory=frozenset)


def _coerce_topic(value: str) -> TopicCategory:
    """Convert a config string to a TopicCategory, or fail fast.

    Args:
        value: Raw topic string.

    Returns:
        TopicCategory: The enum member.

    Raises:
        ConfigurationError: On unknown topic.
    """
    try:
        return TopicCategory(value)
    except ValueError as exc:
        raise ConfigurationError(f"Unknown scope topic: {value!r}") from exc


def _coerce_action(value: str) -> AgentAction:
    """Convert a config string to an AgentAction, or fail fast.

    Args:
        value: Raw action string.

    Returns:
        AgentAction: The enum member.

    Raises:
        ConfigurationError: On unknown action.
    """
    try:
        return AgentAction(value)
    except ValueError as exc:
        raise ConfigurationError(f"Unknown scope action: {value!r}") from exc


def _parse_policy(policy_id: str, body: dict[str, Any]) -> ScopePolicy:
    """Parse and validate one policy, failing fast on problems.

    Args:
        policy_id: The policy id (key).
        body: Raw policy dict.

    Returns:
        ScopePolicy: The parsed policy.

    Raises:
        ConfigurationError: On unknown categories or allow/deny overlap.
    """
    if not body.get("allowed_topics") and not body.get("allowed_actions"):
        raise ConfigurationError(
            f"Scope policy {policy_id!r} has no allowed_topics/allowed_actions."
        )

    allowed_topics = frozenset(_coerce_topic(t) for t in body.get("allowed_topics", []))
    allowed_actions = frozenset(
        _coerce_action(a) for a in body.get("allowed_actions", [])
    )
    restricted_topics = frozenset(
        _coerce_topic(t) for t in body.get("restricted_topics", [])
    )
    escalation_topics = frozenset(
        _coerce_topic(t) for t in body.get("escalation_topics", [])
    )

    # allow/deny overlap → ambiguous, fail-fast.
    overlap = allowed_topics & restricted_topics
    if overlap:
        raise ConfigurationError(
            f"Scope policy {policy_id!r} has topics both allowed and restricted: "
            f"{[t.value for t in overlap]}"
        )
    # escalation topics must be a subset of allowed (they are in-scope by definition).
    esc_not_allowed = escalation_topics - allowed_topics
    if esc_not_allowed:
        raise ConfigurationError(
            f"Scope policy {policy_id!r} escalation topics not in allowed: "
            f"{[t.value for t in esc_not_allowed]}"
        )

    return ScopePolicy(
        policy_id=body.get("policy_id", policy_id),
        allowed_topics=allowed_topics,
        allowed_actions=allowed_actions,
        restricted_topics=restricted_topics,
        escalation_topics=escalation_topics,
    )


def load_scope_policies(path: str | Path) -> dict[str, ScopePolicy]:
    """Load and validate all scope policies, failing fast on any error.

    Args:
        path: Path to scope_policy.yaml.

    Returns:
        dict[str, ScopePolicy]: Policies keyed by id.

    Raises:
        ConfigurationError: On any structural problem.
    """
    policy_path = Path(path)
    if not policy_path.exists():
        raise ConfigurationError(f"Scope policy file not found: {policy_path}")

    raw: dict[str, Any] = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
    policies_raw = raw.get("policies", {})
    if not policies_raw:
        raise ConfigurationError("Scope policy config has no policies.")

    policies: dict[str, ScopePolicy] = {}
    for pid, body in policies_raw.items():
        policies[pid] = _parse_policy(pid, body or {})
    return policies