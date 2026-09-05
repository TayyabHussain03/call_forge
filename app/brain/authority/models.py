"""Authority policy model and fail-fast loader.

AuthorityPolicy = TRUSTED commercial/execution authority config. Scope/catalog se
ALAG. Validator ko ek RESOLVED single policy deta hai (merging future orchestrator).

FAIL-FAST: unknown tier/action/commercial-kind → ConfigurationError. FAIL-CLOSED:
default_tier auto_execute NAHI ho sakta (loader enforce karta hai).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from app.core.constants import AgentAction, CommercialRequestKind
from app.core.exceptions import ConfigurationError


class AuthorityTier(str, Enum):
    """Execution-authority tier assigned to a proposed action/decision.

    DENIED aur HUMAN_APPROVAL_REQUIRED ALAG (locked): human-approvable vs
    permanently-forbidden.
    """

    AUTO_EXECUTE = "auto_execute"                    # autonomous allowed
    POLICY_BOUNDED = "policy_bounded"                # autonomous only within bounds
    HUMAN_APPROVAL_REQUIRED = "human_approval_required"  # human approval se proceed
    DENIED = "denied"                                # not allowed at all


@dataclass(frozen=True)
class DiscountRule:
    """Deterministic discount bound.

    Attributes:
        max_autonomous_percent: <= is percent → within_tier; > → over_tier.
        within_tier: Tier when requested discount within limit.
        over_tier: Tier when over limit.
    """

    max_autonomous_percent: float
    within_tier: AuthorityTier
    over_tier: AuthorityTier


@dataclass(frozen=True)
class PricingDisclosureRule:
    """Pricing-disclosure permission.

    Attributes:
        allowed: Whether disclosure is permitted.
        tier: Tier when allowed (else most-restrictive applies at eval).
    """

    allowed: bool
    tier: AuthorityTier


@dataclass(frozen=True)
class AuthorityPolicy:
    """A single resolved, trusted authority policy.

    Attributes:
        policy_id: Stable identifier (traceability).
        default_tier: Tier for unmapped actions/decisions. FAIL-CLOSED: never
            AUTO_EXECUTE (loader enforces).
        action_tiers: Action → tier mapping.
        discount: Discount bound rule, ya None.
        pricing_disclosure: Disclosure rule, ya None.
        commercial_tiers: Non-discount commercial kind → tier (custom_pricing,
            quotation, scope_change, guarantee, contractual_commitment).
    """

    policy_id: str
    default_tier: AuthorityTier
    action_tiers: dict[AgentAction, AuthorityTier] = field(default_factory=dict)
    discount: DiscountRule | None = None
    pricing_disclosure: PricingDisclosureRule | None = None
    commercial_tiers: dict[CommercialRequestKind, AuthorityTier] = field(
        default_factory=dict
    )


def _coerce_tier(value: str) -> AuthorityTier:
    """Convert a config string to an AuthorityTier, or fail fast.

    Args:
        value: Raw tier string.

    Returns:
        AuthorityTier: The enum member.

    Raises:
        ConfigurationError: On unknown tier.
    """
    try:
        return AuthorityTier(value)
    except ValueError as exc:
        raise ConfigurationError(f"Unknown authority tier: {value!r}") from exc


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
        raise ConfigurationError(f"Unknown authority action: {value!r}") from exc


def _parse_policy(policy_id: str, body: dict[str, Any]) -> AuthorityPolicy:
    """Parse and validate one authority policy, failing fast.

    Args:
        policy_id: The policy id.
        body: Raw policy dict.

    Returns:
        AuthorityPolicy: Parsed policy.

    Raises:
        ConfigurationError: On unknown values or an AUTO_EXECUTE default (unsafe).
    """
    default_raw = body.get("default_tier")
    if default_raw is None:
        raise ConfigurationError(f"Authority policy {policy_id!r} missing default_tier.")
    default_tier = _coerce_tier(default_raw)
    # FAIL-CLOSED: default can never be auto-execute.
    if default_tier == AuthorityTier.AUTO_EXECUTE:
        raise ConfigurationError(
            f"Authority policy {policy_id!r} default_tier cannot be AUTO_EXECUTE "
            "(fail-closed requires a restrictive default)."
        )

    action_tiers = {
        _coerce_action(a): _coerce_tier(t)
        for a, t in (body.get("action_tiers", {}) or {}).items()
    }

    commercial = body.get("commercial", {}) or {}
    discount_rule: DiscountRule | None = None
    disclosure_rule: PricingDisclosureRule | None = None
    commercial_tiers: dict[CommercialRequestKind, AuthorityTier] = {}

    for kind_key, rule in commercial.items():
        rule = rule or {}
        if kind_key == "discount":
            discount_rule = DiscountRule(
                max_autonomous_percent=float(rule["max_autonomous_percent"]),
                within_tier=_coerce_tier(rule["within_tier"]),
                over_tier=_coerce_tier(rule["over_tier"]),
            )
        elif kind_key == "pricing_disclosure":
            disclosure_rule = PricingDisclosureRule(
                allowed=bool(rule["allowed"]),
                tier=_coerce_tier(rule["tier"]),
            )
        else:
            # Other commercial kinds → single tier.
            try:
                kind = CommercialRequestKind(kind_key)
            except ValueError as exc:
                raise ConfigurationError(
                    f"Unknown commercial kind {kind_key!r} in policy {policy_id!r}"
                ) from exc
            commercial_tiers[kind] = _coerce_tier(rule["tier"])

    return AuthorityPolicy(
        policy_id=body.get("policy_id", policy_id),
        default_tier=default_tier,
        action_tiers=action_tiers,
        discount=discount_rule,
        pricing_disclosure=disclosure_rule,
        commercial_tiers=commercial_tiers,
    )


def load_authority_policies(path: str | Path) -> dict[str, AuthorityPolicy]:
    """Load and validate all authority policies, failing fast.

    Args:
        path: Path to authority_policy.yaml.

    Returns:
        dict[str, AuthorityPolicy]: Policies keyed by id.

    Raises:
        ConfigurationError: On any structural problem.
    """
    policy_path = Path(path)
    if not policy_path.exists():
        raise ConfigurationError(f"Authority policy file not found: {policy_path}")

    raw: dict[str, Any] = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
    policies_raw = raw.get("policies", {})
    if not policies_raw:
        raise ConfigurationError("Authority policy config has no policies.")

    return {pid: _parse_policy(pid, body or {}) for pid, body in policies_raw.items()}