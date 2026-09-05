"""Deterministic authority classifier.

Answer deta hai: "is proposed action/decision ko trusted config kitni execution
authority deti hai?" — AUTO_EXECUTE / POLICY_BOUNDED / HUMAN_APPROVAL_REQUIRED /
DENIED. Aur kuch nahi.

CORE INVARIANT: LLM apni authority KABHI decide nahi karta — tier trusted config
se deterministically derive hoti hai. Commercial detail (discount %) UNTRUSTED
proposal data hai; validator config bounds ke against check karta hai, LLM
calculate/authorize NAHI karta.

FAIL-CLOSED: unknown/missing/ambiguous → MOST RESTRICTIVE (human/denied), kabhi
AUTO_EXECUTE nahi.

PRECEDENCE (deterministic): DENIED > HUMAN_APPROVAL_REQUIRED > POLICY_BOUNDED >
AUTO_EXECUTE. Commercial request (agar present) action-tier par takes precedence
kyunki commercial higher-risk hai — aur usmein bhi restrictive jeetta hai.

Validator PURE: koi mutation/LLM/network/DB/raw-text/keyword/execution/state/
service/claim logic. Sirf classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.brain.authority.models import AuthorityPolicy, AuthorityTier
from app.brain.contracts import CommercialRequest
from app.core.constants import AgentAction, CommercialRequestKind


# Restrictiveness order (higher index = more restrictive) — deterministic precedence.
_RESTRICTIVENESS = {
    AuthorityTier.AUTO_EXECUTE: 0,
    AuthorityTier.POLICY_BOUNDED: 1,
    AuthorityTier.HUMAN_APPROVAL_REQUIRED: 2,
    AuthorityTier.DENIED: 3,
}


def _more_restrictive(a: AuthorityTier, b: AuthorityTier) -> AuthorityTier:
    """Return the more restrictive of two tiers.

    Args:
        a: First tier.
        b: Second tier.

    Returns:
        AuthorityTier: The more restrictive tier.
    """
    return a if _RESTRICTIVENESS[a] >= _RESTRICTIVENESS[b] else b


@dataclass(frozen=True)
class AuthorityCheckInput:
    """Minimal, data-only input for authority classification.

    Koi transcript/DB/catalog/repo/state-machine/LLM/raw-pricing-string.

    Attributes:
        proposal_action: The proposed AgentAction (may be None).
        commercial_request: Optional structured commercial detail (untrusted).
        policy: The resolved, trusted AuthorityPolicy.
    """

    proposal_action: AgentAction | None
    commercial_request: CommercialRequest | None
    policy: AuthorityPolicy


@dataclass(frozen=True)
class AuthorityResult:
    """Deterministic authority classification (executes nothing).

    Attributes:
        tier: The assigned authority tier.
        policy_id: Which policy was evaluated (traceability).
        reason: Machine-readable reason.
        within_bounds: For POLICY_BOUNDED/discount evaluations — True/False, ya
            None jab applicable nahi.
        escalation_hint: Machine-readable category for orchestrator routing (not
            wording, not an action), ya None.
    """

    tier: AuthorityTier
    policy_id: str
    reason: str
    within_bounds: bool | None = None
    escalation_hint: str | None = None

    @property
    def is_auto(self) -> bool:
        """Whether autonomous execution is permitted outright.

        Returns:
            bool: True only for AUTO_EXECUTE.
        """
        return self.tier == AuthorityTier.AUTO_EXECUTE


class AuthorityPolicyValidator:
    """Classifies a proposal's execution authority from trusted config.

    Stateless, deterministic, pure. Same input + same policy → same result.
    """

    def classify(self, check: AuthorityCheckInput) -> AuthorityResult:
        """Classify authority tier (fail-closed, restrictive precedence).

        Logic:
            1. Agar commercial_request present → commercial tier evaluate karo.
            2. Action tier evaluate karo (mapping ya default).
            3. Dono ka MORE RESTRICTIVE tier return karo (commercial higher-risk).
            Missing/unknown → most restrictive via default (fail-closed).

        Args:
            check: The authority-check input.

        Returns:
            AuthorityResult: Tier + traceability + bounds/escalation.
        """
        policy = check.policy

        # Action tier: mapping se, warna default (fail-closed restrictive).
        action = check.proposal_action
        if action is not None and action in policy.action_tiers:
            action_tier = policy.action_tiers[action]
            action_reason = f"action {action.value} -> {action_tier.value}"
        else:
            action_tier = policy.default_tier
            action_reason = (
                f"action {action.value if action else None} unmapped -> default "
                f"{action_tier.value}"
            )

        # Koi commercial request nahi → sirf action tier.
        if check.commercial_request is None:
            return AuthorityResult(
                tier=action_tier,
                policy_id=policy.policy_id,
                reason=action_reason,
                escalation_hint=self._hint_for(action_tier),
            )

        # Commercial request present → commercial tier + bounds.
        comm_tier, within, comm_reason = self._classify_commercial(
            policy, check.commercial_request
        )

        # More restrictive of (action, commercial) wins.
        final = _more_restrictive(action_tier, comm_tier)
        return AuthorityResult(
            tier=final,
            policy_id=policy.policy_id,
            reason=f"{comm_reason}; {action_reason}; final={final.value}",
            within_bounds=within,
            escalation_hint=self._hint_for(final),
        )

    def _classify_commercial(
        self, policy: AuthorityPolicy, req: CommercialRequest
    ) -> tuple[AuthorityTier, bool | None, str]:
        """Classify a commercial request deterministically from config.

        Fail-closed: koi rule missing/ambiguous → default_tier (restrictive).

        Args:
            policy: The resolved policy.
            req: The structured commercial request.

        Returns:
            tuple[AuthorityTier, bool | None, str]: (tier, within_bounds, reason).
        """
        kind = req.kind

        if kind == CommercialRequestKind.DISCOUNT:
            rule = policy.discount
            if rule is None or req.requested_discount_percent is None:
                # Missing bound config ya ambiguous discount → fail-closed.
                return policy.default_tier, None, "discount rule/percent missing -> default"
            within = req.requested_discount_percent <= rule.max_autonomous_percent
            tier = rule.within_tier if within else rule.over_tier
            return tier, within, (
                f"discount {req.requested_discount_percent}% "
                f"(max {rule.max_autonomous_percent}%) -> {tier.value}"
            )

        if kind == CommercialRequestKind.PRICING_DISCLOSURE:
            rule = policy.pricing_disclosure
            if rule is None or not rule.allowed:
                # Disclosure not explicitly permitted → most restrictive default.
                return policy.default_tier, None, "pricing disclosure not permitted -> default"
            return rule.tier, None, f"pricing disclosure allowed -> {rule.tier.value}"

        # Other commercial kinds → configured tier, warna default (fail-closed).
        if kind in policy.commercial_tiers:
            tier = policy.commercial_tiers[kind]
            return tier, None, f"commercial {kind.value} -> {tier.value}"
        return policy.default_tier, None, f"commercial {kind.value} unmapped -> default"

    @staticmethod
    def _hint_for(tier: AuthorityTier) -> str | None:
        """Return an escalation hint category for a tier.

        Args:
            tier: The authority tier.

        Returns:
            str | None: Machine-readable hint, ya None for AUTO_EXECUTE.
        """
        if tier == AuthorityTier.HUMAN_APPROVAL_REQUIRED:
            return "escalate_human_approval"
        if tier == AuthorityTier.DENIED:
            return "deny_and_redirect"
        if tier == AuthorityTier.POLICY_BOUNDED:
            return "proceed_within_bounds"
        return None