"""Deterministic claim/service authorization validator.

Answer deta hai: "kya ye exact service/claim is campaign mein authorized hai?"
Ye authority-check hai — NLP nahi, selection nahi, relevance nahi, pricing/
guarantee invent NAHI. Guardrail layer mein baithta hai, ActionValidator/
StateMachine se ALAG (separation of concerns).

FAIL-CLOSED: unknown service ya unknown claim → REJECTED (kabhi silently allowed
nahi). Forbidden claim → REJECTED. Sirf explicitly allowed claim → AUTHORIZED.
Isse LLM invented pricing/guarantees/case-studies authorize nahi ho sakte.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.catalog.scoped_catalog import ScopedCatalog


class ClaimCategory(str, Enum):
    """The category of a claim/service authorization decision."""

    AUTHORIZED = "authorized"              # explicitly allowed
    FORBIDDEN = "forbidden"                # explicitly forbidden
    UNKNOWN_SERVICE = "unknown_service"    # service not in campaign scope
    UNKNOWN_CLAIM = "unknown_claim"        # claim neither allowed nor forbidden


@dataclass(frozen=True)
class ClaimValidationResult:
    """The outcome of validating a service/claim.

    Attributes:
        authorized: True only for AUTHORIZED.
        category: The decision category.
        reason: Machine-readable detail (debug), ya None.
    """

    authorized: bool
    category: ClaimCategory
    reason: str | None = None


class ClaimValidator:
    """Validates services/claims against a campaign-scoped catalog.

    Attributes:
        catalog: The campaign-scoped catalog (authority).
    """

    def __init__(self, scoped_catalog: ScopedCatalog) -> None:
        """Initialize with a campaign-scoped catalog.

        Args:
            scoped_catalog: The scoped catalog providing authority.
        """
        self._catalog = scoped_catalog

    def validate_service(self, service_id: str) -> ClaimValidationResult:
        """Whether a service is authorized in this campaign.

        Args:
            service_id: The service id.

        Returns:
            ClaimValidationResult: AUTHORIZED or UNKNOWN_SERVICE (fail-closed).
        """
        if self._catalog.is_service_authorized(service_id):
            return ClaimValidationResult(True, ClaimCategory.AUTHORIZED)
        return ClaimValidationResult(
            False, ClaimCategory.UNKNOWN_SERVICE,
            reason=f"service not authorized in campaign: {service_id}",
        )

    def validate_claim(
        self, service_id: str, claim: str, subservice_id: str | None = None
    ) -> ClaimValidationResult:
        """Whether an exact claim is authorized for a service/subservice.

        Precedence (fail-closed):
            1. Service not in scope → UNKNOWN_SERVICE.
            2. Claim in forbidden set → FORBIDDEN.
            3. Claim in allowed set → AUTHORIZED.
            4. Neither → UNKNOWN_CLAIM (rejected — LLM invented claim yahan girta).

        Subservice diya ho to uske claims bhi consider hote hain (service-level ke
        saath).

        Args:
            service_id: The service id.
            claim: The exact claim identifier.
            subservice_id: Optional subservice id for subservice-level claims.

        Returns:
            ClaimValidationResult: The authorization decision.
        """
        service = self._catalog.get_service(service_id)
        if service is None:
            return ClaimValidationResult(
                False, ClaimCategory.UNKNOWN_SERVICE,
                reason=f"service not authorized: {service_id}",
            )

        allowed = set(service.allowed_claims)
        forbidden = set(service.forbidden_claims)

        if subservice_id is not None:
            sub = service.subservices.get(subservice_id)
            if sub is None:
                return ClaimValidationResult(
                    False, ClaimCategory.UNKNOWN_SERVICE,
                    reason=f"subservice not found: {subservice_id}",
                )
            allowed |= set(sub.allowed_claims)
            forbidden |= set(sub.forbidden_claims)

        # Forbidden pehle check — forbidden hamesha jeetta hai.
        if claim in forbidden:
            return ClaimValidationResult(
                False, ClaimCategory.FORBIDDEN,
                reason=f"claim forbidden: {claim}",
            )
        if claim in allowed:
            return ClaimValidationResult(True, ClaimCategory.AUTHORIZED)
        # Fail-closed: unknown claim rejected.
        return ClaimValidationResult(
            False, ClaimCategory.UNKNOWN_CLAIM,
            reason=f"claim not authorized: {claim}",
        )