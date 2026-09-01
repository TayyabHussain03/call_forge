"""Contract for action-validation results.

Ye ek chhota, explicit data holder hai jo batata hai ke ek proposed action
current context mein valid hai ya nahi, aur agar nahi to kis category mein. Ismein
KOI business logic nahi — validation logic `action_validator.py` mein hai; ye
sirf uska structured result carry karta hai.

DESIGN: `from_deterministic_rule` field abhi hamesha True hai (prerequisite
checks deterministic hain). Ye future DNC/priority override ke liye pehle se jagah
banata hai (Sitting 2B) — bina contract badle. LLM-derived kuch bhi is result
ko authoritative nahi banata.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ValidationCategory(str, Enum):
    """The category of a validation decision.

    Alag categories caller ko batati hain ke action kyun accept/reject hua, taake
    sahi agla kadam (Sitting 2B: state-aware fallback) chuna ja sake.
    """

    OK = "ok"                                # allowed
    NOT_ELIGIBLE = "not_eligible"            # action is state mein structurally allowed nahi
    MISSING_PREREQUISITE = "missing_prerequisite"  # context precondition poora nahi
    UNKNOWN_REQUIREMENT = "unknown_requirement"    # config/design error: unknown requires-key


class ValidationResult(BaseModel):
    """The outcome of validating a proposed action against context.

    Attributes:
        allowed: True agar action valid hai (category OK), warna False.
        category: Decision ki category (OK / not-eligible / missing-prereq /
            unknown-requirement).
        reason: Optional short machine-readable detail (e.g. missing requirement
            ka naam). PII-free.
        from_deterministic_rule: True jab decision ek deterministic rule se aaya
            (abhi hamesha True). Future override mechanism yahan reflect hoga.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed: bool
    category: ValidationCategory
    reason: str | None = Field(default=None, max_length=200)
    from_deterministic_rule: bool = True

    @classmethod
    def ok(cls) -> ValidationResult:
        """Build an allowed result.

        Returns:
            ValidationResult: allowed=True, category OK.
        """
        return cls(allowed=True, category=ValidationCategory.OK)

    @classmethod
    def rejected(
        cls, category: ValidationCategory, reason: str | None = None
    ) -> ValidationResult:
        """Build a rejected result.

        Args:
            category: The rejection category (must not be OK).
            reason: Optional short detail.

        Returns:
            ValidationResult: allowed=False with the given category.
        """
        return cls(allowed=False, category=category, reason=reason)