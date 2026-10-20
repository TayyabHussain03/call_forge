"""Budget-state serializer.

Persists the processor's `_reasoning_calls` counter so restart cannot
grant extra reasoning capacity. `BudgetPolicyEvaluator` reads this
counter to decide whether the Brain may run; it is therefore
AUTHORITATIVE across turns, not derived.

Attempted-vs-committed semantics: the counter is incremented AFTER a
successful Brain invocation (which is itself inside the turn that may
later roll back). The Slice 51B rollback mechanism restores the live
processor state on commit failure, so `_reasoning_calls` naturally
rolls back with the turn. Persisted value therefore reflects the
committed state.
"""

from __future__ import annotations

from app.application.sales_runtime.serializers.base import (
    PayloadRequirement,
    build_envelope,
    parse_envelope,
)
from app.persistence.errors import InvalidPersistedPayloadError


BUDGET_STATE_KEY = "budget_state"
BUDGET_STATE_VERSION = 1


class BudgetStateSerializer:
    @property
    def key(self) -> str:
        return BUDGET_STATE_KEY

    @property
    def version(self) -> int:
        return BUDGET_STATE_VERSION

    @property
    def requirement(self) -> PayloadRequirement:
        return PayloadRequirement.OPTIONAL_DERIVED

    def serialize(self, reasoning_calls: int) -> str:
        if not isinstance(reasoning_calls, int) or isinstance(
            reasoning_calls, bool,
        ):
            raise ValueError("reasoning_calls must be a non-boolean integer")
        if reasoning_calls < 0:
            raise ValueError("reasoning_calls must be non-negative")
        return build_envelope(self.key, self.version, {
            "reasoning_calls": reasoning_calls,
        })

    def deserialize(self, envelope_json: str) -> int:
        payload = parse_envelope(envelope_json, self.key, (self.version,))
        try:
            value = payload["reasoning_calls"]
            if not isinstance(value, int) or isinstance(value, bool):
                raise InvalidPersistedPayloadError(
                    "reasoning_calls must be a non-boolean integer",
                )
            if value < 0:
                raise InvalidPersistedPayloadError(
                    "reasoning_calls must be non-negative",
                )
            return value
        except KeyError as exc:
            raise InvalidPersistedPayloadError(
                f"budget_state payload missing field: {exc}",
            ) from exc
