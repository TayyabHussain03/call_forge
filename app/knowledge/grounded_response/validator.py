"""Post-construction validation for GRC output integrity."""

from __future__ import annotations

from app.knowledge.grounded_response.contracts import (
    GroundedConversationPlan,
    GroundedResponseInput,
)


def validate_grounded_plan(
    plan: GroundedConversationPlan,
    value: GroundedResponseInput,
) -> None:
    """Raise ValueError if the grounded plan violates structural invariants."""

    selected_ids = {item.evidence.approval_id for item in value.knowledge_context.selected}
    grounded_ids = {item.evidence.approval_id for item in plan.grounded_items}

    if grounded_ids != selected_ids:
        raise ValueError("grounded items must match selected evidence exactly")

    if len(plan.grounded_items) != len(value.knowledge_context.selected):
        raise ValueError("grounded item count must match selected evidence count")
