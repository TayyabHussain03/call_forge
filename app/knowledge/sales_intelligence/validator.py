"""Post-construction validation for SKIE output integrity."""

from __future__ import annotations

from app.knowledge.sales_intelligence.contracts import (
    KnowledgeConversationContext,
    KnowledgePriority,
    SalesKnowledgeInput,
)


def validate_knowledge_context(
    context: KnowledgeConversationContext,
    value: SalesKnowledgeInput,
) -> None:
    """Raise ValueError if the teaching plan violates structural invariants."""

    all_approved_ids = {item.approval_id for item in value.approved_evidence.items}
    selected_ids = {item.evidence.approval_id for item in context.selected}
    suppressed_ids = {item.evidence.approval_id for item in context.suppressed}

    if not selected_ids <= all_approved_ids:
        raise ValueError("selected evidence contains items not in approved set")
    if not suppressed_ids <= all_approved_ids:
        raise ValueError("suppressed evidence contains items not in approved set")

    primary_count = sum(
        1
        for item in context.selected
        if item.priority == KnowledgePriority.PRIMARY
    )
    if context.selected and primary_count != 1:
        raise ValueError("exactly one selected item must be primary")
