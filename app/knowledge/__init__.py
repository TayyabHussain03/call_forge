"""Provider-neutral authoritative knowledge-management foundation."""

from app.knowledge.contracts import (
    ActiveKnowledgeSelection,
    DocumentFormat,
    DocumentStatus,
    KnowledgeBase,
    KnowledgeCategory,
    KnowledgeDocument,
    KnowledgePurpose,
)
from app.knowledge.registry import KnowledgeRegistry, KnowledgeRegistryError

__all__ = [
    "ActiveKnowledgeSelection",
    "DocumentFormat",
    "DocumentStatus",
    "KnowledgeBase",
    "KnowledgeCategory",
    "KnowledgeDocument",
    "KnowledgePurpose",
    "KnowledgeRegistry",
    "KnowledgeRegistryError",
]
