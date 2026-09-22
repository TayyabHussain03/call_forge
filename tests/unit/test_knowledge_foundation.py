"""Focused tests for the provider-neutral knowledge-management foundation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime, timezone

import pytest

from app.knowledge.contracts import (
    ActiveKnowledgeSelection,
    DocumentFormat,
    DocumentStatus,
    KnowledgeBase,
    KnowledgeCategory,
    KnowledgeDocument,
)
from app.knowledge.registry import KnowledgeRegistry, KnowledgeRegistryError


_NOW = datetime(2026, 9, 22, tzinfo=timezone.utc)


def _base(
    business_id: str = "business-a", knowledge_base_id: str = "knowledge-main"
) -> KnowledgeBase:
    return KnowledgeBase(
        knowledge_base_id,
        business_id,
        "Main Knowledge",
        _NOW,
        "Approved business knowledge metadata.",
    )


def _document(
    *,
    business_id: str = "business-a",
    knowledge_base_id: str = "knowledge-main",
    document_id: str = "services",
    version: int = 1,
    version_id: str | None = None,
    status: DocumentStatus = DocumentStatus.DRAFT,
    category: KnowledgeCategory = KnowledgeCategory.SERVICES,
) -> KnowledgeDocument:
    return KnowledgeDocument(
        document_id,
        version_id or f"{document_id}-v{version}",
        knowledge_base_id,
        business_id,
        "Services Guide",
        "Service descriptions maintained by the business.",
        category,
        ("services", "approved"),
        _NOW,
        "user-1",
        version,
        status,
        "a" * 64,
        DocumentFormat.PDF,
        "services.pdf",
    )


def _registry() -> KnowledgeRegistry:
    return KnowledgeRegistry((_base(),))


def test_knowledge_base_creation_is_bounded_and_immutable() -> None:
    knowledge_base = _base()
    assert knowledge_base.business_id == "business-a"
    with pytest.raises(FrozenInstanceError):
        knowledge_base.name = "Changed"  # type: ignore[misc]


@pytest.mark.parametrize("document_format", list(DocumentFormat))
def test_supported_document_formats_are_metadata_only(  # type: ignore[no-untyped-def]
    document_format,
) -> None:
    document = replace(_document(), document_format=document_format)
    assert document.document_format == document_format
    assert "content" not in {item.name for item in fields(document)}


@pytest.mark.parametrize("category", list(KnowledgeCategory))
def test_supported_categories_are_typed(category: KnowledgeCategory) -> None:
    assert replace(_document(), category=category).category == category


def test_invalid_category_and_status_types_fail_closed() -> None:
    with pytest.raises(TypeError):
        replace(_document(), category="services")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        replace(_document(), status="published")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "change",
    [
        {"checksum": "not-a-checksum"},
        {"version": 0},
        {"tags": ("duplicate", "duplicate")},
        {"tags": ("spaces are invalid",)},
        {"uploaded_at": datetime(2026, 1, 1)},
        {"title": ""},
        {"source_name": ""},
    ],
)
def test_document_metadata_validation(change: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(_document(), **change)


def test_duplicate_knowledge_base_and_version_ids_are_rejected() -> None:
    registry = _registry()
    with pytest.raises(KnowledgeRegistryError, match="duplicate"):
        registry.add_knowledge_base(_base())
    registry.add_document_version(_document())
    with pytest.raises(KnowledgeRegistryError, match="duplicate"):
        registry.add_document_version(_document(version=2, version_id="services-v1"))


def test_version_ids_are_unique_across_one_knowledge_base() -> None:
    registry = _registry()
    registry.add_document_version(_document())
    with pytest.raises(KnowledgeRegistryError, match="duplicate"):
        registry.add_document_version(
            _document(document_id="policies", version_id="services-v1")
        )


def test_versions_must_be_contiguous_and_start_as_draft() -> None:
    registry = _registry()
    with pytest.raises(KnowledgeRegistryError, match="next version"):
        registry.add_document_version(_document(version=2))
    with pytest.raises(KnowledgeRegistryError, match="begin as draft"):
        registry.add_document_version(_document(status=DocumentStatus.PUBLISHED))


def test_document_lifecycle_is_draft_then_published_then_archived() -> None:
    registry = _registry()
    original = registry.add_document_version(_document())
    published = registry.publish_document(
        "business-a", "knowledge-main", "services", 1
    )
    archived = registry.archive_document(
        "business-a", "knowledge-main", "services", 1
    )
    assert original.status == DocumentStatus.DRAFT
    assert published.published and published.active and not published.archived
    assert archived.archived and not archived.published and not archived.active


@pytest.mark.parametrize(
    ("initial", "target"),
    [
        (DocumentStatus.DRAFT, DocumentStatus.ARCHIVED),
        (DocumentStatus.DRAFT, DocumentStatus.DRAFT),
        (DocumentStatus.PUBLISHED, DocumentStatus.DRAFT),
        (DocumentStatus.ARCHIVED, DocumentStatus.PUBLISHED),
    ],
)
def test_invalid_status_transitions_are_rejected(  # type: ignore[no-untyped-def]
    initial, target
) -> None:
    registry = _registry()
    registry.add_document_version(_document())
    if initial == DocumentStatus.PUBLISHED:
        registry.publish_document("business-a", "knowledge-main", "services", 1)
    elif initial == DocumentStatus.ARCHIVED:
        registry.publish_document("business-a", "knowledge-main", "services", 1)
        registry.archive_document("business-a", "knowledge-main", "services", 1)
    with pytest.raises(KnowledgeRegistryError, match="invalid"):
        registry.transition_document(
            "business-a", "knowledge-main", "services", 1, target
        )


def test_publishing_new_version_preserves_old_value_and_archives_registry_copy() -> None:
    registry = _registry()
    first_draft = registry.add_document_version(_document())
    first_published = registry.publish_document(
        "business-a", "knowledge-main", "services", 1
    )
    second_draft = registry.add_document_version(_document(version=2))
    second_published = registry.publish_document(
        "business-a", "knowledge-main", "services", 2
    )
    current = registry.versions("business-a", "knowledge-main", "services")
    assert first_draft.status == DocumentStatus.DRAFT
    assert first_published.status == DocumentStatus.PUBLISHED
    assert second_draft.status == DocumentStatus.DRAFT
    assert current[0].status == DocumentStatus.ARCHIVED
    assert current[1] == second_published
    assert registry.published_documents("business-a", "knowledge-main") == (
        second_published,
    )


def test_only_published_documents_are_future_indexing_eligible() -> None:
    registry = _registry()
    registry.add_document_version(_document())
    assert registry.published_documents("business-a", "knowledge-main") == ()
    published = registry.publish_document(
        "business-a", "knowledge-main", "services", 1
    )
    assert registry.published_documents("business-a", "knowledge-main") == (
        published,
    )


def test_active_selection_is_campaign_scoped_and_replaceable() -> None:
    second = _base(knowledge_base_id="knowledge-support")
    registry = KnowledgeRegistry((_base(), second))
    first_selection = ActiveKnowledgeSelection(
        "business-a", "campaign-sales", "knowledge-main"
    )
    second_selection = ActiveKnowledgeSelection(
        "business-a", "campaign-sales", "knowledge-support"
    )
    registry.select_active(first_selection)
    assert registry.active_for_campaign("business-a", "campaign-sales") == (
        first_selection
    )
    registry.select_active(second_selection)
    assert registry.active_for_campaign("business-a", "campaign-sales") == (
        second_selection
    )


def test_multi_business_isolation_allows_same_local_ids_without_leakage() -> None:
    registry = KnowledgeRegistry((_base(), _base("business-b")))
    first = registry.add_document_version(_document())
    second = registry.add_document_version(_document(business_id="business-b"))
    registry.publish_document("business-a", "knowledge-main", "services", 1)
    assert registry.versions("business-a", "knowledge-main", "services")[0] != second
    assert registry.versions("business-b", "knowledge-main", "services") == (second,)
    assert first.business_id != second.business_id


def test_cross_business_document_and_selection_are_rejected() -> None:
    registry = _registry()
    with pytest.raises(KnowledgeRegistryError):
        registry.add_document_version(_document(business_id="business-b"))
    with pytest.raises(KnowledgeRegistryError):
        registry.select_active(
            ActiveKnowledgeSelection(
                "business-b", "campaign-sales", "knowledge-main"
            )
        )


def test_registry_views_cannot_be_mutated() -> None:
    registry = _registry()
    with pytest.raises(TypeError):
        registry.knowledge_bases[("business-a", "other")] = _base()  # type: ignore[index]


def test_knowledge_contracts_have_no_authority_or_retrieval_surface() -> None:
    names = {item.name for item in fields(KnowledgeDocument)}
    assert names.isdisjoint(
        {
            "content",
            "chunks",
            "embedding",
            "vector",
            "evidence",
            "prompt",
            "action",
            "authority",
            "next_state",
        }
    )
