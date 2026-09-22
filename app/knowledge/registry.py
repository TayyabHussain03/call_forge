"""Deterministic in-memory knowledge lifecycle and isolation boundary."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Iterable, Mapping

from app.knowledge.contracts import (
    ActiveKnowledgeSelection,
    DocumentStatus,
    KnowledgeBase,
    KnowledgeDocument,
)


class KnowledgeRegistryError(ValueError):
    """Knowledge ownership, identity, version, or lifecycle validation failed."""


class KnowledgeRegistry:
    """Offline deterministic registry; stores metadata only and performs no I/O."""

    _TRANSITIONS = {
        DocumentStatus.DRAFT: frozenset({DocumentStatus.PUBLISHED}),
        DocumentStatus.PUBLISHED: frozenset({DocumentStatus.ARCHIVED}),
        DocumentStatus.ARCHIVED: frozenset(),
    }

    def __init__(
        self,
        knowledge_bases: Iterable[KnowledgeBase] = (),
        documents: Iterable[KnowledgeDocument] = (),
        selections: Iterable[ActiveKnowledgeSelection] = (),
    ) -> None:
        self._knowledge_bases: dict[tuple[str, str], KnowledgeBase] = {}
        self._documents: dict[tuple[str, str, str], tuple[KnowledgeDocument, ...]] = {}
        self._selections: dict[tuple[str, str], ActiveKnowledgeSelection] = {}
        for knowledge_base in knowledge_bases:
            self.add_knowledge_base(knowledge_base)
        for document in sorted(documents, key=lambda item: item.version):
            self.add_document_version(document)
        for selection in selections:
            self.select_active(selection)

    def add_knowledge_base(self, value: KnowledgeBase) -> KnowledgeBase:
        """Add one business-scoped collection, rejecting duplicate identities."""
        if not isinstance(value, KnowledgeBase):
            raise TypeError("knowledge base must be a KnowledgeBase")
        key = (value.business_id, value.knowledge_base_id)
        if key in self._knowledge_bases:
            raise KnowledgeRegistryError("duplicate knowledge base id")
        self._knowledge_bases[key] = value
        return value

    def add_document_version(self, value: KnowledgeDocument) -> KnowledgeDocument:
        """Append exactly the next immutable version inside its owning business."""
        if not isinstance(value, KnowledgeDocument):
            raise TypeError("document must be a KnowledgeDocument")
        self._require_base(value.business_id, value.knowledge_base_id)
        key = (value.business_id, value.knowledge_base_id, value.document_id)
        versions = self._documents.get(key, ())
        if any(
            item.version_id == value.version_id
            for document_key, document_versions in self._documents.items()
            if document_key[:2] == (value.business_id, value.knowledge_base_id)
            for item in document_versions
        ):
            raise KnowledgeRegistryError("duplicate document version id")
        expected = len(versions) + 1
        if value.version != expected:
            raise KnowledgeRegistryError(
                f"document version must be the next version: {expected}"
            )
        if value.status != DocumentStatus.DRAFT:
            raise KnowledgeRegistryError("new document versions must begin as draft")
        self._documents[key] = (*versions, value)
        return value

    def transition_document(
        self,
        business_id: str,
        knowledge_base_id: str,
        document_id: str,
        version: int,
        status: DocumentStatus,
    ) -> KnowledgeDocument:
        """Apply a legal lifecycle transition and return a new immutable value."""
        key = (business_id, knowledge_base_id, document_id)
        versions = self._documents.get(key)
        if versions is None or not 1 <= version <= len(versions):
            raise KnowledgeRegistryError("unknown document version")
        current = versions[version - 1]
        if status not in self._TRANSITIONS[current.status]:
            raise KnowledgeRegistryError("invalid document status transition")
        updated = replace(current, status=status)
        values = list(versions)
        if status == DocumentStatus.PUBLISHED:
            for index, item in enumerate(values):
                if item.status == DocumentStatus.PUBLISHED:
                    values[index] = replace(item, status=DocumentStatus.ARCHIVED)
        values[version - 1] = updated
        self._documents[key] = tuple(values)
        return updated

    def publish_document(
        self,
        business_id: str,
        knowledge_base_id: str,
        document_id: str,
        version: int,
    ) -> KnowledgeDocument:
        """Publish one draft; any prior published version becomes archived."""
        return self.transition_document(
            business_id,
            knowledge_base_id,
            document_id,
            version,
            DocumentStatus.PUBLISHED,
        )

    def archive_document(
        self,
        business_id: str,
        knowledge_base_id: str,
        document_id: str,
        version: int,
    ) -> KnowledgeDocument:
        """Archive one published version."""
        return self.transition_document(
            business_id,
            knowledge_base_id,
            document_id,
            version,
            DocumentStatus.ARCHIVED,
        )

    def versions(
        self, business_id: str, knowledge_base_id: str, document_id: str
    ) -> tuple[KnowledgeDocument, ...]:
        """Return immutable ordered versions for one business-owned document."""
        return self._documents.get(
            (business_id, knowledge_base_id, document_id), ()
        )

    def published_documents(
        self, business_id: str, knowledge_base_id: str
    ) -> tuple[KnowledgeDocument, ...]:
        """Return only documents eligible for future indexing."""
        self._require_base(business_id, knowledge_base_id)
        return tuple(
            item
            for key, versions in sorted(self._documents.items())
            if key[:2] == (business_id, knowledge_base_id)
            for item in versions
            if item.status == DocumentStatus.PUBLISHED
        )

    def select_active(
        self, selection: ActiveKnowledgeSelection
    ) -> ActiveKnowledgeSelection:
        """Select one existing same-business knowledge base for a campaign."""
        if not isinstance(selection, ActiveKnowledgeSelection):
            raise TypeError("selection must be an ActiveKnowledgeSelection")
        self._require_base(selection.business_id, selection.knowledge_base_id)
        self._selections[(selection.business_id, selection.campaign_id)] = selection
        return selection

    def active_for_campaign(
        self, business_id: str, campaign_id: str
    ) -> ActiveKnowledgeSelection | None:
        """Return only the selection belonging to the requested business."""
        return self._selections.get((business_id, campaign_id))

    @property
    def knowledge_bases(self) -> Mapping[tuple[str, str], KnowledgeBase]:
        """Return a read-only business-scoped knowledge-base view."""
        return MappingProxyType(dict(self._knowledge_bases))

    def _require_base(self, business_id: str, knowledge_base_id: str) -> KnowledgeBase:
        try:
            return self._knowledge_bases[(business_id, knowledge_base_id)]
        except KeyError as exc:
            raise KnowledgeRegistryError(
                "unknown knowledge base for business"
            ) from exc
