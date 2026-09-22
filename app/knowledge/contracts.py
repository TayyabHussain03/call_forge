"""Immutable bounded contracts for business-owned knowledge metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
_CHECKSUM = re.compile(r"^[a-f0-9]{64}$")
_TAG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")


class DocumentFormat(str, Enum):
    """Supported source formats; parsing is deliberately outside this slice."""

    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    TXT = "txt"
    HTML = "html"


class KnowledgeCategory(str, Enum):
    """Extensible bounded taxonomy for business knowledge."""

    SERVICES = "services"
    PRODUCTS = "products"
    POLICIES = "policies"
    PRICING = "pricing"
    FAQS = "faqs"
    CASE_STUDIES = "case_studies"
    INTEGRATIONS = "integrations"
    SALES_PLAYBOOKS = "sales_playbooks"
    TECHNICAL_GUIDES = "technical_guides"
    TRAINING = "training"
    SOP = "sop"
    OTHER = "other"


class DocumentStatus(str, Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


@dataclass(frozen=True)
class KnowledgeBase:
    """One named knowledge collection owned by exactly one business."""

    knowledge_base_id: str
    business_id: str
    name: str
    created_at: datetime
    description: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.knowledge_base_id, "knowledge base id")
        _identifier(self.business_id, "business id")
        _text(self.name, "knowledge base name", 120)
        _timestamp(self.created_at, "knowledge base creation time")
        if self.description is not None:
            _text(self.description, "knowledge base description", 500)


@dataclass(frozen=True)
class KnowledgeDocument:
    """One immutable document version containing metadata but no file contents."""

    document_id: str
    version_id: str
    knowledge_base_id: str
    business_id: str
    title: str
    description: str | None
    category: KnowledgeCategory
    tags: tuple[str, ...]
    uploaded_at: datetime
    uploaded_by: str
    version: int
    status: DocumentStatus
    checksum: str
    document_format: DocumentFormat
    source_name: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.document_id, "document id"),
            (self.version_id, "version id"),
            (self.knowledge_base_id, "knowledge base id"),
            (self.business_id, "business id"),
            (self.uploaded_by, "uploader id"),
        ):
            _identifier(value, name)
        _text(self.title, "document title", 160)
        if self.description is not None:
            _text(self.description, "document description", 500)
        _text(self.source_name, "source name", 200)
        if not isinstance(self.category, KnowledgeCategory):
            raise TypeError("document category must be a KnowledgeCategory")
        if not isinstance(self.status, DocumentStatus):
            raise TypeError("document status must be a DocumentStatus")
        if not isinstance(self.document_format, DocumentFormat):
            raise TypeError("document format must be a DocumentFormat")
        tags = tuple(self.tags)
        if len(tags) > 12 or any(not _TAG.fullmatch(tag) for tag in tags):
            raise ValueError("document tags must be normalized and bounded")
        if len(set(tags)) != len(tags):
            raise ValueError("document tags must be unique")
        if self.version < 1:
            raise ValueError("document version must be positive")
        if not _CHECKSUM.fullmatch(self.checksum):
            raise ValueError("checksum must be a lowercase SHA-256 digest")
        _timestamp(self.uploaded_at, "document upload time")
        object.__setattr__(self, "tags", tags)

    @property
    def active(self) -> bool:
        """Return whether this version is the currently published version."""
        return self.status == DocumentStatus.PUBLISHED

    @property
    def published(self) -> bool:
        """Return whether this version is eligible for future indexing."""
        return self.status == DocumentStatus.PUBLISHED

    @property
    def archived(self) -> bool:
        """Return whether this version is historical and ineligible."""
        return self.status == DocumentStatus.ARCHIVED


@dataclass(frozen=True)
class ActiveKnowledgeSelection:
    """One campaign's active knowledge base, without loading its documents."""

    business_id: str
    campaign_id: str
    knowledge_base_id: str

    def __post_init__(self) -> None:
        _identifier(self.business_id, "business id")
        _identifier(self.campaign_id, "campaign id")
        _identifier(self.knowledge_base_id, "knowledge base id")


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{name} must contain 1-100 safe characters")


def _text(value: str, name: str, limit: int) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must contain 1-{limit} characters")


def _timestamp(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
