"""Immutable bounded contracts for prepared knowledge snapshots."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from app.knowledge.contracts import (
    KnowledgeCategory,
    KnowledgeDocument,
    KnowledgePurpose,
)

_CHECKSUM = re.compile(r"^[a-f0-9]{64}$")


class ProcessingStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    READY_FOR_INDEXING = "ready_for_indexing"
    INDEXED = "indexed"


class DocumentLanguage(str, Enum):
    ENGLISH = "english"
    URDU = "urdu"
    HINDI = "hindi"
    ROMAN_URDU = "roman_urdu"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DocumentProcessingRequest:
    """Ephemeral upload bytes paired with authoritative version metadata."""

    document: KnowledgeDocument
    content: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.document, KnowledgeDocument):
            raise TypeError("processing document must be a KnowledgeDocument")
        if not isinstance(self.content, bytes):
            raise TypeError("document content must be bytes")


@dataclass(frozen=True)
class DocumentSection:
    """One detected logical section before chunking."""

    title: str
    level: int
    order: int
    page_number: int | None
    text: str

    def __post_init__(self) -> None:
        if not self.title.strip() or len(self.title) > 200:
            raise ValueError("section title must be bounded")
        if not 1 <= self.level <= 6:
            raise ValueError("section level must be between one and six")
        if self.order < 0:
            raise ValueError("section order must be non-negative")
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("page number must be positive")
        if not self.text.strip() or len(self.text) > 100_000:
            raise ValueError("section text must be non-empty and bounded")


@dataclass(frozen=True)
class KnowledgeChunk:
    """Immutable ordered text unit for future indexing, not evidence."""

    chunk_id: str
    business_id: str
    knowledge_base_id: str
    document_id: str
    version_id: str
    document_version: int
    page_number: int | None
    section_title: str
    section_level: int
    category: KnowledgeCategory
    purpose: KnowledgePurpose
    language: DocumentLanguage
    chunk_order: int
    checksum: str
    source_reference: str
    text: str

    def __post_init__(self) -> None:
        if self.document_version < 1 or self.chunk_order < 0:
            raise ValueError("chunk version and order are invalid")
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("chunk page number must be positive")
        if not 1 <= self.section_level <= 6:
            raise ValueError("chunk section level is invalid")
        if not _CHECKSUM.fullmatch(self.checksum):
            raise ValueError("chunk checksum must be SHA-256")
        for value, name, limit in (
            (self.chunk_id, "chunk id", 180),
            (self.section_title, "section title", 200),
            (self.source_reference, "source reference", 300),
            (self.text, "chunk text", 2_000),
        ):
            if not value.strip() or len(value) > limit:
                raise ValueError(f"{name} must be non-empty and bounded")


@dataclass(frozen=True)
class KnowledgeSnapshot:
    """Complete deterministic output eligible for a future indexing boundary."""

    business_id: str
    knowledge_base_id: str
    document_id: str
    version_id: str
    document_version: int
    source_checksum: str
    language: DocumentLanguage
    sections: tuple[DocumentSection, ...]
    chunks: tuple[KnowledgeChunk, ...]
    status: ProcessingStatus
    lifecycle: tuple[ProcessingStatus, ...]

    def __post_init__(self) -> None:
        sections = tuple(self.sections)
        chunks = tuple(self.chunks)
        lifecycle = tuple(self.lifecycle)
        if not sections or not chunks:
            raise ValueError("ready snapshot requires sections and chunks")
        if self.status != ProcessingStatus.READY_FOR_INDEXING:
            raise ValueError("processing pipeline may only emit ready snapshots")
        if not _CHECKSUM.fullmatch(self.source_checksum):
            raise ValueError("snapshot source checksum must be SHA-256")
        expected = (
            ProcessingStatus.UPLOADED,
            ProcessingStatus.PROCESSING,
            ProcessingStatus.PROCESSED,
            ProcessingStatus.READY_FOR_INDEXING,
        )
        if lifecycle != expected:
            raise ValueError("processing lifecycle is invalid")
        if tuple(item.order for item in sections) != tuple(range(len(sections))):
            raise ValueError("section ordering must be contiguous")
        if tuple(item.chunk_order for item in chunks) != tuple(range(len(chunks))):
            raise ValueError("chunk ordering must be contiguous")
        if any(
            (
                item.business_id,
                item.knowledge_base_id,
                item.document_id,
                item.version_id,
                item.document_version,
            )
            != (
                self.business_id,
                self.knowledge_base_id,
                self.document_id,
                self.version_id,
                self.document_version,
            )
            for item in chunks
        ):
            raise ValueError("snapshot chunks must belong to the same document version")
        object.__setattr__(self, "sections", sections)
        object.__setattr__(self, "chunks", chunks)
        object.__setattr__(self, "lifecycle", lifecycle)
