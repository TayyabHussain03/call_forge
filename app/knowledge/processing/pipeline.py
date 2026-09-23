"""Deterministic document validation, extraction, cleaning, and chunking pipeline."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import replace
from enum import Enum

from app.knowledge.contracts import KnowledgeDocument
from app.knowledge.processing.contracts import (
    DocumentLanguage,
    DocumentProcessingRequest,
    DocumentSection,
    KnowledgeChunk,
    KnowledgeSnapshot,
    ProcessingStatus,
)
from app.knowledge.processing.language import detect_language
from app.knowledge.processing.parsers import (
    DocumentParseError,
    ParsedBlock,
    detect_format,
    parse_document,
)

_PAGE_NUMBER = re.compile(
    r"^(?:page\s+)?\d+(?:\s+(?:of|/)\s+\d+)?$", re.IGNORECASE
)
_KNOWN_HEADING = re.compile(
    r"^(?:pricing|features?|integrations?|faqs?|frequently asked questions|"
    r"case studies|services?|products?|policies|technical guides?|training|sop)$",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_LIFECYCLE = (
    ProcessingStatus.UPLOADED,
    ProcessingStatus.PROCESSING,
    ProcessingStatus.PROCESSED,
    ProcessingStatus.READY_FOR_INDEXING,
)


class ProcessingFailureKind(str, Enum):
    EMPTY_DOCUMENT = "empty_document"
    TOO_LARGE = "too_large"
    UNSUPPORTED_FORMAT = "unsupported_format"
    FORMAT_MISMATCH = "format_mismatch"
    CORRUPT_DOCUMENT = "corrupt_document"
    ENCRYPTED_DOCUMENT = "encrypted_document"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    DUPLICATE_UPLOAD = "duplicate_upload"
    DUPLICATE_VERSION = "duplicate_version"
    NO_EXTRACTABLE_TEXT = "no_extractable_text"


class DocumentProcessingError(ValueError):
    """Typed fail-closed document processing failure."""

    def __init__(self, message: str, kind: ProcessingFailureKind) -> None:
        super().__init__(message)
        self.kind = kind


class DocumentProcessingPipeline:
    """Offline synchronous preparation with isolated duplicate tracking."""

    def __init__(self, *, max_upload_bytes: int = 10_000_000, max_chunk_chars: int = 1_200) -> None:
        if not 1 <= max_upload_bytes <= 50_000_000:
            raise ValueError("upload limit must be between one and 50 MB")
        if not 200 <= max_chunk_chars <= 2_000:
            raise ValueError("chunk size must be between 200 and 2,000 characters")
        self._max_upload_bytes = max_upload_bytes
        self._max_chunk_chars = max_chunk_chars
        self._checksums: set[tuple[str, str, str]] = set()
        self._snapshots: dict[tuple[str, str, str, int], KnowledgeSnapshot] = {}

    def process(self, request: DocumentProcessingRequest) -> KnowledgeSnapshot:
        """Create one ready snapshot or fail without retaining partial output."""
        document = request.document
        content = request.content
        self._validate_upload(document, content)
        version_key = (
            document.business_id,
            document.knowledge_base_id,
            document.document_id,
            document.version,
        )
        if version_key in self._snapshots:
            raise DocumentProcessingError(
                "document version was already processed",
                ProcessingFailureKind.DUPLICATE_VERSION,
            )
        checksum_key = (
            document.business_id,
            document.knowledge_base_id,
            document.checksum,
        )
        if checksum_key in self._checksums:
            raise DocumentProcessingError(
                "duplicate upload checksum", ProcessingFailureKind.DUPLICATE_UPLOAD
            )
        detected = self._detect(content, document.source_name)
        if detected != document.document_format:
            raise DocumentProcessingError(
                "detected format does not match metadata",
                ProcessingFailureKind.FORMAT_MISMATCH,
            )
        blocks = self._parse(content, detected)
        cleaned = _clean_blocks(blocks)
        if not cleaned or not any(item.text.strip() for item in cleaned):
            raise DocumentProcessingError(
                "document has no extractable text",
                ProcessingFailureKind.NO_EXTRACTABLE_TEXT,
            )
        sections = _build_sections(cleaned, document.title)
        if not sections:
            raise DocumentProcessingError(
                "document has no extractable sections",
                ProcessingFailureKind.NO_EXTRACTABLE_TEXT,
            )
        language = detect_language(" ".join(item.text for item in sections))
        chunks = _build_chunks(
            document, sections, language, self._max_chunk_chars
        )
        snapshot = KnowledgeSnapshot(
            document.business_id,
            document.knowledge_base_id,
            document.document_id,
            document.version_id,
            document.version,
            document.checksum,
            language,
            sections,
            chunks,
            ProcessingStatus.READY_FOR_INDEXING,
            _LIFECYCLE,
        )
        self._checksums.add(checksum_key)
        self._snapshots[version_key] = snapshot
        return snapshot

    def snapshot(
        self,
        business_id: str,
        knowledge_base_id: str,
        document_id: str,
        version: int,
    ) -> KnowledgeSnapshot | None:
        """Return one isolated immutable snapshot without retrieval semantics."""
        return self._snapshots.get(
            (business_id, knowledge_base_id, document_id, version)
        )

    def _validate_upload(self, document: KnowledgeDocument, content: bytes) -> None:
        if not content:
            raise DocumentProcessingError(
                "document upload is empty", ProcessingFailureKind.EMPTY_DOCUMENT
            )
        if len(content) > self._max_upload_bytes:
            raise DocumentProcessingError(
                "document upload exceeds size limit", ProcessingFailureKind.TOO_LARGE
            )
        checksum = hashlib.sha256(content).hexdigest()
        if checksum != document.checksum:
            raise DocumentProcessingError(
                "document checksum does not match upload",
                ProcessingFailureKind.CHECKSUM_MISMATCH,
            )

    @staticmethod
    def _detect(content: bytes, source_name: str):  # type: ignore[no-untyped-def]
        try:
            return detect_format(content, source_name)
        except DocumentParseError as exc:
            kind = (
                ProcessingFailureKind.UNSUPPORTED_FORMAT
                if "unsupported" in str(exc)
                else ProcessingFailureKind.FORMAT_MISMATCH
            )
            raise DocumentProcessingError(str(exc), kind) from exc

    @staticmethod
    def _parse(content: bytes, document_format):  # type: ignore[no-untyped-def]
        try:
            return parse_document(content, document_format)
        except DocumentParseError as exc:
            message = str(exc)
            kind = (
                ProcessingFailureKind.ENCRYPTED_DOCUMENT
                if "encrypted" in message
                else ProcessingFailureKind.CORRUPT_DOCUMENT
            )
            raise DocumentProcessingError(message, kind) from exc


def _clean_blocks(blocks: tuple[ParsedBlock, ...]) -> tuple[ParsedBlock, ...]:
    pages: dict[int, list[str]] = {}
    for block in blocks:
        if block.page_number is not None:
            pages.setdefault(block.page_number, []).extend(
                " ".join(line.split())
                for line in block.text.splitlines()
                if line.strip()
            )
    page_lines = list(pages.values())
    repeated = set()
    if len(page_lines) > 1:
        edges = [
            line
            for lines in page_lines
            for line in ({lines[0], lines[-1]} if lines else set())
        ]
        repeated = {line for line, count in Counter(edges).items() if count > 1}
    result = []
    for block in blocks:
        lines = [" ".join(line.split()) for line in block.text.splitlines()]
        lines = [
            line
            for line in lines
            if line and line not in repeated and not _PAGE_NUMBER.fullmatch(line)
        ]
        text = _join_wrapped_lines(lines)
        if text:
            result.append(replace(block, text=text))
    return tuple(result)


def _join_wrapped_lines(lines: list[str]) -> str:
    paragraphs: list[str] = []
    current = ""
    for line in lines:
        if not current:
            current = line
        elif current.endswith((".", "!", "?", ":", ";")):
            paragraphs.append(current)
            current = line
        else:
            current = f"{current} {line}"
    if current:
        paragraphs.append(current)
    return "\n".join(paragraphs)


def _build_sections(
    blocks: tuple[ParsedBlock, ...], default_title: str
) -> tuple[DocumentSection, ...]:
    sections: list[DocumentSection] = []
    title = default_title
    level = 1
    page: int | None = None
    content: list[str] = []

    def flush() -> None:
        if content:
            sections.append(
                DocumentSection(title, level, len(sections), page, "\n".join(content))
            )
            content.clear()

    for block in blocks:
        lines = block.text.splitlines()
        if block.heading_level is not None:
            flush()
            title, level, page = block.text, block.heading_level, block.page_number
            continue
        for line in lines:
            if _is_heading(line):
                flush()
                title, level, page = line.rstrip(":"), 2, block.page_number
            else:
                if page is None:
                    page = block.page_number
                content.append(line)
    flush()
    return tuple(sections)


def _is_heading(text: str) -> bool:
    value = text.strip()
    return bool(
        value
        and len(value) <= 80
        and (
            _KNOWN_HEADING.fullmatch(value.rstrip(":"))
            or (value.isupper() and any(character.isalpha() for character in value))
        )
    )


def _build_chunks(
    document: KnowledgeDocument,
    sections: tuple[DocumentSection, ...],
    language: DocumentLanguage,
    limit: int,
) -> tuple[KnowledgeChunk, ...]:
    chunks = []
    for section in sections:
        for text in _split_text(section.text, limit):
            order = len(chunks)
            checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
            chunks.append(
                KnowledgeChunk(
                    f"{document.version_id}-chunk-{order}",
                    document.business_id,
                    document.knowledge_base_id,
                    document.document_id,
                    document.version_id,
                    document.version,
                    section.page_number,
                    section.title,
                    section.level,
                    document.category,
                    document.purpose,
                    language,
                    order,
                    checksum,
                    f"{document.source_name}#section-{section.order}-chunk-{order}",
                    text,
                )
            )
    return tuple(chunks)


def _split_text(text: str, limit: int) -> tuple[str, ...]:
    units = [part.strip() for part in _SENTENCE.split(text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for unit in units:
        pieces = (
            tuple(unit[index : index + limit] for index in range(0, len(unit), limit))
            if len(unit) > limit
            else (unit,)
        )
        for piece in pieces:
            candidate = f"{current} {piece}".strip()
            if current and len(candidate) > limit:
                chunks.append(current)
                current = piece
            else:
                current = candidate
    if current:
        chunks.append(current)
    return tuple(chunks)
