"""Focused tests for deterministic offline document preparation."""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime, timezone

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.knowledge.contracts import (
    DocumentFormat,
    DocumentStatus,
    KnowledgeCategory,
    KnowledgeDocument,
    KnowledgePurpose,
)
from app.knowledge.processing.contracts import (
    DocumentLanguage,
    DocumentProcessingRequest,
    ProcessingStatus,
)
from app.knowledge.processing.pipeline import (
    DocumentProcessingError,
    DocumentProcessingPipeline,
    ProcessingFailureKind,
)

_URDU_TEXT = (
    "\u06cc\u06c1 \u06c1\u0645\u0627\u0631\u06cc "
    "\u06a9\u0627\u0631\u0648\u0628\u0627\u0631\u06cc "
    "\u062e\u062f\u0645\u0627\u062a \u06c1\u06cc\u06ba"
)
_HINDI_TEXT = (
    "\u092f\u0939 \u0939\u092e\u093e\u0930\u0940 "
    "\u0935\u094d\u092f\u093e\u0935\u0938\u093e\u092f\u093f\u0915 "
    "\u0938\u0947\u0935\u093e \u0939\u0948"
)


def _document(
    content: bytes,
    document_format: DocumentFormat,
    source_name: str,
    *,
    business_id: str = "business-a",
    document_id: str = "guide",
    version: int = 1,
    purpose: KnowledgePurpose = KnowledgePurpose.SERVICE_KNOWLEDGE,
) -> KnowledgeDocument:
    return KnowledgeDocument(
        document_id,
        f"{document_id}-v{version}",
        "knowledge-main",
        business_id,
        "Business Guide",
        "Uploaded business knowledge.",
        KnowledgeCategory.SERVICES,
        ("business",),
        datetime(2026, 9, 23, tzinfo=timezone.utc),
        "user-1",
        version,
        DocumentStatus.PUBLISHED,
        hashlib.sha256(content).hexdigest(),
        document_format,
        source_name,
        purpose,
    )


def _process(
    content: bytes, document_format: DocumentFormat, source_name: str, **metadata
):  # type: ignore[no-untyped-def]
    document = _document(content, document_format, source_name, **metadata)
    return DocumentProcessingPipeline().process(
        DocumentProcessingRequest(document, content)
    )


def _docx(paragraphs: tuple[tuple[str, str | None], ...]) -> bytes:
    body = []
    for text, style in paragraphs:
        properties = (
            f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        )
        body.append(f"<w:p>{properties}<w:r><w:t>{text}</w:t></w:r></w:p>")
    document = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/'
        'wordprocessingml/2006/main"><w:body>'
        + "".join(body)
        + "</w:body></w:document>"
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", document)
    return output.getvalue()


def _pdf(text: str = "Services and business workflow.", *, encrypted: bool = False) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): font_ref}
            )
        }
    )
    stream = DecodedStreamObject()
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    if encrypted:
        writer.encrypt("password")
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def test_pdf_parsing_produces_ready_snapshot() -> None:
    snapshot = _process(_pdf(), DocumentFormat.PDF, "guide.pdf")
    assert snapshot.status == ProcessingStatus.READY_FOR_INDEXING
    assert "Services" in snapshot.chunks[0].text
    assert snapshot.chunks[0].page_number == 1


def test_docx_parsing_preserves_heading_hierarchy() -> None:
    content = _docx(
        (("Pricing", "Heading1"), ("Plans are provided on request.", None))
    )
    snapshot = _process(content, DocumentFormat.DOCX, "guide.docx")
    assert snapshot.sections[0].title == "Pricing"
    assert snapshot.sections[0].level == 1


def test_markdown_parsing_detects_ordered_sections() -> None:
    content = b"# Services\nWorkflow support.\n\n## FAQ\nAnswers for customers."
    snapshot = _process(content, DocumentFormat.MARKDOWN, "guide.md")
    assert [section.title for section in snapshot.sections] == ["Services", "FAQ"]
    assert [section.level for section in snapshot.sections] == [1, 2]


def test_txt_parsing_detects_known_section_headings() -> None:
    content = b"PRICING\n\nPricing is shared after qualification.\n\nFAQ\n\nCommon answers."
    snapshot = _process(content, DocumentFormat.TXT, "guide.txt")
    assert [section.title for section in snapshot.sections] == ["PRICING", "FAQ"]


def test_html_parsing_excludes_script_and_preserves_headings() -> None:
    content = (
        b"<html><script>ignore me</script><h1>Integrations</h1>"
        b"<p>Supported connections are documented here.</p></html>"
    )
    snapshot = _process(content, DocumentFormat.HTML, "guide.html")
    assert snapshot.sections[0].title == "Integrations"
    assert "ignore me" not in snapshot.chunks[0].text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("This is the business service guide.", DocumentLanguage.ENGLISH),
        (_URDU_TEXT, DocumentLanguage.URDU),
        (_HINDI_TEXT, DocumentLanguage.HINDI),
        ("Aap kaise hain aur kya chahiye", DocumentLanguage.ROMAN_URDU),
        ("\u06cc\u06c1 service business ke liye hai", DocumentLanguage.MIXED),
    ],
)
def test_language_detection(text: str, expected: DocumentLanguage) -> None:
    content = text.encode("utf-8")
    assert _process(content, DocumentFormat.TXT, "guide.txt").language == expected


def test_cleaning_removes_page_numbers_headers_and_broken_wrapping() -> None:
    content = b"Company Guide\n1\nThis line is\nwrapped badly\fCompany Guide\n2\nSecond page text."
    snapshot = _process(content, DocumentFormat.TXT, "guide.txt")
    combined = " ".join(chunk.text for chunk in snapshot.chunks)
    assert "Company Guide" not in combined
    assert "This line is wrapped badly" in combined
    assert "\n1\n" not in combined


def test_chunk_order_metadata_checksum_and_purpose_are_deterministic() -> None:
    content = ("Services\n\n" + "A useful sentence. " * 80).encode()
    document = _document(
        content,
        DocumentFormat.TXT,
        "guide.txt",
        purpose=KnowledgePurpose.SALES_KNOWLEDGE,
    )
    first = DocumentProcessingPipeline(max_chunk_chars=250).process(
        DocumentProcessingRequest(document, content)
    )
    second = DocumentProcessingPipeline(max_chunk_chars=250).process(
        DocumentProcessingRequest(document, content)
    )
    assert first == second
    assert len(first.chunks) > 1
    assert tuple(chunk.chunk_order for chunk in first.chunks) == tuple(
        range(len(first.chunks))
    )
    for chunk in first.chunks:
        assert chunk.purpose == KnowledgePurpose.SALES_KNOWLEDGE
        assert chunk.checksum == hashlib.sha256(chunk.text.encode()).hexdigest()
        assert chunk.document_version == 1


def test_processing_output_is_immutable_and_upload_bytes_are_not_retained() -> None:
    content = b"Services\n\nThis is the business service guide."
    snapshot = _process(content, DocumentFormat.TXT, "guide.txt")
    with pytest.raises(FrozenInstanceError):
        snapshot.status = ProcessingStatus.INDEXED  # type: ignore[misc]
    assert "content" not in {item.name for item in fields(snapshot)}
    assert "content" not in {item.name for item in fields(snapshot.chunks[0])}


def test_processing_lifecycle_stops_at_ready_for_indexing() -> None:
    content = b"This is the business service guide."
    snapshot = _process(content, DocumentFormat.TXT, "guide.txt")
    assert snapshot.lifecycle == (
        ProcessingStatus.UPLOADED,
        ProcessingStatus.PROCESSING,
        ProcessingStatus.PROCESSED,
        ProcessingStatus.READY_FOR_INDEXING,
    )
    assert ProcessingStatus.INDEXED not in snapshot.lifecycle


@pytest.mark.parametrize(
    ("content", "document_format", "source_name", "kind"),
    [
        (b"", DocumentFormat.TXT, "empty.txt", ProcessingFailureKind.EMPTY_DOCUMENT),
        (
            b"not a pdf",
            DocumentFormat.PDF,
            "broken.pdf",
            ProcessingFailureKind.FORMAT_MISMATCH,
        ),
        (
            b"%PDF-corrupt",
            DocumentFormat.PDF,
            "broken.pdf",
            ProcessingFailureKind.CORRUPT_DOCUMENT,
        ),
        (
            b"binary",
            DocumentFormat.TXT,
            "guide.exe",
            ProcessingFailureKind.UNSUPPORTED_FORMAT,
        ),
        (
            b"\xff\xfe\xfd",
            DocumentFormat.TXT,
            "guide.txt",
            ProcessingFailureKind.CORRUPT_DOCUMENT,
        ),
        (
            b"\n\n   \n",
            DocumentFormat.TXT,
            "guide.txt",
            ProcessingFailureKind.NO_EXTRACTABLE_TEXT,
        ),
    ],
)
def test_invalid_documents_fail_closed(  # type: ignore[no-untyped-def]
    content, document_format, source_name, kind
) -> None:
    document = _document(content, document_format, source_name)
    with pytest.raises(DocumentProcessingError) as caught:
        DocumentProcessingPipeline().process(
            DocumentProcessingRequest(document, content)
        )
    assert caught.value.kind == kind


def test_encrypted_pdf_is_rejected() -> None:
    content = _pdf(encrypted=True)
    document = _document(content, DocumentFormat.PDF, "private.pdf")
    with pytest.raises(DocumentProcessingError) as caught:
        DocumentProcessingPipeline().process(
            DocumentProcessingRequest(document, content)
        )
    assert caught.value.kind == ProcessingFailureKind.ENCRYPTED_DOCUMENT


def test_checksum_mismatch_and_duplicate_upload_are_rejected() -> None:
    content = b"This is the business service guide."
    document = _document(content, DocumentFormat.TXT, "guide.txt")
    with pytest.raises(DocumentProcessingError) as mismatch:
        DocumentProcessingPipeline().process(
            DocumentProcessingRequest(replace(document, checksum="b" * 64), content)
        )
    assert mismatch.value.kind == ProcessingFailureKind.CHECKSUM_MISMATCH
    pipeline = DocumentProcessingPipeline()
    pipeline.process(DocumentProcessingRequest(document, content))
    duplicate = replace(document, version=2, version_id="guide-v2")
    with pytest.raises(DocumentProcessingError) as repeated:
        pipeline.process(DocumentProcessingRequest(duplicate, content))
    assert repeated.value.kind == ProcessingFailureKind.DUPLICATE_UPLOAD


def test_processed_version_cannot_be_overwritten() -> None:
    first_content = b"This is the first business service guide."
    replacement_content = b"This is an unauthorized replacement guide."
    first = _document(first_content, DocumentFormat.TXT, "guide.txt")
    replacement = _document(
        replacement_content, DocumentFormat.TXT, "guide.txt"
    )
    pipeline = DocumentProcessingPipeline()
    snapshot = pipeline.process(DocumentProcessingRequest(first, first_content))
    with pytest.raises(DocumentProcessingError) as caught:
        pipeline.process(
            DocumentProcessingRequest(replacement, replacement_content)
        )
    assert caught.value.kind == ProcessingFailureKind.DUPLICATE_VERSION
    assert pipeline.snapshot("business-a", "knowledge-main", "guide", 1) == snapshot


def test_versions_keep_separate_immutable_snapshots() -> None:
    first_content = b"This is the first business service guide."
    second_content = b"This is the second business service guide."
    first = _document(first_content, DocumentFormat.TXT, "guide.txt")
    second = _document(
        second_content, DocumentFormat.TXT, "guide.txt", version=2
    )
    pipeline = DocumentProcessingPipeline()
    first_snapshot = pipeline.process(DocumentProcessingRequest(first, first_content))
    second_snapshot = pipeline.process(
        DocumentProcessingRequest(second, second_content)
    )
    assert pipeline.snapshot("business-a", "knowledge-main", "guide", 1) == (
        first_snapshot
    )
    assert pipeline.snapshot("business-a", "knowledge-main", "guide", 2) == (
        second_snapshot
    )
    assert first_snapshot.chunks != second_snapshot.chunks


def test_duplicate_tracking_and_snapshots_are_business_isolated() -> None:
    content = b"This is the shared business service guide."
    first = _document(content, DocumentFormat.TXT, "guide.txt")
    second = _document(
        content, DocumentFormat.TXT, "guide.txt", business_id="business-b"
    )
    pipeline = DocumentProcessingPipeline()
    first_snapshot = pipeline.process(DocumentProcessingRequest(first, content))
    second_snapshot = pipeline.process(DocumentProcessingRequest(second, content))
    assert first_snapshot.business_id == "business-a"
    assert second_snapshot.business_id == "business-b"
    assert pipeline.snapshot("business-a", "knowledge-main", "guide", 1) != (
        pipeline.snapshot("business-b", "knowledge-main", "guide", 1)
    )


def test_pipeline_has_no_retrieval_embedding_evidence_or_llm_surface() -> None:
    names = set(dir(DocumentProcessingPipeline))
    assert names.isdisjoint(
        {
            "embed",
            "retrieve",
            "search",
            "answer",
            "generate_evidence",
            "call_llm",
        }
    )
