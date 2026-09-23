"""Offline format adapters producing deterministic text blocks without LLMs."""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import PurePath
from xml.etree import ElementTree

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.knowledge.contracts import DocumentFormat


class DocumentParseError(ValueError):
    """Source bytes are corrupt, encrypted, unsupported, or unreadable."""


@dataclass(frozen=True)
class ParsedBlock:
    text: str
    page_number: int | None = None
    heading_level: int | None = None


def detect_format(content: bytes, source_name: str) -> DocumentFormat:
    """Detect supported type from signature plus bounded source extension."""
    suffix = PurePath(source_name).suffix.casefold()
    by_extension = {
        ".pdf": DocumentFormat.PDF,
        ".docx": DocumentFormat.DOCX,
        ".md": DocumentFormat.MARKDOWN,
        ".markdown": DocumentFormat.MARKDOWN,
        ".txt": DocumentFormat.TXT,
        ".html": DocumentFormat.HTML,
        ".htm": DocumentFormat.HTML,
    }.get(suffix)
    if by_extension is None:
        raise DocumentParseError("unsupported document extension")
    if content.startswith(b"%PDF-"):
        detected = DocumentFormat.PDF
    elif content.startswith(b"PK\x03\x04"):
        detected = DocumentFormat.DOCX
    elif by_extension in {
        DocumentFormat.MARKDOWN,
        DocumentFormat.TXT,
        DocumentFormat.HTML,
    }:
        detected = by_extension
    else:
        raise DocumentParseError("document signature is invalid")
    if detected != by_extension:
        raise DocumentParseError("document extension and content disagree")
    return detected


def parse_document(content: bytes, document_format: DocumentFormat) -> tuple[ParsedBlock, ...]:
    """Parse supported bytes into ordered blocks without retaining the upload."""
    parsers = {
        DocumentFormat.PDF: _parse_pdf,
        DocumentFormat.DOCX: _parse_docx,
        DocumentFormat.MARKDOWN: _parse_markdown,
        DocumentFormat.TXT: _parse_text,
        DocumentFormat.HTML: _parse_html,
    }
    try:
        return parsers[document_format](content)
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError("document is corrupt or unreadable") from exc


def _decode(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentParseError("text document must use UTF-8") from exc


def _parse_pdf(content: bytes) -> tuple[ParsedBlock, ...]:
    try:
        reader = PdfReader(io.BytesIO(content), strict=True)
    except (PdfReadError, ValueError, OSError) as exc:
        raise DocumentParseError("PDF is corrupt") from exc
    if reader.is_encrypted:
        raise DocumentParseError("encrypted PDF is unsupported")
    blocks = []
    try:
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            blocks.append(ParsedBlock(text, number))
    except Exception as exc:
        raise DocumentParseError("PDF text extraction failed") from exc
    return tuple(blocks)


def _parse_docx(content: bytes) -> tuple[ParsedBlock, ...]:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = frozenset(archive.namelist())
            if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                raise DocumentParseError("DOCX package is incomplete")
            root = ElementTree.fromstring(archive.read("word/document.xml"))
    except (zipfile.BadZipFile, ElementTree.ParseError, KeyError) as exc:
        raise DocumentParseError("DOCX is corrupt") from exc
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    blocks = []
    page = 1
    for paragraph in root.iter(f"{namespace}p"):
        text = "".join(node.text or "" for node in paragraph.iter(f"{namespace}t"))
        if not text.strip():
            continue
        style = paragraph.find(f".//{namespace}pStyle")
        style_name = style.get(f"{namespace}val", "") if style is not None else ""
        match = re.fullmatch(r"Heading([1-6])", style_name, re.IGNORECASE)
        blocks.append(
            ParsedBlock(text, page, int(match.group(1)) if match else None)
        )
        page += len(paragraph.findall(f".//{namespace}br[@{namespace}type='page']"))
    return tuple(blocks)


def _parse_markdown(content: bytes) -> tuple[ParsedBlock, ...]:
    blocks = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            blocks.append(ParsedBlock("\n".join(buffer)))
            buffer.clear()

    for line in _decode(content).splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*$", line.strip())
        if heading:
            flush()
            blocks.append(ParsedBlock(heading.group(2), None, len(heading.group(1))))
        elif line.strip():
            buffer.append(line)
        else:
            flush()
    flush()
    return tuple(blocks)


def _parse_text(content: bytes) -> tuple[ParsedBlock, ...]:
    pages = _decode(content).split("\f")
    numbered = len(pages) > 1
    return tuple(
        ParsedBlock(part, page_number if numbered else None)
        for page_number, page in enumerate(pages, start=1)
        for part in re.split(r"\n\s*\n", page)
    )


class _HTMLTextParser(HTMLParser):
    _BLOCKS = frozenset({"p", "li", "div", "section", "article"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[ParsedBlock] = []
        self._text: list[str] = []
        self._level: int | None = None
        self._ignored = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[no-untyped-def]
        tag = tag.casefold()
        if tag in {"script", "style", "noscript"}:
            self._ignored += 1
        if self._ignored:
            return
        if re.fullmatch(r"h[1-6]", tag):
            self._flush()
            self._level = int(tag[1])
        elif tag in self._BLOCKS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in {"script", "style", "noscript"} and self._ignored:
            self._ignored -= 1
            return
        if not self._ignored and (
            re.fullmatch(r"h[1-6]", tag) or tag in self._BLOCKS
        ):
            self._flush()

    def handle_data(self, data: str) -> None:
        if not self._ignored:
            self._text.append(data)

    def close(self) -> None:
        super().close()
        self._flush()

    def _flush(self) -> None:
        text = " ".join(" ".join(self._text).split())
        if text:
            self.blocks.append(ParsedBlock(text, None, self._level))
        self._text.clear()
        self._level = None


def _parse_html(content: bytes) -> tuple[ParsedBlock, ...]:
    parser = _HTMLTextParser()
    try:
        parser.feed(_decode(content))
        parser.close()
    except Exception as exc:
        raise DocumentParseError("HTML is corrupt") from exc
    return tuple(parser.blocks)
