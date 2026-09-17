from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass(frozen=True)
class DocumentPage:
    text: str
    page: int
    source: str


def load_document(file_path: Path) -> list[DocumentPage]:
    """Extract text while preserving source and page metadata."""
    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document type: {extension}")

    if extension == ".pdf":
        return _load_pdf(file_path)

    return _load_text(file_path)


def _load_pdf(file_path: Path) -> list[DocumentPage]:
    reader = PdfReader(str(file_path))
    pages: list[DocumentPage] = []

    for page_number, pdf_page in enumerate(reader.pages, start=1):
        text = (pdf_page.extract_text() or "").strip()
        if text:
            pages.append(
                DocumentPage(
                    text=text,
                    page=page_number,
                    source=file_path.name,
                )
            )

    return pages


def _load_text(file_path: Path) -> list[DocumentPage]:
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    return [DocumentPage(text=text, page=1, source=file_path.name)]
