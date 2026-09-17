from dataclasses import dataclass

from app.services.document_loader import DocumentPage


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    source: str
    page: int
    chunk_index: int


def chunk_pages(
    pages: list[DocumentPage],
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[DocumentChunk]:
    """Split document pages into overlapping character-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[DocumentChunk] = []

    for page in pages:
        start = 0
        chunk_index = 0
        text = page.text.strip()

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        source=page.source,
                        page=page.page,
                        chunk_index=chunk_index,
                    )
                )

            if end == len(text):
                break

            start = end - overlap
            chunk_index += 1

    return chunks
