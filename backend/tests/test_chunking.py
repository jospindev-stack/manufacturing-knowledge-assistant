import pytest

from app.services.chunking import chunk_pages
from app.services.document_loader import DocumentPage


def test_chunk_pages_preserves_metadata_and_overlap() -> None:
    pages = [DocumentPage(text="abcdefghij", page=3, source="manual.txt")]

    chunks = chunk_pages(pages, chunk_size=6, overlap=2)

    assert [chunk.text for chunk in chunks] == ["abcdef", "efghij"]
    assert all(chunk.page == 3 for chunk in chunks)
    assert all(chunk.source == "manual.txt" for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]


def test_chunk_pages_rejects_invalid_overlap() -> None:
    pages = [DocumentPage(text="example", page=1, source="manual.txt")]

    with pytest.raises(ValueError, match="overlap"):
        chunk_pages(pages, chunk_size=5, overlap=5)
