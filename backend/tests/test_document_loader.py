from pathlib import Path

import pytest

from app.services.document_loader import load_document


def test_load_text_document(tmp_path: Path) -> None:
    document = tmp_path / "work_instruction.txt"
    document.write_text("Inspect the safety guard before restart.", encoding="utf-8")

    pages = load_document(document)

    assert len(pages) == 1
    assert pages[0].text == "Inspect the safety guard before restart."
    assert pages[0].page == 1
    assert pages[0].source == "work_instruction.txt"


def test_reject_unsupported_document(tmp_path: Path) -> None:
    document = tmp_path / "manual.docx"
    document.write_text("Unsupported", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported document type"):
        load_document(document)
