from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.services.document_loader import SUPPORTED_EXTENSIONS, load_document

router = APIRouter(prefix="/api/documents", tags=["documents"])

UPLOAD_DIR = Path("data/documents")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)) -> dict:
    """Store a supported document and extract its text with page metadata."""
    original_name = Path(file.filename or "document").name
    extension = Path(original_name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Supported formats are PDF, TXT and Markdown.",
        )

    stored_name = f"{uuid4().hex}{extension}"
    destination = UPLOAD_DIR / stored_name

    try:
        destination.write_bytes(await file.read())
        pages = load_document(destination)
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unable to process document: {exc}",
        ) from exc
    finally:
        await file.close()

    if not pages:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No extractable text was found in the document.",
        )

    return {
        "id": destination.stem,
        "filename": original_name,
        "pages": len(pages),
        "characters": sum(len(page.text) for page in pages),
        "status": "extracted",
    }
