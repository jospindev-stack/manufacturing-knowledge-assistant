from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.services.chunking import chunk_pages
from app.services.document_loader import SUPPORTED_EXTENSIONS, load_document
from app.services.embeddings import get_embedding_service
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/api/documents", tags=["documents"])

UPLOAD_DIR = Path("data/documents")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = 500,
    overlap: int = 100,
) -> dict:
    """Extract, chunk, embed and index a supported document."""
    original_name = Path(file.filename or "document").name
    extension = Path(original_name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Supported formats are PDF, TXT and Markdown.",
        )

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="chunk_size must be > 0 and overlap must be >= 0 and smaller than chunk_size.",
        )

    document_id = uuid4().hex
    destination = UPLOAD_DIR / f"{document_id}{extension}"

    try:
        destination.write_bytes(await file.read())
        pages = load_document(destination)

        if not pages:
            raise ValueError("No extractable text was found in the document.")

        # Keep the user-facing source name in vector metadata.
        pages = [
            type(page)(text=page.text, page=page.page, source=original_name)
            for page in pages
        ]
        chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
        embeddings = get_embedding_service().embed_documents(
            [chunk.text for chunk in chunks]
        )
        indexed_chunks = VectorStore().add_chunks(document_id, chunks, embeddings)
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unable to process document: {exc}",
        ) from exc
    finally:
        await file.close()

    return {
        "id": document_id,
        "filename": original_name,
        "pages": len(pages),
        "characters": sum(len(page.text) for page in pages),
        "chunks": indexed_chunks,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "status": "indexed",
    }
