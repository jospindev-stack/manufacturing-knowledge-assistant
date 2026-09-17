from fastapi import APIRouter, HTTPException, Query, status

from app.services.embeddings import get_embedding_service
from app.services.retrieval import RetrievalService
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/api/search", tags=["retrieval"])


@router.get("")
def semantic_search(
    q: str = Query(min_length=1, description="Natural-language search query"),
    top_k: int = Query(default=5, ge=1, le=20),
) -> dict:
    """Return the most semantically similar chunks without invoking an LLM."""
    try:
        service = RetrievalService(get_embedding_service(), VectorStore())
        results = service.retrieve(q, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return {
        "query": q,
        "top_k": top_k,
        "results": [
            {
                "text": result.text,
                "source": result.source,
                "page": result.page,
                "chunk_index": result.chunk_index,
                "similarity": round(result.similarity, 4),
            }
            for result in results
        ],
    }
