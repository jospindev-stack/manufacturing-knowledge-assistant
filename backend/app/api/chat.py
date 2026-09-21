from fastapi import APIRouter, HTTPException, status

from app.core.config import get_settings
from app.schemas.chat import ChatRequest, ChatResponse, SourceResponse
from app.services.embeddings import get_embedding_service
from app.services.generation import GenerationError, OllamaGenerationService
from app.services.rag import RagService
from app.services.retrieval import RetrievalService
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/api/chat", tags=["rag"])


@router.post("", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    """Answer a question using only relevant indexed manufacturing documents."""
    settings = get_settings()
    retrieval = RetrievalService(get_embedding_service(), VectorStore())
    generation = OllamaGenerationService(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
    )
    rag = RagService(
        retrieval_service=retrieval,
        generation_service=generation,
        min_similarity=settings.retrieval_min_similarity,
    )

    try:
        result = rag.answer(\n            payload.question,\n            top_k=payload.top_k or settings.retrieval_top_k,\n        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except GenerationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return ChatResponse(
        answer=result.answer,
        grounded=result.grounded,
        sources=[
            SourceResponse(
                source=source.source,
                page=source.page,
                chunk_index=source.chunk_index,
                similarity=round(source.similarity, 4),
            )
            for source in result.sources
        ],
    )
