from fastapi import APIRouter, HTTPException, status

from app.schemas.evaluation import (
    EvaluationCaseResponse,
    EvaluationRequest,
    EvaluationResponse,
)
from app.services.embeddings import get_embedding_service
from app.services.evaluation import EvaluationCase, RetrievalEvaluationService
from app.services.retrieval import RetrievalService
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


@router.post("/retrieval", response_model=EvaluationResponse)
def evaluate_retrieval(payload: EvaluationRequest) -> EvaluationResponse:
    """Measure Top-K retrieval hit rate and mean reciprocal rank."""
    retrieval = RetrievalService(get_embedding_service(), VectorStore())
    evaluator = RetrievalEvaluationService(retrieval)

    cases = [
        EvaluationCase(
            question=case.question,
            expected_source=case.expected_source,
            expected_page=case.expected_page,
        )
        for case in payload.cases
    ]

    try:
        report = evaluator.evaluate(cases, top_k=payload.top_k)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return EvaluationResponse(
        total=report.total,
        hits=report.hits,
        hit_rate=round(report.hit_rate, 4),
        mean_reciprocal_rank=round(report.mean_reciprocal_rank, 4),
        cases=[
            EvaluationCaseResponse(
                question=case.question,
                expected_source=case.expected_source,
                expected_page=case.expected_page,
                hit=case.hit,
                rank=case.rank,
                best_similarity=(
                    round(case.best_similarity, 4)
                    if case.best_similarity is not None
                    else None
                ),
            )
            for case in report.cases
        ],
    )
