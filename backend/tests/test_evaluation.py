import pytest

from app.services.evaluation import EvaluationCase, RetrievalEvaluationService
from app.services.retrieval import RetrievalResult


class FakeRetrievalService:
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if query == "restart procedure":
            return [
                RetrievalResult(
                    text="Inspect the guard before restart.",
                    source="maintenance.pdf",
                    page=14,
                    chunk_index=2,
                    similarity=0.91,
                )
            ]
        return [
            RetrievalResult(
                text="General safety information.",
                source="safety.pdf",
                page=2,
                chunk_index=0,
                similarity=0.63,
            )
        ]


def test_evaluation_calculates_hit_rate_and_mrr() -> None:
    evaluator = RetrievalEvaluationService(FakeRetrievalService())
    report = evaluator.evaluate(
        [
            EvaluationCase("restart procedure", "maintenance.pdf", 14),
            EvaluationCase("lubrication interval", "maintenance.pdf", 22),
        ],
        top_k=3,
    )

    assert report.total == 2
    assert report.hits == 1
    assert report.hit_rate == pytest.approx(0.5)
    assert report.mean_reciprocal_rank == pytest.approx(0.5)
    assert report.cases[0].rank == 1
    assert report.cases[1].rank is None
