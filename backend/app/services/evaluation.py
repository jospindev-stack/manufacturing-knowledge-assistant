from dataclasses import dataclass

from app.services.retrieval import RetrievalService


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    expected_source: str
    expected_page: int | None = None


@dataclass(frozen=True)
class EvaluationCaseResult:
    question: str
    expected_source: str
    expected_page: int | None
    hit: bool
    rank: int | None
    top_similarity: float | None


@dataclass(frozen=True)
class EvaluationReport:
    total: int
    hits: int
    hit_rate: float
    mean_reciprocal_rank: float
    cases: list[EvaluationCaseResult]


class RetrievalEvaluationService:
    """Evaluate whether expected evidence is present in Top-K retrieval results."""

    def __init__(self, retrieval_service: RetrievalService) -> None:
        self.retrieval_service = retrieval_service

    def evaluate(
        self,
        cases: list[EvaluationCase],
        top_k: int = 5,
    ) -> EvaluationReport:
        results: list[EvaluationCaseResult] = []

        for case in cases:
            retrieved = self.retrieval_service.retrieve(case.question, top_k=top_k)
            rank = None

            for index, item in enumerate(retrieved, start=1):
                source_matches = item.source == case.expected_source
                page_matches = case.expected_page is None or item.page == case.expected_page
                if source_matches and page_matches:
                    rank = index
                    break

            results.append(
                EvaluationCaseResult(
                    question=case.question,
                    expected_source=case.expected_source,
                    expected_page=case.expected_page,
                    hit=rank is not None,
                    rank=rank,
                    top_similarity=(retrieved[0].similarity if retrieved else None),
                )
            )

        hits = sum(result.hit for result in results)
        reciprocal_rank_sum = sum(
            1.0 / result.rank for result in results if result.rank is not None
        )
        total = len(results)

        return EvaluationReport(
            total=total,
            hits=hits,
            hit_rate=(hits / total if total else 0.0),
            mean_reciprocal_rank=(reciprocal_rank_sum / total if total else 0.0),
            cases=results,
        )
