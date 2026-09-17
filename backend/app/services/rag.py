from dataclasses import dataclass

from app.services.generation import OllamaGenerationService
from app.services.retrieval import RetrievalResult, RetrievalService


NOT_FOUND_ANSWER = "I couldn't find enough information in the available documents."


@dataclass(frozen=True)
class SourceCitation:
    source: str
    page: int
    chunk_index: int
    similarity: float


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    sources: list[SourceCitation]
    grounded: bool


class RagService:
    """Coordinate retrieval, relevance validation, context building and generation."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        generation_service: OllamaGenerationService,
        min_similarity: float,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.generation_service = generation_service
        self.min_similarity = min_similarity

    def answer(self, question: str, top_k: int = 5) -> RagAnswer:
        results = self.retrieval_service.retrieve(question, top_k=top_k)
        relevant = [result for result in results if result.similarity >= self.min_similarity]

        if not relevant:
            return RagAnswer(answer=NOT_FOUND_ANSWER, sources=[], grounded=False)

        context = self._build_context(relevant)
        answer = self.generation_service.generate(question=question, context=context)
        sources = self._citations(relevant)

        return RagAnswer(answer=answer, sources=sources, grounded=True)

    @staticmethod
    def _build_context(results: list[RetrievalResult]) -> str:
        sections = []
        for index, result in enumerate(results, start=1):
            sections.append(
                f"[SOURCE {index} | {result.source} | page {result.page}]\n{result.text}"
            )
        return "\n\n".join(sections)

    @staticmethod
    def _citations(results: list[RetrievalResult]) -> list[SourceCitation]:
        citations: list[SourceCitation] = []
        seen: set[tuple[str, int, int]] = set()

        for result in results:
            key = (result.source, result.page, result.chunk_index)
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                SourceCitation(
                    source=result.source,
                    page=result.page,
                    chunk_index=result.chunk_index,
                    similarity=result.similarity,
                )
            )

        return citations
