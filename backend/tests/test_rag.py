from app.services.rag import NOT_FOUND_ANSWER, RagService
from app.services.retrieval import RetrievalResult


class FakeRetrievalService:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return self.results[:top_k]


class FakeGenerationService:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, question: str, context: str) -> str:
        self.calls += 1
        assert question in {"How do I restart?", "Unknown question"}
        assert "maintenance.pdf" in context
        return "Inspect the guard before restarting the machine."


def test_rag_generates_from_relevant_context() -> None:
    retrieval = FakeRetrievalService(
        [
            RetrievalResult(
                text="Inspect the guard before restarting the machine.",
                source="maintenance.pdf",
                page=14,
                chunk_index=2,
                similarity=0.82,
            )
        ]
    )
    generation = FakeGenerationService()
    service = RagService(retrieval, generation, min_similarity=0.35)

    result = service.answer("How do I restart?")

    assert result.grounded is True
    assert generation.calls == 1
    assert result.sources[0].page == 14
    assert result.sources[0].similarity == 0.82


def test_rag_refuses_generation_when_retrieval_is_weak() -> None:
    retrieval = FakeRetrievalService(
        [
            RetrievalResult(
                text="Unrelated content",
                source="maintenance.pdf",
                page=3,
                chunk_index=1,
                similarity=0.12,
            )
        ]
    )
    generation = FakeGenerationService()
    service = RagService(retrieval, generation, min_similarity=0.35)

    result = service.answer("Unknown question")

    assert result.grounded is False
    assert result.answer == NOT_FOUND_ANSWER
    assert result.sources == []
    assert generation.calls == 0
