from dataclasses import dataclass

from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore


@dataclass(frozen=True)
class RetrievalResult:
    text: str
    source: str
    page: int
    chunk_index: int
    similarity: float


class RetrievalService:
    """Embed a user query and retrieve semantically similar document chunks."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        query_embedding = self.embedding_service.embed_query(query)
        matches = self.vector_store.search(query_embedding, top_k=top_k)

        return [
            RetrievalResult(
                text=match["text"],
                source=str(match["metadata"]["source"]),
                page=int(match["metadata"]["page"]),
                chunk_index=int(match["metadata"]["chunk_index"]),
                similarity=float(match["similarity"]),
            )
            for match in matches
        ]
