from pathlib import Path

import chromadb

from app.services.chunking import DocumentChunk


class VectorStore:
    """Persistent ChromaDB storage for document chunks and their embeddings."""

    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "manufacturing_knowledge",
    ) -> None:
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        document_id: str,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return 0

        sources = {chunk.source for chunk in chunks}
        if len(sources) != 1:
            raise ValueError("all chunks in one indexing operation must share the same source")

        source = next(iter(sources))

        # Re-uploading the same logical document should replace its previous
        # vectors rather than accumulate duplicates that distort Top-K ranking.
        self._collection.delete(where={"source": source})

        ids = [f"{document_id}:{chunk.chunk_index}" for chunk in chunks]
        metadatas = [
            {
                "document_id": document_id,
                "source": chunk.source,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ]

        self._collection.add(
            ids=ids,
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        count = self._collection.count()
        if count == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        return [
            {
                "text": document,
                "metadata": metadata,
                "distance": distance,
                "similarity": 1.0 - distance,
            }
            for document, metadata, distance in zip(documents, metadatas, distances)
        ]
