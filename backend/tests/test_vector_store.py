from app.services.chunking import DocumentChunk
from app.services.vector_store import VectorStore


def test_reindexing_same_source_replaces_previous_chunks(tmp_path) -> None:
    store = VectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_reindexing",
    )

    first_chunks = [
        DocumentChunk(
            text="Old maintenance instruction",
            source="maintenance_manual.md",
            page=1,
            chunk_index=0,
        ),
        DocumentChunk(
            text="Old lubrication instruction",
            source="maintenance_manual.md",
            page=1,
            chunk_index=1,
        ),
    ]
    store.add_chunks("doc-old", first_chunks, [[1.0, 0.0], [0.9, 0.1]])

    replacement_chunks = [
        DocumentChunk(
            text="Current maintenance instruction",
            source="maintenance_manual.md",
            page=1,
            chunk_index=0,
        )
    ]
    store.add_chunks("doc-new", replacement_chunks, [[1.0, 0.0]])

    results = store.search([1.0, 0.0], top_k=5)

    assert len(results) == 1
    assert results[0]["text"] == "Current maintenance instruction"
    assert results[0]["metadata"]["document_id"] == "doc-new"
