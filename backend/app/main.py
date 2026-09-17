from fastapi import FastAPI

from app.api.documents import router as documents_router

app = FastAPI(
    title="Manufacturing Knowledge Assistant",
    description="Local-first RAG API for manufacturing documentation.",
    version="0.1.0",
)

app.include_router(documents_router)


@app.get("/api/health", tags=["health"])
def health_check() -> dict[str, str]:
    """Return the API health status."""
    return {
        "status": "ok",
        "service": "Manufacturing Knowledge Assistant",
    }
