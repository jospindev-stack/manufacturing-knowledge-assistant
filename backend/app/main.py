from fastapi import FastAPI

app = FastAPI(
    title="Manufacturing Knowledge Assistant",
    description="Local-first RAG API for manufacturing documentation.",
    version="0.1.0",
)


@app.get("/api/health", tags=["health"])
def health_check() -> dict[str, str]:
    """Return the API health status."""
    return {
        "status": "ok",
        "service": "Manufacturing Knowledge Assistant",
    }
