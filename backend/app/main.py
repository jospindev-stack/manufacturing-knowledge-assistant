from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.evaluation import router as evaluation_router
from app.api.search import router as search_router

app = FastAPI(
    title="Manufacturing Knowledge Assistant",
    description="Local-first RAG API for manufacturing documentation.",
    version="0.5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(search_router)
app.include_router(chat_router)
app.include_router(evaluation_router)


@app.get("/api/health", tags=["health"])
def health_check() -> dict[str, str]:
    """Return the API health status."""
    return {
        "status": "ok",
        "service": "Manufacturing Knowledge Assistant",
    }
