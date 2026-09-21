from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceResponse(BaseModel):
    source: str
    page: int
    chunk_index: int
    similarity: float


class ChatResponse(BaseModel):
    answer: str
    grounded: bool
    sources: list[SourceResponse]
