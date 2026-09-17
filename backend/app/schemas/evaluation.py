from pydantic import BaseModel, Field


class EvaluationCaseRequest(BaseModel):
    question: str = Field(min_length=1)
    expected_source: str = Field(min_length=1)
    expected_page: int | None = Field(default=None, ge=1)


class EvaluationRequest(BaseModel):
    cases: list[EvaluationCaseRequest] = Field(min_length=1, max_length=100)
    top_k: int = Field(default=5, ge=1, le=20)


class EvaluationCaseResponse(BaseModel):
    question: str
    expected_source: str
    expected_page: int | None
    hit: bool
    rank: int | None
    best_similarity: float | None


class EvaluationResponse(BaseModel):
    total: int
    hits: int
    hit_rate: float
    mean_reciprocal_rank: float
    cases: list[EvaluationCaseResponse]
