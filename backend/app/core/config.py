from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings configurable through environment variables."""

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    retrieval_top_k: int = 5
    retrieval_min_similarity: float = 0.35

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
