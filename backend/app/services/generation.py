import json
from urllib import error, request


NOT_FOUND_SIGNAL = "__NOT_FOUND__"

SYSTEM_INSTRUCTIONS = f"""You are a manufacturing documentation assistant.
Answer the user's question using ONLY the supplied context.
Do not use outside knowledge and do not invent procedures, values, warnings, or facts.
If the supplied context does not contain the answer, return exactly {NOT_FOUND_SIGNAL} and nothing else.
Keep supported answers concise and operational. Source citations are handled separately by the application.
"""


class GenerationError(RuntimeError):
    """Raised when the local generation service cannot produce a response."""


class OllamaGenerationService:
    """Small explicit Ollama HTTP client; no orchestration framework is required."""

    def __init__(self, base_url: str, model: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, question: str, context: str) -> str:
        prompt = (
            f"{SYSTEM_INSTRUCTIONS}\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{question}\n\n"
            "ANSWER:"
        )
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            }
        ).encode("utf-8")

        http_request = request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise GenerationError(f"Ollama generation failed: {exc}") from exc

        answer = str(body.get("response", "")).strip()
        if not answer:
            raise GenerationError("Ollama returned an empty response.")

        return answer
