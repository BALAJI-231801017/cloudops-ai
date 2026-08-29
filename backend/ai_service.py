"""Ollama-backed analysis constrained to supplied monitoring evidence."""

import ollama

from backend.config import Settings


class AiServiceError(RuntimeError):
    """Raised when Ollama is unavailable."""


class AiService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def analyze(self, question: str, evidence: dict):
        prompt = (
            "You are CloudOps AI. Explain only the supplied monitoring evidence. "
            "State uncertainty when evidence is missing. Give a concise diagnosis, likely causes, "
            "and practical next actions.\n\n"
            f"Question: {question}\n\nEvidence: {evidence}"
        )
        try:
            response = ollama.Client(host=self.settings.ollama_host).chat(
                model=self.settings.ollama_model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"]
        except Exception as error:
            raise AiServiceError(
                "Unable to reach Ollama. Start Ollama and confirm the configured model is available."
            ) from error
