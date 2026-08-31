"""Ollama/Llama-backed infrastructure diagnosis grounded strictly in structured evidence."""

import json
import logging
import re
from typing import Any
import ollama

from backend.config import Settings

logger = logging.getLogger("cloudops.ai_service")


class AiServiceError(RuntimeError):
    """Raised when Ollama inference cannot be completed."""


SYSTEM_PROMPT = """You are CloudOps AI, an expert cloud infrastructure troubleshooting assistant.
Your responsibility is to analyze the provided monitoring evidence and explain the incident with high fidelity.

CRITICAL OPERATIONAL RULES:
1. Grounding: Rely EXCLUSIVELY on the telemetry metrics, threshold violations, and detector evidence provided.
2. Anti-Hallucination: Never invent metrics, server logs, network traces, AWS resource IDs, or operational events not in the evidence.
3. Authority: Never alter or override the deterministic incident severity or classification provided in the evidence.
4. Advisory: Recommend safe, actionable diagnostic and remediation steps. Never claim to have taken autonomous remediation actions.
5. Uncertainty: If the evidence is insufficient to identify an exact root cause with certainty, state the limitation clearly.

Output Format:
You MUST respond with a valid JSON object matching this schema:
{
  "incident_summary": "Concise summary of the situation based on evidence",
  "probable_root_cause": "Most probable technical explanation for the observed metrics",
  "confidence": 0.85,
  "supporting_evidence": ["Direct observation from metrics 1", "Direct observation 2"],
  "recommended_actions": ["Safe verification or mitigation step 1", "Step 2"],
  "limitations": ["Any unmonitored factors or data gaps"]
}
"""


class AiService:
    """Interface to local Llama inference through Ollama with structured output parsing."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: ollama.Client | None = None

    def _get_client(self) -> ollama.Client:
        if self._client is None:
            self._client = ollama.Client(host=self.settings.ollama_host)
        return self._client

    def is_available(self) -> bool:
        """Check if local Ollama daemon is reachable and responding."""
        try:
            client = self._get_client()
            client.list()
            return True
        except Exception:
            return False

    def analyze(self, question: str, evidence: dict[str, Any]) -> dict[str, Any]:
        """Generate an evidence-grounded diagnosis using Ollama/Llama.

        Args:
            question: The user prompt or operator inquiry.
            evidence: Structured dictionary containing metrics, anomaly scores, and issues.
        """
        evidence_json = json.dumps(evidence, indent=2, default=str)
        user_message = (
            f"Operator Question: {question}\n\n"
            f"Structured Monitoring Telemetry and Evidence:\n{evidence_json}\n\n"
            "Produce a structured JSON diagnosis strictly adhering to the specified schema."
        )

        try:
            client = self._get_client()
            logger.info("Dispatching prompt to Ollama model: %s at %s", self.settings.ollama_model, self.settings.ollama_host)
            response = client.chat(
                model=self.settings.ollama_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                options={"temperature": 0.2, "top_p": 0.9}
            )
            raw_content = response.get("message", {}).get("content", "")
            return self._parse_json_response(raw_content, question, evidence)

        except Exception as error:
            logger.warning("Ollama call failed: %s", error)
            raise AiServiceError(
                f"Unable to reach local Ollama daemon at '{self.settings.ollama_host}'. "
                f"Ensure Ollama is running (`ollama serve`) and model '{self.settings.ollama_model}' is pulled."
            ) from error

    @staticmethod
    def _parse_json_response(content: str, question: str, evidence: dict[str, Any]) -> dict[str, Any]:
        """Extract and validate JSON from model output with structured fallback."""
        cleaned = content.strip()
        # Remove markdown code block fences if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
            cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and "incident_summary" in parsed:
                # Ensure all schema keys are present
                parsed.setdefault("confidence", 0.75)
                parsed.setdefault("supporting_evidence", evidence.get("evidence_statements", []))
                parsed.setdefault("recommended_actions", ["Check running processes and review error logs"])
                parsed.setdefault("limitations", [])
                return parsed
        except json.JSONDecodeError:
            logger.info("LLM did not return strict JSON; constructing structured wrapper around response text.")

        # Robust fallback preserving raw LLM reasoning
        return {
            "incident_summary": cleaned if cleaned else f"Incident analysis for: {question}",
            "probable_root_cause": "Derived from provided metrics and anomaly detector output.",
            "confidence": 0.60,
            "supporting_evidence": evidence.get("evidence_statements", []),
            "recommended_actions": [
                "Inspect high CPU/Memory consuming processes",
                "Review recent application deployment logs and error traces",
                "Verify database connection pools and upstream API latency"
            ],
            "limitations": [
                "Response was generated in free-text format and wrapped in standard schema."
            ],
            "raw_output": cleaned
        }
