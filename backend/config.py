"""Runtime configuration loaded from environment variables."""

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    aws_region: str = os.getenv("AWS_REGION", "ap-south-1")
    aws_profile: str | None = os.getenv("CLOUDOPS_AWS_PROFILE") or None
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    health_data_path: str = os.getenv("HEALTH_DATA_PATH", "health_data.csv")
    model_path: str = os.getenv("MODEL_PATH", "backend/ml/model.pkl")
    frontend_origin: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
