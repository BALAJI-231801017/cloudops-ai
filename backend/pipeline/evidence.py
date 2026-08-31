"""Evidence Object Builder for grounding Generative AI / LLM root-cause analysis."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from backend.pipeline.classifier import AnomalyResult


@dataclass
class EvidenceObject:
    """Canonical monitoring evidence object.

    This serves as the single source of truth for the LLM diagnosis layer.
    The LLM is strictly constrained to interpret only the fields present in this object.
    """
    timestamp_utc: str
    incident_type: str
    severity: str
    health_score: float
    is_anomaly: bool
    primary_detector: str
    anomaly_score: float
    current_metrics: dict[str, float]
    affected_metrics: list[str]
    threshold_issues: list[dict[str, Any]]
    evidence_statements: list[str]
    baseline_telemetry: dict[str, Any] | None
    ml_telemetry: dict[str, Any] | None
    cloudwatch_metrics: dict[str, Any] | None
    data_source: str  # "cloudwatch_live", "csv_telemetry", "simulation"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceBuilder:
    """Constructs verifiable evidence objects from telemetry and anomaly classification."""

    @staticmethod
    def build(
        metrics: dict[str, float],
        anomaly_result: AnomalyResult,
        data_source: str = "csv_telemetry",
        cloudwatch_data: dict[str, Any] | None = None,
    ) -> EvidenceObject:
        now_str = datetime.now(timezone.utc).isoformat()

        return EvidenceObject(
            timestamp_utc=now_str,
            incident_type=anomaly_result.incident_type.value,
            severity=anomaly_result.severity.value,
            health_score=anomaly_result.health_score,
            is_anomaly=anomaly_result.is_anomaly,
            primary_detector=anomaly_result.primary_detector,
            anomaly_score=anomaly_result.anomaly_score,
            current_metrics=metrics,
            affected_metrics=anomaly_result.affected_metrics,
            threshold_issues=anomaly_result.issues,
            evidence_statements=anomaly_result.evidence_statements,
            baseline_telemetry=anomaly_result.baseline_result,
            ml_telemetry=anomaly_result.ml_result,
            cloudwatch_metrics=cloudwatch_data,
            data_source=data_source,
        )
