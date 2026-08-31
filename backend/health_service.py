"""CSV-backed health telemetry service integrating Preprocessing, Baseline, and ML Anomaly Detection."""

import logging
from pathlib import Path
from typing import Any
import pandas as pd

from backend.ml.detector import IsolationForestDetector, MLAnomalyResult
from backend.pipeline.classifier import AnomalyResult, IncidentClassifier
from backend.pipeline.evidence import EvidenceBuilder, EvidenceObject
from backend.pipeline.preprocessor import MetricPreprocessor, PreprocessingError

logger = logging.getLogger("cloudops.health_service")


class HealthServiceError(RuntimeError):
    """Raised when health data cannot be read, validated, or processed."""


class HealthService:
    """Orchestrates ingestion, preprocessing, baseline rules, ML detection, and incident classification."""

    def __init__(
        self,
        data_path: str = "health_data.csv",
        detector: IsolationForestDetector | None = None,
        classifier: IncidentClassifier | None = None,
    ):
        self.data_path = Path(data_path)
        self.preprocessor = MetricPreprocessor()
        self.detector = detector
        self.classifier = classifier or IncidentClassifier()

    def analyze(self) -> dict[str, Any]:
        """Process the health dataset and return comprehensive health and anomaly analysis."""
        try:
            data = pd.read_csv(self.data_path)
        except FileNotFoundError as error:
            raise HealthServiceError(f"Health data file not found: {self.data_path}") from error
        except Exception as error:
            raise HealthServiceError(f"Unable to read health data: {error}") from error

        try:
            preprocessed = self.preprocessor.preprocess_dataframe(data)
        except PreprocessingError as error:
            raise HealthServiceError(f"Telemetry validation failure: {error}") from error

        latest_metrics = preprocessed.latest_values
        latest_vector = preprocessed.latest_row

        # 1. Baseline statistical detector (CPU rolling window)
        baseline_res = self._detect_cpu_anomaly(data)

        # 2. Isolation Forest ML Anomaly Detection
        ml_res: MLAnomalyResult | None = None
        if self.detector and self.detector.is_trained():
            try:
                ml_res = self.detector.predict_vector(latest_vector)
            except Exception as err:
                logger.warning("ML inference failed: %s", err)

        # 3. Deterministic Incident Classification
        anomaly_result: AnomalyResult = self.classifier.classify(
            metrics=latest_metrics,
            baseline_result=baseline_res,
            ml_result=ml_res
        )

        # 4. Build Evidence Object
        evidence: EvidenceObject = EvidenceBuilder.build(
            metrics=latest_metrics,
            anomaly_result=anomaly_result,
            data_source="csv_telemetry"
        )

        # Format historical records for charting
        recent_history = []
        tail_df = data.tail(12)
        for _, row in tail_df.iterrows():
            recent_history.append({
                "timestamp": str(row.get("timestamp", "")),
                "cpu_usage": float(row.get("cpu_usage", 0.0)),
                "memory_usage": float(row.get("memory_usage", 0.0)),
                "error_rate": float(row.get("error_rate", 0.0)),
                "response_time": float(row.get("response_time", 0.0)),
                "status": str(row.get("status", "Healthy"))
            })

        return {
            "source": str(self.data_path),
            "latest": latest_metrics,
            "health_score": anomaly_result.health_score,
            "severity": anomaly_result.severity.value,
            "incident_type": anomaly_result.incident_type.value,
            "is_anomaly": anomaly_result.is_anomaly,
            "primary_detector": anomaly_result.primary_detector,
            "anomaly_score": anomaly_result.anomaly_score,
            "affected_metrics": anomaly_result.affected_metrics,
            "issues": anomaly_result.issues,
            "evidence_statements": anomaly_result.evidence_statements,
            "anomaly": baseline_res,
            "ml_result": anomaly_result.ml_result,
            "history": recent_history,
            "evidence": evidence.to_dict()
        }

    @staticmethod
    def _detect_cpu_anomaly(data: pd.DataFrame) -> dict[str, Any]:
        """Classical statistical dynamic baseline: CPU > mean + max(15, 2*std)."""
        if len(data) < 4:
            return {"detected": False, "detector": "statistical_baseline", "reason": "Insufficient history for baseline comparison."}

        baseline = data["cpu_usage"].iloc[:-1]
        current = float(data["cpu_usage"].iloc[-1])
        average = float(baseline.mean())
        standard_deviation = float(baseline.std(ddof=0))
        threshold = average + max(15.0, standard_deviation * 2.0)

        return {
            "detected": bool(current > threshold),
            "detector": "statistical_baseline",
            "metric": "cpu_usage",
            "current": round(current, 2),
            "baseline_average": round(average, 2),
            "threshold": round(threshold, 2),
        }
