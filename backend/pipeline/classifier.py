"""Deterministic incident classification and unified anomaly schema."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from backend.ml.detector import MLAnomalyResult


class SeverityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class IncidentType(str, Enum):
    NORMAL = "NORMAL"
    CPU_PRESSURE = "CPU_PRESSURE"
    MEMORY_PRESSURE = "MEMORY_PRESSURE"
    ERROR_SPIKE = "ERROR_SPIKE"
    LATENCY_DEGRADATION = "LATENCY_DEGRADATION"
    RESOURCE_SATURATION = "RESOURCE_SATURATION"
    MULTI_METRIC_ANOMALY = "MULTI_METRIC_ANOMALY"
    UNKNOWN_ANOMALY = "UNKNOWN_ANOMALY"


# Standard two-tier thresholds: (Warning, Critical)
DEFAULT_THRESHOLDS: dict[str, tuple[float, float]] = {
    "cpu_usage": (60.0, 80.0),
    "memory_usage": (65.0, 80.0),
    "error_rate": (5.0, 10.0),
    "response_time": (2.0, 3.5),
}


@dataclass
class MetricIssue:
    metric: str
    severity: str
    value: float
    threshold: float
    message: str


@dataclass
class AnomalyResult:
    """Unified anomaly detection and incident classification schema."""
    is_anomaly: bool
    incident_type: IncidentType
    severity: SeverityLevel
    primary_detector: str                  # "isolation_forest", "statistical_baseline", "rule_threshold"
    anomaly_score: float                   # Normalized [0.0, 1.0]
    affected_metrics: list[str]
    issues: list[dict[str, Any]]
    evidence_statements: list[str]
    health_score: float                    # [0.0, 100.0]
    baseline_result: dict[str, Any] | None = None
    ml_result: dict[str, Any] | None = None


class IncidentClassifier:
    """Deterministic, transparent incident classifier based on explicit rules and ML evidence.

    Principles:
    1. ML detects multivariate statistical anomalies.
    2. Deterministic rules categorize the incident and calculate official severity.
    3. LLM explains the evidence and advises remediation without changing severity.
    """

    def __init__(self, thresholds: dict[str, tuple[float, float]] | None = None):
        self.thresholds = thresholds or dict(DEFAULT_THRESHOLDS)

    def evaluate_metric_issues(self, metrics: dict[str, float]) -> tuple[list[MetricIssue], float]:
        """Evaluate individual metrics against thresholds and compute health score penalty."""
        issues: list[MetricIssue] = []
        penalties: list[float] = []

        for metric, (warn_thresh, crit_thresh) in self.thresholds.items():
            if metric not in metrics:
                continue
            val = float(metrics[metric])

            if val >= crit_thresh:
                penalties.append(25.0)
                issues.append(MetricIssue(
                    metric=metric,
                    severity="critical",
                    value=val,
                    threshold=crit_thresh,
                    message=f"{metric.replace('_', ' ').title()} is critical ({val} >= {crit_thresh})"
                ))
            elif val >= warn_thresh:
                # Fractional penalty between warning and critical
                frac = (val - warn_thresh) / (crit_thresh - warn_thresh)
                penalties.append(10.0 + frac * 10.0)
                issues.append(MetricIssue(
                    metric=metric,
                    severity="warning",
                    value=val,
                    threshold=warn_thresh,
                    message=f"{metric.replace('_', ' ').title()} is elevated ({val} >= {warn_thresh})"
                ))
            else:
                # Low penalty if near warning
                if warn_thresh > 0:
                    penalties.append(min(1.0, val / warn_thresh) * 2.0)

        health_score = round(max(0.0, min(100.0, 100.0 - sum(penalties))), 1)
        return issues, health_score

    def classify(
        self,
        metrics: dict[str, float],
        baseline_result: dict[str, Any] | None = None,
        ml_result: MLAnomalyResult | None = None,
    ) -> AnomalyResult:
        """Classify telemetry into an incident type with severity and evidence statements."""
        issues, health_score = self.evaluate_metric_issues(metrics)

        crit_metrics = {issue.metric for issue in issues if issue.severity == "critical"}
        warn_metrics = {issue.metric for issue in issues if issue.severity == "warning"}
        all_affected = sorted(list(crit_metrics | warn_metrics))

        evidence: list[str] = []
        for issue in issues:
            evidence.append(issue.message)

        # Baseline detector check
        baseline_detected = bool(baseline_result and baseline_result.get("detected", False))
        if baseline_detected:
            evidence.append(
                f"Statistical baseline detected CPU anomaly: current {baseline_result.get('current')}% "
                f"exceeds dynamic baseline {baseline_result.get('threshold')}%"
            )

        # ML detector check
        ml_detected = bool(ml_result and ml_result.is_anomaly)
        ml_score = ml_result.anomaly_score if ml_result else 0.0

        if ml_result:
            if ml_result.is_anomaly:
                evidence.append(
                    f"Isolation Forest identified multivariate anomaly (Score: {ml_result.anomaly_score:.3f}, "
                    f"Severity: {ml_result.severity})"
                )
            else:
                evidence.append(f"Isolation Forest verified normal multivariate operation (Score: {ml_result.anomaly_score:.3f})")

        # Deterministic Incident Categorization Logic
        if len(crit_metrics) >= 3 or (len(crit_metrics) >= 2 and len(warn_metrics) >= 1):
            incident_type = IncidentType.MULTI_METRIC_ANOMALY
            severity = SeverityLevel.CRITICAL
        elif "cpu_usage" in crit_metrics and "memory_usage" in crit_metrics:
            incident_type = IncidentType.RESOURCE_SATURATION
            severity = SeverityLevel.CRITICAL
        elif "cpu_usage" in crit_metrics:
            incident_type = IncidentType.CPU_PRESSURE
            severity = SeverityLevel.CRITICAL
        elif "memory_usage" in crit_metrics:
            incident_type = IncidentType.MEMORY_PRESSURE
            severity = SeverityLevel.CRITICAL
        elif "error_rate" in crit_metrics:
            incident_type = IncidentType.ERROR_SPIKE
            severity = SeverityLevel.CRITICAL
        elif "response_time" in crit_metrics:
            incident_type = IncidentType.LATENCY_DEGRADATION
            severity = SeverityLevel.CRITICAL
        elif len(warn_metrics) >= 2:
            incident_type = IncidentType.RESOURCE_SATURATION if ("cpu_usage" in warn_metrics and "memory_usage" in warn_metrics) else IncidentType.MULTI_METRIC_ANOMALY
            severity = SeverityLevel.HIGH
        elif "cpu_usage" in warn_metrics:
            incident_type = IncidentType.CPU_PRESSURE
            severity = SeverityLevel.MEDIUM
        elif "memory_usage" in warn_metrics:
            incident_type = IncidentType.MEMORY_PRESSURE
            severity = SeverityLevel.MEDIUM
        elif "error_rate" in warn_metrics:
            incident_type = IncidentType.ERROR_SPIKE
            severity = SeverityLevel.HIGH
        elif "response_time" in warn_metrics:
            incident_type = IncidentType.LATENCY_DEGRADATION
            severity = SeverityLevel.MEDIUM
        elif ml_detected or baseline_detected:
            incident_type = IncidentType.UNKNOWN_ANOMALY
            severity = SeverityLevel.MEDIUM if ml_score < 0.6 else SeverityLevel.HIGH
        else:
            incident_type = IncidentType.NORMAL
            severity = SeverityLevel.LOW

        is_anomaly = bool(incident_type != IncidentType.NORMAL or ml_detected or baseline_detected)

        # Primary detector determination
        if ml_result and ml_result.is_anomaly:
            primary_detector = "isolation_forest"
            anomaly_score = ml_score
        elif baseline_detected:
            primary_detector = "statistical_baseline"
            anomaly_score = 0.75
        elif issues:
            primary_detector = "rule_threshold"
            anomaly_score = max([0.6 if i.severity == "warning" else 0.9 for i in issues])
        else:
            primary_detector = "isolation_forest" if ml_result else "rule_threshold"
            anomaly_score = ml_score

        return AnomalyResult(
            is_anomaly=is_anomaly,
            incident_type=incident_type,
            severity=severity,
            primary_detector=primary_detector,
            anomaly_score=round(float(anomaly_score), 3),
            affected_metrics=all_affected,
            issues=[
                {"metric": i.metric, "severity": i.severity, "value": i.value, "threshold": i.threshold, "message": i.message}
                for i in issues
            ],
            evidence_statements=evidence,
            health_score=health_score,
            baseline_result=baseline_result,
            ml_result={
                "is_anomaly": ml_result.is_anomaly,
                "anomaly_score": ml_result.anomaly_score,
                "raw_decision_score": ml_result.raw_decision_score,
                "severity": ml_result.severity,
                "detector_name": ml_result.detector_name,
                "model_version": ml_result.model_version,
                "feature_contributions": ml_result.feature_contributions
            } if ml_result else None
        )
