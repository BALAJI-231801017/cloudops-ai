"""CSV-backed health scoring and anomaly detection for local development."""

from pathlib import Path

import pandas as pd


class HealthServiceError(RuntimeError):
    """Raised when health data is unavailable or invalid."""


class HealthService:
    REQUIRED_COLUMNS = {"cpu_usage", "memory_usage", "error_rate", "response_time", "status"}

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def analyze(self):
        try:
            data = pd.read_csv(self.data_path)
        except FileNotFoundError as error:
            raise HealthServiceError(f"Health data file not found: {self.data_path}") from error
        except Exception as error:
            raise HealthServiceError(f"Unable to read health data: {error}") from error

        if data.empty or not self.REQUIRED_COLUMNS.issubset(data.columns):
            raise HealthServiceError("Health data must contain at least one record and all required metric columns.")

        latest = data.iloc[-1]
        thresholds = {
            "cpu_usage": (60, 80),
            "memory_usage": (60, 80),
            "error_rate": (5, 10),
            "response_time": (2, 3),
        }
        issues = []
        penalties = []
        for metric, (warning, critical) in thresholds.items():
            value = float(latest[metric])
            penalties.append(min(value / critical, 1) * 25)
            if value > critical:
                issues.append({"metric": metric, "severity": "critical", "value": value})
            elif value > warning:
                issues.append({"metric": metric, "severity": "warning", "value": value})

        anomaly = self._detect_cpu_anomaly(data)
        severity = "critical" if any(issue["severity"] == "critical" for issue in issues) else "warning" if issues else "healthy"
        return {
            "source": "health_data.csv",
            "latest": {column: self._serialize(latest[column]) for column in data.columns},
            "health_score": round(max(0, 100 - sum(penalties)), 1),
            "severity": severity,
            "issues": issues,
            "anomaly": anomaly,
        }

    @staticmethod
    def _serialize(value):
        return value.item() if hasattr(value, "item") else value

    @staticmethod
    def _detect_cpu_anomaly(data):
        if len(data) < 4:
            return {"detected": False, "reason": "Insufficient history for baseline comparison."}
        baseline = data["cpu_usage"].iloc[:-1]
        current = float(data["cpu_usage"].iloc[-1])
        average = float(baseline.mean())
        standard_deviation = float(baseline.std(ddof=0))
        threshold = average + max(15, standard_deviation * 2)
        return {
            "detected": current > threshold,
            "metric": "cpu_usage",
            "current": current,
            "baseline_average": round(average, 2),
            "threshold": round(threshold, 2),
        }
