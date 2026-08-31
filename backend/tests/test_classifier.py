"""Tests for deterministic incident classification and unified anomaly schema."""

import unittest
from backend.ml.detector import MLAnomalyResult
from backend.pipeline.classifier import IncidentClassifier, IncidentType, SeverityLevel


class TestIncidentClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = IncidentClassifier()

    def test_normal_telemetry(self):
        metrics = {"cpu_usage": 35.0, "memory_usage": 45.0, "error_rate": 0.5, "response_time": 0.8}
        res = self.classifier.classify(metrics)
        self.assertEqual(res.incident_type, IncidentType.NORMAL)
        self.assertEqual(res.severity, SeverityLevel.LOW)
        self.assertFalse(res.is_anomaly)
        self.assertGreater(res.health_score, 85.0)

    def test_cpu_pressure_classification(self):
        metrics = {"cpu_usage": 92.0, "memory_usage": 50.0, "error_rate": 1.0, "response_time": 1.2}
        res = self.classifier.classify(metrics)
        self.assertEqual(res.incident_type, IncidentType.CPU_PRESSURE)
        self.assertEqual(res.severity, SeverityLevel.CRITICAL)
        self.assertTrue(res.is_anomaly)
        self.assertIn("cpu_usage", res.affected_metrics)

    def test_resource_saturation_classification(self):
        metrics = {"cpu_usage": 88.0, "memory_usage": 92.0, "error_rate": 1.0, "response_time": 1.5}
        res = self.classifier.classify(metrics)
        self.assertEqual(res.incident_type, IncidentType.RESOURCE_SATURATION)
        self.assertEqual(res.severity, SeverityLevel.CRITICAL)

    def test_error_spike_classification(self):
        metrics = {"cpu_usage": 45.0, "memory_usage": 50.0, "error_rate": 22.0, "response_time": 1.5}
        res = self.classifier.classify(metrics)
        self.assertEqual(res.incident_type, IncidentType.ERROR_SPIKE)
        self.assertEqual(res.severity, SeverityLevel.CRITICAL)

    def test_multi_metric_cascade_classification(self):
        metrics = {"cpu_usage": 95.0, "memory_usage": 90.0, "error_rate": 15.0, "response_time": 6.5}
        res = self.classifier.classify(metrics)
        self.assertEqual(res.incident_type, IncidentType.MULTI_METRIC_ANOMALY)
        self.assertEqual(res.severity, SeverityLevel.CRITICAL)
        self.assertEqual(len(res.affected_metrics), 4)

    def test_ml_anomaly_integration(self):
        # Normal individual metrics but ML flagged strange multivariate cluster
        metrics = {"cpu_usage": 58.0, "memory_usage": 62.0, "error_rate": 4.5, "response_time": 1.8}
        fake_ml = MLAnomalyResult(
            is_anomaly=True,
            anomaly_score=0.65,
            raw_decision_score=-0.12,
            severity="HIGH",
            detector_name="isolation_forest",
            model_version="test-1",
            feature_contributions={"cpu_usage": 1.8}
        )
        res = self.classifier.classify(metrics, ml_result=fake_ml)
        self.assertTrue(res.is_anomaly)
        self.assertEqual(res.primary_detector, "isolation_forest")
        self.assertEqual(res.incident_type, IncidentType.UNKNOWN_ANOMALY)


if __name__ == "__main__":
    unittest.main()
