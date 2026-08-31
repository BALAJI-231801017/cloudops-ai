"""Comprehensive test suite for the Flask REST API."""

import unittest
from backend.app import create_app
from backend.config import Settings
from backend.ml.detector import IsolationForestDetector


class FakeAwsService:
    def list_instances(self):
        return [{"id": "i-1234567890abcdef0", "name": "cloudops-test", "state": "running"}]

    def get_instance_metrics(self, instance_id, hours=1):
        return {
            "instance_id": instance_id,
            "window_hours": hours,
            "metrics": {
                "CPUUtilization": [{"timestamp": "2026-08-31T00:00:00Z", "average": 45.2, "unit": "Percent"}],
                "NetworkIn": [],
                "NetworkOut": []
            }
        }


class FakeHealthService:
    def analyze(self):
        return {
            "source": "mock",
            "latest": {"cpu_usage": 92.0, "memory_usage": 50.0, "error_rate": 1.0, "response_time": 1.1},
            "health_score": 75.0,
            "severity": "CRITICAL",
            "incident_type": "CPU_PRESSURE",
            "is_anomaly": True,
            "primary_detector": "rule_threshold",
            "anomaly_score": 0.85,
            "affected_metrics": ["cpu_usage"],
            "issues": [{"metric": "cpu_usage", "severity": "critical", "value": 92.0}],
            "evidence_statements": ["CPU usage is critical (92.0 >= 80.0)"],
            "evidence": {
                "incident_type": "CPU_PRESSURE",
                "severity": "CRITICAL",
                "evidence_statements": ["CPU is 92%"]
            }
        }


class FakeAiService:
    def is_available(self):
        return True

    def analyze(self, question, evidence):
        return {
            "incident_summary": f"Diagnosis for: {question}",
            "probable_root_cause": "High CPU utilization detected.",
            "confidence": 0.88,
            "supporting_evidence": ["CPU is 92%"],
            "recommended_actions": ["Investigate worker processes"],
            "limitations": []
        }


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        app = create_app(
            settings=Settings(),
            aws_service=FakeAwsService(),
            health_service=FakeHealthService(),
            ai_service=FakeAiService(),
        )
        self.client = app.test_client()

    def test_health_check(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "healthy")
        self.assertIn("components", response.json)

    def test_instances(self):
        response = self.client.get("/api/instances")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["instances"][0]["id"], "i-1234567890abcdef0")

    def test_metrics_rejects_invalid_hours(self):
        response = self.client.get("/api/metrics/i-1234567890abcdef0?hours=25")
        self.assertEqual(response.status_code, 400)

    def test_metrics_rejects_invalid_instance_id(self):
        response = self.client.get("/api/metrics/not-an-instance-id")
        self.assertEqual(response.status_code, 400)

    def test_health_analysis_endpoint(self):
        response = self.client.get("/api/health-analysis")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["incident_type"], "CPU_PRESSURE")

    def test_ai_analysis_includes_evidence(self):
        response = self.client.post("/api/analyze", json={
            "question": "What is the CPU issue?",
            "instance_id": "i-1234567890abcdef0"
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("analysis", response.json)
        self.assertEqual(response.json["analysis"]["probable_root_cause"], "High CPU utilization detected.")

    def test_simulate_valid_scenario(self):
        response = self.client.post("/api/simulate", json={"scenario": "cpu_spike"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["simulation_mode"])
        self.assertEqual(response.json["scenario"], "cpu_spike")
        self.assertTrue(response.json["is_anomaly"])

    def test_simulate_invalid_scenario(self):
        response = self.client.post("/api/simulate", json={"scenario": "nonexistent_scenario"})
        self.assertEqual(response.status_code, 400)

    def test_evaluation_endpoint(self):
        response = self.client.get("/api/evaluation")
        self.assertEqual(response.status_code, 200)
        self.assertIn("models", response.json)


if __name__ == "__main__":
    unittest.main()
