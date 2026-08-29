import unittest

from backend.app import create_app
from backend.config import Settings


class FakeAwsService:
    def list_instances(self):
        return [{"id": "i-123", "name": "demo", "state": "running"}]

    def get_instance_metrics(self, instance_id, hours=1):
        return {"instance_id": instance_id, "window_hours": hours, "metrics": {"CPUUtilization": []}}


class FakeHealthService:
    def analyze(self):
        return {"health_score": 88.0, "severity": "healthy", "issues": []}


class FakeAiService:
    def analyze(self, question, evidence):
        return f"Analysis for: {question}"


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
        self.assertEqual(response.json["status"], "ok")

    def test_instances(self):
        response = self.client.get("/api/instances")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["instances"][0]["id"], "i-123")

    def test_metrics_rejects_invalid_hours(self):
        response = self.client.get("/api/metrics/i-123?hours=25")
        self.assertEqual(response.status_code, 400)

    def test_ai_analysis_includes_evidence(self):
        response = self.client.post("/api/analyze", json={"question": "What changed?", "instance_id": "i-123"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("cloudwatch_metrics", response.json["evidence"])


if __name__ == "__main__":
    unittest.main()
