"""End-to-end integration test: Preprocessing -> ML -> Classifier -> Evidence."""

import unittest
import numpy as np

from backend.ml.detector import IsolationForestDetector
from backend.pipeline.classifier import IncidentClassifier, IncidentType
from backend.pipeline.evidence import EvidenceBuilder
from backend.pipeline.preprocessor import MetricPreprocessor


class TestEndToEndPipeline(unittest.TestCase):
    def setUp(self):
        # Quick fit of Isolation Forest
        rng = np.random.default_rng(42)
        X_nominal = rng.normal(40.0, 5.0, (100, 4))
        self.detector = IsolationForestDetector(contamination=0.05, random_state=42)
        self.detector.fit(X_nominal)

        self.preprocessor = MetricPreprocessor()
        self.classifier = IncidentClassifier()

    def test_full_pipeline_flow(self):
        raw_telemetry = {
            "cpu_usage": 94.5,
            "memory_usage": 88.2,
            "error_rate": 14.0,
            "response_time": 5.2
        }

        # 1. Preprocess
        vector, cleaned = self.preprocessor.preprocess_single(raw_telemetry)
        self.assertEqual(vector.shape, (4,))

        # 2. ML Inference
        ml_res = self.detector.predict_vector(vector)
        self.assertTrue(ml_res.is_anomaly)

        # 3. Classify Incident
        anom_res = self.classifier.classify(metrics=cleaned, ml_result=ml_res)
        self.assertEqual(anom_res.incident_type, IncidentType.MULTI_METRIC_ANOMALY)
        self.assertTrue(anom_res.is_anomaly)

        # 4. Evidence Building
        evidence = EvidenceBuilder.build(
            metrics=cleaned,
            anomaly_result=anom_res,
            data_source="test_pipeline"
        )
        evidence_dict = evidence.to_dict()

        self.assertIn("timestamp_utc", evidence_dict)
        self.assertEqual(evidence_dict["incident_type"], "MULTI_METRIC_ANOMALY")
        self.assertGreater(len(evidence_dict["evidence_statements"]), 0)
        self.assertIn("ml_telemetry", evidence_dict)


if __name__ == "__main__":
    unittest.main()
