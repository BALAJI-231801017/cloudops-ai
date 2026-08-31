"""Tests for Isolation Forest ML Anomaly Detector."""

import tempfile
import unittest
from pathlib import Path
import numpy as np

from backend.ml.detector import IsolationForestDetector, ModelNotTrainedError, MLAnomalyResult


class TestIsolationForestDetector(unittest.TestCase):
    def setUp(self):
        # 100 samples of normal metrics
        rng = np.random.default_rng(42)
        cpus = rng.normal(40.0, 5.0, 100)
        mems = rng.normal(50.0, 5.0, 100)
        errs = rng.normal(0.8, 0.2, 100)
        lats = rng.normal(1.0, 0.2, 100)
        self.X_nominal = np.column_stack([cpus, mems, errs, lats])
        self.detector = IsolationForestDetector(contamination=0.05, random_state=42)

    def test_untrained_model_raises_error(self):
        detector = IsolationForestDetector()
        self.assertFalse(detector.is_trained())
        with self.assertRaises(ModelNotTrainedError):
            detector.predict_vector(np.array([40.0, 50.0, 0.8, 1.0]))

    def test_model_fit_and_inference(self):
        self.detector.fit(self.X_nominal)
        self.assertTrue(self.detector.is_trained())

        # Test normal sample
        normal_sample = np.array([40.0, 50.0, 0.8, 1.0])
        res_norm = self.detector.predict_vector(normal_sample)
        self.assertIsInstance(res_norm, MLAnomalyResult)
        self.assertFalse(res_norm.is_anomaly)
        self.assertLess(res_norm.anomaly_score, 0.35)
        self.assertEqual(res_norm.severity, "LOW")

        # Test extreme outlier sample
        spike_sample = np.array([98.0, 95.0, 25.0, 12.0])
        res_spike = self.detector.predict_vector(spike_sample)
        self.assertTrue(res_spike.is_anomaly)
        self.assertGreaterEqual(res_spike.anomaly_score, 0.35)
        self.assertIn(res_spike.severity, ["MEDIUM", "HIGH", "CRITICAL"])

    def test_model_save_and_load(self):
        self.detector.fit(self.X_nominal)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "test_model.pkl"
            self.detector.save(model_path)
            self.assertTrue(model_path.exists())

            loaded_detector = IsolationForestDetector()
            loaded_detector.load(model_path)
            self.assertTrue(loaded_detector.is_trained())

            res1 = self.detector.predict_vector(np.array([40.0, 50.0, 0.8, 1.0]))
            res2 = loaded_detector.predict_vector(np.array([40.0, 50.0, 0.8, 1.0]))
            self.assertEqual(res1.anomaly_score, res2.anomaly_score)
            self.assertEqual(res1.is_anomaly, res2.is_anomaly)


if __name__ == "__main__":
    unittest.main()
