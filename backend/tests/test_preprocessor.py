"""Tests for metric telemetry preprocessing pipeline."""

import unittest
import numpy as np
import pandas as pd

from backend.pipeline.preprocessor import (
    MetricPreprocessor,
    PreprocessingError,
    PreprocessedMetrics,
    FEATURE_COLUMNS,
)


class TestMetricPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = MetricPreprocessor(max_consecutive_gaps=2)

    def test_valid_dataframe_preprocessing(self):
        df = pd.DataFrame({
            "timestamp": ["T-2", "T-1", "T-0"],
            "cpu_usage": [30.0, 50.0, 85.0],
            "memory_usage": [40.0, 60.0, 75.0],
            "error_rate": [0.5, 1.2, 5.0],
            "response_time": [0.8, 1.1, 2.5],
            "status": ["Healthy", "Healthy", "Warning"]
        })
        res = self.preprocessor.preprocess_dataframe(df)
        self.assertIsInstance(res, PreprocessedMetrics)
        self.assertEqual(res.feature_matrix.shape, (3, 4))
        self.assertEqual(res.feature_names, FEATURE_COLUMNS)
        self.assertEqual(res.latest_values["cpu_usage"], 85.0)
        self.assertFalse(res.has_gaps)

    def test_missing_required_column_raises_error(self):
        df = pd.DataFrame({
            "cpu_usage": [30.0],
            "memory_usage": [40.0]
            # missing error_rate, response_time
        })
        with self.assertRaises(PreprocessingError):
            self.preprocessor.preprocess_dataframe(df)

    def test_nan_interpolation_within_tolerance(self):
        df = pd.DataFrame({
            "cpu_usage": [30.0, np.nan, 50.0],
            "memory_usage": [40.0, 45.0, 50.0],
            "error_rate": [0.5, 0.6, 0.7],
            "response_time": [0.8, 0.9, 1.0],
        })
        res = self.preprocessor.preprocess_dataframe(df)
        self.assertTrue(res.has_gaps)
        self.assertEqual(res.feature_matrix.shape, (3, 4))
        # Value should be interpolated
        self.assertFalse(np.isnan(res.feature_matrix).any())

    def test_clamping_to_physical_boundaries(self):
        df = pd.DataFrame({
            "cpu_usage": [-15.0, 145.0],
            "memory_usage": [120.0, -5.0],
            "error_rate": [-1.0, 200.0],
            "response_time": [-2.0, 500.0],
        })
        res = self.preprocessor.preprocess_dataframe(df)
        # Clamped to valid range
        self.assertEqual(res.feature_matrix[0, 0], 0.0)    # cpu lower
        self.assertEqual(res.feature_matrix[1, 0], 100.0)  # cpu upper
        self.assertEqual(res.feature_matrix[0, 1], 100.0)  # mem upper
        self.assertEqual(res.feature_matrix[1, 3], 300.0)  # lat upper

    def test_preprocess_single_dictionary(self):
        record = {
            "cpu_usage": "92.5",
            "memory_usage": 80,
            "error_rate": "12.0",
            "response_time": 4.5
        }
        vec, cleaned = self.preprocessor.preprocess_single(record)
        self.assertEqual(vec.shape, (4,))
        self.assertEqual(cleaned["cpu_usage"], 92.5)
        self.assertEqual(cleaned["memory_usage"], 80.0)


if __name__ == "__main__":
    unittest.main()
