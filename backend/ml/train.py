"""Offline training script for the Isolation Forest anomaly detector."""

import logging
from pathlib import Path
import numpy as np
import pandas as pd

from backend.ml.detector import IsolationForestDetector
from backend.pipeline.preprocessor import MetricPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cloudops.train")


def generate_baseline_training_csv(output_path: Path, n_samples: int = 500, random_seed: int = 42) -> pd.DataFrame:
    """Generate and persist nominal/healthy historical telemetry for model fitting."""
    rng = np.random.default_rng(random_seed)

    # Simulating 500 hourly records of a standard healthy cloud workload with daily cyclicality
    hours = np.arange(n_samples)
    diurnal_cycle = 10.0 * np.sin(2 * np.pi * hours / 24.0)

    cpu = np.clip(35.0 + diurnal_cycle + rng.normal(0, 4.5, n_samples), 15.0, 68.0)
    memory = np.clip(45.0 + 0.5 * diurnal_cycle + rng.normal(0, 3.5, n_samples), 25.0, 70.0)
    error_rate = np.clip(0.5 + rng.exponential(0.3, n_samples), 0.0, 3.0)
    response_time = np.clip(0.8 + 0.01 * cpu + rng.exponential(0.2, n_samples), 0.2, 2.2)

    df = pd.DataFrame({
        "timestamp": [f"2026-08-01T{h%24:02d}:00:00Z" for h in hours],
        "cpu_usage": np.round(cpu, 2),
        "memory_usage": np.round(memory, 2),
        "error_rate": np.round(error_rate, 2),
        "response_time": np.round(response_time, 2),
        "status": "Healthy"
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Generated %d nominal training samples at: %s", n_samples, output_path)
    return df


def train_model(
    training_data_path: str = "backend/data/training_baseline.csv",
    model_output_path: str = "backend/ml/model.pkl",
    contamination: float = 0.05,
    random_state: int = 42
) -> IsolationForestDetector:
    """Train and persist the Isolation Forest detector."""
    csv_path = Path(training_data_path)
    if not csv_path.exists():
        logger.info("Training data not found at %s. Generating synthetic baseline dataset...", csv_path)
        generate_baseline_training_csv(csv_path, n_samples=500, random_seed=random_state)

    logger.info("Loading training data from %s...", csv_path)
    raw_df = pd.read_csv(csv_path)

    preprocessor = MetricPreprocessor()
    preprocessed = preprocessor.preprocess_dataframe(raw_df)

    logger.info("Feature matrix shape: %s with features: %s", preprocessed.feature_matrix.shape, preprocessed.feature_names)
    logger.info("Fitting IsolationForest (contamination=%.3f, random_state=%d)...", contamination, random_state)

    detector = IsolationForestDetector(contamination=contamination, random_state=random_state)
    detector.fit(preprocessed.feature_matrix, feature_names=preprocessed.feature_names)

    model_path = Path(model_output_path)
    detector.save(model_path)
    logger.info("Model saved successfully to %s", model_path)

    # Verification run
    sample_normal = np.array([40.0, 50.0, 0.5, 0.9])
    res_normal = detector.predict_vector(sample_normal)
    logger.info("Sanity Check - Normal sample [40, 50, 0.5, 0.9] -> Anomaly: %s, Score: %.3f, Severity: %s",
                res_normal.is_anomaly, res_normal.anomaly_score, res_normal.severity)

    sample_spike = np.array([96.0, 92.0, 15.0, 6.0])
    res_spike = detector.predict_vector(sample_spike)
    logger.info("Sanity Check - Extreme spike [96, 92, 15, 6.0] -> Anomaly: %s, Score: %.3f, Severity: %s",
                res_spike.is_anomaly, res_spike.anomaly_score, res_spike.severity)

    return detector


if __name__ == "__main__":
    train_model()
