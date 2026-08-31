"""Unsupervised multivariate anomaly detector using Isolation Forest and StandardScaler."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class ModelNotTrainedError(RuntimeError):
    """Raised when inference is attempted on an untrained Isolation Forest model."""


@dataclass
class MLAnomalyResult:
    """Standardized output of the Isolation Forest ML detector."""
    is_anomaly: bool
    anomaly_score: float         # Normalized [0.0, 1.0]
    raw_decision_score: float    # Raw sklearn decision_function value
    severity: str                # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    detector_name: str           # "isolation_forest"
    model_version: str
    feature_contributions: dict[str, float]  # Scaled distance from median per feature


class IsolationForestDetector:
    """Multivariate anomaly detection pipeline combining StandardScaler and Isolation Forest.

    Isolation Forest operates by isolating anomalies through random feature splitting.
    Because anomalies require fewer splits to isolate in feature space, their tree path
    lengths are notably shorter than nominal operating points.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        random_state: int = 42,
        n_estimators: int = 100,
        model_version: str = "iforest-v1.0"
    ):
        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.model_version = model_version
        self.scaler: StandardScaler | None = None
        self.model: IsolationForest | None = None
        self.feature_names: list[str] = ["cpu_usage", "memory_usage", "error_rate", "response_time"]
        self._is_trained = False

    def is_trained(self) -> bool:
        """Check if the model and scaler are fitted and ready for inference."""
        return self._is_trained and self.model is not None and self.scaler is not None

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> "IsolationForestDetector":
        """Fit StandardScaler and IsolationForest on nominal historical data.

        Args:
            X: 2D NumPy array of shape (N, 4) representing clean/nominal baseline telemetry.
            feature_names: Optional list of feature names.
        """
        if X is None or len(X) < 10:
            raise ValueError("Training requires at least 10 telemetry observations.")

        if feature_names:
            self.feature_names = list(feature_names)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            max_samples="auto",
            n_jobs=-1
        )
        self.model.fit(X_scaled)
        self._is_trained = True
        return self

    def predict_vector(self, vector: np.ndarray) -> MLAnomalyResult:
        """Run anomaly inference on a single 1D or 2D feature vector."""
        if not self.is_trained():
            raise ModelNotTrainedError(
                "Isolation Forest model has not been trained or loaded. Run training script first."
            )

        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        assert self.scaler is not None
        assert self.model is not None

        scaled = self.scaler.transform(vector)
        raw_score = float(self.model.decision_function(scaled)[0])
        pred = int(self.model.predict(scaled)[0])  # +1 for inlier, -1 for outlier

        # Normalize score to [0.0, 1.0] where 1.0 = most anomalous
        # decision_function typically lies in [-0.5, +0.5]
        anomaly_score = float(np.clip(-raw_score / 0.40, 0.0, 1.0))

        is_anomaly = bool(pred == -1 or anomaly_score >= 0.35)

        if anomaly_score >= 0.70:
            severity = "CRITICAL"
        elif anomaly_score >= 0.50:
            severity = "HIGH"
        elif anomaly_score >= 0.35:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        # Calculate z-score feature deviation as contribution indicator
        contributions: dict[str, float] = {}
        for idx, feat in enumerate(self.feature_names):
            contributions[feat] = round(float(scaled[0, idx]), 2)

        return MLAnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=round(anomaly_score, 3),
            raw_decision_score=round(raw_score, 4),
            severity=severity,
            detector_name="isolation_forest",
            model_version=self.model_version,
            feature_contributions=contributions
        )

    def save(self, file_path: str | Path) -> None:
        """Persist trained scaler and model to a joblib artifact."""
        if not self.is_trained():
            raise ModelNotTrainedError("Cannot save an untrained model.")
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scaler": self.scaler,
            "model": self.model,
            "feature_names": self.feature_names,
            "contamination": self.contamination,
            "random_state": self.random_state,
            "n_estimators": self.n_estimators,
            "model_version": self.model_version,
            "is_trained": True
        }
        joblib.dump(payload, path)

    def load(self, file_path: str | Path) -> "IsolationForestDetector":
        """Load trained scaler and model from a joblib artifact."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at: {path}")

        payload = joblib.load(path)
        self.scaler = payload["scaler"]
        self.model = payload["model"]
        self.feature_names = payload.get("feature_names", self.feature_names)
        self.contamination = payload.get("contamination", self.contamination)
        self.random_state = payload.get("random_state", self.random_state)
        self.n_estimators = payload.get("n_estimators", self.n_estimators)
        self.model_version = payload.get("model_version", self.model_version)
        self._is_trained = True
        return self
