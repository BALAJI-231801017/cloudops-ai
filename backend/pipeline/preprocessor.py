"""Deterministic metric preprocessing and feature construction pipeline."""

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd


class PreprocessingError(ValueError):
    """Raised when metric telemetry cannot be preprocessed or validated."""


FEATURE_COLUMNS = ["cpu_usage", "memory_usage", "error_rate", "response_time"]

METRIC_BOUNDS: dict[str, tuple[float, float]] = {
    "cpu_usage": (0.0, 100.0),
    "memory_usage": (0.0, 100.0),
    "error_rate": (0.0, 100.0),
    "response_time": (0.0, 300.0),
}


@dataclass(frozen=True)
class PreprocessedMetrics:
    """Standardized output of the metric preprocessing pipeline."""
    feature_matrix: np.ndarray          # Shape: (N, 4)
    feature_names: list[str]            # ["cpu_usage", "memory_usage", "error_rate", "response_time"]
    latest_row: np.ndarray              # Shape: (4,)
    latest_values: dict[str, float]     # {"cpu_usage": 92.0, ...}
    n_rows: int
    has_gaps: bool
    timestamps: list[str]


class MetricPreprocessor:
    """Validates, cleans, clamps, and constructs feature matrices from metric telemetry."""

    def __init__(self, max_consecutive_gaps: int = 2):
        self.max_consecutive_gaps = max_consecutive_gaps
        self.feature_names = list(FEATURE_COLUMNS)

    def preprocess_dataframe(self, df: pd.DataFrame) -> PreprocessedMetrics:
        """Preprocess a pandas DataFrame containing metric timeseries."""
        if df is None or df.empty:
            raise PreprocessingError("Telemetry DataFrame is empty or None.")

        # Check required columns
        missing_cols = [col for col in self.feature_names if col not in df.columns]
        if missing_cols:
            raise PreprocessingError(f"Missing required metric columns: {missing_cols}")

        cleaned_df = df.copy()

        # Check consecutive NaN values
        has_gaps = False
        for col in self.feature_names:
            # Coerce to numeric
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")
            nan_mask = cleaned_df[col].isna()
            if nan_mask.any():
                has_gaps = True
                # Check maximum consecutive NaNs
                consecutive = nan_mask.astype(int).groupby((~nan_mask).cumsum()).sum()
                if (consecutive > self.max_consecutive_gaps).any():
                    raise PreprocessingError(
                        f"Metric '{col}' contains more than {self.max_consecutive_gaps} consecutive missing values."
                    )
                # Forward fill then backward fill small gaps
                cleaned_df[col] = cleaned_df[col].ffill().bfill()
                # If still NaN (e.g. all NaNs in column)
                if cleaned_df[col].isna().any():
                    raise PreprocessingError(f"Metric '{col}' could not be imputed.")

        # Clamping to valid physical boundaries
        for col, (min_val, max_val) in METRIC_BOUNDS.items():
            cleaned_df[col] = cleaned_df[col].clip(lower=min_val, upper=max_val).astype(np.float64)

        feature_matrix = cleaned_df[self.feature_names].to_numpy(dtype=np.float64)
        latest_row = feature_matrix[-1]
        latest_values = {
            col: float(latest_row[idx]) for idx, col in enumerate(self.feature_names)
        }

        timestamps = (
            cleaned_df["timestamp"].astype(str).tolist()
            if "timestamp" in cleaned_df.columns
            else [f"T-{len(cleaned_df) - 1 - i}" for i in range(len(cleaned_df))]
        )

        return PreprocessedMetrics(
            feature_matrix=feature_matrix,
            feature_names=self.feature_names,
            latest_row=latest_row,
            latest_values=latest_values,
            n_rows=len(cleaned_df),
            has_gaps=has_gaps,
            timestamps=timestamps,
        )

    def preprocess_single(self, record: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
        """Preprocess a single telemetry dictionary."""
        missing = [col for col in self.feature_names if col not in record]
        if missing:
            raise PreprocessingError(f"Missing required metric keys: {missing}")

        cleaned_values: dict[str, float] = {}
        for col in self.feature_names:
            try:
                val = float(record[col])
            except (ValueError, TypeError) as err:
                raise PreprocessingError(f"Metric '{col}' has invalid non-numeric value: {record[col]}") from err
            min_val, max_val = METRIC_BOUNDS[col]
            clamped = max(min_val, min(max_val, val))
            cleaned_values[col] = clamped

        vector = np.array([cleaned_values[col] for col in self.feature_names], dtype=np.float64)
        return vector, cleaned_values
