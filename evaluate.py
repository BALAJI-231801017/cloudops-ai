"""Comprehensive model evaluation benchmark: Baseline Detector vs. Isolation Forest.

Evaluates detectors against ground-truth labels on the synthetic evaluation dataset.
Calculates Precision, Recall, F1-score, False Positive Rate (FPR), and per-scenario detection rate.
"""

import json
import logging
from pathlib import Path
import time
from typing import Any
import numpy as np
import pandas as pd

from backend.data.synthetic_eval import generate_evaluation_dataset
from backend.ml.detector import IsolationForestDetector
from backend.pipeline.classifier import IncidentClassifier
from backend.pipeline.preprocessor import MetricPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cloudops.evaluate")


def run_benchmark(
    n_samples_per_scenario: int = 20,
    model_path: str = "backend/ml/model.pkl",
    output_json_path: str = "evaluation_results.json",
    random_seed: int = 42
) -> dict[str, Any]:
    """Execute evaluation benchmark and return metrics dictionary."""
    logger.info("Generating synthetic evaluation dataset (8 scenarios, %d samples each)...", n_samples_per_scenario)
    eval_df = generate_evaluation_dataset(samples_per_scenario=n_samples_per_scenario, random_seed=random_seed)

    preprocessor = MetricPreprocessor()
    preprocessed = preprocessor.preprocess_dataframe(eval_df)

    # Load Isolation Forest detector
    detector = IsolationForestDetector()
    path = Path(model_path)
    if path.exists():
        detector.load(path)
        logger.info("Loaded pre-trained Isolation Forest model from %s", path)
    else:
        logger.warning("Model file not found at %s. Fitting on nominal baseline...", path)
        from backend.ml.train import train_model
        detector = train_model(model_output_path=str(path))

    classifier = IncidentClassifier()

    y_true = eval_df["ground_truth_anomaly"].to_numpy(dtype=int)
    total_samples = len(y_true)

    # 1. Evaluate Baseline Detector (Rule/Threshold-based: CPU > 80%)
    baseline_preds: list[int] = []
    t0 = time.perf_counter()
    for idx in range(total_samples):
        cpu_val = float(eval_df.loc[idx, "cpu_usage"])
        # Classical rule baseline only inspects CPU spikes
        baseline_preds.append(1 if cpu_val > 80.0 else 0)
    baseline_latency_ms = ((time.perf_counter() - t0) / total_samples) * 1000.0
    y_pred_baseline = np.array(baseline_preds, dtype=int)

    # 2. Evaluate Isolation Forest ML Detector
    iforest_preds: list[int] = []
    iforest_scores: list[float] = []
    t0 = time.perf_counter()
    for idx in range(total_samples):
        vec = preprocessed.feature_matrix[idx]
        res = detector.predict_vector(vec)
        iforest_preds.append(1 if res.is_anomaly else 0)
        iforest_scores.append(res.anomaly_score)
    iforest_latency_ms = ((time.perf_counter() - t0) / total_samples) * 1000.0
    y_pred_iforest = np.array(iforest_preds, dtype=int)

    # 3. Evaluate Combined Production Pipeline (Classifier + ML + Rules)
    pipeline_preds: list[int] = []
    t0 = time.perf_counter()
    for idx in range(total_samples):
        row_metrics = {
            "cpu_usage": float(eval_df.loc[idx, "cpu_usage"]),
            "memory_usage": float(eval_df.loc[idx, "memory_usage"]),
            "error_rate": float(eval_df.loc[idx, "error_rate"]),
            "response_time": float(eval_df.loc[idx, "response_time"])
        }
        vec = preprocessed.feature_matrix[idx]
        ml_res = detector.predict_vector(vec)
        anom_res = classifier.classify(metrics=row_metrics, ml_result=ml_res)
        pipeline_preds.append(1 if anom_res.is_anomaly else 0)
    pipeline_latency_ms = ((time.perf_counter() - t0) / total_samples) * 1000.0
    y_pred_pipeline = np.array(pipeline_preds, dtype=int)

    def calc_metrics(y_actual: np.ndarray, y_hat: np.ndarray, latency: float) -> dict[str, float]:
        tp = int(np.sum((y_actual == 1) & (y_hat == 1)))
        fp = int(np.sum((y_actual == 0) & (y_hat == 1)))
        tn = int(np.sum((y_actual == 0) & (y_hat == 0)))
        fn = int(np.sum((y_actual == 1) & (y_hat == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        accuracy = (tp + tn) / len(y_actual) if len(y_actual) > 0 else 0.0

        return {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "detection_rate": round(recall, 4),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
            "accuracy": round(accuracy, 4),
            "avg_latency_ms": round(latency, 4)
        }

    baseline_metrics = calc_metrics(y_true, y_pred_baseline, baseline_latency_ms)
    iforest_metrics = calc_metrics(y_true, y_pred_iforest, iforest_latency_ms)
    pipeline_metrics = calc_metrics(y_true, y_pred_pipeline, pipeline_latency_ms)

    # Per-scenario detection breakdown
    scenario_breakdown: dict[str, dict[str, Any]] = {}
    unique_scenarios = list(eval_df["scenario"].unique())

    for sc in unique_scenarios:
        mask = eval_df["scenario"] == sc
        sub_true = y_true[mask]
        sub_base = y_pred_baseline[mask]
        sub_if = y_pred_iforest[mask]
        sub_pipe = y_pred_pipeline[mask]

        is_anom_sc = int(sub_true[0]) == 1
        scenario_breakdown[sc] = {
            "is_anomaly_scenario": is_anom_sc,
            "sample_count": int(np.sum(mask)),
            "baseline_detected_count": int(np.sum(sub_base == 1)),
            "iforest_detected_count": int(np.sum(sub_if == 1)),
            "pipeline_detected_count": int(np.sum(sub_pipe == 1)),
            "baseline_detection_rate": round(float(np.mean(sub_base == 1) if is_anom_sc else np.mean(sub_base == 0)), 3),
            "iforest_detection_rate": round(float(np.mean(sub_if == 1) if is_anom_sc else np.mean(sub_if == 0)), 3),
            "pipeline_detection_rate": round(float(np.mean(sub_pipe == 1) if is_anom_sc else np.mean(sub_pipe == 0)), 3),
        }

    results = {
        "benchmark_timestamp": pd.Timestamp.now().isoformat(),
        "total_eval_samples": total_samples,
        "anomalous_samples": int(np.sum(y_true == 1)),
        "nominal_samples": int(np.sum(y_true == 0)),
        "models": {
            "baseline_cpu_threshold": baseline_metrics,
            "isolation_forest_ml": iforest_metrics,
            "cloudops_unified_pipeline": pipeline_metrics
        },
        "scenario_breakdown": scenario_breakdown
    }

    # Save to json file
    out_file = Path(output_json_path)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print summary table using ASCII
    print("\n" + "=" * 80)
    print(" CLOUDOPS AI - ANOMALY DETECTION BENCHMARK EVALUATION")
    print("=" * 80)
    print(f"Dataset: Synthetic Evaluation Dataset ({total_samples} samples across 8 distinct scenarios)")
    print("-" * 80)
    header = f"{'Metric':<22} | {'Baseline (CPU Rule)':<20} | {'Isolation Forest (ML)':<22} | {'Unified Pipeline':<18}"
    print(header)
    print("-" * 80)
    for m in ["precision", "recall", "f1_score", "false_positive_rate", "accuracy", "avg_latency_ms"]:
        b_val = f"{baseline_metrics[m]:.4f}" if "latency" not in m else f"{baseline_metrics[m]:.3f} ms"
        i_val = f"{iforest_metrics[m]:.4f}" if "latency" not in m else f"{iforest_metrics[m]:.3f} ms"
        p_val = f"{pipeline_metrics[m]:.4f}" if "latency" not in m else f"{pipeline_metrics[m]:.3f} ms"
        print(f"{m.replace('_', ' ').title():<22} | {b_val:<20} | {i_val:<22} | {p_val:<18}")
    print("-" * 80)
    print("\nScenario Breakdown (Detection Rate on Ground Truth):")
    print(f"{'Scenario':<22} | {'Type':<10} | {'Baseline':<10} | {'Isolation Forest':<18} | {'Pipeline':<10}")
    print("-" * 80)
    for sc, info in scenario_breakdown.items():
        stype = "ANOMALY" if info["is_anomaly_scenario"] else "NORMAL"
        b_rate = f"{info['baseline_detection_rate']*100:.1f}%"
        i_rate = f"{info['iforest_detection_rate']*100:.1f}%"
        p_rate = f"{info['pipeline_detection_rate']*100:.1f}%"
        print(f"{sc:<22} | {stype:<10} | {b_rate:<10} | {i_rate:<18} | {p_rate:<10}")
    print("=" * 80 + "\n")

    return results


if __name__ == "__main__":
    run_benchmark()
