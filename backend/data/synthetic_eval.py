"""Synthetic evaluation dataset generator for benchmark evaluation.

NOTE: This data is explicitly synthetic and used solely for testing and evaluating
the anomaly detection detectors (Baseline vs Isolation Forest) against ground truth labels.
It does not represent real AWS production telemetry.
"""

from typing import Any
import numpy as np
import pandas as pd

EVAL_SCENARIOS = {
    "normal": {
        "description": "Stable steady-state operation within standard operational parameters",
        "cpu_range": (25.0, 55.0),
        "mem_range": (35.0, 60.0),
        "err_range": (0.2, 1.8),
        "lat_range": (0.4, 1.4),
        "is_anomaly": 0,
        "incident_type": "NORMAL",
        "severity": "LOW"
    },
    "cpu_spike": {
        "description": "High CPU utilization spike caused by compute-intensive workload",
        "cpu_range": (88.0, 99.0),
        "mem_range": (40.0, 55.0),
        "err_range": (0.5, 2.0),
        "lat_range": (0.8, 1.8),
        "is_anomaly": 1,
        "incident_type": "CPU_PRESSURE",
        "severity": "CRITICAL"
    },
    "memory_spike": {
        "description": "Memory exhaustion nearing OOM limits",
        "cpu_range": (30.0, 50.0),
        "mem_range": (90.0, 99.0),
        "err_range": (0.5, 2.0),
        "lat_range": (0.9, 2.0),
        "is_anomaly": 1,
        "incident_type": "MEMORY_PRESSURE",
        "severity": "CRITICAL"
    },
    "error_spike": {
        "description": "Elevated HTTP 5xx / application unhandled error rate",
        "cpu_range": (40.0, 60.0),
        "mem_range": (45.0, 60.0),
        "err_range": (15.0, 35.0),
        "lat_range": (1.2, 3.0),
        "is_anomaly": 1,
        "incident_type": "ERROR_SPIKE",
        "severity": "CRITICAL"
    },
    "latency_spike": {
        "description": "Severe response time degradation (slow queries / downstream bottleneck)",
        "cpu_range": (45.0, 60.0),
        "mem_range": (50.0, 65.0),
        "err_range": (1.0, 3.0),
        "lat_range": (5.5, 12.0),
        "is_anomaly": 1,
        "incident_type": "LATENCY_DEGRADATION",
        "severity": "CRITICAL"
    },
    "cpu_latency": {
        "description": "Simultaneous CPU saturation and response latency degradation",
        "cpu_range": (86.0, 96.0),
        "mem_range": (50.0, 65.0),
        "err_range": (1.0, 3.5),
        "lat_range": (6.0, 10.5),
        "is_anomaly": 1,
        "incident_type": "RESOURCE_SATURATION",
        "severity": "CRITICAL"
    },
    "memory_latency": {
        "description": "Heavy swap / memory pressure leading to elevated response time",
        "cpu_range": (35.0, 55.0),
        "mem_range": (88.0, 98.0),
        "err_range": (1.0, 3.0),
        "lat_range": (5.0, 9.5),
        "is_anomaly": 1,
        "incident_type": "RESOURCE_SATURATION",
        "severity": "CRITICAL"
    },
    "multi_metric": {
        "description": "Compound cascade failure: high CPU, near OOM, high errors, and severe latency",
        "cpu_range": (90.0, 99.0),
        "mem_range": (89.0, 98.0),
        "err_range": (14.0, 32.0),
        "lat_range": (6.5, 15.0),
        "is_anomaly": 1,
        "incident_type": "MULTI_METRIC_ANOMALY",
        "severity": "CRITICAL"
    }
}


def generate_evaluation_dataset(samples_per_scenario: int = 20, random_seed: int = 42) -> pd.DataFrame:
    """Generate a reproducible synthetic evaluation DataFrame with ground truth labels."""
    rng = np.random.default_rng(random_seed)
    records: list[dict[str, Any]] = []

    for scenario_name, cfg in EVAL_SCENARIOS.items():
        cpu_min, cpu_max = cfg["cpu_range"]
        mem_min, mem_max = cfg["mem_range"]
        err_min, err_max = cfg["err_range"]
        lat_min, lat_max = cfg["lat_range"]

        cpus = rng.uniform(cpu_min, cpu_max, samples_per_scenario)
        mems = rng.uniform(mem_min, mem_max, samples_per_scenario)
        errs = rng.uniform(err_min, err_max, samples_per_scenario)
        lats = rng.uniform(lat_min, lat_max, samples_per_scenario)

        for i in range(samples_per_scenario):
            records.append({
                "scenario": scenario_name,
                "cpu_usage": round(float(cpus[i]), 2),
                "memory_usage": round(float(mems[i]), 2),
                "error_rate": round(float(errs[i]), 2),
                "response_time": round(float(lats[i]), 2),
                "status": "Healthy" if cfg["is_anomaly"] == 0 else "Critical",
                "ground_truth_anomaly": cfg["is_anomaly"],
                "expected_incident": cfg["incident_type"],
                "expected_severity": cfg["severity"],
            })

    return pd.DataFrame(records)


def get_single_scenario_sample(scenario_name: str, random_seed: int | None = None) -> dict[str, Any]:
    """Get a single realistic metric dictionary for an interactive simulation scenario."""
    if scenario_name not in EVAL_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(EVAL_SCENARIOS.keys())}")

    cfg = EVAL_SCENARIOS[scenario_name]
    rng = np.random.default_rng(random_seed)

    return {
        "cpu_usage": round(float(rng.uniform(*cfg["cpu_range"])), 1),
        "memory_usage": round(float(rng.uniform(*cfg["mem_range"])), 1),
        "error_rate": round(float(rng.uniform(*cfg["err_range"])), 1),
        "response_time": round(float(rng.uniform(*cfg["lat_range"])), 2),
        "scenario": scenario_name,
        "description": cfg["description"]
    }
