"""CloudOps AI - Interactive CLI Demonstration Tool.

Executes the end-to-end monitoring pipeline from the command line:
Metric Ingestion -> Preprocessing -> Isolation Forest ML -> Classification -> LLM Root-Cause Analysis.
"""

import sys
import pandas as pd

from backend.ai_service import AiService, AiServiceError
from backend.config import Settings
from backend.health_service import HealthService
from backend.ml.detector import IsolationForestDetector


def main():
    print("=" * 70)
    print(" CLOUDOPS AI - INFRASTRUCTURE HEALTH & INCIDENT DIAGNOSTIC CLI")
    print("=" * 70)

    settings = Settings()

    # Load ML detector
    detector = IsolationForestDetector()
    try:
        detector.load(settings.model_path)
        print(f"[ML Engine] Loaded Isolation Forest detector ({settings.model_path})")
    except Exception:
        print("[ML Engine] Model artifact not found. Run: python -m backend.ml.train")

    health_svc = HealthService(data_path=settings.health_data_path, detector=detector)
    ai_svc = AiService(settings)

    print(f"[Telemetry] Ingesting health records from: {settings.health_data_path}")
    try:
        report = health_svc.analyze()
    except Exception as e:
        print(f"[ERROR] Failed to process telemetry: {e}")
        sys.exit(1)

    latest = report["latest"]
    print("\n--- Current Metric State ---")
    print(f" CPU Utilization : {latest['cpu_usage']}%")
    print(f" Memory Usage    : {latest['memory_usage']}%")
    print(f" Error Rate      : {latest['error_rate']}%")
    print(f" Response Time   : {latest['response_time']}s")
    print(f" Health Score    : {report['health_score']}/100 ({report['severity']})")

    print("\n--- Detection & Classification ---")
    print(f" Incident Type   : {report['incident_type']}")
    print(f" Primary Detector: {report['primary_detector']}")
    print(f" Anomaly Score   : {report['anomaly_score']}")
    print(f" Anomaly State   : {'ANOMALOUS' if report['is_anomaly'] else 'NOMINAL'}")

    print("\n--- Evidence Statements ---")
    for stmt in report["evidence_statements"]:
        print(f" * {stmt}")

    print("\n--- LLM Root-Cause Analysis (Llama 3.2 via Ollama) ---")
    try:
        diagnosis = ai_svc.analyze(
            question="Diagnose current incident and recommend technical remediation.",
            evidence=report["evidence"]
        )
        print(f"\nSummary: {diagnosis.get('incident_summary')}")
        print(f"Root Cause: {diagnosis.get('probable_root_cause')}")
        print(f"Confidence: {int(diagnosis.get('confidence', 0) * 100)}%")
        print("Recommended Actions:")
        for action in diagnosis.get("recommended_actions", []):
            print(f" - {action}")
    except AiServiceError as err:
        print(f"[NOTE] Local LLM offline ({err}). Deterministic classification displayed above.")

    print("=" * 70)


if __name__ == "__main__":
    main()
