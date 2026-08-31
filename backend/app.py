"""Production-ready Flask REST API for CloudOps AI."""

import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any

from flask import Flask, jsonify, request

from backend.ai_service import AiService, AiServiceError
from backend.aws_service import AwsService, AwsServiceError
from backend.config import Settings
from backend.data.synthetic_eval import EVAL_SCENARIOS, get_single_scenario_sample
from backend.health_service import HealthService, HealthServiceError
from backend.ml.detector import IsolationForestDetector, MLAnomalyResult
from backend.pipeline.classifier import AnomalyResult, IncidentClassifier
from backend.pipeline.evidence import EvidenceBuilder
from backend.pipeline.preprocessor import MetricPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
logger = logging.getLogger("cloudops.api")

EC2_INSTANCE_ID_PATTERN = re.compile(r"^i-[0-9a-fA-F]{8,17}$")


def create_app(
    settings: Settings | None = None,
    aws_service: AwsService | None = None,
    health_service: HealthService | None = None,
    ai_service: AiService | None = None,
    detector: IsolationForestDetector | None = None,
    classifier: IncidentClassifier | None = None,
) -> Flask:
    """Application factory for CloudOps AI Flask backend."""
    settings = settings or Settings()

    # Model initialization
    if detector is None:
        detector = IsolationForestDetector()
        model_p = Path(settings.model_path)
        if model_p.exists():
            try:
                detector.load(model_p)
                logger.info("Loaded trained Isolation Forest detector from: %s", model_p)
            except Exception as e:
                logger.warning("Could not load model from %s: %s", model_p, e)
        else:
            logger.warning("ML model not found at %s. Baseline rules active. Run python -m backend.ml.train", model_p)

    classifier = classifier or IncidentClassifier()
    aws_service = aws_service or AwsService(settings)
    health_service = health_service or HealthService(
        data_path=settings.health_data_path,
        detector=detector,
        classifier=classifier
    )
    ai_service = ai_service or AiService(settings)
    preprocessor = MetricPreprocessor()

    app = Flask(__name__)

    # Observability: Track request start time and duration
    @app.before_request
    def start_timer():
        request.environ["_cloudops_start_time"] = time.perf_counter()

    @app.after_request
    def record_metrics_and_cors(response):
        # Latency tracking
        start_t = request.environ.get("_cloudops_start_time")
        duration_ms = ((time.perf_counter() - start_t) * 1000.0) if start_t else 0.0
        logger.info(
            "%s %s -> %s (%.2f ms)",
            request.method,
            request.path,
            response.status_code,
            duration_ms
        )

        # CORS
        allowed_origins = {
            settings.frontend_origin,
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        }
        origin = request.headers.get("Origin")
        if origin in allowed_origins or not origin:
            response.headers["Access-Control-Allow-Origin"] = origin if origin else "*"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    # 1. API Health & Observability endpoint
    @app.get("/api/health")
    def health_check():
        ollama_ok = ai_service.is_available()
        ml_ok = detector.is_trained() if detector else False
        data_ok = Path(settings.health_data_path).exists()

        return jsonify({
            "status": "healthy",
            "service": "cloudops-ai-api",
            "version": "2.0.0",
            "region": settings.aws_region,
            "components": {
                "isolation_forest_model": "loaded" if ml_ok else "not_loaded",
                "ollama_inference": "connected" if ollama_ok else "offline",
                "health_telemetry_csv": "available" if data_ok else "missing"
            }
        })

    # 2. AWS EC2 Instance Discovery
    @app.get("/api/instances")
    def list_instances():
        instances = aws_service.list_instances()
        return jsonify({"instances": instances, "region": settings.aws_region, "count": len(instances)})

    # 3. AWS CloudWatch Metrics
    @app.get("/api/metrics/<instance_id>")
    def get_metrics(instance_id: str):
        if not EC2_INSTANCE_ID_PATTERN.match(instance_id):
            return jsonify({"error": f"Invalid EC2 instance ID format: '{instance_id}'"}), 400

        hours = request.args.get("hours", default=1, type=int)
        if hours is None or not (1 <= hours <= 24):
            return jsonify({"error": "Query parameter 'hours' must be an integer between 1 and 24"}), 400

        data = aws_service.get_instance_metrics(instance_id, hours)
        return jsonify(data)

    # 4. Health Analysis (CSV / Local Telemetry)
    @app.get("/api/health-analysis")
    def health_analysis():
        return jsonify(health_service.analyze())

    # 5. Full Pipeline Analysis & Grounded AI Diagnosis
    @app.post("/api/analyze")
    def analyze():
        payload = request.get_json(silent=True) or {}
        question = payload.get("question", "What is the probable root cause of the current application health state?").strip()
        if not question:
            return jsonify({"error": "Field 'question' cannot be empty"}), 400
        if len(question) > 500:
            return jsonify({"error": "Field 'question' exceeds maximum length of 500 characters"}), 400

        instance_id = payload.get("instance_id")
        if instance_id and not EC2_INSTANCE_ID_PATTERN.match(instance_id):
            return jsonify({"error": f"Invalid EC2 instance ID format: '{instance_id}'"}), 400

        # Execute monitoring pipeline
        health_data = health_service.analyze()
        evidence_dict = health_data.get("evidence", {})

        cloudwatch_data = None
        if instance_id:
            try:
                cloudwatch_data = aws_service.get_instance_metrics(instance_id, hours=1)
                evidence_dict["cloudwatch_metrics"] = cloudwatch_data
            except Exception as e:
                logger.warning("CloudWatch retrieval error for %s: %s", instance_id, e)
                evidence_dict["cloudwatch_error"] = str(e)

        # Call AI service with graceful fallback
        ai_available = True
        analysis_result = None
        try:
            analysis_result = ai_service.analyze(question=question, evidence=evidence_dict)
        except AiServiceError as err:
            logger.info("AI Service unavailable: %s", err)
            ai_available = False
            analysis_result = {
                "incident_summary": "AI Diagnosis is currently unavailable (Ollama is offline).",
                "probable_root_cause": "Deterministic incident classification is active below.",
                "confidence": 0.0,
                "supporting_evidence": health_data.get("evidence_statements", []),
                "recommended_actions": [
                    "Inspect affected metrics identified by the deterministic classifier",
                    "Verify memory and CPU utilization trends",
                    "Check server logs for recent error exceptions"
                ],
                "limitations": [str(err)]
            }

        return jsonify({
            "ai_available": ai_available,
            "analysis": analysis_result,
            "evidence": evidence_dict,
            "health_summary": {
                "health_score": health_data.get("health_score"),
                "severity": health_data.get("severity"),
                "incident_type": health_data.get("incident_type"),
                "primary_detector": health_data.get("primary_detector"),
                "anomaly_score": health_data.get("anomaly_score")
            }
        })

    # 6. Interactive Scenario Simulation Runner
    @app.post("/api/simulate")
    def simulate_scenario():
        payload = request.get_json(silent=True) or {}
        scenario_name = payload.get("scenario", "cpu_spike")
        if scenario_name not in EVAL_SCENARIOS:
            return jsonify({
                "error": f"Unknown simulation scenario '{scenario_name}'",
                "available_scenarios": list(EVAL_SCENARIOS.keys())
            }), 400

        # Retrieve synthetic telemetry for this scenario
        sample = get_single_scenario_sample(scenario_name)
        metrics = {
            "cpu_usage": sample["cpu_usage"],
            "memory_usage": sample["memory_usage"],
            "error_rate": sample["error_rate"],
            "response_time": sample["response_time"],
        }

        # 1. Preprocess
        vector, cleaned_metrics = preprocessor.preprocess_single(metrics)

        # 2. Run Isolation Forest ML Anomaly Detector
        ml_res: MLAnomalyResult | None = None
        if detector and detector.is_trained():
            ml_res = detector.predict_vector(vector)

        # 3. Deterministic Incident Classification
        anomaly_res = classifier.classify(metrics=cleaned_metrics, ml_result=ml_res)

        # 4. Construct Evidence Object
        evidence = EvidenceBuilder.build(
            metrics=cleaned_metrics,
            anomaly_result=anomaly_res,
            data_source="simulation"
        )
        evidence_dict = evidence.to_dict()

        # 5. Attempt AI Diagnosis
        ai_available = True
        analysis_result = None
        try:
            analysis_result = ai_service.analyze(
                question=f"Analyze simulated scenario: {scenario_name}",
                evidence=evidence_dict
            )
        except AiServiceError:
            ai_available = False
            analysis_result = {
                "incident_summary": f"Simulated {scenario_name.replace('_', ' ')} scenario analyzed deterministically.",
                "probable_root_cause": sample.get("description", "Synthetic simulation pattern."),
                "confidence": 0.85,
                "supporting_evidence": anomaly_res.evidence_statements,
                "recommended_actions": [
                    "Investigate root cause of simulated metric surge",
                    "Verify alert triggers for this metric threshold"
                ],
                "limitations": ["Simulation mode (synthetic telemetry)"]
            }

        return jsonify({
            "simulation_mode": True,
            "scenario": scenario_name,
            "description": sample.get("description"),
            "metrics": cleaned_metrics,
            "health_score": anomaly_res.health_score,
            "severity": anomaly_res.severity.value,
            "incident_type": anomaly_res.incident_type.value,
            "is_anomaly": anomaly_res.is_anomaly,
            "primary_detector": anomaly_res.primary_detector,
            "anomaly_score": anomaly_res.anomaly_score,
            "affected_metrics": anomaly_res.affected_metrics,
            "evidence_statements": anomaly_res.evidence_statements,
            "ml_result": anomaly_res.ml_result,
            "ai_available": ai_available,
            "analysis": analysis_result,
            "evidence": evidence_dict
        })

    # 7. Model Evaluation Results Benchmark Endpoint
    @app.get("/api/evaluation")
    def get_evaluation():
        eval_file = Path("evaluation_results.json")
        if eval_file.exists():
            try:
                with open(eval_file, "r", encoding="utf-8") as f:
                    return jsonify(json.load(f))
            except Exception as e:
                logger.warning("Could not read %s: %s", eval_file, e)

        # If not on disk, run benchmark live
        from evaluate import run_benchmark
        results = run_benchmark(n_samples_per_scenario=20, output_json_path=str(eval_file))
        return jsonify(results)

    # Error handling
    @app.errorhandler(AwsServiceError)
    def handle_aws_error(error):
        return jsonify({"error": str(error), "service": "aws"}), 503

    @app.errorhandler(HealthServiceError)
    def handle_health_error(error):
        return jsonify({"error": str(error), "service": "health"}), 500

    @app.errorhandler(AiServiceError)
    def handle_ai_error(error):
        return jsonify({"error": str(error), "service": "ai"}), 503

    return app


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    create_app().run(host="0.0.0.0", port=5000, debug=debug_mode)
