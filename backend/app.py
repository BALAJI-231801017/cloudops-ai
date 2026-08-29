"""Flask REST API for CloudOps AI."""

import os

from flask import Flask, jsonify, request

from backend.ai_service import AiService, AiServiceError
from backend.aws_service import AwsService, AwsServiceError
from backend.config import Settings
from backend.health_service import HealthService, HealthServiceError


def create_app(settings=None, aws_service=None, health_service=None, ai_service=None):
    settings = settings or Settings()
    aws_service = aws_service or AwsService(settings)
    health_service = health_service or HealthService(settings.health_data_path)
    ai_service = ai_service or AiService(settings)
    app = Flask(__name__)

    @app.after_request
    def add_frontend_cors_headers(response):
        allowed_origins = {
            os.getenv("FRONTEND_ORIGIN", "http://localhost:5173"),
            "http://127.0.0.1:5173",
        }
        origin = request.headers.get("Origin")
        if origin in allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @app.get("/api/health")
    def health_check():
        return jsonify({"status": "ok", "service": "cloudops-api"})

    @app.get("/api/instances")
    def list_instances():
        return jsonify({"instances": aws_service.list_instances(), "region": settings.aws_region})

    @app.get("/api/metrics/<instance_id>")
    def get_metrics(instance_id):
        hours = request.args.get("hours", default=1, type=int)
        if hours is None or not 1 <= hours <= 24:
            return jsonify({"error": "hours must be an integer from 1 through 24"}), 400
        return jsonify(aws_service.get_instance_metrics(instance_id, hours))

    @app.get("/api/health-analysis")
    def health_analysis():
        return jsonify(health_service.analyze())

    @app.post("/api/analyze")
    def analyze():
        payload = request.get_json(silent=True) or {}
        question = payload.get("question", "What needs attention in this environment?").strip()
        if not question:
            return jsonify({"error": "question must not be empty"}), 400

        evidence = {"health_analysis": health_service.analyze()}
        instance_id = payload.get("instance_id")
        if instance_id:
            evidence["cloudwatch_metrics"] = aws_service.get_instance_metrics(instance_id)
        return jsonify({"analysis": ai_service.analyze(question, evidence), "evidence": evidence})

    @app.errorhandler(AwsServiceError)
    @app.errorhandler(HealthServiceError)
    @app.errorhandler(AiServiceError)
    def service_error(error):
        return jsonify({"error": str(error)}), 503

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000, debug=True)
