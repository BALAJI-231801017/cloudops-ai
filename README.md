# ☁️ CloudOps AI
### AI-Powered Cloud Infrastructure Monitoring & Root-Cause Analysis

CloudOps AI is an intelligent cloud observability and incident analysis platform. It combines **unsupervised multivariate Machine Learning (Isolation Forest)**, **deterministic incident classification**, and **evidence-grounded local Generative AI (Llama 3.2 via Ollama)** to detect performance degradation, categorize failures, and generate actionable technical recommendations across AWS CloudWatch telemetry and application health streams.

---

## 🎯 Core Architectural Principle

> **"ML detects abnormal behavior. Deterministic logic classifies the incident. The LLM explains the evidence and recommends safe advisory actions."**

---

## 🏗️ System Architecture

The system maintains a clean separation between **Live AWS Telemetry**, **Local CSV Telemetry**, and **Synthetic Simulation Telemetry**:

```
                       AWS Cloud Infrastructure
                                  │
                   ┌──────────────┴──────────────┐
                   ▼                             ▼
           Amazon EC2 API               AWS CloudWatch API
      (Instance Metadata & State)     (CPUUtilization, NetworkIn/Out)
                   │                             │
                   └──────────────┬──────────────┘
                                  ▼
                        ┌──────────────────┐
                        │    AwsService    │ ── (boto3 IAM Credential Chain)
                        └─────────┬────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
            ▼                     ▼                     ▼
   [Live AWS Telemetry]    [Local CSV Stream]    [Simulation Runner]
   (Read-Only CloudWatch)   (4D Metric Stream)    (8 Scenario Generator)
            │                     │                     │
            │                     ▼                     ▼
            │           ┌───────────────────┐           │
            │           │ MetricPreprocessor│ ◄─────────┘
            │           └─────────┬─────────┘
            │                     │
            │               ┌─────┴──────────────────┐
            │               ▼                        ▼
            │       ┌───────────────┐        ┌───────────────┐
            │       │ Baseline Rule │        │IsolationForest│
            │       │ (Mean + 2*Std)│        │  (ML Engine)  │
            │       └───────┬───────┘        └───────┬───────┘
            │               │                        │
            │               └───────────┬────────────┘
            │                           ▼
            │               ┌───────────────────────┐
            │               │  IncidentClassifier   │ ── (8 Deterministic Categories)
            │               └───────────┬───────────┘
            │                           ▼
            │               ┌───────────────────────┐
            └─────────────► │    EvidenceBuilder    │ ── (Canonical Evidence Object)
                            └───────────┬───────────┘
                                        ▼
                            ┌───────────────────────┐
                            │  Llama 3.2 / Ollama   │ ── (Evidence-Grounded Advisory)
                            └───────────┬───────────┘
                                        ▼
                            ┌───────────────────────┐
                            │    Flask REST API     │ ── (Port 5000: Routes, CORS, Latency)
                            └───────────┬───────────┘
                                        ▼
                            ┌───────────────────────┐
                            │    React Dashboard    │ ── (Port 5173: Real-Time UI, Benchmarks)
                            └───────────────────────┘
```

---

## 📡 Telemetry Sources & Data Paths

The application handles three distinct telemetry sources without conflating them:

1. **Live AWS Telemetry (`Live AWS Mode`)**:
   - Ingests `CPUUtilization`, `NetworkIn`, and `NetworkOut` from Amazon CloudWatch via `boto3`.
   - Discovers monitored EC2 instances via EC2 DescribeInstances API.
   - *Note on Memory:* Standard EC2 CloudWatch metrics do **not** include OS memory utilization without the CloudWatch Unified Agent. The system does not fabricate synthetic memory values for raw CloudWatch data.
2. **Local CSV Telemetry (`Flask + CSV Mode`)**:
   - Ingests 4-dimensional time-series records: `cpu_usage` (%), `memory_usage` (%), `error_rate` (%), and `response_time` (s).
   - Feeds the full `MetricPreprocessor` $\rightarrow$ `IsolationForestDetector` $\rightarrow$ `IncidentClassifier` pipeline.
3. **Synthetic Simulation Telemetry (`Simulation Mode`)**:
   - Evaluates 8 controlled degradation scenarios live through the identical preprocessing, ML inference, and classification pipeline.

---

## 🔬 Machine Learning Anomaly Detection

### Algorithm: Isolation Forest
Isolation Forest identifies anomalies by isolating observations through recursive random feature partitioning. Because anomalies are sparse and distinct in feature space, they require significantly fewer splits to isolate, resulting in shorter average path lengths in tree ensembles.

- **Feature Space**: $\mathbf{x} = [\text{cpu\_usage}, \text{memory\_usage}, \text{error\_rate}, \text{response\_time}] \in \mathbb{R}^4$.
- **Preprocessing & Scaling**: `StandardScaler` standardizes features to zero-mean and unit-variance. While tree partitioning is scale-invariant across individual features, standard scaling is maintained in the preprocessing pipeline to compute feature $z$-score deviations ($(x - \mu)/\sigma$) for feature contribution explanations.
- **Contamination**: Configured to `0.05` (5% expected anomalies in nominal training baseline).
- **Application-Level Anomaly Score Normalization**:
  Isolation Forest's `decision_function(x)` returns continuous values (negative for outliers, positive for inliers). The application normalizes this into an empirical $[0.0, 1.0]$ severity score:
  $$\text{Normalized Anomaly Score} = \text{clip}\left(\frac{-\text{decision\_function}(\mathbf{x})}{0.40}, 0.0, 1.0\right)$$
  - $\text{Score} \ge 0.70 \rightarrow \text{CRITICAL}$
  - $\text{Score} \ge 0.50 \rightarrow \text{HIGH}$
  - $\text{Score} \ge 0.35 \rightarrow \text{MEDIUM}$
  - $\text{Score} < 0.35 \rightarrow \text{LOW (Nominal)}$

---

## 📊 Synthetic Evaluation Benchmark

The detectors were evaluated on a controlled **Synthetic Evaluation Benchmark** (160 samples across 8 distinct scenarios with ground-truth labels). Reproducible via `python -m evaluate`.

| Metric | Baseline (CPU Rule) | Isolation Forest (ML) | CloudOps Unified Pipeline |
|---|---|---|---|
| **Precision** | `1.0000` | `0.9429` | `0.9459` |
| **Recall / Anomaly Detection Rate** | `0.4286` | `0.9429` | `1.0000` |
| **F1 Score** | `0.6000` | `0.9429` | `0.9722` |
| **False Positive Rate (FPR)** | `0.0000` | `0.4000` | `0.4000` |
| **Overall Accuracy** | `0.5000` | `0.9000` | `0.9500` |
| **Avg Inference Latency** | `0.016 ms` | `31.365 ms` | `31.433 ms` |

### Per-Scenario Classification Behavior:
- `normal` (Nominal): Baseline 100.0% · ML 60.0% · Pipeline 60.0%
- `cpu_spike` (Anomaly): Baseline 100.0% · ML 70.0% · Pipeline 100.0%
- `memory_spike` (Anomaly): Baseline 0.0% (missed) · ML 90.0% · Pipeline 100.0%
- `error_spike` (Anomaly): Baseline 0.0% (missed) · ML 100.0% · Pipeline 100.0%
- `latency_spike` (Anomaly): Baseline 0.0% (missed) · ML 100.0% · Pipeline 100.0%
- `cpu_latency` (Anomaly): Baseline 100.0% · ML 100.0% · Pipeline 100.0%
- `memory_latency` (Anomaly): Baseline 0.0% (missed) · ML 100.0% · Pipeline 100.0%
- `multi_metric` (Anomaly): Baseline 100.0% · ML 100.0% · Pipeline 100.0%

### Technical Analysis of Results:
1. **Univariate Rule Blindness**: The baseline CPU rule achieves only **42.86% Recall** because it cannot detect memory pressure, latency degradation, or error spikes.
2. **Analysis of the 40% False Positive Rate (FPR)**: In the nominal scenario, the default sensitivity threshold ($0.35$) on synthetic noise flagged 8 of 20 nominal points as mild anomalies. The model intentionally prioritizes high recall ($100\%$ on true failure scenarios). In production, this can be mitigated using **temporal persistence windows** (e.g., requiring 3 consecutive anomalous samples) and threshold calibration.

---

## 🤖 Evidence-Grounded Generative AI Diagnosis

The LLM layer utilizes local **Llama 3.2 (3B)** via **Ollama**.

### Grounding & Guardrail Controls:
1. **Structured Context Injection**: The prompt receives the immutable `EvidenceObject` containing exact metric values, $z$-score feature deviations, threshold violations, and detector scores.
2. **Prompt Constraints**: System instructions forbid inventing unmonitored metrics, phantom server logs, or fake AWS resource IDs.
3. **Structured Schema Output**: The LLM outputs a technical JSON payload:
   ```json
   {
     "incident_summary": "High CPU utilization (94.5%) combined with memory saturation (88.2%).",
     "probable_root_cause": "Resource contention caused by thread lock or traffic surge.",
     "confidence": 0.85,
     "supporting_evidence": ["CPU usage at 94.5% exceeds critical threshold (80.0%)"],
     "recommended_actions": ["Inspect CPU-heavy worker processes", "Review database connection pool"],
     "limitations": ["Single-host metrics; cluster-wide logs not present in evidence."]
   }
   ```
4. **LLM-Reported Confidence Estimate**: The confidence field represents the model's self-assessed estimate based on evidence sufficiency (not a statistically calibrated probability).
5. **Strictly Advisory**: The LLM recommends safe technical actions for engineer review; it never executes autonomous remediation.

---

## ☁️ AWS Telemetry & Security

- **Programmatic Integration**: Uses `boto3` for `ec2:DescribeInstances` and `cloudwatch:GetMetricStatistics`.
- **Credential Resolution**: Resolves credentials via standard AWS credential chain (IAM instance roles, environment variables, or local profile `CLOUDOPS_AWS_PROFILE`). No credentials are ever hardcoded.
- **Read-Only Scope**: The service requires strictly read-only monitoring permissions.
- **Frontend Isolation**: AWS credentials and secret keys are never exposed to the client-side browser.

> **Deployment Status**: CloudOps AI is **integrated with AWS CloudWatch** as a telemetry consumer and is **Dockerized and deployment-ready**. The CloudOps AI application itself runs locally / in Docker and is **not currently hosted on AWS**.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.12+
- Node.js 18+ & npm
- [Ollama](https://ollama.ai/): `ollama run llama3.2:3b`

### 1. Backend Setup
```bash
# Install Python dependencies
pip install -r backend/requirements.txt

# Train the Isolation Forest model on nominal baseline telemetry
python -m backend.ml.train

# Start the Flask REST API server
python -m backend.app
# Backend runs on http://127.0.0.1:5000
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
# Frontend runs on http://localhost:5173
```

### 3. Model Benchmark Evaluation
```bash
python -m evaluate
```

### 4. Automated Test Suite
```bash
python -m pytest backend/tests/ -v
```

---

## 🐳 Docker Deployment

The repository includes a containerized setup for the Flask API and Ollama service:

```bash
docker-compose up --build
```

- **Flask API Container**: `http://localhost:5000` (Health checked via `/api/health`).
- **Ollama Container**: `http://localhost:11434`.
  - *Model Initialization*: On first run of the Ollama container, download the model:
    ```bash
    docker exec -it cloudops-ai-ollama ollama pull llama3.2:3b
    ```
- **React Frontend**: Started locally via `npm run dev` in `frontend/`.

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check verifying ML model, Ollama daemon, and CSV telemetry status |
| `GET` | `/api/instances` | Discover monitored AWS EC2 instances |
| `GET` | `/api/metrics/<id>?hours=1` | Retrieve CloudWatch metrics (`CPUUtilization`, `NetworkIn`, `NetworkOut`) |
| `GET` | `/api/health-analysis` | Full pipeline analysis over local CSV telemetry |
| `POST` | `/api/analyze` | Request evidence-grounded AI root-cause diagnosis |
| `POST` | `/api/simulate` | Execute interactive synthetic incident scenario |
| `GET` | `/api/evaluation` | Retrieve benchmark evaluation metrics |

---

## 📁 Repository Structure

```
cloudops-ai/
├── backend/
│   ├── data/
│   │   ├── synthetic_eval.py       # Benchmark evaluation dataset generator
│   │   └── training_baseline.csv   # Nominal training data (500 samples)
│   ├── ml/
│   │   ├── detector.py             # Isolation Forest ML Detector
│   │   ├── model.pkl               # Serialized StandardScaler + IsolationForest
│   │   └── train.py                # Model training and verification script
│   ├── pipeline/
│   │   ├── classifier.py           # Deterministic incident classification
│   │   ├── evidence.py             # Evidence object builder for LLM
│   │   └── preprocessor.py         # Telemetry validation, imputation, and clamping
│   ├── tests/
│   │   ├── test_api.py             # Flask API endpoint unit tests
│   │   ├── test_classifier.py      # Classification logic unit tests
│   │   ├── test_detector.py        # ML Isolation Forest unit tests
│   │   ├── test_pipeline.py        # End-to-end integration tests
│   │   └── test_preprocessor.py    # Metric preprocessor unit tests
│   ├── ai_service.py               # Ollama / Llama 3.2 interface
│   ├── app.py                      # Flask REST API factory & routes
│   ├── aws_service.py              # AWS EC2 & CloudWatch integration
│   ├── config.py                   # Environment configuration dataclass
│   ├── health_service.py           # Telemetry ingestion orchestrator
│   └── requirements.txt            # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx                 # React Dashboard SPA
│   │   ├── main.jsx                # React DOM entry point
│   │   └── styles.css              # Dark-mode dashboard stylesheet
│   ├── index.html                  # HTML shell
│   ├── package.json                # Frontend dependencies (React + Vite)
│   └── vite.config.js              # Vite dev server with /api proxy
├── .dockerignore
├── .env.example                    # Environment variable template
├── .gitignore
├── app.py                          # Standalone CLI demonstration tool
├── docker-compose.yml              # Container orchestration (API + Ollama)
├── Dockerfile                      # Production container build
├── evaluate.py                     # Benchmark evaluation execution script
├── health_data.csv                 # 72-hour demonstration telemetry
├── README.md                       # Comprehensive technical documentation
├── requirements.txt                # Unified dependency definitions
└── test_ai.py                      # Local Ollama connectivity verification
```

---

## ⚖️ Known Limitations & Future Work

1. **Pull-Based Telemetry**: Telemetry is ingested on request. A production deployment would use streaming ingestion via message queues (e.g., Apache Kafka or AWS Kinesis).
2. **False Positive Rate Calibration**: The default $0.35$ anomaly threshold produces false positives on noisy nominal telemetry ($FPR = 0.40$). Alert suppression windows or dynamic threshold calibration can be added.
3. **CloudWatch Memory Ingestion**: Standard CloudWatch EC2 metrics do not expose memory without the CloudWatch Agent. Future iterations could ingest CloudWatch Custom Metrics.
4. **Model Storage**: Model artifacts are stored locally via `joblib`. Production scaling would use an artifact registry such as AWS S3.
