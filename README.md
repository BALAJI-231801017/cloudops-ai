# ☁️ CloudOps AI
### AI-Powered Cloud Infrastructure Monitoring & Root-Cause Analysis

CloudOps AI is an intelligent cloud observability and automated incident analysis platform. It pairs **unsupervised multivariate Machine Learning (Isolation Forest)** with **deterministic incident classification** and **grounded local Generative AI (Llama 3.2 via Ollama)** to detect performance degradation, pinpoint root causes, and advise remediation steps across AWS CloudWatch telemetry and application health streams.

---

## 🎯 Core Architectural Principle

> **"ML detects abnormal behavior. Deterministic logic classifies the incident. The LLM explains the evidence and recommends safe advisory actions."**

```
   AWS Infrastructure (EC2)
             │
       AWS CloudWatch (Telemetry)
             │
             ▼
   ┌────────────────────┐
   │  Metric Collector  │ ── (CPU, Memory, Error Rate, Latency)
   └─────────┬──────────┘
             ▼
   ┌────────────────────┐
   │ MetricPreprocessor │ ── (Validation, Imputation, Clamping, Feature Matrix)
   └─────────┬──────────┘
             │
       ┌─────┴──────────────────┐
       ▼                        ▼
┌───────────────┐     ┌──────────────────────┐
│ Baseline Rule │     │ Isolation Forest ML  │
│  (Threshold)  │     │  (StandardScaler)    │
└───────┬───────┘     └──────────┬───────────┘
        │                        │
        └───────────┬────────────┘
                    ▼
       ┌────────────────────────┐
       │  Incident Classifier   │ ── (Deterministic Rules & Official Severity)
       └────────────┬───────────┘
                    ▼
       ┌────────────────────────┐
       │    Evidence Builder    │ ── (Canonical Telemetry & Anomaly Evidence)
       └────────────┬───────────┘
                    ▼
       ┌────────────────────────┐
       │   Llama 3.2 / Ollama   │ ── (Evidence-Grounded Root-Cause Diagnosis)
       └────────────┬───────────┘
                    ▼
       ┌────────────────────────┐
       │     Flask REST API     │ ── (Port 5000: Observability, Endpoints, CORS)
       └────────────┬───────────┘
                    ▼
       ┌────────────────────────┐
       │    React Dashboard     │ ── (Port 5173: Real-Time Charts, Simulator, Benchmark)
       └────────────────────────┘
```

---

## 🚀 Key Capabilities

- **Multivariate ML Anomaly Detection**: Unsupervised Isolation Forest model trained on nominal baseline telemetry to detect subtle multi-metric correlations.
- **Dynamic Baseline Comparison**: Dual detection strategy comparing rule-based thresholds against multivariate ML path lengths.
- **Deterministic Incident Categorization**: Transparent classification into 8 incident categories (`RESOURCE_SATURATION`, `CPU_PRESSURE`, `MEMORY_PRESSURE`, `ERROR_SPIKE`, `LATENCY_DEGRADATION`, `MULTI_METRIC_ANOMALY`, `UNKNOWN_ANOMALY`, `NORMAL`).
- **Evidence-Grounded GenAI Diagnosis**: Local Llama 3.2 model via Ollama constrained strictly to telemetry evidence (zero fabricated metrics or phantom cloud resources).
- **Graceful Degradation**: If Ollama or AWS is unreachable, monitoring, anomaly scoring, and classification continue uninterrupted.
- **Interactive Scenario Simulator**: 8 built-in incident scenarios evaluated live through the exact same preprocessing and ML pipeline.
- **AWS CloudWatch & EC2 Telemetry**: Read-only integration with AWS EC2 instance discovery and CloudWatch metric statistics using IAM role authentication.
- **Dockerized & Deployment-Ready**: Containerized backend with pre-trained model artifacts and `docker-compose` orchestration.

---

## 🔬 Machine Learning Architecture

### Algorithm: Isolation Forest
Isolation Forest detects anomalies by randomly partitioning feature dimensions. Because anomalies are few and distinct in metric space, they require significantly fewer recursive splits to isolate, resulting in shorter average path lengths in tree ensembles.

- **Feature Matrix**: $\mathbf{X} \in \mathbb{R}^{N \times 4}$ composed of:
  1. `cpu_usage` (%)
  2. `memory_usage` (%)
  3. `error_rate` (%)
  4. `response_time` (seconds)
- **Feature Normalization**: `StandardScaler` standardizes features to zero-mean and unit-variance prior to tree construction to prevent scale dominance.
- **Contamination**: Set to `0.05` (5% expected anomalies in nominal training baseline).
- **Anomaly Score Normalization**:
  $$\text{Score} = \text{clip}\left(\frac{-\text{decision\_function}(\mathbf{x})}{0.40}, 0.0, 1.0\right)$$
  - $\text{Score} \ge 0.70 \rightarrow \text{CRITICAL}$
  - $\text{Score} \ge 0.50 \rightarrow \text{HIGH}$
  - $\text{Score} \ge 0.35 \rightarrow \text{MEDIUM}$
  - $\text{Score} < 0.35 \rightarrow \text{LOW (Nominal)}$

---

## 📊 Measured Benchmark Evaluation

The models were evaluated on a synthetic benchmark dataset (160 samples across 8 distinct scenarios with ground-truth labels). Reproducible via `python -m evaluate`.

| Metric | Baseline (CPU Rule) | Isolation Forest (ML) | CloudOps Unified Pipeline |
|---|---|---|---|
| **Precision** | `1.0000` | `0.9429` | `0.9459` |
| **Recall / Detection Rate** | `0.4286` | `0.9429` | `1.0000` |
| **F1 Score** | `0.6000` | `0.9429` | `0.9722` |
| **False Positive Rate** | `0.0000` | `0.4000` | `0.4000` |
| **Overall Accuracy** | `0.5000` | `0.9000` | `0.9500` |
| **Avg Inference Latency** | `0.013 ms` | `32.080 ms` | `31.140 ms` |

### Per-Scenario Detection Rate:
- `normal` (Nominal): Baseline 100.0% · ML 60.0% · Pipeline 60.0%
- `cpu_spike` (Anomaly): Baseline 100.0% · ML 70.0% · Pipeline 100.0%
- `memory_spike` (Anomaly): Baseline 0.0% (missed) · ML 90.0% · Pipeline 100.0%
- `error_spike` (Anomaly): Baseline 0.0% (missed) · ML 100.0% · Pipeline 100.0%
- `latency_spike` (Anomaly): Baseline 0.0% (missed) · ML 100.0% · Pipeline 100.0%
- `cpu_latency` (Anomaly): Baseline 100.0% · ML 100.0% · Pipeline 100.0%
- `memory_latency` (Anomaly): Baseline 0.0% (missed) · ML 100.0% · Pipeline 100.0%
- `multi_metric` (Anomaly): Baseline 100.0% · ML 100.0% · Pipeline 100.0%

*Key Insight:* The baseline CPU rule achieves only **42.86% Recall** because it cannot detect memory, latency, or error anomalies. The hybrid Unified Pipeline achieves **100% Recall** across all incident scenarios.

---

## 🤖 Generative AI: Grounded Root-Cause Diagnosis

The LLM layer utilizes local **Llama 3.2 (3B)** via **Ollama**.

### Anti-Hallucination Controls:
1. **Strict Context Injection**: The prompt receives a structured JSON `EvidenceObject` containing exact metric values, z-score deviations, threshold violations, and detector scores.
2. **System Prompt Guardrails**: Explicit constraints forbid inventing telemetry, external logs, or phantom AWS resource IDs.
3. **Structured JSON Output**: The model outputs a strict JSON payload:
   ```json
   {
     "incident_summary": "High CPU utilization (94.5%) combined with memory saturation (88.2%).",
     "probable_root_cause": "Resource contention caused by thread lock or traffic surge.",
     "confidence": 0.85,
     "supporting_evidence": ["CPU usage at 94.5% exceeds critical threshold (80.0%)"],
     "recommended_actions": ["Inspect high CPU worker threads", "Review database connection pool"],
     "limitations": ["Single-host metrics; cluster-wide logs not present in evidence."]
   }
   ```
4. **Resilient Fallback Parsing**: Regex-based JSON extractor with structured schema fallback if the LLM emits free-form text.

---

## ☁️ AWS Telemetry Integration

CloudOps AI communicates with AWS via `boto3`:
- `EC2:DescribeInstances`: Discovers monitored instances, availability zones, state, and IP addresses.
- `CloudWatch:GetMetricStatistics`: Queries time-series history for `CPUUtilization`, `NetworkIn`, and `NetworkOut`.
- **Security & IAM**: Uses AWS default credential chain (IAM roles when on EC2, or `~/.aws/credentials`). No credentials are hardcoded.

> **Deployment Status Clarification**: CloudOps AI is **integrated with AWS CloudWatch** as a monitoring consumer and is **deployment-ready**. The CloudOps AI monitoring application itself runs locally / on Docker and is not currently hosted on AWS.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.12+
- Node.js 18+ & npm
- [Ollama](https://ollama.ai/) (for local LLM diagnosis): `ollama run llama3.2:3b`

### 1. Backend Setup
```bash
# Install Python dependencies
pip install -r backend/requirements.txt

# Train the Isolation Forest model on nominal baseline data
python -m backend.ml.train

# Start the Flask API server
python -m backend.app
# Backend runs at http://127.0.0.1:5000
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
# Frontend runs at http://localhost:5173
```

### 3. Running Model Benchmark Evaluation
```bash
python -m evaluate
```

### 4. Running Automated Test Suite
```bash
python -m pytest backend/tests/ -v
```

---

## 🐳 Docker Deployment

Run the complete stack with Docker Compose:

```bash
docker-compose up --build
```

- API container: `http://localhost:5000` (Health checked at `/api/health`)
- Ollama container: `http://localhost:11434`

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service health & dependency status (Ollama, ML model, CSV) |
| `GET` | `/api/instances` | Discover monitored AWS EC2 instances |
| `GET` | `/api/metrics/<id>?hours=1` | AWS CloudWatch time-series metrics |
| `GET` | `/api/health-analysis` | Full pipeline analysis of telemetry data |
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
├── docker-compose.yml              # Container orchestration
├── Dockerfile                      # Production container build
├── evaluate.py                     # Benchmark evaluation execution script
├── health_data.csv                 # 72-hour demonstration telemetry
├── README.md                       # Comprehensive technical documentation
├── requirements.txt                # Unified dependency definitions
└── test_ai.py                      # Local Ollama connectivity verification
```

---

## 🛡️ Observability & Security

- **Structured Logging**: Timed request logging with latency in milliseconds (`time.perf_counter()`).
- **Read-Only AWS Access**: Requires only `ec2:DescribeInstances` and `cloudwatch:GetMetricStatistics`.
- **Zero Hardcoded Secrets**: Configuration driven via environment variables and IAM role resolution.
- **Frontend Isolation**: AWS credentials and secret keys are never exposed to the client-side browser.

---

## ⚖️ Known Limitations & Future Work

- **Batch Telemetry vs Streaming**: Metric ingestion is currently periodic pull-based; future revisions can incorporate WebSocket streaming or Kafka consumers.
- **CloudWatch Memory Metric**: Memory utilization on EC2 requires the CloudWatch Unified Agent; fallback estimation is used when agent metric is absent.
- **Model Storage**: Model artifacts are stored locally via `joblib`; production scaling would utilize an artifact registry like AWS S3 or MLflow.
