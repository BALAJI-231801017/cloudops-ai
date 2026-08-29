# ☁️ CloudOps AI

An AI-powered cloud application health monitoring and troubleshooting assistant.

CloudOps AI analyzes application health metrics, detects potential performance problems, visualizes historical trends, calculates an application health score, and uses a local LLM to provide troubleshooting recommendations.

## Flask API

The existing Streamlit dashboard remains available as the fallback UI in `streamlit_app.py`. A separate Flask backend now exposes the application logic for a future React dashboard.

Install the backend dependencies and start it from the project root:

```bash
pip install -r backend/requirements.txt
python -m backend.app
```

Available endpoints:

- `GET /api/health` — API readiness check
- `GET /api/instances` — discover EC2 instances
- `GET /api/metrics/<instance_id>?hours=1` — CPU, network-in, and network-out CloudWatch history
- `GET /api/health-analysis` — score and analyze the CSV health data
- `POST /api/analyze` — run Ollama against health evidence, with optional `instance_id`

Example request body:

```json
{
  "question": "Why is this instance slow?",
  "instance_id": "i-0123456789abcdef0"
}
```

### AWS authentication

The backend never stores credentials in code. On EC2, boto3 automatically uses the attached IAM role. For local development, it uses the default AWS credential chain; set `CLOUDOPS_AWS_PROFILE` only when you need a named local profile. `AWS_REGION` defaults to `ap-south-1` and can be overridden. Set `OLLAMA_HOST` and `OLLAMA_MODEL` when the local Ollama defaults do not apply.

## React dashboard

The React dashboard in `frontend/` is the planned primary UI. It starts in **Demo Mode**, showing local EC2/CloudWatch-style data without contacting AWS or creating cloud resources.

```bash
cd frontend
npm install
npm run dev
```

Use the **Live AWS API** toggle only after the local Flask API is running and valid read-only AWS credentials are available. Leave it off for a no-cost local demo.

### Demo modes

The dashboard supports three clear data sources:

- **Demo data — no backend**: fully simulated monitoring data for a quick presentation.
- **Flask + CSV — no AWS**: React calls Flask, which runs the real health-scoring and anomaly-detection code against `health_data.csv`.
- **Live AWS API — read only**: React calls Flask, which reads EC2 and CloudWatch only when credentials are configured.

This makes it possible to demonstrate the complete React-to-Flask data flow without creating an AWS resource or incurring cloud charges.

## Interview architecture

```text
React dashboard
      |
      | REST API
      v
Flask backend -----> CSV health history (local demo)
      |                       |
      |                       v
      |                health score + anomaly detection
      |
      +-----> Ollama (local diagnosis)
      |
      +-----> AWS CloudWatch + EC2 (optional, IAM role, read only)
```

For a cost-free demonstration, use **Flask + CSV — no AWS** and explain that the optional AWS adapter uses the same API contract and IAM role-based authentication when deployed.

## 🚀 Features

- Application health monitoring
- CPU, memory, error-rate and response-time analysis
- Rule-based problem detection
- Application health score
- Historical health trend visualization
- AI-powered application diagnosis
- Interactive CloudOps AI assistant
- Natural-language questions about application health
- Local AI inference using Ollama and Llama 3.2
- Streamlit web dashboard

## 🏗️ Architecture

```text
                 health_data.csv
                       |
                       v
                  Python / Pandas
                       |
             +---------+---------+
             |                   |
             v                   v
       Health Rules         Health History
             |                   |
             v                   v
       Problem Detection    Trend Visualization
             |
             v
        Health Score
             |
             v
       Ollama / Llama 3.2
             |
             v
       AI Diagnosis
             |
             v
        Streamlit UI
