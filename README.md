# ☁️ CloudOps AI

An AI-powered cloud application health monitoring and troubleshooting assistant.

CloudOps AI analyzes application health metrics, detects potential performance problems, visualizes historical trends, calculates an application health score, and uses a local LLM to provide troubleshooting recommendations.

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