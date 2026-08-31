import React, { useEffect, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:5000";

const DEMO_HEALTH = {
  health_score: 32.5,
  severity: "CRITICAL",
  incident_type: "RESOURCE_SATURATION",
  is_anomaly: true,
  primary_detector: "isolation_forest",
  anomaly_score: 0.842,
  affected_metrics: ["cpu_usage", "memory_usage", "response_time"],
  latest: { cpu_usage: 94.5, memory_usage: 88.2, error_rate: 3.4, response_time: 4.8, status: "Critical" },
  anomaly: { detected: true, baseline_average: 58.2, threshold: 75.0, metric: "cpu_usage", current: 94.5 },
  ml_result: {
    is_anomaly: true,
    anomaly_score: 0.842,
    severity: "CRITICAL",
    detector_name: "isolation_forest",
    model_version: "iforest-v1.0",
    feature_contributions: { cpu_usage: 2.85, memory_usage: 2.41, error_rate: 0.82, response_time: 2.15 }
  },
  issues: [
    { metric: "cpu_usage", severity: "critical", value: 94.5, threshold: 80.0, message: "Cpu Usage is critical (94.5 >= 80.0)" },
    { metric: "memory_usage", severity: "critical", value: 88.2, threshold: 80.0, message: "Memory Usage is critical (88.2 >= 80.0)" },
    { metric: "response_time", severity: "critical", value: 4.8, threshold: 3.5, message: "Response Time is critical (4.8 >= 3.5)" }
  ],
  evidence_statements: [
    "Cpu Usage is critical (94.5 >= 80.0)",
    "Memory Usage is critical (88.2 >= 80.0)",
    "Response Time is critical (4.8 >= 3.5)",
    "Statistical baseline detected CPU anomaly: current 94.5% exceeds dynamic baseline 75.0%",
    "Isolation Forest identified multivariate anomaly (Normalized Score: 0.842, Severity: CRITICAL)"
  ],
  history: [
    { timestamp: "10:00", cpu_usage: 42, memory_usage: 52, error_rate: 0.8, response_time: 0.9 },
    { timestamp: "11:00", cpu_usage: 55, memory_usage: 58, error_rate: 1.1, response_time: 1.1 },
    { timestamp: "12:00", cpu_usage: 68, memory_usage: 65, error_rate: 1.8, response_time: 1.4 },
    { timestamp: "13:00", cpu_usage: 82, memory_usage: 74, error_rate: 2.2, response_time: 2.1 },
    { timestamp: "14:00", cpu_usage: 91, memory_usage: 84, error_rate: 2.9, response_time: 3.8 },
    { timestamp: "15:00", cpu_usage: 94.5, memory_usage: 88.2, error_rate: 3.4, response_time: 4.8 }
  ]
};

const DEMO_INSTANCES = [{
  id: "i-0987654321fedcba0",
  name: "cloudops-demo-instance",
  state: "running",
  type: "t3.medium",
  availability_zone: "ap-south-1a"
}];

const DEMO_METRICS = {
  metrics: {
    CPUUtilization: [32, 45, 62, 78, 89, 95].map((average, index) => ({ timestamp: `T-${5 - index}m`, average })),
    NetworkIn: [20, 24, 35, 48, 52, 60].map((average, index) => ({ timestamp: `T-${5 - index}m`, average })),
    NetworkOut: [12, 15, 22, 31, 38, 42].map((average, index) => ({ timestamp: `T-${5 - index}m`, average }))
  }
};

const SCENARIOS = [
  { id: "normal", name: "1. Normal Baseline", desc: "Steady nominal workload (CPU ~35%, Mem ~45%, Latency ~0.8s)" },
  { id: "cpu_spike", name: "2. CPU Spike", desc: "Isolated compute burst (CPU ~95%, nominal memory/errors)" },
  { id: "memory_spike", name: "3. Memory Pressure", desc: "Near-OOM saturation (Memory ~95%, normal CPU)" },
  { id: "error_spike", name: "4. Error Spike", desc: "Elevated HTTP 5xx / application faults (Error rate ~25%)" },
  { id: "latency_spike", name: "5. Latency Degradation", desc: "Severe response delay (Response time ~8.5s)" },
  { id: "cpu_latency", name: "6. CPU + Latency", desc: "Dual saturation: compute lock and request queuing" },
  { id: "memory_latency", name: "7. Memory + Latency", desc: "Memory paging/swap contention elevating latency" },
  { id: "multi_metric", name: "8. Cascade Failure", desc: "Compound breakdown across CPU, Memory, Errors, and Latency" }
];

function apiPath(path) {
  return `${API_BASE_URL}${path}`;
}

function MetricCard({ label, value, unit = "", severity = "healthy" }) {
  const toneClass = severity === "CRITICAL" || severity === "critical"
    ? "tone-critical"
    : severity === "HIGH" || severity === "WARNING" || severity === "warning"
    ? "tone-warning"
    : "tone-healthy";

  return (
    <article className={`metric-card ${toneClass}`}>
      <span className="metric-label">{label}</span>
      <strong className="metric-val">{value} <small>{unit}</small></strong>
    </article>
  );
}

function MiniChart({ points }) {
  if (!points || points.length === 0) return <p className="subtle">No telemetry points</p>;
  const values = points.map((p) => p.average ?? p.cpu_usage ?? 0);
  const max = Math.max(...values, 100);
  const coordinates = values
    .map((val, idx) => `${(idx / Math.max(values.length - 1, 1)) * 100},${100 - (val / max) * 85}`)
    .join(" ");

  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="sparkline" aria-label="Metric Trend">
      <polyline points={coordinates} />
    </svg>
  );
}

export default function App() {
  const [dataSource, setDataSource] = useState("demo");
  const [instances, setInstances] = useState(DEMO_INSTANCES);
  const [health, setHealth] = useState(DEMO_HEALTH);
  const [metrics, setMetrics] = useState(DEMO_METRICS);
  const [selectedInstance, setSelectedInstance] = useState(DEMO_INSTANCES[0].id);
  const [statusMsg, setStatusMsg] = useState("Demo Mode: Standalone client preview (no backend requests).");
  const [selectedScenario, setSelectedScenario] = useState("cpu_spike");
  const [isSimulating, setIsSimulating] = useState(false);
  const [showEvaluation, setShowEvaluation] = useState(false);
  const [evalResults, setEvalResults] = useState(null);
  const [aiQuestion, setAiQuestion] = useState("What is the probable root cause and what immediate technical remediation is recommended?");
  const [aiDiagnosis, setAiDiagnosis] = useState({
    incident_summary: "Severe CPU and memory saturation detected concurrently with elevated response time.",
    probable_root_cause: "Resource contention caused by intensive background job or connection pool exhaustion.",
    confidence: 0.85,
    supporting_evidence: [
      "CPU usage at 94.5% exceeds critical threshold (80.0%)",
      "Memory utilization at 88.2% exceeds critical threshold (80.0%)",
      "Isolation Forest normalized anomaly score of 0.842 indicates strong multivariate anomaly"
    ],
    recommended_actions: [
      "Profile high-CPU threads or top worker processes",
      "Inspect database connection pool saturation and query latency",
      "Review recent application code deployments for thread leaks"
    ],
    limitations: ["Metrics are from local telemetry; cluster-wide logs require log aggregation."]
  });
  const [aiLoading, setAiLoading] = useState(false);

  // Load telemetry based on data source
  async function loadDashboard(source = dataSource) {
    if (source === "demo") {
      setInstances(DEMO_INSTANCES);
      setHealth(DEMO_HEALTH);
      setMetrics(DEMO_METRICS);
      setSelectedInstance(DEMO_INSTANCES[0].id);
      setStatusMsg("Demo Mode — Purely client-side preview with simulated critical state (no network calls).");
      return;
    }

    if (source === "simulation") {
      runSimulation(selectedScenario);
      return;
    }

    setStatusMsg(source === "csv" ? "Querying Flask API for CSV Telemetry..." : "Connecting to AWS CloudWatch & EC2 via Flask API...");

    try {
      const healthRes = await fetch(apiPath("/api/health-analysis"));
      if (!healthRes.ok) throw new Error(`Flask API error (${healthRes.status})`);
      const healthData = await healthRes.json();
      setHealth(healthData);

      if (source === "csv") {
        setInstances(DEMO_INSTANCES);
        setMetrics(DEMO_METRICS);
        setStatusMsg("Flask + CSV Mode: 4-feature telemetry processed by Preprocessor → Isolation Forest ML → Incident Classifier.");
        return;
      }

      // Live AWS Mode
      const instRes = await fetch(apiPath("/api/instances"));
      if (!instRes.ok) throw new Error("Could not discover EC2 instances from AWS.");
      const instPayload = await instRes.json();
      const discovered = instPayload.instances || [];
      setInstances(discovered);

      if (discovered.length > 0) {
        const targetId = discovered[0].id;
        setSelectedInstance(targetId);
        const metricRes = await fetch(apiPath(`/api/metrics/${targetId}?hours=1`));
        if (metricRes.ok) {
          const metricData = await metricRes.json();
          setMetrics(metricData);
        }
      }
      setStatusMsg(`Live AWS Mode: Discovered ${discovered.length} EC2 instance(s) in region ${instPayload.region}. (CloudWatch metrics: CPU, NetworkIn, NetworkOut)`);
    } catch (err) {
      setStatusMsg(`Connection Note: ${err.message}. Switch to Demo or Simulation mode if Flask / AWS is offline.`);
    }
  }

  // Run interactive simulation
  async function runSimulation(scenarioId = selectedScenario) {
    setIsSimulating(true);
    setStatusMsg(`Running simulation scenario '${scenarioId}' through full ML + Classification pipeline...`);
    try {
      const res = await fetch(apiPath("/api/simulate"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario: scenarioId })
      });
      if (!res.ok) throw new Error(`Simulation API error (${res.status})`);
      const data = await res.json();
      setHealth({
        health_score: data.health_score,
        severity: data.severity,
        incident_type: data.incident_type,
        is_anomaly: data.is_anomaly,
        primary_detector: data.primary_detector,
        anomaly_score: data.anomaly_score,
        affected_metrics: data.affected_metrics,
        latest: data.metrics,
        evidence_statements: data.evidence_statements,
        issues: data.evidence_statements.map((s) => ({ message: s })),
        ml_result: data.ml_result,
        anomaly: { detected: data.is_anomaly, current: data.metrics.cpu_usage, threshold: 80.0 }
      });
      if (data.analysis) {
        setAiDiagnosis(data.analysis);
      }
      setStatusMsg(`Simulation Mode: Injected scenario '${scenarioId}' through Preprocessor → Isolation Forest ML → Classifier.`);
    } catch (err) {
      setStatusMsg(`Simulation Error: ${err.message}. Ensure Flask backend is running on port 5000.`);
    } finally {
      setIsSimulating(false);
    }
  }

  // Fetch benchmark evaluation metrics
  async function loadEvaluation() {
    try {
      const res = await fetch(apiPath("/api/evaluation"));
      if (res.ok) {
        const data = await res.json();
        setEvalResults(data);
      }
    } catch (e) {
      console.warn("Evaluation API offline:", e);
    }
  }

  // Trigger LLM analysis
  async function triggerAiAnalysis() {
    if (dataSource === "demo") {
      setAiDiagnosis({
        incident_summary: "Demo Diagnosis: High CPU utilization (94.5%) with memory pressure (88.2%) and response lag.",
        probable_root_cause: "High concurrency traffic spike causing thread contention and swap thrashing.",
        confidence: 0.88,
        supporting_evidence: [
          "CPU Usage (94.5%) > 80.0% critical threshold",
          "Memory Usage (88.2%) > 80.0% critical threshold",
          "Isolation Forest normalized anomaly score = 0.842"
        ],
        recommended_actions: [
          "Inspect CPU-heavy worker processes with `top` or CloudWatch Container Insights",
          "Verify application heap allocation and garbage collection pauses",
          "Consider horizontal scaling if traffic surge is sustained"
        ],
        limitations: ["Client-side demo simulation."]
      });
      return;
    }

    setAiLoading(true);
    try {
      const res = await fetch(apiPath("/api/analyze"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: aiQuestion,
          ...(dataSource === "aws" ? { instance_id: selectedInstance } : {})
        })
      });
      if (!res.ok) throw new Error(`AI Analysis failed (${res.status})`);
      const payload = await res.json();
      setAiDiagnosis(payload.analysis);
      setStatusMsg(payload.ai_available ? "AI Root-Cause Diagnosis complete (Llama 3.2 via Ollama)." : "AI Service unavailable (Deterministic rules active).");
    } catch (err) {
      setStatusMsg(`AI Diagnosis Note: ${err.message}`);
    } finally {
      setAiLoading(false);
    }
  }

  useEffect(() => {
    loadDashboard(dataSource);
    loadEvaluation();
  }, [dataSource]);

  const currentCpu = health.latest?.cpu_usage ?? 0;
  const currentMem = health.latest?.memory_usage ?? 0;
  const currentErr = health.latest?.error_rate ?? 0;
  const currentLat = health.latest?.response_time ?? 0;

  return (
    <div className="app-shell">
      {/* Header Bar */}
      <header className="app-header">
        <div className="header-brand">
          <div className="brand-badge">CLOUDOPS AI</div>
          <h1>Infrastructure Health & AI Root-Cause Analysis</h1>
          <p className="subtle">Unsupervised Isolation Forest ML · Deterministic Classifier · Grounded Llama 3.2 · AWS Telemetry</p>
        </div>
        <div className="header-controls">
          <div className="control-group">
            <label htmlFor="source-select">Data Source:</label>
            <select
              id="source-select"
              value={dataSource}
              onChange={(e) => setDataSource(e.target.value)}
              className="select-input"
            >
              <option value="demo">Demo Mode (Client Preview)</option>
              <option value="csv">Flask + CSV Telemetry (4 Features)</option>
              <option value="simulation">Simulation Runner (8 Scenarios)</option>
              <option value="aws">Live AWS Telemetry (CloudWatch Read-Only)</option>
            </select>
          </div>
          <button onClick={() => loadDashboard()} className="btn btn-primary">Refresh</button>
          <button onClick={() => setShowEvaluation(!showEvaluation)} className="btn btn-secondary">
            {showEvaluation ? "Hide Benchmark" : "Model Benchmark"}
          </button>
        </div>
      </header>

      {/* Status Notice Banner */}
      <div className={`status-banner ${dataSource === "aws" ? "banner-aws" : dataSource === "simulation" ? "banner-sim" : "banner-info"}`}>
        <span className="status-indicator"></span>
        <span>{statusMsg}</span>
      </div>

      {/* Model Benchmark Evaluation Modal / Panel */}
      {showEvaluation && evalResults && (
        <section className="eval-panel">
          <div className="panel-header">
            <h2>Synthetic Evaluation Benchmark (Baseline vs Isolation Forest)</h2>
            <button onClick={() => setShowEvaluation(false)} className="btn-close">✕</button>
          </div>
          <p className="subtle">
            Evaluated on synthetic evaluation dataset ({evalResults.total_eval_samples} samples across 8 scenarios with ground truth labels).
          </p>
          <div className="table-wrapper">
            <table className="eval-table">
              <thead>
                <tr>
                  <th>Performance Metric</th>
                  <th>Baseline (CPU Rule)</th>
                  <th>Isolation Forest (ML)</th>
                  <th>CloudOps Unified Pipeline</th>
                </tr>
              </thead>
              <tbody>
                {["precision", "recall", "f1_score", "false_positive_rate", "accuracy", "avg_latency_ms"].map((m) => (
                  <tr key={m}>
                    <td className="metric-name">{m.replaceAll("_", " ").toUpperCase()}</td>
                    <td>{evalResults.models.baseline_cpu_threshold?.[m] ?? "-"}</td>
                    <td className="highlight">{evalResults.models.isolation_forest_ml?.[m] ?? "-"}</td>
                    <td className="highlight-success">{evalResults.models.cloudops_unified_pipeline?.[m] ?? "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h3 style={{ marginTop: "1rem" }}>Per-Scenario Accuracy (Recall on Anomaly, Specificity on Nominal)</h3>
          <div className="scenario-grid">
            {Object.entries(evalResults.scenario_breakdown || {}).map(([scName, scData]) => (
              <div key={scName} className="scenario-card">
                <strong>{scName}</strong>
                <span className={scData.is_anomaly_scenario ? "badge badge-crit" : "badge badge-ok"}>
                  {scData.is_anomaly_scenario ? "ANOMALY" : "NOMINAL"}
                </span>
                <div className="sc-rates">
                  <div>Baseline: {(scData.baseline_accuracy_rate * 100).toFixed(0)}%</div>
                  <div>Isolation Forest: <strong>{(scData.iforest_accuracy_rate * 100).toFixed(0)}%</strong></div>
                  <div>Pipeline: <strong style={{ color: "#10b981" }}>{(scData.pipeline_accuracy_rate * 100).toFixed(0)}%</strong></div>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Interactive Simulation Runner Control */}
      {dataSource === "simulation" && (
        <section className="simulation-toolbar">
          <div className="sim-title">
            <strong>Interactive Incident Simulator</strong>
            <span className="subtle">Injects controlled synthetic failure patterns through Preprocessor → Isolation Forest → Classifier</span>
          </div>
          <div className="sim-controls">
            <select
              value={selectedScenario}
              onChange={(e) => {
                setSelectedScenario(e.target.value);
                runSimulation(e.target.value);
              }}
              className="select-input"
            >
              {SCENARIOS.map((sc) => (
                <option key={sc.id} value={sc.id}>{sc.name}</option>
              ))}
            </select>
            <button
              onClick={() => runSimulation(selectedScenario)}
              disabled={isSimulating}
              className="btn btn-warning"
            >
              {isSimulating ? "Evaluating..." : "Run Scenario"}
            </button>
          </div>
        </section>
      )}

      {/* Primary KPI Overview Cards */}
      <section className="kpi-grid">
        <article className={`health-card severity-${health.severity?.toLowerCase()}`}>
          <span className="card-label">System Health Score</span>
          <div className="score-display">
            <strong>{Math.round(health.health_score ?? 100)}</strong>
            <small>/100</small>
          </div>
          <span className={`status-badge badge-${health.severity?.toLowerCase()}`}>
            {health.severity || "HEALTHY"}
          </span>
        </article>

        <MetricCard label="CPU Utilization" value={currentCpu} unit="%" severity={currentCpu >= 80 ? "CRITICAL" : currentCpu >= 60 ? "WARNING" : "HEALTHY"} />
        <MetricCard label="Memory Utilization" value={currentMem} unit="%" severity={currentMem >= 80 ? "CRITICAL" : currentMem >= 65 ? "WARNING" : "HEALTHY"} />
        <MetricCard label="Error Rate" value={currentErr} unit="%" severity={currentErr >= 10 ? "CRITICAL" : currentErr >= 5 ? "WARNING" : "HEALTHY"} />
        <MetricCard label="Response Latency" value={currentLat} unit="s" severity={currentLat >= 3.5 ? "CRITICAL" : currentLat >= 2.0 ? "WARNING" : "HEALTHY"} />
      </section>

      {/* Main Analysis Grid */}
      <main className="dashboard-grid">
        {/* Panel 1: Anomaly Detection Engine */}
        <article className="dash-card">
          <div className="card-header">
            <span className="eyebrow">DETECTION ENGINE</span>
            <h2>Multivariate ML & Statistical Baseline</h2>
          </div>
          <div className="card-body">
            <div className="detector-status-box">
              <div className="detector-row">
                <span>Isolation Forest ML Detector:</span>
                <strong className={health.ml_result?.is_anomaly ? "text-crit" : "text-ok"}>
                  {health.ml_result?.is_anomaly ? "● ANOMALY DETECTED" : "● NOMINAL"}
                </strong>
              </div>
              <div className="detector-row">
                <span>Normalized Anomaly Score [0.0 - 1.0]:</span>
                <strong>{health.anomaly_score ?? health.ml_result?.anomaly_score ?? 0.0}</strong>
              </div>
              <div className="detector-row">
                <span>Statistical Baseline (CPU):</span>
                <strong className={health.anomaly?.detected ? "text-crit" : "text-ok"}>
                  {health.anomaly?.detected ? "● CPU ANOMALY" : "● NOMINAL"}
                </strong>
              </div>
            </div>

            {health.ml_result?.feature_contributions && (
              <div className="contributions-box">
                <span className="subtle">Feature Z-Score Deviations:</span>
                <div className="contrib-chips">
                  {Object.entries(health.ml_result.feature_contributions).map(([f, z]) => (
                    <span key={f} className={`chip ${Math.abs(z) > 1.5 ? "chip-crit" : "chip-norm"}`}>
                      {f.replace("_", " ")}: <strong>{z > 0 ? `+${z}` : z}σ</strong>
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div className="chart-container">
              <span className="subtle">Recent Utilization Trend</span>
              <MiniChart points={health.history || metrics.metrics?.CPUUtilization || []} />
            </div>
          </div>
        </article>

        {/* Panel 2: Incident Classification & Evidence */}
        <article className="dash-card">
          <div className="card-header">
            <span className="eyebrow">DETERMINISTIC CLASSIFICATION</span>
            <h2>{health.incident_type ? health.incident_type.replace(/_/g, " ") : "SYSTEM HEALTHY"}</h2>
          </div>
          <div className="card-body">
            <div className="incident-meta">
              <div>Severity: <span className={`badge badge-${health.severity?.toLowerCase()}`}>{health.severity}</span></div>
              <div>Primary Detector: <strong>{health.primary_detector || "isolation_forest"}</strong></div>
              <div>Affected: <strong>{health.affected_metrics?.join(", ") || "None"}</strong></div>
            </div>

            <div className="evidence-list-container">
              <span className="subtle">Grounded Evidence Statements:</span>
              <ul className="evidence-list">
                {(health.evidence_statements || []).map((stmt, idx) => (
                  <li key={idx}>
                    <span className="bullet">▸</span> {stmt}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </article>

        {/* Panel 3: Generative AI Root-Cause Diagnosis (Ollama/Llama 3.2) */}
        <article className="dash-card card-ai">
          <div className="card-header">
            <div className="ai-tag">🤖 LLAMA 3.2 / OLLAMA</div>
            <h2>Evidence-Grounded AI Diagnosis</h2>
          </div>
          <div className="card-body">
            <div className="ai-qa-box">
              <input
                type="text"
                value={aiQuestion}
                onChange={(e) => setAiQuestion(e.target.value)}
                placeholder="Ask CloudOps AI about this incident..."
                className="ai-input"
              />
              <button
                onClick={triggerAiAnalysis}
                disabled={aiLoading}
                className="btn btn-ai"
              >
                {aiLoading ? "Analyzing..." : "Diagnose"}
              </button>
            </div>

            <div className="ai-result-box">
              <div className="ai-summary">
                <strong>Incident Summary:</strong>
                <p>{aiDiagnosis.incident_summary}</p>
              </div>

              <div className="ai-root-cause">
                <strong>Probable Root Cause:</strong>
                <p>{aiDiagnosis.probable_root_cause}</p>
                <div className="confidence-meter">
                  <span>LLM-Reported Confidence Estimate: {(aiDiagnosis.confidence * 100).toFixed(0)}%</span>
                  <div className="bar"><div className="fill" style={{ width: `${aiDiagnosis.confidence * 100}%` }}></div></div>
                </div>
              </div>

              {aiDiagnosis.recommended_actions?.length > 0 && (
                <div className="ai-actions">
                  <strong>Recommended Advisory Actions:</strong>
                  <ul>
                    {aiDiagnosis.recommended_actions.map((act, i) => (
                      <li key={i}>{act}</li>
                    ))}
                  </ul>
                </div>
              )}

              {aiDiagnosis.limitations?.length > 0 && (
                <div className="ai-limits subtle">
                  <em>Limitations: {aiDiagnosis.limitations.join("; ")}</em>
                </div>
              )}
            </div>
          </div>
        </article>

        {/* Panel 4: Monitored EC2 Infrastructure */}
        <article className="dash-card">
          <div className="card-header">
            <span className="eyebrow">INFRASTRUCTURE TELEMETRY</span>
            <h2>Monitored EC2 Instances ({instances.length})</h2>
          </div>
          <div className="card-body">
            <div className="instance-list">
              {instances.map((inst) => (
                <div
                  key={inst.id}
                  className={`instance-item ${inst.id === selectedInstance ? "selected" : ""}`}
                  onClick={() => setSelectedInstance(inst.id)}
                >
                  <div className="inst-main">
                    <strong>{inst.name || "Instance"}</strong>
                    <small>{inst.id} · {inst.type} · {inst.availability_zone}</small>
                  </div>
                  <span className={`state-badge state-${inst.state}`}>{inst.state}</span>
                </div>
              ))}
            </div>
          </div>
        </article>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <div>CloudOps AI · Final Year Artificial Intelligence & Data Science Engineering Project</div>
        <div>Architecture: ML Anomaly Detection → Deterministic Classification → Grounded Llama 3.2 Advisory</div>
      </footer>
    </div>
  );
}
