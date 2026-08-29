import { useEffect, useMemo, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:5000";

const demoHealth = {
  health_score: 31.5,
  severity: "critical",
  latest: { cpu_usage: 95, memory_usage: 89, error_rate: 15, response_time: 5.6, status: "Critical" },
  anomaly: { detected: true, baseline_average: 60.4, threshold: 78.2 },
  issues: [
    { metric: "cpu_usage", severity: "critical", value: 95 },
    { metric: "memory_usage", severity: "critical", value: 89 },
    { metric: "error_rate", severity: "critical", value: 15 },
    { metric: "response_time", severity: "critical", value: 5.6 }
  ]
};

const demoInstances = [{
  id: "i-demo-cloudops-01",
  name: "cloudops-demo-server",
  state: "running",
  type: "t3.micro",
  availability_zone: "ap-south-1a"
}];

const demoMetrics = {
  metrics: {
    CPUUtilization: [32, 38, 53, 71, 86, 95].map((average, index) => ({ timestamp: `T-${5 - index}`, average })),
    NetworkIn: [18, 20, 26, 33, 41, 45].map((average, index) => ({ timestamp: `T-${5 - index}`, average })),
    NetworkOut: [9, 10, 14, 19, 23, 28].map((average, index) => ({ timestamp: `T-${5 - index}`, average }))
  }
};

function apiPath(path) {
  return `${API_BASE_URL}${path}`;
}

function MetricCard({ label, value, unit, tone = "neutral" }) {
  return <article className={`metric-card ${tone}`}><span>{label}</span><strong>{value}{unit}</strong></article>;
}

function MiniChart({ points }) {
  const values = points.map((point) => point.average);
  const max = Math.max(...values, 1);
  const coordinates = values.map((value, index) => `${(index / (values.length - 1)) * 100},${100 - (value / max) * 85}`).join(" ");
  return <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-label="CPU trend"><polyline points={coordinates} /></svg>;
}

function App() {
  const [dataSource, setDataSource] = useState("demo");
  const [instances, setInstances] = useState(demoInstances);
  const [health, setHealth] = useState(demoHealth);
  const [metrics, setMetrics] = useState(demoMetrics);
  const [selectedInstance, setSelectedInstance] = useState(demoInstances[0].id);
  const [status, setStatus] = useState("Demo mode — no AWS calls or cloud resources.");
  const [analysis, setAnalysis] = useState("CPU, memory, error rate, and response time are above the configured thresholds. Investigate the busiest processes, recent deployments, and application errors before resizing the instance.");

  const currentCpu = useMemo(() => {
    const values = metrics.metrics?.CPUUtilization || [];
    return values.at(-1)?.average ?? health.latest.cpu_usage;
  }, [health, metrics]);

  async function loadDashboard() {
    if (dataSource === "demo") {
      setInstances(demoInstances);
      setHealth(demoHealth);
      setMetrics(demoMetrics);
      setSelectedInstance(demoInstances[0].id);
      setStatus("Demo mode — no AWS calls or cloud resources.");
      return;
    }

    setStatus(dataSource === "local" ? "Loading CSV health analysis from Flask..." : "Loading read-only data from Flask...");
    try {
      const healthResponse = await fetch(apiPath("/api/health-analysis"));
      if (!healthResponse.ok) throw new Error("Flask API could not load health data.");
      const nextHealth = await healthResponse.json();
      if (dataSource === "local") {
        setHealth(nextHealth);
        setInstances(demoInstances);
        setMetrics(demoMetrics);
        setSelectedInstance(demoInstances[0].id);
        setStatus("Local API mode — React is using Flask and CSV health data. No AWS calls.");
        return;
      }
      const instancesResponse = await fetch(apiPath("/api/instances"));
      if (!instancesResponse.ok) throw new Error("Flask API could not load AWS data.");
      const instancePayload = await instancesResponse.json();
      const nextInstances = instancePayload.instances;
      setHealth(nextHealth);
      setInstances(nextInstances);
      if (nextInstances.length) {
        const instanceId = nextInstances[0].id;
        setSelectedInstance(instanceId);
        const metricsResponse = await fetch(apiPath(`/api/metrics/${instanceId}?hours=1`));
        if (metricsResponse.ok) setMetrics(await metricsResponse.json());
      }
      setStatus(`Live read-only API mode — ${nextInstances.length} instance(s) discovered.`);
    } catch (error) {
      setStatus(`${error.message} Switch back to Demo mode to continue without AWS.`);
    }
  }

  async function requestAnalysis() {
    if (dataSource === "demo") {
      setAnalysis("Demo diagnosis: high CPU and memory point to resource pressure. Check running processes, error logs, and recent deployments. Scale only if the elevated load persists after optimization.");
      return;
    }
    setStatus("Requesting local Ollama analysis through Flask...");
    try {
      const response = await fetch(apiPath("/api/analyze"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: "Why is this instance unhealthy?", ...(dataSource === "aws" ? { instance_id: selectedInstance } : {}) })
      });
      if (!response.ok) throw new Error("AI analysis is unavailable. Ensure Ollama is running locally.");
      const payload = await response.json();
      setAnalysis(payload.analysis);
      setStatus("Analysis complete.");
    } catch (error) {
      setStatus(error.message);
    }
  }

  useEffect(() => { loadDashboard(); }, [dataSource]);

  return <main>
    <header>
      <div><p className="eyebrow">CLOUDOPS AI</p><h1>Infrastructure health, explained.</h1><p className="subtle">React dashboard · Flask API · local Ollama analysis</p></div>
      <div className="actions"><label className="toggle">Data source <select value={dataSource} onChange={(event) => setDataSource(event.target.value)}><option value="demo">Demo data — no backend</option><option value="local">Flask + CSV — no AWS</option><option value="aws">Live AWS API — read only</option></select></label><button onClick={loadDashboard}>Refresh</button></div>
    </header>

    <p className="status">{status}</p>
    <section className="overview">
      <article className={`health-score ${health.severity}`}><span>Overall health</span><strong>{Math.round(health.health_score)}<small>/100</small></strong><b>{health.severity}</b></article>
      <MetricCard label="CPU" value={Math.round(currentCpu)} unit="%" tone="critical" />
      <MetricCard label="Memory" value={health.latest.memory_usage} unit="%" tone="critical" />
      <MetricCard label="Error rate" value={health.latest.error_rate} unit="%" tone="critical" />
      <MetricCard label="Response time" value={health.latest.response_time} unit="s" tone="critical" />
    </section>

    <section className="grid">
      <article className="panel chart"><div className="panel-heading"><div><p className="eyebrow">CLOUDWATCH TREND</p><h2>CPU utilization</h2></div><b>{Math.round(currentCpu)}%</b></div><MiniChart points={metrics.metrics?.CPUUtilization || []} /><div className="chart-labels"><span>1 hour ago</span><span>now</span></div></article>
      <article className="panel"><p className="eyebrow">ANOMALY DETECTION</p><h2>{health.anomaly?.detected ? "Unexpected CPU spike" : "No anomaly detected"}</h2><p>Baseline: {health.anomaly?.baseline_average ?? "n/a"}% · threshold: {health.anomaly?.threshold ?? "n/a"}%</p><ul>{health.issues.map((issue) => <li key={issue.metric}>{issue.metric.replaceAll("_", " ")}: {issue.value} ({issue.severity})</li>)}</ul></article>
      <article className="panel instances"><p className="eyebrow">EC2 INSTANCES</p><h2>{instances.length} monitored</h2>{instances.map((instance) => <button className={instance.id === selectedInstance ? "instance selected" : "instance"} key={instance.id} onClick={() => setSelectedInstance(instance.id)}><span><b>{instance.name}</b><small>{instance.id} · {instance.type}</small></span><em>{instance.state}</em></button>)}</article>
      <article className="panel ai"><div><p className="eyebrow">OLLAMA DIAGNOSIS</p><h2>What should I check first?</h2></div><p>{analysis}</p><button onClick={requestAnalysis}>Analyze health</button></article>
    </section>
  </main>;
}

export default App;
