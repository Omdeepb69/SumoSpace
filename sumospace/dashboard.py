"""
SumoSpace Benchmark Dashboard — Data Collector & HTML Generator
================================================================
Collects structured telemetry from ExecutionTrace objects and generates
a self-contained HTML dashboard for failure analysis.
"""
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class BenchmarkRun:
    """A single benchmark task execution result."""
    task_name: str
    model: str
    timestamp: float
    success: bool
    duration_ms: float
    intent_predicted: str
    intent_correct: str | None = None
    failure_category: str | None = None
    routing_trace: dict = field(default_factory=dict)
    steps: list[dict] = field(default_factory=list)
    total_retries: int = 0
    recovery_successes: int = 0
    error: str = ""


class BenchmarkCollector:
    """Collects ExecutionTrace data into structured benchmark results."""

    def __init__(self, output_dir: str = "./benchmark_results"):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._runs: list[BenchmarkRun] = []

    def record(self, trace, task_name: str, model: str, correct_intent: str | None = None):
        """Extract telemetry from an ExecutionTrace and record it."""
        steps = []
        total_retries = 0
        recovery_successes = 0

        for st in trace.step_traces:
            step_data = {
                "tool": st.tool,
                "success": st.result.success,
                "duration_ms": st.duration_ms,
                "error": st.result.error if not st.result.success else "",
                "recoverable": getattr(st.result, "recoverable", False),
                "retry_hint": getattr(st.result, "retry_hint", ""),
            }
            steps.append(step_data)
            if not st.result.success:
                total_retries += 1
            if not st.result.success and getattr(st.result, "recoverable", False):
                # Check if a later step with the same tool succeeded
                pass  # Simplified — full recovery tracking would need more context

        # Count recovery: tool failed then later succeeded
        tool_failures = {}
        for st in trace.step_traces:
            if not st.result.success:
                tool_failures[st.tool] = True
            elif st.tool in tool_failures:
                recovery_successes += 1
                del tool_failures[st.tool]

        routing_trace = {}
        if hasattr(trace, "classification") and trace.classification:
            routing_trace = getattr(trace.classification, "routing_trace", {})

        failure_cat = None
        if hasattr(trace, "failure_category") and trace.failure_category:
            failure_cat = trace.failure_category.value if hasattr(trace.failure_category, "value") else str(trace.failure_category)

        run = BenchmarkRun(
            task_name=task_name,
            model=model,
            timestamp=time.time(),
            success=trace.success,
            duration_ms=trace.duration_ms,
            intent_predicted=trace.intent.value if hasattr(trace.intent, "value") else str(trace.intent),
            intent_correct=correct_intent,
            failure_category=failure_cat,
            routing_trace=routing_trace,
            steps=steps,
            total_retries=total_retries,
            recovery_successes=recovery_successes,
            error=trace.error,
        )
        self._runs.append(run)
        return run

    def save(self, filename: str = "benchmark_results.json"):
        """Save all collected runs to JSON."""
        path = self._output_dir / filename
        data = [asdict(r) for r in self._runs]
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return str(path)

    def load(self, filename: str = "benchmark_results.json"):
        """Load runs from a JSON file."""
        path = self._output_dir / filename
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self._runs = [BenchmarkRun(**d) for d in data]
        return self._runs

    def generate_dashboard(self, filename: str = "dashboard.html") -> str:
        """Generate a self-contained HTML dashboard."""
        data_json = json.dumps([asdict(r) for r in self._runs], indent=2, default=str)
        
        html = _DASHBOARD_TEMPLATE.replace("__BENCHMARK_DATA__", data_json)
        
        path = self._output_dir / filename
        path.write_text(html, encoding="utf-8")
        return str(path)


_DASHBOARD_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SumoSpace Benchmark Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  
  * { margin: 0; padding: 0; box-sizing: border-box; }
  
  :root {
    --bg: #0a0e17;
    --surface: #111827;
    --surface-2: #1f2937;
    --border: #374151;
    --text: #e5e7eb;
    --text-dim: #9ca3af;
    --accent: #818cf8;
    --accent-2: #6366f1;
    --green: #34d399;
    --red: #f87171;
    --yellow: #fbbf24;
    --orange: #fb923c;
  }
  
  body {
    font-family: 'Inter', -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: 24px;
  }
  
  .header {
    text-align: center;
    margin-bottom: 32px;
    padding: 24px;
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(129,140,248,0.05));
    border: 1px solid var(--border);
    border-radius: 16px;
  }
  
  .header h1 {
    font-size: 28px;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  
  .header .subtitle {
    color: var(--text-dim);
    margin-top: 6px;
    font-size: 14px;
  }
  
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
  }
  
  .kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
  }
  
  .kpi-card:hover {
    transform: translateY(-2px);
    border-color: var(--accent);
  }
  
  .kpi-card .value {
    font-size: 36px;
    font-weight: 700;
    line-height: 1.1;
  }
  
  .kpi-card .label {
    font-size: 12px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 6px;
  }
  
  .kpi-card.success .value { color: var(--green); }
  .kpi-card.fail .value { color: var(--red); }
  .kpi-card.warn .value { color: var(--yellow); }
  .kpi-card.accent .value { color: var(--accent); }
  
  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-bottom: 32px;
  }
  
  @media (max-width: 900px) { .grid-2 { grid-template-columns: 1fr; } }
  
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
  }
  
  .card h2 {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 16px;
    color: var(--accent);
  }
  
  .chart-container {
    position: relative;
    height: 280px;
  }
  
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }
  
  th, td {
    padding: 10px 12px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }
  
  th {
    color: var(--text-dim);
    font-weight: 500;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.5px;
  }
  
  .tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
  }
  
  .tag-pass { background: rgba(52,211,153,0.15); color: var(--green); }
  .tag-fail { background: rgba(248,113,113,0.15); color: var(--red); }
  
  .heatmap-row {
    display: flex;
    align-items: center;
    margin-bottom: 8px;
    gap: 12px;
  }
  
  .heatmap-label {
    width: 160px;
    font-size: 12px;
    color: var(--text-dim);
    text-align: right;
    flex-shrink: 0;
  }
  
  .heatmap-bar {
    height: 28px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    padding-left: 8px;
    font-size: 12px;
    font-weight: 600;
    min-width: 40px;
    transition: width 0.6s ease;
  }
  
  .empty-state {
    text-align: center;
    padding: 48px;
    color: var(--text-dim);
    font-size: 14px;
  }
</style>
</head>
<body>

<div class="header">
  <h1>⚡ SumoSpace Benchmark Dashboard</h1>
  <p class="subtitle">Failure Analytics · Tool Reliability · Routing Intelligence</p>
</div>

<div id="kpis" class="kpi-grid"></div>

<div class="grid-2">
  <div class="card">
    <h2>🔥 Failure Heatmap</h2>
    <div id="failure-heatmap"></div>
  </div>
  <div class="card">
    <h2>🛠️ Tool Reliability</h2>
    <div class="chart-container"><canvas id="tool-chart"></canvas></div>
  </div>
</div>

<div class="grid-2">
  <div class="card">
    <h2>🔄 Retry Intelligence</h2>
    <div class="chart-container"><canvas id="retry-chart"></canvas></div>
  </div>
  <div class="card">
    <h2>🧭 Routing Accuracy</h2>
    <div id="routing-matrix"></div>
  </div>
</div>

<div class="card" style="margin-bottom:32px">
  <h2>📊 Run History</h2>
  <table id="run-table">
    <thead>
      <tr>
        <th>Task</th>
        <th>Model</th>
        <th>Status</th>
        <th>Intent</th>
        <th>Failure</th>
        <th>Duration</th>
        <th>Steps</th>
        <th>Retries</th>
      </tr>
    </thead>
    <tbody></tbody>
  </table>
</div>

<script>
const DATA = __BENCHMARK_DATA__;

// ── KPIs ──
function renderKPIs() {
  const el = document.getElementById('kpis');
  if (!DATA.length) { el.innerHTML = '<div class="empty-state">No benchmark data yet.</div>'; return; }
  
  const total = DATA.length;
  const passed = DATA.filter(r => r.success).length;
  const rate = total ? ((passed / total) * 100).toFixed(1) : 0;
  const avgDuration = total ? (DATA.reduce((s, r) => s + r.duration_ms, 0) / total / 1000).toFixed(1) : 0;
  const totalRetries = DATA.reduce((s, r) => s + r.total_retries, 0);
  const totalRecoveries = DATA.reduce((s, r) => s + r.recovery_successes, 0);
  const recoveryRate = totalRetries ? ((totalRecoveries / totalRetries) * 100).toFixed(0) : 'N/A';

  el.innerHTML = `
    <div class="kpi-card success"><div class="value">${rate}%</div><div class="label">Success Rate</div></div>
    <div class="kpi-card accent"><div class="value">${total}</div><div class="label">Total Runs</div></div>
    <div class="kpi-card warn"><div class="value">${avgDuration}s</div><div class="label">Avg Duration</div></div>
    <div class="kpi-card ${totalRetries > total ? 'fail' : 'success'}"><div class="value">${totalRetries}</div><div class="label">Total Retries</div></div>
    <div class="kpi-card accent"><div class="value">${recoveryRate}${recoveryRate !== 'N/A' ? '%' : ''}</div><div class="label">Recovery Rate</div></div>
  `;
}

// ── Failure Heatmap ──
function renderFailureHeatmap() {
  const el = document.getElementById('failure-heatmap');
  const failures = DATA.filter(r => !r.success && r.failure_category);
  if (!failures.length) { el.innerHTML = '<div class="empty-state">No failures recorded.</div>'; return; }
  
  const counts = {};
  failures.forEach(r => { counts[r.failure_category] = (counts[r.failure_category] || 0) + 1; });
  
  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const max = sorted[0][1];
  
  const colors = {
    'routing_failure': '#f87171',
    'parsing_failure': '#fb923c',
    'invalid_edit': '#fbbf24',
    'hallucinated_tool': '#a78bfa',
    'critic_deadlock': '#818cf8',
    'context_overflow': '#34d399',
    'timeout': '#94a3b8',
    'malformed_params': '#f472b6',
    'unknown': '#6b7280',
  };
  
  el.innerHTML = sorted.map(([cat, count]) => {
    const pct = ((count / failures.length) * 100).toFixed(0);
    const width = Math.max(15, (count / max) * 100);
    const color = colors[cat] || '#6b7280';
    return `<div class="heatmap-row">
      <div class="heatmap-label">${cat}</div>
      <div class="heatmap-bar" style="width:${width}%;background:${color}22;color:${color};border:1px solid ${color}44">${count} (${pct}%)</div>
    </div>`;
  }).join('');
}

// ── Tool Reliability ──
function renderToolChart() {
  const allSteps = DATA.flatMap(r => r.steps || []);
  if (!allSteps.length) return;
  
  const toolStats = {};
  allSteps.forEach(s => {
    if (!toolStats[s.tool]) toolStats[s.tool] = { success: 0, fail: 0 };
    s.success ? toolStats[s.tool].success++ : toolStats[s.tool].fail++;
  });
  
  const labels = Object.keys(toolStats);
  const rates = labels.map(t => {
    const total = toolStats[t].success + toolStats[t].fail;
    return total ? ((toolStats[t].success / total) * 100).toFixed(1) : 0;
  });
  
  new Chart(document.getElementById('tool-chart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Success %',
        data: rates,
        backgroundColor: rates.map(r => r >= 80 ? '#34d39944' : r >= 50 ? '#fbbf2444' : '#f8717144'),
        borderColor: rates.map(r => r >= 80 ? '#34d399' : r >= 50 ? '#fbbf24' : '#f87171'),
        borderWidth: 1,
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, max: 100, ticks: { color: '#9ca3af' }, grid: { color: '#1f293740' } },
        x: { ticks: { color: '#9ca3af', font: { size: 11 } }, grid: { display: false } }
      }
    }
  });
}

// ── Retry Intelligence ──
function renderRetryChart() {
  const models = [...new Set(DATA.map(r => r.model))];
  if (!models.length) return;
  
  const modelData = models.map(m => {
    const runs = DATA.filter(r => r.model === m);
    const avgRetries = runs.length ? (runs.reduce((s, r) => s + r.total_retries, 0) / runs.length).toFixed(1) : 0;
    const successRate = runs.length ? ((runs.filter(r => r.success).length / runs.length) * 100).toFixed(0) : 0;
    return { model: m, avgRetries: parseFloat(avgRetries), successRate: parseFloat(successRate) };
  });
  
  new Chart(document.getElementById('retry-chart'), {
    type: 'bar',
    data: {
      labels: modelData.map(d => d.model),
      datasets: [
        {
          label: 'Avg Retries',
          data: modelData.map(d => d.avgRetries),
          backgroundColor: '#818cf844',
          borderColor: '#818cf8',
          borderWidth: 1,
          borderRadius: 4,
        },
        {
          label: 'Success %',
          data: modelData.map(d => d.successRate),
          backgroundColor: '#34d39944',
          borderColor: '#34d399',
          borderWidth: 1,
          borderRadius: 4,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#9ca3af' } } },
      scales: {
        y: { beginAtZero: true, ticks: { color: '#9ca3af' }, grid: { color: '#1f293740' } },
        x: { ticks: { color: '#9ca3af' }, grid: { display: false } }
      }
    }
  });
}

// ── Routing Matrix ──
function renderRoutingMatrix() {
  const el = document.getElementById('routing-matrix');
  const withRouting = DATA.filter(r => r.intent_correct);
  if (!withRouting.length) {
    el.innerHTML = '<div class="empty-state">No routing ground-truth labels yet.<br>Set <code>correct_intent</code> when recording.</div>';
    return;
  }
  
  const matrix = {};
  withRouting.forEach(r => {
    const key = r.intent_correct;
    if (!matrix[key]) matrix[key] = {};
    matrix[key][r.intent_predicted] = (matrix[key][r.intent_predicted] || 0) + 1;
  });
  
  const allIntents = [...new Set([...Object.keys(matrix), ...Object.values(matrix).flatMap(v => Object.keys(v))])];
  
  let html = '<table><thead><tr><th>Actual \\ Predicted</th>';
  allIntents.forEach(i => html += `<th>${i}</th>`);
  html += '</tr></thead><tbody>';
  
  Object.entries(matrix).forEach(([actual, preds]) => {
    const total = Object.values(preds).reduce((s, v) => s + v, 0);
    html += `<tr><td><strong>${actual}</strong></td>`;
    allIntents.forEach(pred => {
      const count = preds[pred] || 0;
      const pct = total ? ((count / total) * 100).toFixed(0) : 0;
      const isCorrect = actual === pred;
      const bg = isCorrect ? 'rgba(52,211,153,0.15)' : count > 0 ? 'rgba(248,113,113,0.15)' : 'transparent';
      html += `<td style="background:${bg};text-align:center">${count > 0 ? `${count} (${pct}%)` : '-'}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  el.innerHTML = html;
}

// ── Run History Table ──
function renderRunTable() {
  const tbody = document.querySelector('#run-table tbody');
  if (!DATA.length) { tbody.innerHTML = '<tr><td colspan="8" class="empty-state">No runs yet.</td></tr>'; return; }
  
  tbody.innerHTML = DATA.map(r => `<tr>
    <td>${r.task_name}</td>
    <td>${r.model}</td>
    <td><span class="tag ${r.success ? 'tag-pass' : 'tag-fail'}">${r.success ? 'PASS' : 'FAIL'}</span></td>
    <td>${r.intent_predicted}</td>
    <td>${r.failure_category || '-'}</td>
    <td>${(r.duration_ms / 1000).toFixed(1)}s</td>
    <td>${(r.steps || []).length}</td>
    <td>${r.total_retries}</td>
  </tr>`).join('');
}

// ── Init ──
renderKPIs();
renderFailureHeatmap();
renderToolChart();
renderRetryChart();
renderRoutingMatrix();
renderRunTable();
</script>
</body>
</html>
"""
