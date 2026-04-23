"use strict";

// ── DOM refs ────────────────────────────────────────────────────────────────
const uploadArea    = document.getElementById("uploadArea");
const fileInput     = document.getElementById("fileInput");
const fileChip      = document.getElementById("fileChip");
const modelSelect   = document.getElementById("modelSelect");
const modelHint     = document.getElementById("modelHint");
const runBtn        = document.getElementById("runBtn");
const statusMsg     = document.getElementById("statusMsg");

const emptyState      = document.getElementById("emptyState");
const pipelineSection = document.getElementById("pipelineSection");
const stepsList       = document.getElementById("stepsList");
const resultsSection  = document.getElementById("resultsSection");

const riskBanner    = document.getElementById("riskBanner");
const riskText      = document.getElementById("riskText");
const riskBadge     = document.getElementById("riskBadge");
const summaryCards  = document.getElementById("summaryCards");
const metricsSection = document.getElementById("metricsSection");
const metricCards   = document.getElementById("metricCards");
const confusionSection = document.getElementById("confusionSection");
const confusionMatrix  = document.getElementById("confusionMatrix");

let timelineChart = null;
let metricsChart  = null;

// ── Model hints ─────────────────────────────────────────────────────────────
const MODEL_HINTS = {
  xgboost:       "Gradient-boosted trees — fast, handles class imbalance well. Best overall on SKAB.",
  random_forest: "Ensemble of decision trees — robust and interpretable. Slightly lower recall but very stable.",
};

function updateHint() {
  modelHint.textContent = MODEL_HINTS[modelSelect.value] || "";
}
modelSelect.addEventListener("change", updateHint);
updateHint();

// ── File upload ─────────────────────────────────────────────────────────────
uploadArea.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.classList.add("dragover");
});
uploadArea.addEventListener("dragleave", () => uploadArea.classList.remove("dragover"));
uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("dragover");
  if (e.dataTransfer.files[0]) {
    fileInput.files = e.dataTransfer.files;
    setFile(e.dataTransfer.files[0]);
  }
});

function setFile(file) {
  fileChip.textContent = file.name;
  fileChip.classList.remove("d-none");
  uploadArea.classList.add("has-file");
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function showStatus(msg, type = "info") {
  statusMsg.textContent = msg;
  statusMsg.className = `status-msg ${type}`;
  statusMsg.classList.remove("d-none");
}
function hideStatus() { statusMsg.classList.add("d-none"); }

function show(el)  { el.classList.remove("d-none"); }
function hide(el)  { el.classList.add("d-none"); }

function scoreClass(val) {
  if (val >= 0.75) return "high";
  if (val >= 0.50) return "medium";
  return "low";
}

// ── Render pipeline steps (placeholder before response) ─────────────────────
const STEP_DEFS = [
  { id: 1, name: "File Loaded",          icon: "📁" },
  { id: 2, name: "Feature Engineering",  icon: "⚙️" },
  { id: 3, name: "Normalisation",        icon: "⚖️" },
  { id: 4, name: "Model Running",        icon: "🤖" },
  { id: 5, name: "Results Ready",        icon: "📊" },
];

function buildStepItems() {
  stepsList.innerHTML = "";
  STEP_DEFS.forEach(({ id, name, icon }) => {
    const div = document.createElement("div");
    div.className = "step-item";
    div.id = `step-${id}`;
    div.innerHTML = `
      <div class="step-icon-wrap">${icon}</div>
      <div class="flex-grow-1">
        <div class="step-name">${name}</div>
        <div class="step-detail" id="step-detail-${id}">Waiting…</div>
      </div>
      <div class="step-spinner"></div>
      <div class="step-check">✅</div>
    `;
    stepsList.appendChild(div);
  });
}

async function animateSteps(stepsData) {
  for (const step of stepsData) {
    const el     = document.getElementById(`step-${step.id}`);
    const detail = document.getElementById(`step-detail-${step.id}`);

    // Mark previous as done
    if (step.id > 1) {
      const prev = document.getElementById(`step-${step.id - 1}`);
      if (prev) { prev.classList.remove("active"); prev.classList.add("done"); }
    }

    el.classList.add("active");
    detail.innerHTML = step.detail;

    await sleep(step.id === 4 ? 900 : 600);
  }
  // Mark last step done
  const last = document.getElementById(`step-${stepsData.at(-1).id}`);
  if (last) { last.classList.remove("active"); last.classList.add("done"); }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Risk banner ──────────────────────────────────────────────────────────────
function renderRiskBanner(anomalyRate) {
  riskBanner.className = "risk-banner mb-4";
  if (anomalyRate < 10) {
    riskBanner.classList.add("low");
    riskText.textContent  = "Pump Operating Normally";
    riskBadge.textContent = "✅ Low Risk";
  } else if (anomalyRate < 30) {
    riskBanner.classList.add("medium");
    riskText.textContent  = "Potential Irregularity Detected";
    riskBadge.textContent = "⚠️ Medium Risk";
  } else {
    riskText.textContent  = "Failure Likely — Inspect Pump";
    riskBadge.textContent = "🔴 High Risk";
  }
}

// ── Summary cards ─────────────────────────────────────────────────────────────
function renderSummaryCards(data) {
  summaryCards.innerHTML = `
    <div class="col-6 col-md-3">
      <div class="summary-card">
        <span class="summary-label">Total Readings</span>
        <div class="summary-value">${data.total_rows.toLocaleString()}</div>
      </div>
    </div>
    <div class="col-6 col-md-3">
      <div class="summary-card">
        <span class="summary-label">Anomalies Found</span>
        <div class="summary-value text-danger">${data.anomaly_count.toLocaleString()}</div>
      </div>
    </div>
    <div class="col-6 col-md-3">
      <div class="summary-card">
        <span class="summary-label">Normal Readings</span>
        <div class="summary-value text-success">${data.normal_count.toLocaleString()}</div>
      </div>
    </div>
    <div class="col-6 col-md-3">
      <div class="summary-card">
        <span class="summary-label">Anomaly Rate</span>
        <div class="summary-value">${data.anomaly_rate}%</div>
      </div>
    </div>
  `;
}

// ── Metric cards + bar chart ──────────────────────────────────────────────────
function renderMetrics(metrics) {
  const items = [
    { key: "f1",        label: "F1 Score",  tip: "Balance of precision and recall" },
    { key: "precision", label: "Precision", tip: "Of flagged anomalies, how many were real?" },
    { key: "recall",    label: "Recall",    tip: "Of real anomalies, how many were caught?" },
    { key: "roc_auc",   label: "ROC-AUC",   tip: "Overall ranking ability (1 = perfect)" },
  ];

  metricCards.innerHTML = items
    .filter(({ key }) => metrics[key] != null)
    .map(({ key, label, tip }) => {
      const val = metrics[key];
      const cls = scoreClass(val);
      return `
        <div class="col-6 col-md-3">
          <div class="metric-card" title="${tip}">
            <div class="metric-name">${label}</div>
            <div class="metric-value ${cls}">${(val * 100).toFixed(1)}%</div>
          </div>
        </div>`;
    }).join("");

  // Bar chart
  const labels = items.filter(({ key }) => metrics[key] != null).map(i => i.label);
  const values = items.filter(({ key }) => metrics[key] != null).map(({ key }) => +(metrics[key] * 100).toFixed(1));

  if (metricsChart) metricsChart.destroy();
  metricsChart = new Chart(document.getElementById("metricsChart"), {
    type: "bar",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: values.map(v => v >= 75 ? "#16a34a" : v >= 50 ? "#d97706" : "#dc2626"),
        borderRadius: 8,
        barThickness: 28,
      }],
    },
    options: {
      indexAxis: "y",
      plugins: { legend: { display: false } },
      scales: {
        x: { min: 0, max: 100, ticks: { callback: v => v + "%" } },
        y: { ticks: { font: { weight: "600" } } },
      },
    },
  });
}

// ── Anomaly timeline chart ────────────────────────────────────────────────────
function renderTimeline(timeline) {
  const labels = timeline.map(p => p.i);
  const probs  = timeline.map(p => +(p.prob * 100).toFixed(1));
  const colors = timeline.map(p => p.pred === 1 ? "rgba(220,38,38,.7)" : "rgba(22,163,74,.6)");

  if (timelineChart) timelineChart.destroy();
  timelineChart = new Chart(document.getElementById("timelineChart"), {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Anomaly Probability",
          data: probs,
          borderColor: colors,
          backgroundColor: "transparent",
          pointBackgroundColor: colors,
          pointRadius: timeline.length > 300 ? 0 : 3,
          borderWidth: 1.5,
          tension: 0.2,
        },
      ],
    },
    options: {
      animation: { duration: 600 },
      plugins: {
        legend: { display: false },
        annotation: {
          annotations: {
            threshold: {
              type: "line",
              yMin: 50, yMax: 50,
              borderColor: "#dc2626",
              borderWidth: 1.5,
              borderDash: [6, 3],
              label: {
                content: "Threshold (50%)",
                enabled: true,
                position: "end",
                color: "#dc2626",
                font: { size: 11 },
              },
            },
          },
        },
      },
      scales: {
        x: { display: false },
        y: {
          min: 0, max: 100,
          ticks: { callback: v => v + "%" },
          title: { display: true, text: "Anomaly Probability (%)" },
        },
      },
    },
  });
}

// ── Confusion matrix ──────────────────────────────────────────────────────────
function renderConfusion(cm) {
  const [[tn, fp], [fn, tp]] = cm;
  const total = tn + fp + fn + tp;
  const pct = v => `<small>${((v / total) * 100).toFixed(1)}% of data</small>`;

  confusionMatrix.innerHTML = `
    <div></div>
    <div class="cm-header">Predicted: Normal</div>
    <div class="cm-header">Predicted: Anomaly</div>

    <div class="cm-header" style="writing-mode:vertical-lr;transform:rotate(180deg)">Actual: Normal</div>
    <div class="cm-cell cm-tn">${tn.toLocaleString()}${pct(tn)}<small style="color:#166534">✓ True Negative</small></div>
    <div class="cm-cell cm-fp">${fp.toLocaleString()}${pct(fp)}<small style="color:#b91c1c">✗ False Positive</small></div>

    <div class="cm-header" style="writing-mode:vertical-lr;transform:rotate(180deg)">Actual: Anomaly</div>
    <div class="cm-cell cm-fn">${fn.toLocaleString()}${pct(fn)}<small style="color:#c2410c">✗ Missed</small></div>
    <div class="cm-cell cm-tp">${tp.toLocaleString()}${pct(tp)}<small style="color:#1e40af">✓ True Positive</small></div>
  `;
}

// ── Main run handler ──────────────────────────────────────────────────────────
runBtn.addEventListener("click", async () => {
  hideStatus();

  if (!fileInput.files[0]) {
    showStatus("Please upload a CSV file first.", "error");
    return;
  }

  // Reset UI
  hide(emptyState);
  hide(resultsSection);
  buildStepItems();
  show(pipelineSection);

  runBtn.disabled = true;
  runBtn.textContent = "Running…";

  const formData = new FormData();
  formData.append("file",  fileInput.files[0]);
  formData.append("model", modelSelect.value);

  let data;
  try {
    const res = await fetch("/api/predict", { method: "POST", body: formData });
    data = await res.json();
    if (!res.ok) throw new Error(data.error || "Server error");
  } catch (err) {
    showStatus(`Error: ${err.message}`, "error");
    runBtn.disabled = false;
    runBtn.textContent = "Run Prediction";
    return;
  }

  // Animate steps (staggered, using real detail text from backend)
  await animateSteps(data.steps);
  await sleep(400);

  // Render results
  renderRiskBanner(data.anomaly_rate);
  renderSummaryCards(data);

  if (data.has_labels && data.metrics && Object.keys(data.metrics).length) {
    renderMetrics(data.metrics);
    show(metricsSection);
  }

  renderTimeline(data.timeline);

  if (data.has_labels && data.confusion) {
    renderConfusion(data.confusion);
    show(confusionSection);
  } else {
    hide(confusionSection);
  }

  show(resultsSection);
  showStatus(`Prediction complete — model: ${data.model} · file: ${data.filename}`, "info");

  runBtn.disabled = false;
  runBtn.textContent = "Run Prediction";
});
