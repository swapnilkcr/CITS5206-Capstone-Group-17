"use strict";

// ── DOM refs ─────────────────────────────────────────────────────────────────
const uploadArea     = document.getElementById("uploadArea");
const fileInput      = document.getElementById("fileInput");
const fileChip       = document.getElementById("fileChip");
const uploadTitle    = document.getElementById("uploadTitle");
const modelSelect    = document.getElementById("modelSelect");
const modelHint      = document.getElementById("modelHint");
const modeHint       = document.getElementById("modeHint");
const runBtn         = document.getElementById("runBtn");
const statusMsg      = document.getElementById("statusMsg");

const emptyState      = document.getElementById("emptyState");
const pipelineSection = document.getElementById("pipelineSection");
const stepsList       = document.getElementById("stepsList");
const singleResults   = document.getElementById("singleResults");
const zipResults      = document.getElementById("zipResults");

// Single-file result elements
const riskBanner    = document.getElementById("riskBanner");
const riskText      = document.getElementById("riskText");
const riskBadge     = document.getElementById("riskBadge");
const summaryCards  = document.getElementById("summaryCards");
const metricsSection = document.getElementById("metricsSection");
const metricCards   = document.getElementById("metricCards");
const confusionSection = document.getElementById("confusionSection");
const confusionMatrix  = document.getElementById("confusionMatrix");

// ZIP result elements
const zipBanner       = document.getElementById("zipBanner");
const zipTitle        = document.getElementById("zipTitle");
const zipBadge        = document.getElementById("zipBadge");
const zipSummaryCards = document.getElementById("zipSummaryCards");
const zipMetricCards  = document.getElementById("zipMetricCards");
const perFileBody     = document.getElementById("perFileBody");
const zipConfusionMatrix = document.getElementById("zipConfusionMatrix");

let timelineChart    = null;
let metricsChart     = null;
let zipTimelineChart = null;
let zipMetricsChart  = null;
let foldChart        = null;

// ── Mode (single / zip) ───────────────────────────────────────────────────────
let currentMode = "single";

function setMode(mode) {
  currentMode = mode;
  document.getElementById("btnSingle").classList.toggle("active", mode === "single");
  document.getElementById("btnZip").classList.toggle("active", mode === "zip");

  if (mode === "single") {
    fileInput.accept  = ".csv";
    uploadTitle.textContent = "Drag & drop CSV here";
    modeHint.textContent = "Upload one CSV — evaluated with Leave-One-Out against all SKAB training files.";
  } else {
    fileInput.accept  = ".zip";
    uploadTitle.textContent = "Drag & drop ZIP here";
    modeHint.textContent = "Upload a ZIP of CSVs — evaluated with Group K-Fold (same as the research notebook).";
  }

  // Reset file selection
  fileInput.value = "";
  fileChip.classList.add("d-none");
  uploadArea.classList.remove("has-file");
}

// ── Model hints ───────────────────────────────────────────────────────────────
const MODEL_HINTS = {
  xgboost:          "Gradient-boosted trees — fast, handles class imbalance well. Best overall on SKAB.",
  random_forest:    "Ensemble of decision trees — robust and stable. Slightly lower recall but very consistent.",
  isolation_forest: "Unsupervised anomaly detection — no labels needed. Trained only on normal pump behaviour; flags readings that deviate from the baseline.",
  lstm_autoencoder: "LSTM Autoencoder (Thant) — unsupervised deep learning. Learns normal sensor patterns then flags windows with high reconstruction error. No labels needed.",
  transformer:      "Transformer Autoencoder (Zeyang) — unsupervised deep learning. Uses self-attention over 60-timestep windows to learn normal behaviour; high reconstruction error flags anomalies. No labels needed.",
  adapted_lstm:     "Adapted LSTM / ALSS-SVDD (Julie) — semi-supervised deep SVDD. LSTM encoder learns a centre for normal behaviour; anomalies are windows far from that centre. Trained on labelled valve1 data.",
};
function updateHint() { modelHint.textContent = MODEL_HINTS[modelSelect.value] || ""; }
modelSelect.addEventListener("change", updateHint);
updateHint();

// ── File upload ───────────────────────────────────────────────────────────────
uploadArea.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });
uploadArea.addEventListener("dragover",  (e) => { e.preventDefault(); uploadArea.classList.add("dragover"); });
uploadArea.addEventListener("dragleave", ()  => uploadArea.classList.remove("dragover"));
uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("dragover");
  if (e.dataTransfer.files[0]) { fileInput.files = e.dataTransfer.files; setFile(e.dataTransfer.files[0]); }
});

function setFile(file) {
  fileChip.textContent = file.name;
  fileChip.classList.remove("d-none");
  uploadArea.classList.add("has-file");
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function showStatus(msg, type = "info") {
  statusMsg.textContent = msg;
  statusMsg.className   = `status-msg ${type}`;
  statusMsg.classList.remove("d-none");
}
function hideStatus() { statusMsg.classList.add("d-none"); }
function show(el) { el.classList.remove("d-none"); }
function hide(el) { el.classList.add("d-none"); }
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function scoreClass(val) {
  if (val >= 0.75) return "high";
  if (val >= 0.50) return "medium";
  return "low";
}
function f1BadgeClass(val) {
  if (val >= 0.75) return "f1-high";
  if (val >= 0.40) return "f1-medium";
  return "f1-low";
}

// ── Pipeline steps ────────────────────────────────────────────────────────────
function buildStepItems(stepsData) {
  stepsList.innerHTML = "";
  stepsData.forEach(({ id, name, icon }) => {
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
      <div class="step-check">✅</div>`;
    stepsList.appendChild(div);
  });
}

async function animateSteps(stepsData) {
  for (const step of stepsData) {
    const el     = document.getElementById(`step-${step.id}`);
    const detail = document.getElementById(`step-detail-${step.id}`);
    if (step.id > 1) {
      const prev = document.getElementById(`step-${step.id - 1}`);
      if (prev) { prev.classList.remove("active"); prev.classList.add("done"); }
    }
    el.classList.add("active");
    detail.innerHTML = step.detail;
    await sleep(step.id === 4 ? 900 : 600);
  }
  const last = document.getElementById(`step-${stepsData.at(-1).id}`);
  if (last) { last.classList.remove("active"); last.classList.add("done"); }
}

// ── Single-file rendering ─────────────────────────────────────────────────────
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

function renderSummaryCards(data, container) {
  container.innerHTML = `
    <div class="col-6 col-md-3"><div class="summary-card">
      <span class="summary-label">Total Readings</span>
      <div class="summary-value">${data.total_rows.toLocaleString()}</div>
    </div></div>
    <div class="col-6 col-md-3"><div class="summary-card">
      <span class="summary-label">Anomalies Found</span>
      <div class="summary-value text-danger">${data.anomaly_count.toLocaleString()}</div>
    </div></div>
    <div class="col-6 col-md-3"><div class="summary-card">
      <span class="summary-label">Normal Readings</span>
      <div class="summary-value text-success">${data.normal_count.toLocaleString()}</div>
    </div></div>
    <div class="col-6 col-md-3"><div class="summary-card">
      <span class="summary-label">Anomaly Rate</span>
      <div class="summary-value">${data.anomaly_rate}%</div>
    </div></div>`;
}

const METRIC_EXPLAIN = {
  f1:        "Overall balance between catching faults and avoiding false alarms",
  precision: "Of every fault flagged, how many were actually real faults",
  recall:    "Of every real fault, how many did the model catch",
  roc_auc:   "How well the model separates normal readings from faulty ones (higher = better)",
};

function renderMetricCards(metrics, container, chartId) {
  const items = [
    { key: "f1",        label: "F1 Score"  },
    { key: "precision", label: "Precision" },
    { key: "recall",    label: "Recall"    },
    { key: "roc_auc",   label: "ROC-AUC"  },
  ].filter(({ key }) => metrics[key] != null);

  container.innerHTML = items.map(({ key, label }) => {
    const val = metrics[key];
    const cls = scoreClass(val);
    return `<div class="col-6 col-md-3"><div class="metric-card">
      <div class="metric-name">${label}</div>
      <div class="metric-value ${cls}">${(val * 100).toFixed(1)}%</div>
      <div class="metric-explain">${METRIC_EXPLAIN[key]}</div>
    </div></div>`;
  }).join("");

  const canvas = document.getElementById(chartId);
  if (!canvas) return;
  const existingChart = Chart.getChart(canvas);
  if (existingChart) existingChart.destroy();

  const labels = items.map(i => i.label);
  const values = items.map(({ key }) => +(metrics[key] * 100).toFixed(1));

  new Chart(canvas, {
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

function renderTimeline(timeline, canvasId) {
  const labels = timeline.map(p => p.i);
  const probs  = timeline.map(p => +(p.prob * 100).toFixed(1));
  const colors = timeline.map(p => p.pred === 1 ? "rgba(220,38,38,.7)" : "rgba(22,163,74,.6)");

  const canvas = document.getElementById(canvasId);
  const existingChart = Chart.getChart(canvas);
  if (existingChart) existingChart.destroy();

  new Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Anomaly Probability",
        data: probs,
        borderColor: colors,
        backgroundColor: "transparent",
        pointBackgroundColor: colors,
        pointRadius: timeline.length > 300 ? 0 : 3,
        borderWidth: 1.5,
        tension: 0.2,
      }],
    },
    options: {
      animation: { duration: 600 },
      plugins: { legend: { display: false } },
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

function renderConfusion(cm, container) {
  const [[tn, fp], [fn, tp]] = cm;
  const total = tn + fp + fn + tp;
  const pct = v => `${((v / total) * 100).toFixed(1)}%`;
  container.innerHTML = `
    <div></div>
    <div class="cm-header">Model said: Normal</div>
    <div class="cm-header">Model said: Fault</div>
    <div class="cm-header" style="writing-mode:vertical-lr;transform:rotate(180deg)">Was: Normal</div>
    <div class="cm-cell cm-tn">
      <div class="cm-count">${tn.toLocaleString()}</div>
      <div class="cm-pct">${pct(tn)}</div>
      <div class="cm-tag cm-tag-tn">✓ Correct — no fault</div>
      <div class="cm-desc">Normal reading, correctly left alone</div>
    </div>
    <div class="cm-cell cm-fp">
      <div class="cm-count">${fp.toLocaleString()}</div>
      <div class="cm-pct">${pct(fp)}</div>
      <div class="cm-tag cm-tag-fp">⚠ False alarm</div>
      <div class="cm-desc">Normal reading, incorrectly flagged as fault</div>
    </div>
    <div class="cm-header" style="writing-mode:vertical-lr;transform:rotate(180deg)">Was: Fault</div>
    <div class="cm-cell cm-fn">
      <div class="cm-count">${fn.toLocaleString()}</div>
      <div class="cm-pct">${pct(fn)}</div>
      <div class="cm-tag cm-tag-fn">✗ Missed fault</div>
      <div class="cm-desc">Real fault that the model failed to catch</div>
    </div>
    <div class="cm-cell cm-tp">
      <div class="cm-count">${tp.toLocaleString()}</div>
      <div class="cm-pct">${pct(tp)}</div>
      <div class="cm-tag cm-tag-tp">✓ Caught fault</div>
      <div class="cm-desc">Real fault, correctly detected</div>
    </div>`;
}

// ── ZIP-specific rendering ────────────────────────────────────────────────────
function renderFoldChart(foldF1s) {
  const canvas = document.getElementById("foldChart");
  const existingChart = Chart.getChart(canvas);
  if (existingChart) existingChart.destroy();

  new Chart(canvas, {
    type: "bar",
    data: {
      labels: foldF1s.map((_, i) => `Fold ${i + 1}`),
      datasets: [{
        data: foldF1s.map(v => +(v * 100).toFixed(1)),
        backgroundColor: foldF1s.map(v => v >= 0.75 ? "#16a34a" : v >= 0.40 ? "#d97706" : "#dc2626"),
        borderRadius: 8,
        barThickness: 36,
      }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        y: {
          min: 0, max: 100,
          ticks: { callback: v => v + "%" },
          title: { display: true, text: "F1 Score (%)" },
        },
      },
    },
  });
}

function renderPerFileTable(perFile) {
  perFileBody.innerHTML = perFile.map(row => {
    const badgeClass = f1BadgeClass(row.f1);
    return `<tr>
      <td><strong>${row.file}</strong></td>
      <td>${row.rows.toLocaleString()}</td>
      <td>${row.anomaly_rate}%</td>
      <td><span class="f1-badge ${badgeClass}">${(row.f1 * 100).toFixed(1)}%</span></td>
      <td>${(row.precision * 100).toFixed(1)}%</td>
      <td>${(row.recall * 100).toFixed(1)}%</td>
    </tr>`;
  }).join("");
}

function renderZipBanner(data) {
  zipBanner.className = "risk-banner mb-4";
  document.querySelector("#zipBanner .risk-label").textContent =
    data.n_folds > 0 ? "Dataset Evaluation (Group K-Fold)" : "Dataset Evaluation (Isolation Forest)";
  const f1 = data.overall.f1;
  if (f1 >= 0.75) {
    zipBanner.classList.add("low");
    zipTitle.textContent = "Strong Model Performance";
    zipBadge.textContent = `✅ F1 ${(f1 * 100).toFixed(1)}%`;
  } else if (f1 >= 0.40) {
    zipBanner.classList.add("medium");
    zipTitle.textContent = "Moderate Model Performance";
    zipBadge.textContent = `⚠️ F1 ${(f1 * 100).toFixed(1)}%`;
  } else {
    zipTitle.textContent = "Low Model Performance";
    zipBadge.textContent = `🔴 F1 ${(f1 * 100).toFixed(1)}%`;
  }
}

// ── Main run handler ──────────────────────────────────────────────────────────
runBtn.addEventListener("click", async () => {
  hideStatus();

  if (!fileInput.files[0]) {
    showStatus(`Please upload a ${currentMode === "zip" ? "ZIP" : "CSV"} file first.`, "error");
    return;
  }

  hide(emptyState);
  hide(singleResults);
  hide(zipResults);

  runBtn.disabled      = true;
  runBtn.textContent   = "Running…";

  const formData = new FormData();
  formData.append("file",  fileInput.files[0]);
  formData.append("model", modelSelect.value);

  const endpoint = currentMode === "zip" ? "/api/evaluate" : "/api/predict";

  let data;
  try {
    const res = await fetch(endpoint, { method: "POST", body: formData });
    data = await res.json();
    if (!res.ok) throw new Error(data.error || "Server error");
  } catch (err) {
    showStatus(`Error: ${err.message}`, "error");
    runBtn.disabled    = false;
    runBtn.textContent = "Run Prediction";
    show(emptyState);
    return;
  }

  buildStepItems(data.steps);
  show(pipelineSection);
  await animateSteps(data.steps);
  await sleep(400);

  if (currentMode === "single") {
    // Single file results
    renderRiskBanner(data.anomaly_rate);
    renderSummaryCards(data, summaryCards);

    if (data.has_labels && data.metrics && Object.keys(data.metrics).length) {
      renderMetricCards(data.metrics, metricCards, "metricsChart");
      show(metricsSection);
    } else {
      hide(metricsSection);
    }

    renderTimeline(data.timeline, "timelineChart");

    if (data.has_labels && data.confusion) {
      renderConfusion(data.confusion, confusionMatrix);
      show(confusionSection);
    } else {
      hide(confusionSection);
    }

    show(singleResults);
    showStatus(`Done · ${data.model} · ${data.filename}`, "info");

  } else {
    // ZIP dataset results
    renderZipBanner(data);

    zipSummaryCards.innerHTML = `
      <div class="col-6 col-md-3"><div class="summary-card">
        <span class="summary-label">Files</span>
        <div class="summary-value">${data.n_files}</div>
      </div></div>
      <div class="col-6 col-md-3"><div class="summary-card">
        <span class="summary-label">Total Rows</span>
        <div class="summary-value">${data.total_rows.toLocaleString()}</div>
      </div></div>
      <div class="col-6 col-md-3"><div class="summary-card">
        <span class="summary-label">Anomalies</span>
        <div class="summary-value text-danger">${data.anomaly_count.toLocaleString()}</div>
      </div></div>
      <div class="col-6 col-md-3"><div class="summary-card">
        <span class="summary-label">${data.n_folds > 0 ? "K-Fold Splits" : "Method"}</span>
        <div class="summary-value">${data.n_folds > 0 ? data.n_folds : "Unsupervised"}</div>
      </div></div>`;

    renderMetricCards(data.overall, zipMetricCards, "zipMetricsChart");

    // Fold chart — hide for Isolation Forest (no cross-validation folds)
    const foldSection = document.querySelector("#zipResults .mb-4:has(#foldChart)");
    if (data.fold_f1s && data.fold_f1s.length > 0) {
      renderFoldChart(data.fold_f1s);
      if (foldSection) foldSection.style.display = "";
    } else {
      if (foldSection) foldSection.style.display = "none";
    }

    renderTimeline(data.timeline, "zipTimelineChart");
    renderPerFileTable(data.per_file);
    renderConfusion(data.confusion, zipConfusionMatrix);

    show(zipResults);
    showStatus(`Done · ${data.model} · ${data.n_files} files · Overall F1: ${(data.overall.f1 * 100).toFixed(1)}%`, "info");
  }

  runBtn.disabled    = false;
  runBtn.textContent = "Run Prediction";
});
