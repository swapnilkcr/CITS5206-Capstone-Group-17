const fileInput = document.getElementById("datasetFile");
const uploadArea = document.getElementById("uploadArea");
const selectedFileName = document.getElementById("selectedFileName");
const algorithmSelect = document.getElementById("algorithm");
const algorithmHint = document.getElementById("algorithmHint");
const sequenceLengthInput = document.getElementById("sequenceLength");
const predictBtn = document.getElementById("predictBtn");
const statusMessage = document.getElementById("statusMessage");

const emptyState = document.getElementById("emptyState");
const loadingState = document.getElementById("loadingState");
const resultSection = document.getElementById("resultSection");

const predictionBanner = document.getElementById("predictionBanner");
const predictionText = document.getElementById("predictionText");
const predictionBadge = document.getElementById("predictionBadge");

const resultAlgorithm = document.getElementById("resultAlgorithm");
const resultFile = document.getElementById("resultFile");
const resultRatio = document.getElementById("resultRatio");
const resultsTableBody = document.getElementById("resultsTableBody");
const windowCount = document.getElementById("windowCount");

const algorithmDescriptions = {
  if: "Good baseline model for unsupervised anomaly detection.",
  ecod: "Lightweight statistical anomaly detector based on empirical distributions.",
  transformer: "Sequence-based deep learning model for more complex temporal patterns."
};

function updateAlgorithmHint() {
  const selected = algorithmSelect.value;
  algorithmHint.textContent = algorithmDescriptions[selected] || "";
}

function showStatus(message, type) {
  statusMessage.textContent = message;
  statusMessage.className = `status-message ${type}`;
  statusMessage.classList.remove("d-none");
}

function hideStatus() {
  statusMessage.classList.add("d-none");
}

function updateFileName(file) {
  if (file) {
    selectedFileName.textContent = file.name;
  } else {
    selectedFileName.textContent = "No file selected";
  }
}

function setViewState(state) {
  emptyState.classList.add("d-none");
  loadingState.classList.add("d-none");
  resultSection.classList.add("d-none");

  if (state === "empty") emptyState.classList.remove("d-none");
  if (state === "loading") loadingState.classList.remove("d-none");
  if (state === "result") resultSection.classList.remove("d-none");
}

function formatAlgorithmName(value) {
  if (value === "if") return "Isolation Forest";
  if (value === "ecod") return "ECOD";
  if (value === "transformer") return "Anomaly Transformer";
  return value;
}

function renderPredictionBanner(anomalyRatio) {
  predictionBanner.classList.remove("normal", "warning");

  if (anomalyRatio < 10) {
    predictionBanner.classList.add("normal");
    predictionText.textContent = "Pump Operating Normally";
    predictionBadge.textContent = "Low Risk";
  } else if (anomalyRatio < 25) {
    predictionBanner.classList.add("warning");
    predictionText.textContent = "Potential Irregularity Detected";
    predictionBadge.textContent = "Medium Risk";
  } else {
    predictionText.textContent = "Failure Likely";
    predictionBadge.textContent = "High Risk";
  }
}

function renderResultsTable(rows) {
  resultsTableBody.innerHTML = "";

  rows.forEach((row) => {
    const tr = document.createElement("tr");

    const statusClass = row.pred.toLowerCase() === "anomaly" ? "anomaly" : "normal";

    tr.innerHTML = `
      <td>${row.window}</td>
      <td>${row.score.toFixed(2)}</td>
      <td>
        <span class="table-status ${statusClass}">
          ${row.pred}
        </span>
      </td>
    `;

    resultsTableBody.appendChild(tr);
  });

  windowCount.textContent = `${rows.length} windows`;
}

function generateMockResults() {
  return [
    { window: 1, score: 0.21, pred: "Normal" },
    { window: 2, score: 0.32, pred: "Normal" },
    { window: 3, score: 0.88, pred: "Anomaly" },
    { window: 4, score: 0.91, pred: "Anomaly" },
    { window: 5, score: 0.27, pred: "Normal" },
    { window: 6, score: 0.79, pred: "Anomaly" }
  ];
}

function calculateAnomalyRatio(rows) {
  const anomalies = rows.filter((row) => row.pred === "Anomaly").length;
  return Math.round((anomalies / rows.length) * 100);
}

async function runPrediction() {
  hideStatus();

  const file = fileInput.files[0];
  const algorithm = algorithmSelect.value;
  const sequenceLength = sequenceLengthInput.value.trim();

  if (!file) {
    showStatus("Please upload a CSV file first.", "error");
    return;
  }

  if (!sequenceLength || Number(sequenceLength) < 1) {
    showStatus("Sequence length must be at least 1.", "error");
    return;
  }

  setViewState("loading");
  predictBtn.disabled = true;
  predictBtn.textContent = "Processing...";

  try {
    await new Promise((resolve) => setTimeout(resolve, 1200));

    const mockResults = generateMockResults();
    const anomalyRatio = calculateAnomalyRatio(mockResults);

    resultAlgorithm.textContent = formatAlgorithmName(algorithm);
    resultFile.textContent = file.name;
    resultRatio.textContent = `${anomalyRatio}%`;

    renderPredictionBanner(anomalyRatio);
    renderResultsTable(mockResults);

    setViewState("result");
    showStatus("Prediction completed successfully.", "info");

    /*
      BACKEND INTEGRATION LATER:

      const formData = new FormData();
      formData.append("file", file);
      formData.append("algorithm", algorithm);
      formData.append("sequenceLength", sequenceLength);

      const response = await fetch("YOUR_BACKEND_ENDPOINT", {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      Then update UI using actual backend response.
    */
  } catch (error) {
    console.error(error);
    setViewState("empty");
    showStatus("Something went wrong while processing the dataset.", "error");
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Run Prediction";
  }
}

uploadArea.addEventListener("click", () => {
  fileInput.click();
});

fileInput.addEventListener("change", () => {
  updateFileName(fileInput.files[0]);
});

uploadArea.addEventListener("dragover", (event) => {
  event.preventDefault();
  uploadArea.classList.add("dragover");
});

uploadArea.addEventListener("dragleave", () => {
  uploadArea.classList.remove("dragover");
});

uploadArea.addEventListener("drop", (event) => {
  event.preventDefault();
  uploadArea.classList.remove("dragover");

  const files = event.dataTransfer.files;
  if (files.length > 0) {
    fileInput.files = files;
    updateFileName(files[0]);
  }
});

algorithmSelect.addEventListener("change", updateAlgorithmHint);
predictBtn.addEventListener("click", runPrediction);

updateAlgorithmHint();
setViewState("empty");