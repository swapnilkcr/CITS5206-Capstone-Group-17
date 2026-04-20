document.getElementById("predictBtn").addEventListener("click", async () => {
  const fileInput = document.getElementById("datasetFile");
  const algorithm = document.getElementById("algorithm").value;
  const sequenceLength = document.getElementById("sequenceLength").value;

  if (!fileInput.files.length) {
    alert("Please upload a CSV file.");
    return;
  }

  const file = fileInput.files[0];

  // Demo UI output for now
  document.getElementById("emptyState").classList.add("d-none");
  document.getElementById("resultSection").classList.remove("d-none");

  document.getElementById("resultAlgorithm").textContent = algorithm.toUpperCase();
  document.getElementById("resultFile").textContent = file.name;
  document.getElementById("resultRatio").textContent = "18%";

  const predictionBox = document.getElementById("predictionText");
  predictionBox.className = "alert alert-danger";
  predictionBox.textContent = "Failure Likely";

  const tableBody = document.getElementById("resultsTableBody");
  tableBody.innerHTML = "";

  const mockResults = [
    { window: 1, score: 0.21, pred: "Normal" },
    { window: 2, score: 0.32, pred: "Normal" },
    { window: 3, score: 0.88, pred: "Anomaly" },
    { window: 4, score: 0.91, pred: "Anomaly" },
    { window: 5, score: 0.27, pred: "Normal" }
  ];

  mockResults.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.window}</td>
      <td>${row.score}</td>
      <td>${row.pred}</td>
    `;
    tableBody.appendChild(tr);
  });

  // Later:
  // send file + algorithm + sequenceLength to backend using fetch()
});