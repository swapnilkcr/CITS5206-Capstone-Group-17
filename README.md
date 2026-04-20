# Predictive Maintenance on SKAB using Isolation Forest

This project implements an **Isolation Forest-based anomaly detection pipeline** on the **SKAB (Skoltech Anomaly Benchmark)** dataset for predictive maintenance experiments.

The goal is to detect abnormal pump behaviour using sensor readings from the dataset and evaluate how well an unsupervised model can identify anomalies across multiple files.

## Project Overview

This notebook-based workflow was developed in **Google Colab** and uses:

- **Python**
- **Pandas / NumPy** for data processing
- **Scikit-learn** for scaling and Isolation Forest
- **Matplotlib** for plotting results

The approach follows a structured evaluation pipeline:

1. Train on **anomaly-free SKAB data** only
2. Engineer **rolling mean** and **rolling standard deviation** features
3. Scale data using `StandardScaler`
4. Train an **Isolation Forest** model on normal data
5. Set anomaly threshold from **training scores**
6. Test on **all labeled SKAB files**
7. Evaluate performance per file using:
   - F1-score
   - Precision
   - Recall
   - ROC-AUC
   - Confusion Matrix

## Dataset

The project uses the **SKAB dataset**, which contains multivariate sensor data for predictive maintenance and anomaly detection experiments.

Training data:
- `SKAB/anomaly-free/anomaly-free.csv`

Testing data:
- all labeled CSV files under the SKAB folders such as:
  - `valve1`
  - `valve2`
  - `other`

## Features Used

Base sensor features:

- `Accelerometer1RMS`
- `Accelerometer2RMS`
- `Current`
- `Pressure`
- `Temperature`
- `Thermocouple`
- `Voltage`
- `Volume Flow RateRMS`

Engineered features:
- rolling mean
- rolling standard deviation

These rolling features help capture short-term trends in the time-series data.

## Methodology

### 1. Training

Only anomaly-free data is used for training to simulate an unsupervised anomaly detection setting.

### 2. Feature Engineering

For each file, rolling statistics are computed per feature using a fixed window size.

### 3. Scaling

A `StandardScaler` is fitted on the anomaly-free training data and then reused for all test files.

### 4. Isolation Forest

The model is trained on scaled normal data.

Example parameters used:

```python
IsolationForest(
    n_estimators=300,
    contamination=0.01,
    random_state=42,
    n_jobs=-1
)
```

### 5. Threshold Selection

The anomaly threshold is selected from the **training score distribution**, not from the test data.

Thresholds tested:
- 95th percentile
- 97th percentile
- 99th percentile

This was done to study the precision-recall trade-off.

## Key Findings

- Lower thresholds gave **very high recall** but also produced many **false positives**.
- Higher thresholds improved **precision** but reduced **recall**.
- The **97th percentile threshold** provided the best balance in this experiment.

This shows a common trade-off in anomaly detection:

- **High recall** means most anomalies are detected
- **Low precision** means many false alarms are generated

## Results Summary

The model was evaluated file-by-file across the SKAB test folders.

Example observations:

- Some files achieved relatively strong F1 scores
- Some files remained difficult for Isolation Forest
- Performance was inconsistent across different anomaly patterns

This suggests that Isolation Forest is a useful **unsupervised baseline**, but supervised models such as **XGBoost** or **Random Forest** may provide more stable performance when labeled anomaly data is available.

## How to Run in Google Colab

### 1. Upload the dataset
Upload `SKAB.zip` to Colab and unzip it:

```python
from google.colab import files
uploaded = files.upload()

!unzip -o SKAB.zip
!ls SKAB
```

### 2. Install dependencies if needed

```python
!pip install scikit-learn pandas matplotlib
```

### 3. Run the notebook cells in order
The workflow includes:
- imports
- defining base features
- rolling feature generation
- training on anomaly-free data
- threshold selection
- evaluation on all test files
- plotting and summary tables

## Repository Structure

A suggested structure for this project:

```text
project/
│
├── README.md
├── notebooks/
│   └── isolation_forest_skab.ipynb
├── data/
│   └── SKAB.zip
└── results/
    ├── summary_metrics.csv
    └── plots/
```

## Limitations

- Isolation Forest does not explicitly model temporal dependencies like sequence models do.
- Performance varies across different SKAB files.
- The model may generate false positives depending on threshold choice.
- Some anomaly files are harder to separate from normal behaviour.

## Future Improvements

Possible next steps:

- compare against supervised models such as XGBoost and Random Forest
- reduce noisy features through feature selection
- try different rolling window sizes
- compare row-based vs sequence-based anomaly detection
- test advanced time-series models such as Autoencoders or Transformer-based methods

## Conclusion

This project demonstrates a complete unsupervised anomaly detection pipeline for predictive maintenance using Isolation Forest on the SKAB dataset.

It provides a strong baseline for comparison with supervised approaches and highlights the importance of threshold tuning, feature engineering, and per-file evaluation when working with real-world style anomaly detection tasks.
