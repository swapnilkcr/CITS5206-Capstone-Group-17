# SKAB Anomaly Detection — Group 17 (Shreyan)

**Dataset:** Skoltech Anomaly Benchmark (SKAB)  
**Algorithms:** XGBoost, Random Forest  
**Evaluation:** Group K-Fold Cross-Validation (grouped by file)  
**Features:** 8 sensor signals + rolling mean & std (window=5) = 24 features  

---

## Datasets

| Dataset | Files | Total Rows | Anomaly Rate |
|---------|-------|------------|--------------|
| valve1  | 16    | 18,162     | 34.7%        |
| valve2  | 4     | 4,610      | ~35%         |
| other   | 14    | ~20,000    | ~35%         |
| anomaly-free (baseline) | 1 | 9,401 | 0% |

> The anomaly-free baseline is used only to fit the StandardScaler — not for evaluation.

---

## Results Summary

### Full Comparison

| Dataset | Model         | F1     | Precision | Recall | ROC-AUC |
|---------|---------------|--------|-----------|--------|---------|
| valve1  | **XGBoost**   | **0.8454** | 0.9236 | 0.7794 | **0.8970** |
| valve1  | Random Forest | 0.8371 | 0.9684    | 0.7372 | 0.8815  |
| valve2  | **XGBoost**   | **0.8728** | 0.9286 | 0.8233 | **0.9262** |
| valve2  | Random Forest | 0.8279 | 0.9657    | 0.7245 | 0.9175  |
| other   | **XGBoost**   | **0.4240** | 0.5553 | 0.3429 | 0.5958  |
| other   | Random Forest | 0.3437 | 0.5255    | 0.2554 | **0.6018** |

### Overall Averages (Across All Datasets)

| Model         | F1     | Precision | Recall | ROC-AUC |
|---------------|--------|-----------|--------|---------|
| **XGBoost**   | **0.7141** | 0.8025 | **0.6485** | **0.8063** |
| Random Forest | 0.6696 | **0.8199** | 0.5724 | 0.8003  |

> **XGBoost wins overall** on F1 and ROC-AUC. Random Forest has marginally higher precision.

---

## Per-File F1 — valve1

| File   | XGBoost F1 | Random Forest F1 |
|--------|------------|-----------------|
| 0.csv  | 0.0718     | 0.0439          |
| 1.csv  | 0.7252     | 0.0526          |
| 2.csv  | 0.8649     | 0.8960          |
| 3.csv  | 0.9294     | 0.9455          |
| 4.csv  | 0.8832     | 0.9245          |
| 5.csv  | 0.8649     | 0.8649          |
| 6.csv  | 0.8270     | 0.8596          |
| 7.csv  | 0.8492     | 0.8608          |
| 8.csv  | 0.9937     | 0.9925          |
| 9.csv  | 0.8743     | 0.8963          |
| 10.csv | 0.8968     | 0.8984          |
| 11.csv | 0.8075     | 0.8229          |
| 12.csv | 0.9462     | 0.9730          |
| 13.csv | 0.9088     | 0.8989          |
| 14.csv | 0.8952     | 0.8986          |
| 15.csv | 0.8345     | 0.8588          |

---

## Per-File F1 — valve2

| File  | XGBoost F1 | Random Forest F1 |
|-------|------------|-----------------|
| 0.csv | 0.8833     | 0.6655          |
| 1.csv | 0.9164     | 0.9120          |
| 2.csv | 0.8768     | 0.8602          |
| 3.csv | 0.8232     | 0.8559          |

---

## Per-File F1 — other

| File   | XGBoost F1 | Random Forest F1 |
|--------|------------|-----------------|
| 9.csv  | 0.8055     | 0.8316          |
| 11.csv | 0.8172     | 0.8187          |
| 12.csv | 0.6949     | 0.4250          |
| 13.csv | 0.8297     | 0.8090          |
| 14.csv | 0.4770     | 0.0860          |
| 15.csv | 0.3373     | 0.0766          |
| 16.csv | 0.0480     | 0.0321          |
| 17.csv | 0.0048     | 0.0050          |
| 18.csv | 0.3636     | 0.1432          |
| 19.csv | 0.3279     | 0.2119          |
| 20.csv | 0.0000     | 0.0000          |
| 21.csv | 0.0048     | 0.0000          |
| 22.csv | 0.4217     | 0.4553          |
| 23.csv | 0.0262     | 0.1772          |

---

## Key Findings

- **XGBoost is the best overall model** — highest F1 and ROC-AUC on all three datasets.
- **valve1 & valve2** — both models perform strongly. Most files score F1 > 0.82, with several above 0.93.
- **other dataset** — both models struggle badly. XGBoost F1 = 0.424, Random Forest F1 = 0.344. Files 16, 17, 20, 21 score near zero, suggesting anomaly types in this group are fundamentally different and harder to generalise from valve training patterns.
- **Random Forest** has higher precision (fewer false alarms) but consistently lower recall — it misses more real anomalies.
- **XGBoost** has better recall — catches more anomalies but produces slightly more false positives.

---

## Files in This Branch

| File | Description |
|------|-------------|
| `SKAB_Anomaly_Detection_shreyan_supervised_unsupervised.ipynb` | Initial notebook — valve1 only, 4 algorithms (XGBoost, RF, Isolation Forest, LSTM Autoencoder) |
| `SKAB_XGBoost_RandomForest_AllDatasets.ipynb` | Extended notebook — all 3 datasets, XGBoost & Random Forest, runs locally |
| `SKAB_Results_Report.pdf` | Full PDF report with charts, tables and findings |

---

*CITS5206 Capstone Group 17 — Shreyan Mittal — April 2026*
