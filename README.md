# CITS5206 Capstone (2026 SEM-1) — Group 17

Anomaly detection for **water pump maintenance** using the **SKAB** dataset ([waico/SKAB](https://github.com/waico/SKAB)), with support for **unseen client CSVs** (e.g. Zenodo NeWater Pump data).

---

## Quick start — run the web UI

```bash
# 1. Clone and enter project root
cd CITS5206-Capstone-Group-17

# 2. Python env (3.10+ recommended)
pip install -r requirements.txt

# macOS only — XGBoost needs OpenMP (one-time)
brew install libomp

# 3. Data + checkpoints (see below)
#    - SKAB CSVs under data/
#    - Model weights under pre_trained_model/artifacts/

# 4. Start server (from repo root)
python app.py
```

Open: **http://127.0.0.1:5001**

Health check: **http://127.0.0.1:5001/api/health**

Sample CSVs: `ui/test_data/valve1_sample.csv`, `ui/test_data/valve2_sample.csv`  
Merged unseen demo: `unseen_data/processed/newater_pump1_merged.csv`

---

## Repository layout

| Path | Role |
|------|------|
| **`app.py`** | Flask entry point (run from **repo root**). Loads checkpoints only — no training on startup. |
| **`ui/`** | UI Python modules + static frontend |
| `ui/model_pipeline.py` | Loads `pre_trained_model/registry.yaml` and artifacts (sklearn before PyTorch on macOS) |
| `ui/data_pipeline.py` | Unseen CSV column mapping → SKAB 8-channel contract |
| `ui/torch_models.py` | LSTM / Transformer / Adapted LSTM inference |
| `ui/constants.py` | Feature columns and hyperparameters |
| `ui/frontend/` | Dashboard HTML, CSS, JS (`index.html`, `docs.html`) |
| `ui/test_data/` | Small sample CSVs for quick tests |
| **`pre_trained_model/`** | Offline training + `registry.yaml` + `artifacts/` |
| **`data/`** | SKAB CSVs (`valve1/`, `valve2/`, `other/`, `anomaly-free/`) |
| **`unseen_data/`** | Client / Zenodo raw downloads and merged wide CSVs |
| `scripts/` | `download_newater_pump1.py`, `prepare_newater_pump1.py`, `model_registry.py` |
| `ipynb_scripts/` | Team notebooks and experiments |

---

## UI features

- **Single CSV** — LOGO-style evaluation for known SKAB files; unseen uploads with optional **8-column mapping**
- **ZIP dataset** — Group K-Fold (supervised) or full-dataset scoring (unsupervised)
- **Unseen dataset** — banner + column picker + **migration risk** popup → [retrain guide](ui/frontend/docs.html#retrain-guide)
- **Algorithms**: XGBoost, Random Forest, Isolation Forest, LSTM AE, Transformer AE, Adapted LSTM

---

## Unseen client data (NeWater Pump 1)

```bash
# Download from Zenodo 13808085 + merge (~350 MB, several minutes)
python scripts/download_newater_pump1.py --also-merge
```

Output: `unseen_data/processed/newater_pump1_merged.csv` — upload in the UI as an unseen dataset.

Details: [`unseen_data/README.md`](unseen_data/README.md)

---

## Training vs running the UI

```
Train (offline)                         Run UI
──────────────                          ──────
notebooks / pre_trained_model/*/train.py   python app.py   (repo root)
        ↓                                       ↓
pre_trained_model/artifacts/               http://127.0.0.1:5001
registry.yaml
```

The UI **does not train on startup**. For known SKAB files with labels, XGBoost/RF may run **LOGO** on predict (evaluation only).

---

## Prerequisites

### SKAB data

```
data/
  anomaly-free/anomaly-free.csv
  valve1/*.csv
  valve2/*.csv
  other/*.csv
```

### Checkpoints

Place weights under `pre_trained_model/artifacts/` per [`pre_trained_model/registry.yaml`](pre_trained_model/registry.yaml).

```bash
python scripts/model_registry.py validate
python scripts/model_registry.py list
```

See [`pre_trained_model/README.md`](pre_trained_model/README.md).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: constants` | Run `python app.py` from **repo root**, not `ui/` |
| Port 5001 in use | `lsof -ti:5001 \| xargs kill -9` |
| XGBoost on macOS | `brew install libomp` |
| Crash after XGBoost + PyTorch | Restart `python app.py` |
| Missing checkpoint | Train offline, copy `.joblib` to `artifacts/`, update registry |

---

## Model registry (team)

1. Save artifacts under `pre_trained_model/artifacts/...`
2. Add entry to `pre_trained_model/registry.yaml`
3. `python scripts/model_registry.py validate`

---

## Dataset notes

- Keep large CSVs and weights out of git (see `.gitignore`).
- Cite Zenodo NeWater data: [10.5281/zenodo.13808085](https://doi.org/10.5281/zenodo.13808085) (CC BY-SA 4.0).
