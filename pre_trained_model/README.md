# `pre_trained_model/` — training & checkpoints

Offline training lives here. The **web UI** (`../app.py` at repo root) only **loads** files from `artifacts/` via `registry.yaml` — it never trains when you run `python app.py`.

---

## Folder map

| Folder / file | Who / what |
|---------------|------------|
| `registry.yaml` | Master list of all deployable models (paths, owners, metrics). |
| `artifacts/sklearn/` | Scaler, XGBoost, Random Forest, Isolation Forest (`.joblib`). |
| `artifacts/lstm/` | LSTM autoencoder UI bundle. |
| `artifacts/transformer/` | Transformer autoencoder UI bundle. |
| `artifacts/adapted_lstm/` | Adapted LSTM (ALSS-SVDD) UI bundle. |
| `Adapted-LSTM/` | Training code (`train.py`), dataset splits, optional `outputs/*/best_model.pt`. |
| `Transformer/` | Training code; native save path `models/best_model.pth` + `scaler.pkl`. |
| `LSTM/` | LSTM autoencoder experiment script. |
| `ipynb_scripts/` | XGBoost / Random Forest notebooks (Shreyan). |

**Native training outputs** (e.g. `Transformer/models/best_model.pth`) are not read by the UI directly. Export to the `.joblib` paths in `registry.yaml`, or ask the UI maintainer to add a loader.

---

## Train → register → run UI

```bash
# 1. Train (example — Adapted LSTM)
cd pre_trained_model/Adapted-LSTM
python train.py

# 2. Export / copy weights into artifacts/ (paths must match registry.yaml)
#    e.g. cp ... artifacts/adapted_lstm/adapted_lstm.joblib

# 3. Register in registry.yaml (or use scripts/model_registry.py add)

# 4. Validate
cd ../..   # project root
python scripts/model_registry.py validate
python scripts/model_registry.py list

# 5. Start UI
python app.py
# → http://127.0.0.1:5001
```

---

## `registry.yaml` fields

Each model entry should include:

- `model_id` — stable id (e.g. `lstm_ae_skab_v1`)
- `owner` — who trained it
- `framework` — `sklearn`, `xgboost`, `pytorch`, …
- `artifact_uri` — path relative to repo root (e.g. `pre_trained_model/artifacts/lstm/lstm_autoencoder.joblib`)
- `dataset` — what data / split was used
- `metrics` — F1, etc. if available
- `notes` — input features, window size, threshold strategy

---

## macOS note

The UI loads **sklearn/XGBoost joblib files before PyTorch** to avoid crashes. You do not need to change anything when training here; this only affects `ui/model_pipeline.py` at runtime.

---

More on the full project: [../README.md](../README.md).
