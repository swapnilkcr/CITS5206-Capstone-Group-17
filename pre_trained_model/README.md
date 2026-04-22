## `pre_trained_model/`

This folder is a shared place for the team to **register** trained models that run on the **same dataset**.

### What goes where

- **Registry (tracked)**: `pre_trained_model/registry.yaml`
- **Artifacts (usually not tracked)**: `pre_trained_model/artifacts/<member>/<model_id>/...`

### Basic workflow (team)

1. Save your trained model artifacts under `pre_trained_model/artifacts/<your_name>/<model_id>/`
2. Add an entry to `pre_trained_model/registry.yaml`
3. Validate and list models:

```bash
python scripts/model_registry.py validate
python scripts/model_registry.py list
```

<<<<<<< HEAD
### What a good registry entry must include

When adding a new model to the registry, clearly document:

- **Author/owner**
- **Algorithm + framework** (e.g. LSTM + Deep SVDD, PyTorch/TensorFlow)
- **Dataset + split** (what data was used and how it was split)
- **Inputs**
  - window shape \(T \times D\)
  - features used / preprocessing assumptions
  - label usage (unsupervised / semi-supervised / supervised)
- **Outputs**
  - anomaly score definition
  - binary label definition (thresholding)
- **Threshold selection**
  - simplest: choose threshold from **validation normal** scores (e.g. 95% / 99% quantile)
  - if labeled validation anomalies exist: tune threshold for **F1** (or cost-weighted metric)
- **Artifacts**
  - local path under `pre_trained_model/artifacts/...` or an external download link in `artifact_uri`

### Example model write-up (Julie Lei — Adapted-LSTM)

If your model is not ready to be moved into `pre_trained_model/` yet, keep its draft/notes in a top-level folder.

- **Draft folder**: `Adapted-LSTM/`
- **Author**: Julie Lei
- **Summary**: Task-specific LSTM that combines **semi-supervised learning** with **Deep SVDD** to leverage many normal samples and few anomalies.

Core idea:
- LSTM encoder \(f(x)\) learns a temporal representation for a window \(x\in \mathbb{R}^{T\times D}\)
- Learn a center \(c\) so normal windows cluster around \(c\)
- Push labeled anomalies away with a margin loss:

\[
L_{anomaly} = \max\bigl(0,\ m - \lVert f(x_{anomaly}) - c \rVert\bigr)^2
\]

Inference:
- score \(s(x)=\lVert f(x)-c \rVert\)
- if `score > threshold` → anomaly else normal

Optional extension:
- **contrastive regularization** (normal–normal pull, normal–anomaly push)

=======
>>>>>>> origin/Julie-work
### Notes

- If artifacts are too large, keep them out of git and store them externally (e.g. cloud drive). Put the download link in the registry under `artifact_uri`.
- Use a **stable** `model_id` (e.g. `lstm_ae_skab_v0_9_julie_001`) so other teammates can reference it reliably.

