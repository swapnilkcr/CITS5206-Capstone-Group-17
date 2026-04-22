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

### Notes

- If artifacts are too large, keep them out of git and store them externally (e.g. cloud drive). Put the download link in the registry under `artifact_uri`.
- Use a **stable** `model_id` (e.g. `lstm_ae_skab_v0_9_julie_001`) so other teammates can reference it reliably.

