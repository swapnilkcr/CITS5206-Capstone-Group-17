# CITS5206 Capstone (2026 SEM-1) — Group 17

## Project overview

This repository is for **CITS5206 Capstone (2026 Semester 1)**, **Group 17**.

- **Project**: Implementation of anomaly detection for **water pump maintenance**
- **Dataset**: **SKAB (Skoltech Anomaly Benchmark)** — see [waico/SKAB](https://github.com/waico/SKAB)

## Dataset storage (important)

We keep datasets under **`data/`** so that project code stays clean and datasets can be swapped/updated without mixing with source code.

- **Current dataset**: SKAB is located at `external/SKAB/`
- **Rule for future datasets**: put them under `external/<DATASET_NAME>/` (and add to `.gitignore` if needed)

If you are working on a GPU server, you can either:
- clone datasets directly on the server under the project’s `external/`, or
- sync them from your local machine (but avoid committing large CSVs/weights to git).

## Repository structure

- **`ipynb_scripts/`**
  - Team members’ Jupyter notebooks (experiments, EDA, training runs, etc.)
  - Keep notebooks here (do not leave notebooks in the repo root)
- **`pre_trained_model/`**
  - Shared location for team-trained models and weights
  - Includes a **registry** so everyone can discover and reuse models consistently
  - Registry file: `pre_trained_model/registry.yaml`
  - Helper script: `scripts/model_registry.py`
- **`data/`**
  - External datasets and third-party checkouts
  - Current: `data/SKAB/`
- **`scripts/`**
  - Small helper scripts (e.g., model registry tooling)

## Model registry (team workflow)

When you train a new model and want others to reuse it:

1. Put artifacts (weights, configs, etc.) under:
   - `pre_trained_model/artifacts/<your_name>/<model_id>/...`
2. Register the model in:
   - `pre_trained_model/registry.yaml`

The registry entry should clearly state:
- **author/owner**
- **algorithm/framework**
- **expected input** (features, windowing, preprocessing assumptions)
- **output** (score/probability/label, thresholds if applicable)
- **metrics** (F1, FAR/MAR, NAB, etc. as relevant)
- **artifact location** (`artifact_uri` local path or external download link)

You can validate/list the registry with:

```bash
python scripts/model_registry.py validate
python scripts/model_registry.py list
```

