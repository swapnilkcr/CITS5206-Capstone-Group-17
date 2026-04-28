## Adapted-LSTM (task-specific)

- **Author**: Julie Lei
- **Model type**: Task-specific LSTM encoder with Semi-Supervised Deep SVDD (+ optional contrastive regularization)

### High-level logic

```mermaid
flowchart TD
    A[Input Time Series Window<br/>(T × D)] --> B[LSTM Encoder]
    B --> C[Embedding z]
    C --> D[SVDD Loss: cluster normal]
    C --> E[Margin Loss: push anomaly]
    C --> F[Contrastive: structure space]
    D --> G[Anomaly score]
    E --> G
    F --> G
```

### Idea (Semi-supervised LSTM + Deep SVDD)

The model will learn a compact **LSTM-based representation** where **normal** windows cluster around a learned **center** \(c\), while **labeled anomalies** are pushed away using a margin constraint.

#### What the LSTM does

- Extract **temporal patterns**
- Compress each time-series window into a **latent representation** \(f(x)\)

### Loss design (core)

- **Normal objective (SVDD-style)**: pull normal representations toward the center \(c\)
- **Anomaly margin objective (semi-supervised)**:

\[
L_{anomaly} = \max\bigl(0,\ m - \lVert f(x_{anomaly}) - c \rVert\bigr)^2
\]

Meaning:
- Anomalies are **not allowed to be close** to the center
- If an anomaly is within the margin \(m\), we apply a penalty
- If it is already farther than \(m\), no penalty is applied

### Inference

- **Anomaly score**: \(s(x)=\lVert f(x)-c \rVert\) (larger = more anomalous)
- **Decision rule**:
  - if `score > threshold` → anomaly
  - else → normal

### Threshold selection (how to choose)

Simple and robust approach:

- Use a **validation set of normal windows**
- Compute scores on that normal-only validation set, then set `threshold` as a high quantile, e.g.:
  - 95th percentile (more sensitive)
  - 99th percentile (fewer false alarms)

If you have labeled anomalies in validation, you can tune the threshold to maximize **F1** (or minimize false alarms, depending on requirements).

### Inputs / Outputs

#### Input

- **X**: time-series window of shape **\(T \times D\)**
- **label**: optional (`normal` / `anomaly`)

#### Output

Training produces:
- **learned encoder** (LSTM-based)
- **center** \(c\)

Inference produces:
- **anomaly score**
- **binary label**

## Code layout

- `config.py`: hyperparameters (window size, latent dim, margin, weights)
- `skab_dataset.py`: SKAB window dataset helper (CSV read/concat/windowing)
- `prepare_dataset.py`: build cleaned train/val/test CSVs under `Adapted-LSTM/dataset/` (does not modify `data/`)
- `model.py`: LSTM encoder + wrapper module
- `loss.py`: SVDD + anomaly margin losses
- `utils.py`: threshold helper + contrastive loss (optional)
- `train.py`: training loop sketch (center init + loss composition)
- `inference.py`: score + thresholding helper

## Environment (one-click install)

Minimal dependencies are listed in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Sync from the current server environment (ssm)

If you want to snapshot whatever is currently installed in your active server environment:

```bash
./sync_env_from_ssm.sh
```

This writes `requirements.lock.txt` (equivalent to `pip freeze`) so teammates can reproduce the same environment.

## Dataset preparation (recommended workflow)

We keep raw CSVs under `data/` untouched. To generate cleaned and time-wise splits for training:

```bash
python Adapted-LSTM/prepare_dataset.py
```

Outputs are written to:

- `Adapted-LSTM/dataset/` (CSV + `manifest.json`)

## Visualize a training run

Each `train.py` run writes JSON histories under `Adapted-LSTM/outputs/<run_id>/`.

To export CSVs (and PNGs if matplotlib is installed):

```bash
python Adapted-LSTM/visualize_run.py --run-dir Adapted-LSTM/outputs/<run_id>
```

### Why this model (advantages)

- **vs Autoencoder (AE)**:
  - does not rely on reconstruction
  - latent space tends to be more stable for scoring

- **vs pure SVDD (fully unsupervised)**:
  - uses anomaly labels when available (semi-supervised)

- **vs Transformer baselines**:
  - typically less sensitive to hyperparameters
  - often more stable on smaller/limited compute regimes

### Optional extension: contrastive regularization

Contrastive learning can be added to shape the latent space:

- normal–normal → **pull together**
- normal–anomaly → **push apart**

