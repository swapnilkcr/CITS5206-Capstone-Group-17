#!/usr/bin/env python3
"""
Pump Failure Detection — Flask backend
- Single file: LOGO evaluation (train on all other SKAB files, test on uploaded)
- ZIP dataset: Group K-Fold (exactly mirrors the notebook evaluation)
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR.parent / "data"
MODELS_DIR = BASE_DIR / "models_cache"
MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Feature config (mirrors Shreyan's notebook exactly)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "Accelerometer1RMS", "Accelerometer2RMS", "Current",
    "Pressure", "Temperature", "Thermocouple",
    "Voltage", "Volume Flow RateRMS",
]
EXTENDED_COLS = (
    FEATURE_COLS
    + [f"{c}_roll_mean" for c in FEATURE_COLS]
    + [f"{c}_roll_std"  for c in FEATURE_COLS]
)

LSTM_SEQ_LEN        = 30
LSTM_LATENT         = 16

TRANSFORMER_SEQ_LEN  = 30
TRANSFORMER_D_MODEL  = 32
TRANSFORMER_NHEAD    = 2
TRANSFORMER_LAYERS   = 1
TRANSFORMER_FFN_DIM  = 64


# ---------------------------------------------------------------------------
# LSTM Autoencoder (Thant's architecture, rewritten in PyTorch)
# ---------------------------------------------------------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, latent_dim: int = LSTM_LATENT):
        super().__init__()
        self.encoder1 = nn.LSTM(n_features, 64,         batch_first=True)
        self.encoder2 = nn.LSTM(64,         latent_dim, batch_first=True)
        self.decoder1 = nn.LSTM(latent_dim, latent_dim, batch_first=True)
        self.decoder2 = nn.LSTM(latent_dim, 64,         batch_first=True)
        self.out      = nn.Linear(64, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _  = self.encoder1(x)
        _, (h, _) = self.encoder2(out)
        h = h.squeeze(0).unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder1(h)
        out, _ = self.decoder2(out)
        return self.out(out)


def _make_sequences(data: np.ndarray, seq_len: int) -> np.ndarray:
    return np.stack([data[i:i + seq_len] for i in range(len(data) - seq_len + 1)]).astype(np.float32)


def _recon_errors(model: LSTMAutoencoder, seqs: np.ndarray, batch: int = 128) -> np.ndarray:
    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            x    = torch.tensor(seqs[i:i + batch])
            recon = model(x)
            err  = ((recon - x) ** 2).mean(dim=(1, 2)).numpy()
            errors.extend(err)
    return np.array(errors)


def train_lstm_autoencoder() -> tuple:
    print("Training LSTM Autoencoder on anomaly-free baseline…")
    baseline_df   = read_skab_csv(DATA_DIR / "anomaly-free" / "anomaly-free.csv")
    baseline_feat = add_rolling_features(baseline_df)
    X_base = _scaler.transform(baseline_feat[EXTENDED_COLS]).astype(np.float32)
    seqs   = _make_sequences(X_base, LSTM_SEQ_LEN)

    n_features = seqs.shape[2]
    model      = LSTMAutoencoder(n_features, LSTM_LATENT)
    optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion  = nn.MSELoss()

    split     = int(len(seqs) * 0.8)
    X_tr      = torch.tensor(seqs[:split])
    X_val     = torch.tensor(seqs[split:])
    best_val  = float("inf")
    patience  = 5
    wait      = 0

    model.train()
    for epoch in range(30):
        perm   = torch.randperm(len(X_tr))
        ep_loss = 0.0
        for i in range(0, len(X_tr), 32):
            batch = X_tr[perm[i:i + 32]]
            recon = model(batch)
            loss  = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()

        with torch.no_grad():
            val_loss = criterion(model(X_val), X_val).item()

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    threshold = float(np.percentile(_recon_errors(model, seqs), 95))
    joblib.dump({"state_dict": best_state, "n_features": n_features, "threshold": threshold},
                MODELS_DIR / "lstm_autoencoder.joblib")
    print(f"  LSTM Autoencoder trained. Threshold: {threshold:.6f}")
    return model, threshold


def load_lstm_autoencoder() -> tuple:
    data      = joblib.load(MODELS_DIR / "lstm_autoencoder.joblib")
    model     = LSTMAutoencoder(data["n_features"], LSTM_LATENT)
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model, data["threshold"]


def lstm_predict(X_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (preds, probas, seq_indices) where seq_indices[i] is the row in the
    original DataFrame that sequence i corresponds to (last timestep of window).
    First LSTM_SEQ_LEN-1 rows have no prediction and are filled as normal (0).
    """
    n_rows = len(X_scaled)
    if n_rows < LSTM_SEQ_LEN:
        return np.zeros(n_rows, int), np.zeros(n_rows), np.arange(n_rows)

    seqs   = _make_sequences(X_scaled.astype(np.float32), LSTM_SEQ_LEN)
    errors = _recon_errors(_lstm_model, seqs)

    # normalise errors to [0,1] as a proxy probability
    e_min, e_max = errors.min(), errors.max()
    probas_seq   = (errors - e_min) / (e_max - e_min + 1e-9)
    preds_seq    = (errors > _lstm_threshold).astype(int)

    # pad the first seq_len-1 rows as normal
    pad        = LSTM_SEQ_LEN - 1
    full_preds = np.zeros(n_rows, dtype=int)
    full_probas = np.zeros(n_rows)
    full_preds[pad:]  = preds_seq
    full_probas[pad:] = probas_seq
    return full_preds, full_probas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def add_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.copy()
    for col in FEATURE_COLS:
        if col in df.columns:
            df[f"{col}_roll_mean"] = df[col].rolling(window, min_periods=1).mean()
            df[f"{col}_roll_std"]  = df[col].rolling(window, min_periods=1).std().fillna(0)
    return df


def read_skab_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", parse_dates=["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def parse_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    for sep in (";", ",", "\t"):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=sep)
            if len(df.columns) > 3:
                return df
        except Exception:
            continue
    raise ValueError("Could not parse CSV — try semicolon or comma separated.")


def file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def make_model(model_choice: str, y_train: np.ndarray):
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos = neg / pos if pos > 0 else 1.0
    if model_choice == "xgboost":
        return XGBClassifier(
            n_estimators=150, max_depth=6,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
    return RandomForestClassifier(
        n_estimators=150, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )


def compute_metrics(y_true: np.ndarray, preds: np.ndarray, probas: np.ndarray) -> dict:
    unique = np.unique(y_true)
    return {
        "f1":        round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "roc_auc":   (
            round(float(roc_auc_score(y_true, probas)), 4)
            if len(unique) > 1 else None
        ),
    }


# ---------------------------------------------------------------------------
# Index all SKAB files by MD5 hash at startup
# ---------------------------------------------------------------------------
def build_skab_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for dataset in ("valve1", "valve2", "other"):
        d = DATA_DIR / dataset
        if d.exists():
            for f in sorted(d.glob("*.csv")):
                index[file_hash(f.read_bytes())] = f
    print(f"  SKAB index: {len(index)} files indexed.")
    return index


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------
def train_isolation_forest():
    print("Training Isolation Forest on anomaly-free baseline…")
    baseline_df   = read_skab_csv(DATA_DIR / "anomaly-free" / "anomaly-free.csv")
    baseline_feat = add_rolling_features(baseline_df)
    X_train = _scaler.transform(baseline_feat[EXTENDED_COLS])
    clf = IsolationForest(
        n_estimators=300, contamination=0.01,
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_train)
    scores_train = -clf.decision_function(X_train)
    threshold    = float(np.percentile(scores_train, 99))
    joblib.dump({"model": clf, "threshold": threshold}, MODELS_DIR / "isolation_forest.joblib")
    print(f"  Isolation Forest trained. Threshold: {threshold:.4f}")
    return clf, threshold


def if_predict(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = -_if_model.decision_function(X)
    preds  = (scores > _if_threshold).astype(int)
    s_min, s_max = scores.min(), scores.max()
    probas = (scores - s_min) / (s_max - s_min + 1e-9)
    return preds, probas


def run_isolation_forest_eval(combined: pd.DataFrame) -> dict:
    feat_df = add_rolling_features(combined)
    X       = _scaler.transform(feat_df[EXTENDED_COLS])
    y       = feat_df["anomaly"].values
    groups  = feat_df["source_file"].values

    preds, probas = if_predict(X)
    overall = compute_metrics(y, preds, probas)

    per_file = []
    for fname in sorted(np.unique(groups)):
        mask = groups == fname
        m    = compute_metrics(y[mask], preds[mask], probas[mask])
        per_file.append({
            "file":         fname,
            "rows":         int(mask.sum()),
            "anomaly_rate": round(float(y[mask].mean()) * 100, 1),
            "f1":           m["f1"],
            "precision":    m["precision"],
            "recall":       m["recall"],
            "roc_auc":      m["roc_auc"],
        })

    cm   = confusion_matrix(y, preds).tolist()
    step = max(1, len(probas) // 1000)
    timeline = [
        {"i": int(i), "prob": round(float(probas[i]), 4), "pred": int(preds[i])}
        for i in range(0, len(probas), step)
    ]

    return {
        "overall":       overall,
        "fold_f1s":      [],
        "per_file":      per_file,
        "confusion":     cm,
        "timeline":      timeline,
        "n_folds":       0,
        "total_rows":    len(y),
        "anomaly_count": int(preds.sum()),
        "normal_count":  int((preds == 0).sum()),
        "anomaly_rate":  round(float(preds.mean()) * 100, 1),
    }


def run_lstm_eval(combined: pd.DataFrame) -> dict:
    feat_df = add_rolling_features(combined)
    feat_df = feat_df.dropna(subset=["anomaly"] + EXTENDED_COLS)
    feat_df["anomaly"] = feat_df["anomaly"].astype(int)
    X      = _scaler.transform(feat_df[EXTENDED_COLS])
    y      = feat_df["anomaly"].values
    groups = feat_df["source_file"].values

    preds, probas = lstm_predict(X)
    overall = compute_metrics(y, preds, probas)

    per_file = []
    for fname in sorted(np.unique(groups)):
        mask = groups == fname
        m    = compute_metrics(y[mask], preds[mask], probas[mask])
        per_file.append({
            "file":         fname,
            "rows":         int(mask.sum()),
            "anomaly_rate": round(float(y[mask].mean()) * 100, 1),
            "f1":           m["f1"],
            "precision":    m["precision"],
            "recall":       m["recall"],
            "roc_auc":      m["roc_auc"],
        })

    cm   = confusion_matrix(y, preds).tolist()
    step = max(1, len(probas) // 1000)
    timeline = [
        {"i": int(i), "prob": round(float(probas[i]), 4), "pred": int(preds[i])}
        for i in range(0, len(probas), step)
    ]

    return {
        "overall":       overall,
        "fold_f1s":      [],
        "per_file":      per_file,
        "confusion":     cm,
        "timeline":      timeline,
        "n_folds":       0,
        "total_rows":    len(y),
        "anomaly_count": int(preds.sum()),
        "normal_count":  int((preds == 0).sum()),
        "anomaly_rate":  round(float(preds.mean()) * 100, 1),
    }


# ---------------------------------------------------------------------------
# Transformer Autoencoder (Zeyang's architecture, integrated for UI)
# ---------------------------------------------------------------------------
class TransformerAE(nn.Module):
    def __init__(self, input_dim: int = 8):
        super().__init__()
        self.embedding = nn.Linear(input_dim, TRANSFORMER_D_MODEL)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=TRANSFORMER_D_MODEL, nhead=TRANSFORMER_NHEAD,
            dim_feedforward=TRANSFORMER_FFN_DIM, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=TRANSFORMER_LAYERS)
        self.decoder     = nn.Linear(TRANSFORMER_D_MODEL, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.transformer(x)
        return self.decoder(x)


def _transformer_recon_errors(model: TransformerAE, seqs: np.ndarray, batch: int = 128) -> np.ndarray:
    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            x     = torch.tensor(seqs[i:i + batch])
            recon = model(x)
            err   = ((x - recon) ** 2).mean(dim=(1, 2)).numpy()
            errors.extend(err)
    return np.array(errors)


def train_transformer() -> tuple:
    print("Training Transformer Autoencoder on anomaly-free baseline…")
    baseline_df = read_skab_csv(DATA_DIR / "anomaly-free" / "anomaly-free.csv")

    # Transformer uses raw 8 features with its own scaler (no rolling features)
    raw = baseline_df[FEATURE_COLS].values.astype(np.float32)
    t_scaler = StandardScaler()
    raw_scaled = t_scaler.fit_transform(raw)

    seqs = _make_sequences(raw_scaled, TRANSFORMER_SEQ_LEN)

    model     = TransformerAE(input_dim=len(FEATURE_COLS))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    split   = int(len(seqs) * 0.8)
    X_tr    = torch.tensor(seqs[:split])
    X_val   = torch.tensor(seqs[split:])
    best_val  = float("inf")
    patience  = 5
    wait      = 0
    best_state = None

    model.train()
    for epoch in range(50):
        perm    = torch.randperm(len(X_tr))
        ep_loss = 0.0
        for i in range(0, len(X_tr), 32):
            batch = X_tr[perm[i:i + 32]]
            loss  = criterion(model(batch), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()

        with torch.no_grad():
            val_loss = criterion(model(X_val), X_val).item()

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    threshold = float(np.percentile(_transformer_recon_errors(model, seqs), 95))
    joblib.dump(
        {"state_dict": best_state, "scaler": t_scaler, "threshold": threshold},
        MODELS_DIR / "transformer.joblib",
    )
    print(f"  Transformer Autoencoder trained. Threshold: {threshold:.6f}")
    return model, t_scaler, threshold


def load_transformer() -> tuple:
    data      = joblib.load(MODELS_DIR / "transformer.joblib")
    model     = TransformerAE(input_dim=len(FEATURE_COLS))
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model, data["scaler"], data["threshold"]


def transformer_predict(df_raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    raw    = df_raw[FEATURE_COLS].values.astype(np.float32)
    scaled = _transformer_scaler.transform(raw)
    n_rows = len(scaled)

    if n_rows < TRANSFORMER_SEQ_LEN:
        return np.zeros(n_rows, dtype=int), np.zeros(n_rows)

    seqs   = _make_sequences(scaled, TRANSFORMER_SEQ_LEN)
    errors = _transformer_recon_errors(_transformer_model, seqs)

    e_min, e_max = errors.min(), errors.max()
    probas_seq   = (errors - e_min) / (e_max - e_min + 1e-9)
    preds_seq    = (errors > _transformer_threshold).astype(int)

    pad          = TRANSFORMER_SEQ_LEN - 1
    full_preds   = np.zeros(n_rows, dtype=int)
    full_probas  = np.zeros(n_rows)
    full_preds[pad:]  = preds_seq
    full_probas[pad:] = probas_seq
    return full_preds, full_probas


def run_transformer_eval(combined: pd.DataFrame) -> dict:
    combined = combined.copy()
    combined = combined.dropna(subset=["anomaly"] + FEATURE_COLS)
    combined["anomaly"] = combined["anomaly"].astype(int)
    y      = combined["anomaly"].values
    groups = combined["source_file"].values

    preds, probas = transformer_predict(combined)
    overall = compute_metrics(y, preds, probas)

    per_file = []
    for fname in sorted(np.unique(groups)):
        mask = groups == fname
        m    = compute_metrics(y[mask], preds[mask], probas[mask])
        per_file.append({
            "file":         fname,
            "rows":         int(mask.sum()),
            "anomaly_rate": round(float(y[mask].mean()) * 100, 1),
            "f1":           m["f1"],
            "precision":    m["precision"],
            "recall":       m["recall"],
            "roc_auc":      m["roc_auc"],
        })

    cm   = confusion_matrix(y, preds).tolist()
    step = max(1, len(probas) // 1000)
    timeline = [
        {"i": int(i), "prob": round(float(probas[i]), 4), "pred": int(preds[i])}
        for i in range(0, len(probas), step)
    ]

    return {
        "overall":       overall,
        "fold_f1s":      [],
        "per_file":      per_file,
        "confusion":     cm,
        "timeline":      timeline,
        "n_folds":       0,
        "total_rows":    len(y),
        "anomaly_count": int(preds.sum()),
        "normal_count":  int((preds == 0).sum()),
        "anomaly_rate":  round(float(preds.mean()) * 100, 1),
    }


# ---------------------------------------------------------------------------
# Adapted LSTM — ALSS-SVDD (Julie's architecture)
# ---------------------------------------------------------------------------
ALSTM_WINDOW   = 100
ALSTM_STRIDE   = 10
ALSTM_HIDDEN   = 64
ALSTM_LATENT   = 32
ALSTM_EPOCHS   = 20
ALSTM_PATIENCE = 5
ALSTM_MARGIN   = 2.0
ALSTM_DIR      = BASE_DIR.parent / "pre_trained_model" / "Adapted-LSTM" / "dataset"


class _ALSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(len(FEATURE_COLS), ALSTM_HIDDEN, batch_first=True)
        self.fc   = nn.Linear(ALSTM_HIDDEN, ALSTM_LATENT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


def _alstm_windows(data: np.ndarray, labels: np.ndarray | None = None):
    xs, ys = [], []
    for i in range(0, len(data) - ALSTM_WINDOW + 1, ALSTM_STRIDE):
        xs.append(data[i:i + ALSTM_WINDOW])
        if labels is not None:
            ys.append(1 if labels[i:i + ALSTM_WINDOW].max() > 0 else 0)
    X = np.array(xs, dtype=np.float32)
    Y = np.array(ys, dtype=np.int64) if labels is not None else None
    return X, Y


def _alstm_svdd_loss(z, center):
    return ((z - center) ** 2).sum(dim=1).mean()


def _alstm_margin_loss(z, center, margin):
    dist = ((z - center) ** 2).sum(dim=1)
    return torch.relu(margin - dist).pow(2).mean()


def _alstm_contrastive_loss(z, y, temperature=0.2):
    import torch.nn.functional as F
    z   = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature
    lbl = (y.unsqueeze(1) == y.unsqueeze(0)).float()
    exp = torch.exp(sim)
    loss = -torch.log((exp * lbl).sum(dim=1) / (exp.sum(dim=1) + 1e-8))
    return loss.mean()


def _load_alstm_split(fname: str, scaler=None):
    path = ALSTM_DIR / fname
    df   = pd.read_csv(path, sep=";", parse_dates=["datetime"])
    df   = df.sort_values("datetime").reset_index(drop=True)
    df   = df.dropna(subset=FEATURE_COLS + ["anomaly"])
    raw  = df[FEATURE_COLS].values.astype(np.float32)
    lbl  = df["anomaly"].values.astype(np.float32)
    if scaler is None:
        scaler = StandardScaler()
        raw = scaler.fit_transform(raw)
        return raw, lbl, scaler
    return scaler.transform(raw), lbl, scaler


def train_adapted_lstm() -> tuple:
    print("Training Adapted LSTM (ALSS-SVDD) on valve1 split…")

    raw_tr, lbl_tr, a_scaler = _load_alstm_split("valve1_train.csv")
    raw_val, lbl_val, _      = _load_alstm_split("valve1_val.csv", a_scaler)

    X_tr, Y_tr   = _alstm_windows(raw_tr,  lbl_tr)
    X_val, Y_val = _alstm_windows(raw_val, lbl_val)

    encoder = _ALSTMEncoder()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

    # init SVDD center from normal training windows
    with torch.no_grad():
        zs = []
        for i in range(0, len(X_tr), 64):
            xb = torch.tensor(X_tr[i:i + 64])
            yb = torch.tensor(Y_tr[i:i + 64])
            z  = encoder(xb)
            zs.append(z[yb == 0])
        center = torch.cat(zs).mean(dim=0)

    best_f1, best_state, best_thr, patience = -1.0, None, 0.95, 0
    for epoch in range(ALSTM_EPOCHS):
        encoder.train()
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), 64):
            idx = perm[i:i + 64]
            xb  = torch.tensor(X_tr[idx])
            yb  = torch.tensor(Y_tr[idx])
            z   = encoder(xb)
            z_n = z[yb == 0];  z_a = z[yb == 1]
            l1  = _alstm_svdd_loss(z_n, center) if len(z_n) > 0 else torch.tensor(0.0)
            l2  = _alstm_margin_loss(z_a, center, ALSTM_MARGIN) if len(z_a) > 0 else torch.tensor(0.0)
            l3  = _alstm_contrastive_loss(z, yb.float())
            loss = l1 + l2 + 0.5 * l3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # threshold from normal val windows
        encoder.eval()
        with torch.no_grad():
            val_scores, val_y = [], []
            for i in range(0, len(X_val), 64):
                xb = torch.tensor(X_val[i:i + 64])
                yb = torch.tensor(Y_val[i:i + 64])
                z  = encoder(xb)
                s  = ((z - center) ** 2).sum(dim=1)
                val_scores.append(s.numpy());  val_y.append(yb.numpy())
            val_scores = np.concatenate(val_scores)
            val_y      = np.concatenate(val_y)

        thr = float(np.percentile(val_scores[val_y == 0], 95)) if (val_y == 0).any() else float(np.percentile(val_scores, 95))
        preds = (val_scores > thr).astype(int)
        val_f1 = float(f1_score(val_y, preds, zero_division=0))

        if val_f1 > best_f1 + 1e-6:
            best_f1    = val_f1
            best_state = {k: v.clone() for k, v in encoder.state_dict().items()}
            best_thr   = thr
            patience   = 0
        else:
            patience += 1
            if patience >= ALSTM_PATIENCE:
                break

    encoder.load_state_dict(best_state)
    joblib.dump(
        {"state_dict": best_state, "center": center.numpy(), "threshold": best_thr, "scaler": a_scaler},
        MODELS_DIR / "adapted_lstm.joblib",
    )
    print(f"  Adapted LSTM trained. Best val F1: {best_f1:.4f}. Threshold: {best_thr:.6f}")
    return encoder, center, best_thr, a_scaler


def load_adapted_lstm() -> tuple:
    data    = joblib.load(MODELS_DIR / "adapted_lstm.joblib")
    encoder = _ALSTMEncoder()
    encoder.load_state_dict(data["state_dict"])
    encoder.eval()
    center  = torch.tensor(data["center"])
    return encoder, center, data["threshold"], data["scaler"]


def adapted_lstm_predict(df_raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    raw    = df_raw[FEATURE_COLS].values.astype(np.float32)
    scaled = _alstm_scaler.transform(raw)
    n_rows = len(scaled)

    if n_rows < ALSTM_WINDOW:
        return np.zeros(n_rows, dtype=int), np.zeros(n_rows)

    X, _ = _alstm_windows(scaled)
    _alstm_encoder.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            xb = torch.tensor(X[i:i + 64])
            z  = _alstm_encoder(xb)
            s  = ((z - _alstm_center) ** 2).sum(dim=1)
            scores.extend(s.numpy())
    scores = np.array(scores)

    # Spread each window's score to ALL rows it covers (max pooling across overlaps)
    row_scores = np.zeros(n_rows)
    for i, s in enumerate(scores):
        start = i * ALSTM_STRIDE
        end   = min(start + ALSTM_WINDOW, n_rows)
        row_scores[start:end] = np.maximum(row_scores[start:end], s)

    s_min, s_max = row_scores.min(), row_scores.max()
    full_probas  = (row_scores - s_min) / (s_max - s_min + 1e-9)
    full_preds   = (row_scores > _alstm_threshold).astype(int)

    return full_preds, full_probas


def run_adapted_lstm_eval(combined: pd.DataFrame) -> dict:
    combined = combined.copy()
    combined = combined.dropna(subset=["anomaly"] + FEATURE_COLS)
    combined["anomaly"] = combined["anomaly"].astype(int)
    y      = combined["anomaly"].values
    groups = combined["source_file"].values

    preds, probas = adapted_lstm_predict(combined)
    overall = compute_metrics(y, preds, probas)

    per_file = []
    for fname in sorted(np.unique(groups)):
        mask = groups == fname
        m    = compute_metrics(y[mask], preds[mask], probas[mask])
        per_file.append({
            "file":         fname,
            "rows":         int(mask.sum()),
            "anomaly_rate": round(float(y[mask].mean()) * 100, 1),
            "f1":           m["f1"],
            "precision":    m["precision"],
            "recall":       m["recall"],
            "roc_auc":      m["roc_auc"],
        })

    cm   = confusion_matrix(y, preds).tolist()
    step = max(1, len(probas) // 1000)
    timeline = [
        {"i": int(i), "prob": round(float(probas[i]), 4), "pred": int(preds[i])}
        for i in range(0, len(probas), step)
    ]

    return {
        "overall":       overall,
        "fold_f1s":      [],
        "per_file":      per_file,
        "confusion":     cm,
        "timeline":      timeline,
        "n_folds":       0,
        "total_rows":    len(y),
        "anomaly_count": int(preds.sum()),
        "normal_count":  int((preds == 0).sum()),
        "anomaly_rate":  round(float(preds.mean()) * 100, 1),
    }


def fit_or_load_scaler() -> StandardScaler:
    scaler_path = MODELS_DIR / "scaler.joblib"
    if scaler_path.exists():
        return joblib.load(scaler_path)
    baseline_df   = read_skab_csv(DATA_DIR / "anomaly-free" / "anomaly-free.csv")
    baseline_feat = add_rolling_features(baseline_df)
    scaler = StandardScaler()
    scaler.fit(baseline_feat[EXTENDED_COLS])
    joblib.dump(scaler, scaler_path)
    return scaler


# ---------------------------------------------------------------------------
# LOGO model (for single-file evaluation)
# ---------------------------------------------------------------------------
def train_logo_model(model_choice: str, exclude_path: Path):
    train_dfs: list[pd.DataFrame] = []
    for dataset in ("valve1", "valve2", "other"):
        d = DATA_DIR / dataset
        if not d.exists():
            continue
        for csv_file in sorted(d.glob("*.csv")):
            if csv_file.resolve() == exclude_path.resolve():
                continue
            df = read_skab_csv(csv_file)
            if "anomaly" not in df.columns:
                continue
            df["source_file"] = csv_file.name
            train_dfs.append(df)

    combined = pd.concat(train_dfs, ignore_index=True)
    feat_df  = add_rolling_features(combined)
    feat_df  = feat_df.dropna(subset=["anomaly"] + EXTENDED_COLS)
    feat_df["anomaly"] = feat_df["anomaly"].astype(int)
    X = _scaler.transform(feat_df[EXTENDED_COLS])
    y = feat_df["anomaly"].values

    clf = make_model(model_choice, y)
    clf.fit(X, y)
    return clf


# ---------------------------------------------------------------------------
# Group K-Fold (for ZIP dataset evaluation — mirrors notebook exactly)
# ---------------------------------------------------------------------------
def run_group_kfold(
    combined: pd.DataFrame,
    model_choice: str,
    n_splits: int = 5,
) -> dict:
    feat_df = add_rolling_features(combined)
    feat_df = feat_df.dropna(subset=["anomaly"] + EXTENDED_COLS)
    feat_df["anomaly"] = feat_df["anomaly"].astype(int)
    X      = _scaler.transform(feat_df[EXTENDED_COLS])
    y      = feat_df["anomaly"].values
    groups = feat_df["source_file"].values

    unique_groups = np.unique(groups)
    actual_splits = min(n_splits, len(unique_groups))

    all_preds  = np.zeros(len(y))
    all_probas = np.zeros(len(y))
    fold_f1s: list[float] = []

    gkf = GroupKFold(n_splits=actual_splits)
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        clf = make_model(model_choice, y[train_idx])
        clf.fit(X[train_idx], y[train_idx])
        all_preds[test_idx]  = clf.predict(X[test_idx])
        all_probas[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
        fold_f1 = float(f1_score(y[test_idx], all_preds[test_idx], zero_division=0))
        fold_f1s.append(round(fold_f1, 4))

    # Overall metrics
    overall = compute_metrics(y, all_preds, all_probas)

    # Per-file metrics
    per_file = []
    for fname in sorted(unique_groups):
        mask = groups == fname
        m = compute_metrics(y[mask], all_preds[mask], all_probas[mask])
        per_file.append({
            "file":         fname,
            "rows":         int(mask.sum()),
            "anomaly_rate": round(float(y[mask].mean()) * 100, 1),
            "f1":           m["f1"],
            "precision":    m["precision"],
            "recall":       m["recall"],
            "roc_auc":      m["roc_auc"],
        })

    # Confusion matrix
    cm = confusion_matrix(y, all_preds).tolist()

    # Timeline (downsampled to 1000 pts)
    step = max(1, len(all_probas) // 1000)
    timeline = [
        {"i": int(i), "prob": round(float(all_probas[i]), 4), "pred": int(all_preds[i])}
        for i in range(0, len(all_probas), step)
    ]

    return {
        "overall":   overall,
        "fold_f1s":  fold_f1s,
        "per_file":  per_file,
        "confusion": cm,
        "timeline":  timeline,
        "n_folds":   actual_splits,
        "total_rows":    len(y),
        "anomaly_count": int(all_preds.sum()),
        "normal_count":  int((all_preds == 0).sum()),
        "anomaly_rate":  round(float(all_preds.mean()) * 100, 1),
    }


# ---------------------------------------------------------------------------
# Fallback full model (for truly unknown single files)
# ---------------------------------------------------------------------------
def train_and_save_full() -> None:
    print("Training fallback full model…")
    all_dfs: list[pd.DataFrame] = []
    for dataset in ("valve1", "valve2", "other"):
        d = DATA_DIR / dataset
        if d.exists():
            for csv_file in sorted(d.glob("*.csv")):
                df = read_skab_csv(csv_file)
                if "anomaly" not in df.columns:
                    continue
                df["source_file"] = csv_file.name
                all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    feat_df  = add_rolling_features(combined)
    feat_df  = feat_df.dropna(subset=["anomaly"] + EXTENDED_COLS)
    feat_df["anomaly"] = feat_df["anomaly"].astype(int)
    X = _scaler.transform(feat_df[EXTENDED_COLS])
    y = feat_df["anomaly"].values

    for key, clf in [
        ("xgboost",       make_model("xgboost", y)),
        ("random_forest", make_model("random_forest", y)),
    ]:
        clf.fit(X, y)
        joblib.dump(clf, MODELS_DIR / f"{key}_full.joblib")
    print("  Fallback models saved.")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
print("=" * 60)
print("Pump Failure Detection — starting up…")
_scaler      = fit_or_load_scaler()
_skab_index  = build_skab_index()

if not (MODELS_DIR / "xgboost_full.joblib").exists():
    train_and_save_full()

_full_models = {
    "xgboost":       joblib.load(MODELS_DIR / "xgboost_full.joblib"),
    "random_forest": joblib.load(MODELS_DIR / "random_forest_full.joblib"),
}

_if_path = MODELS_DIR / "isolation_forest.joblib"
if _if_path.exists():
    _if_data      = joblib.load(_if_path)
    _if_model     = _if_data["model"]
    _if_threshold = _if_data["threshold"]
    print(f"  Isolation Forest loaded. Threshold: {_if_threshold:.4f}")
else:
    _if_model, _if_threshold = train_isolation_forest()

_lstm_path = MODELS_DIR / "lstm_autoencoder.joblib"
if _lstm_path.exists():
    _lstm_model, _lstm_threshold = load_lstm_autoencoder()
    print(f"  LSTM Autoencoder loaded. Threshold: {_lstm_threshold:.6f}")
else:
    _lstm_model, _lstm_threshold = train_lstm_autoencoder()

_transformer_path = MODELS_DIR / "transformer.joblib"
if _transformer_path.exists():
    _transformer_model, _transformer_scaler, _transformer_threshold = load_transformer()
    print(f"  Transformer loaded. Threshold: {_transformer_threshold:.6f}")
else:
    _transformer_model, _transformer_scaler, _transformer_threshold = train_transformer()

_alstm_path = MODELS_DIR / "adapted_lstm.joblib"
if _alstm_path.exists():
    _alstm_encoder, _alstm_center, _alstm_threshold, _alstm_scaler = load_adapted_lstm()
    print(f"  Adapted LSTM loaded. Threshold: {_alstm_threshold:.6f}")
else:
    _alstm_encoder, _alstm_center, _alstm_threshold, _alstm_scaler = train_adapted_lstm()

print("Ready.")
print("=" * 60)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)


@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "skab_files": len(_skab_index)})


# ---------------------------------------------------------------------------
# Single-file prediction (LOGO)
# ---------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    model_choice = request.form.get("model", "xgboost")
    valid_models = {*_full_models, "isolation_forest", "lstm_autoencoder", "transformer", "adapted_lstm"}
    if model_choice not in valid_models:
        return jsonify({"error": f"Unknown model: {model_choice}"}), 400

    raw_bytes    = request.files["file"].read()
    filename     = request.files["file"].filename or "uploaded.csv"
    uploaded_md5 = file_hash(raw_bytes)
    matched_path = _skab_index.get(uploaded_md5)
    is_known     = matched_path is not None

    try:
        df = parse_uploaded_csv(raw_bytes)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        return jsonify({"error": f"Missing required columns: {missing}"}), 400

    has_labels = "anomaly" in df.columns
    if model_choice == "isolation_forest":
        model_label = "Isolation Forest"
    elif model_choice == "xgboost":
        model_label = "XGBoost"
    elif model_choice == "lstm_autoencoder":
        model_label = "LSTM Autoencoder"
    elif model_choice == "transformer":
        model_label = "Transformer Autoencoder"
    elif model_choice == "adapted_lstm":
        model_label = "Adapted LSTM (ALSS-SVDD)"
    else:
        model_label = "Random Forest"

    step1 = {
        "id": 1, "name": "File Loaded", "icon": "📁",
        "detail": (
            f"<strong>{len(df):,} rows</strong> · "
            f"<strong>{len(FEATURE_COLS)} sensor columns</strong> detected. "
            + (f"Recognised as <strong>{matched_path.parent.name}/{matched_path.name}</strong>."
               if is_known else
               ("Ground-truth labels found." if has_labels else "No labels — anomaly scores only."))
        ),
    }

    feat_df = add_rolling_features(df)
    step2 = {
        "id": 2, "name": "Feature Engineering", "icon": "⚙️",
        "detail": (
            f"Computed <strong>5-point rolling mean &amp; std</strong> for each of the "
            f"{len(FEATURE_COLS)} sensors → <strong>{len(EXTENDED_COLS)} features</strong> total."
        ),
    }

    try:
        X = _scaler.transform(feat_df[EXTENDED_COLS])
    except Exception as exc:
        return jsonify({"error": f"Scaling failed: {exc}"}), 500

    step3 = {
        "id": 3, "name": "Normalisation", "icon": "⚖️",
        "detail": (
            "All sensor readings scaled to a common range using the pump's "
            "<strong>anomaly-free baseline</strong>."
        ),
    }

    if model_choice == "isolation_forest":
        step4 = {
            "id": 4, "name": "Isolation Forest Scoring", "icon": "🌲",
            "detail": (
                "Unsupervised anomaly detection — trained only on normal pump behaviour. "
                f"Each of the {len(df):,} readings scored; points above the 99th-percentile "
                "threshold are flagged as anomalies."
            ),
        }
        preds, probas = if_predict(X)
    elif model_choice == "lstm_autoencoder":
        step4 = {
            "id": 4, "name": "LSTM Autoencoder Scoring", "icon": "🧠",
            "detail": (
                "Unsupervised detection — LSTM encodes then reconstructs each "
                f"{LSTM_SEQ_LEN}-timestep window; high reconstruction error flags anomalies. "
                f"First {LSTM_SEQ_LEN - 1} rows have no window yet and are marked normal."
            ),
        }
        preds, probas = lstm_predict(X)
    elif model_choice == "transformer":
        step4 = {
            "id": 4, "name": "Transformer Autoencoder Scoring", "icon": "⚡",
            "detail": (
                "Unsupervised detection — Transformer encodes then reconstructs each "
                f"{TRANSFORMER_SEQ_LEN}-timestep window; high reconstruction error flags anomalies. "
                f"First {TRANSFORMER_SEQ_LEN - 1} rows have no window yet and are marked normal."
            ),
        }
        preds, probas = transformer_predict(df)
    elif model_choice == "adapted_lstm":
        step4 = {
            "id": 4, "name": "Adapted LSTM (ALSS-SVDD) Scoring", "icon": "🔬",
            "detail": (
                "Semi-supervised deep SVDD — LSTM encoder maps each "
                f"{ALSTM_WINDOW}-timestep window to a latent space; "
                "distance from the learned normal centre flags anomalies."
            ),
        }
        preds, probas = adapted_lstm_predict(df)
    elif is_known and has_labels:
        step4 = {
            "id": 4, "name": "Leave-One-Out Training", "icon": "🤖",
            "detail": (
                f"Training <strong>{model_label}</strong> on all SKAB files "
                "<em>except</em> this one — then predicting on this file for honest metrics."
            ),
        }
        clf    = train_logo_model(model_choice, matched_path)
        preds  = clf.predict(X)
        probas = clf.predict_proba(X)[:, 1]
    else:
        step4 = {
            "id": 4, "name": "Model Running", "icon": "🤖",
            "detail": (
                f"<strong>{model_label}</strong> examined each of the {len(df):,} readings "
                "and assigned an anomaly probability score."
            ),
        }
        clf    = _full_models[model_choice]
        preds  = clf.predict(X)
        probas = clf.predict_proba(X)[:, 1]

    anomaly_count = int(preds.sum())
    normal_count  = len(preds) - anomaly_count
    anomaly_rate  = round(float(preds.mean()) * 100, 1)

    metrics: dict  = {}
    confusion_mat: list | None = None

    if has_labels:
        y_true = df["anomaly"].dropna().astype(int).values
        if len(y_true) == len(preds):
            metrics       = compute_metrics(y_true, preds, probas)
            confusion_mat = confusion_matrix(y_true, preds).tolist()

    step5 = {
        "id": 5, "name": "Results Ready", "icon": "📊",
        "detail": (
            f"<strong>{anomaly_count:,}</strong> anomalies out of "
            f"<strong>{len(df):,}</strong> readings ({anomaly_rate}%)."
            + (f" F1: <strong>{metrics['f1']}</strong>." if metrics else "")
        ),
    }

    step_size = max(1, len(probas) // 1000)
    timeline  = [
        {"i": int(i), "prob": round(float(probas[i]), 4), "pred": int(preds[i])}
        for i in range(0, len(probas), step_size)
    ]

    return jsonify({
        "steps":         [step1, step2, step3, step4, step5],
        "metrics":       metrics,
        "has_labels":    has_labels,
        "is_known":      is_known,
        "timeline":      timeline,
        "total_rows":    len(df),
        "anomaly_count": anomaly_count,
        "normal_count":  normal_count,
        "anomaly_rate":  anomaly_rate,
        "model":         model_label,
        "confusion":     confusion_mat,
        "filename":      filename,
    })


# ---------------------------------------------------------------------------
# ZIP dataset evaluation (Group K-Fold — matches notebook exactly)
# ---------------------------------------------------------------------------
@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    model_choice = request.form.get("model", "xgboost")
    valid_models = {*_full_models, "isolation_forest", "lstm_autoencoder", "transformer", "adapted_lstm"}
    if model_choice not in valid_models:
        return jsonify({"error": f"Unknown model: {model_choice}"}), 400

    raw_bytes = request.files["file"].read()
    filename  = request.files["file"].filename or "dataset.zip"

    if not zipfile.is_zipfile(io.BytesIO(raw_bytes)):
        return jsonify({"error": "Uploaded file is not a valid ZIP."}), 400

    # Extract CSVs from zip
    dfs: list[pd.DataFrame] = []
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX")]
        if not csv_names:
            return jsonify({"error": "No CSV files found inside the ZIP."}), 400

        for name in sorted(csv_names):
            try:
                data = zf.read(name)
                df   = parse_uploaded_csv(data)
                df["source_file"] = Path(name).name
                dfs.append(df)
            except Exception:
                continue

    if not dfs:
        return jsonify({"error": "Could not parse any CSVs in the ZIP."}), 400

    combined = pd.concat(dfs, ignore_index=True)

    missing = [c for c in FEATURE_COLS if c not in combined.columns]
    if missing:
        return jsonify({"error": f"Missing required columns: {missing}"}), 400

    if "anomaly" not in combined.columns:
        return jsonify({"error": "No 'anomaly' column found — ZIP evaluation requires labelled data."}), 400

    if model_choice == "isolation_forest":
        model_label = "Isolation Forest"
    elif model_choice == "xgboost":
        model_label = "XGBoost"
    elif model_choice == "lstm_autoencoder":
        model_label = "LSTM Autoencoder"
    elif model_choice == "transformer":
        model_label = "Transformer Autoencoder"
    elif model_choice == "adapted_lstm":
        model_label = "Adapted LSTM (ALSS-SVDD)"
    else:
        model_label = "Random Forest"

    n_files      = combined["source_file"].nunique()
    n_rows       = len(combined)
    anomaly_rate = round(float(combined["anomaly"].mean()) * 100, 1)

    if model_choice == "isolation_forest":
        step4_detail = (
            "<strong>Isolation Forest</strong> (unsupervised) scored all readings using "
            "the anomaly-free baseline model — no cross-validation needed."
        )
        step4_name = "Isolation Forest Scoring"
        step4_icon = "🌲"
    elif model_choice == "lstm_autoencoder":
        step4_detail = (
            "<strong>LSTM Autoencoder</strong> (unsupervised) scores each "
            f"{LSTM_SEQ_LEN}-timestep window by reconstruction error — "
            "no cross-validation needed."
        )
        step4_name = "LSTM Autoencoder Scoring"
        step4_icon = "🧠"
    elif model_choice == "transformer":
        step4_detail = (
            "<strong>Transformer Autoencoder</strong> (unsupervised) scores each "
            f"{TRANSFORMER_SEQ_LEN}-timestep window by reconstruction error — "
            "no cross-validation needed."
        )
        step4_name = "Transformer Autoencoder Scoring"
        step4_icon = "⚡"
    elif model_choice == "adapted_lstm":
        step4_detail = (
            "<strong>Adapted LSTM (ALSS-SVDD)</strong> (semi-supervised) scores each "
            f"{ALSTM_WINDOW}-timestep window by distance to learned normal centre — "
            "no cross-validation needed."
        )
        step4_name = "Adapted LSTM Scoring"
        step4_icon = "🔬"
    else:
        step4_detail = (
            f"<strong>{model_label}</strong> evaluated with <strong>Group K-Fold CV</strong> "
            "(grouped by file) — same method as the research notebook. "
            "Each file is tested only on models that never saw it."
        )
        step4_name = "Group K-Fold Running"
        step4_icon = "🔄"

    steps = [
        {
            "id": 1, "name": "ZIP Extracted", "icon": "📦",
            "detail": (
                f"<strong>{n_files} CSV files</strong> extracted · "
                f"<strong>{n_rows:,} rows</strong> total · "
                f"anomaly rate: <strong>{anomaly_rate}%</strong>."
            ),
        },
        {
            "id": 2, "name": "Feature Engineering", "icon": "⚙️",
            "detail": (
                f"Computed 5-point rolling mean &amp; std for each of the "
                f"{len(FEATURE_COLS)} sensors → <strong>{len(EXTENDED_COLS)} features</strong>."
            ),
        },
        {
            "id": 3, "name": "Normalisation", "icon": "⚖️",
            "detail": "All sensor readings scaled using the anomaly-free baseline.",
        },
        {
            "id": 4, "name": step4_name, "icon": step4_icon,
            "detail": step4_detail,
        },
        {
            "id": 5, "name": "Results Ready", "icon": "📊",
            "detail": "Per-file and overall metrics computed.",
        },
    ]

    try:
        if model_choice == "isolation_forest":
            result = run_isolation_forest_eval(combined)
        elif model_choice == "lstm_autoencoder":
            result = run_lstm_eval(combined)
        elif model_choice == "transformer":
            result = run_transformer_eval(combined)
        elif model_choice == "adapted_lstm":
            result = run_adapted_lstm_eval(combined)
        else:
            result = run_group_kfold(combined, model_choice)
    except Exception as exc:
        return jsonify({"error": f"Evaluation failed: {exc}"}), 500

    steps[4]["detail"] = (
        f"Overall F1: <strong>{result['overall']['f1']}</strong> · "
        f"Recall: <strong>{result['overall']['recall']}</strong> · "
        f"Precision: <strong>{result['overall']['precision']}</strong>."
    )

    return jsonify({
        "steps":         steps,
        "overall":       result["overall"],
        "fold_f1s":      result["fold_f1s"],
        "per_file":      result["per_file"],
        "confusion":     result["confusion"],
        "timeline":      result["timeline"],
        "total_rows":    result["total_rows"],
        "anomaly_count": result["anomaly_count"],
        "normal_count":  result["normal_count"],
        "anomaly_rate":  result["anomaly_rate"],
        "n_folds":       result["n_folds"],
        "model":         model_label,
        "filename":      filename,
        "n_files":       n_files,
    })


if __name__ == "__main__":
    app.run(debug=False, port=5001, threaded=True)
