"""PyTorch model definitions and checkpoint loaders (import after sklearn artifacts)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(1)

from constants import (
    ALSTM_HIDDEN,
    ALSTM_LATENT,
    ALSTM_STRIDE,
    ALSTM_WINDOW,
    FEATURE_COLS,
    LSTM_LATENT,
    LSTM_SEQ_LEN,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_FFN_DIM,
    TRANSFORMER_LAYERS,
    TRANSFORMER_NHEAD,
    TRANSFORMER_SEQ_LEN,
)


def make_sequences(data: np.ndarray, seq_len: int) -> np.ndarray:
    return np.stack([data[i:i + seq_len] for i in range(len(data) - seq_len + 1)]).astype(np.float32)


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, latent_dim: int = LSTM_LATENT):
        super().__init__()
        self.encoder1 = nn.LSTM(n_features, 64, batch_first=True)
        self.encoder2 = nn.LSTM(64, latent_dim, batch_first=True)
        self.decoder1 = nn.LSTM(latent_dim, latent_dim, batch_first=True)
        self.decoder2 = nn.LSTM(latent_dim, 64, batch_first=True)
        self.out = nn.Linear(64, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.encoder1(x)
        _, (h, _) = self.encoder2(out)
        h = h.squeeze(0).unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder1(h)
        out, _ = self.decoder2(out)
        return self.out(out)


def lstm_recon_errors(model: LSTMAutoencoder, seqs: np.ndarray, batch: int = 128) -> np.ndarray:
    model.eval()
    errors: list[float] = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            x = torch.tensor(seqs[i:i + batch])
            recon = model(x)
            err = ((recon - x) ** 2).mean(dim=(1, 2)).numpy()
            errors.extend(err)
    return np.array(errors)


def load_lstm_bundle(path: Path) -> tuple[LSTMAutoencoder, float]:
    data = joblib.load(path)
    model = LSTMAutoencoder(data["n_features"], LSTM_LATENT)
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model, float(data["threshold"])


def lstm_predict(
    model: LSTMAutoencoder, threshold: float, X_scaled: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = len(X_scaled)
    if n_rows < LSTM_SEQ_LEN:
        return np.zeros(n_rows, int), np.zeros(n_rows)

    seqs = make_sequences(X_scaled.astype(np.float32), LSTM_SEQ_LEN)
    errors = lstm_recon_errors(model, seqs)
    e_min, e_max = errors.min(), errors.max()
    probas_seq = (errors - e_min) / (e_max - e_min + 1e-9)
    preds_seq = (errors > threshold).astype(int)
    pad = LSTM_SEQ_LEN - 1
    full_preds = np.zeros(n_rows, dtype=int)
    full_probas = np.zeros(n_rows)
    full_preds[pad:] = preds_seq
    full_probas[pad:] = probas_seq
    return full_preds, full_probas


class TransformerAE(nn.Module):
    def __init__(self, input_dim: int = 8):
        super().__init__()
        self.embedding = nn.Linear(input_dim, TRANSFORMER_D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TRANSFORMER_D_MODEL,
            nhead=TRANSFORMER_NHEAD,
            dim_feedforward=TRANSFORMER_FFN_DIM,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=TRANSFORMER_LAYERS)
        self.decoder = nn.Linear(TRANSFORMER_D_MODEL, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.transformer(x)
        return self.decoder(x)


def transformer_recon_errors(model: TransformerAE, seqs: np.ndarray, batch: int = 128) -> np.ndarray:
    model.eval()
    errors: list[float] = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            x = torch.tensor(seqs[i:i + batch])
            recon = model(x)
            err = ((x - recon) ** 2).mean(dim=(1, 2)).numpy()
            errors.extend(err)
    return np.array(errors)


def load_transformer_bundle(path: Path) -> tuple[TransformerAE, StandardScaler, float]:
    data = joblib.load(path)
    model = TransformerAE(input_dim=len(FEATURE_COLS))
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model, data["scaler"], float(data["threshold"])


def transformer_predict(
    model: TransformerAE,
    scaler: StandardScaler,
    threshold: float,
    df_raw,
) -> tuple[np.ndarray, np.ndarray]:
    raw = df_raw[FEATURE_COLS].values.astype(np.float32)
    scaled = scaler.transform(raw)
    n_rows = len(scaled)
    if n_rows < TRANSFORMER_SEQ_LEN:
        return np.zeros(n_rows, dtype=int), np.zeros(n_rows)

    seqs = make_sequences(scaled, TRANSFORMER_SEQ_LEN)
    errors = transformer_recon_errors(model, seqs)
    e_min, e_max = errors.min(), errors.max()
    probas_seq = (errors - e_min) / (e_max - e_min + 1e-9)
    preds_seq = (errors > threshold).astype(int)
    pad = TRANSFORMER_SEQ_LEN - 1
    full_preds = np.zeros(n_rows, dtype=int)
    full_probas = np.zeros(n_rows)
    full_preds[pad:] = preds_seq
    full_probas[pad:] = probas_seq
    return full_preds, full_probas


class ALSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(len(FEATURE_COLS), ALSTM_HIDDEN, batch_first=True)
        self.fc = nn.Linear(ALSTM_HIDDEN, ALSTM_LATENT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


def alstm_windows(data: np.ndarray, labels: np.ndarray | None = None):
    xs, ys = [], []
    for i in range(0, len(data) - ALSTM_WINDOW + 1, ALSTM_STRIDE):
        xs.append(data[i:i + ALSTM_WINDOW])
        if labels is not None:
            ys.append(1 if labels[i:i + ALSTM_WINDOW].max() > 0 else 0)
    X = np.array(xs, dtype=np.float32)
    Y = np.array(ys, dtype=np.int64) if labels is not None else None
    return X, Y


def load_adapted_lstm_from_pt(pt_path: Path, scaler: StandardScaler) -> tuple:
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    encoder = ALSTMEncoder()
    encoder.load_state_dict(ckpt["state_dict"])
    encoder.eval()
    center = ckpt["center"] if isinstance(ckpt["center"], torch.Tensor) else torch.tensor(ckpt["center"])
    return encoder, center, float(ckpt["thr"]), scaler


def load_adapted_lstm_bundle(joblib_path: Path, pretrained_dir: Path) -> tuple:
    if joblib_path.exists():
        data = joblib.load(joblib_path)
        encoder = ALSTMEncoder()
        encoder.load_state_dict(data["state_dict"])
        encoder.eval()
        return encoder, torch.tensor(data["center"]), float(data["threshold"]), data["scaler"]

    matches = sorted(
        pretrained_dir.glob("Adapted-LSTM/outputs/*/best_model.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if matches:
        scaler_path = joblib_path.parent / "scaler.joblib"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Adapted LSTM scaler required with .pt: {scaler_path}")
        return load_adapted_lstm_from_pt(matches[0], joblib.load(scaler_path))

    raise FileNotFoundError(f"Adapted LSTM checkpoint not found: {joblib_path}")


def adapted_lstm_predict(
    encoder: ALSTMEncoder,
    center: torch.Tensor,
    threshold: float,
    scaler: StandardScaler,
    df_raw,
) -> tuple[np.ndarray, np.ndarray]:
    raw = df_raw[FEATURE_COLS].values.astype(np.float32)
    scaled = scaler.transform(raw)
    n_rows = len(scaled)
    if n_rows < ALSTM_WINDOW:
        return np.zeros(n_rows, dtype=int), np.zeros(n_rows)

    X, _ = alstm_windows(scaled)
    encoder.eval()
    scores: list[float] = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            xb = torch.tensor(X[i:i + 64])
            z = encoder(xb)
            s = ((z - center) ** 2).sum(dim=1)
            scores.extend(s.numpy())
    scores_arr = np.array(scores)

    row_scores = np.zeros(n_rows)
    for i, s in enumerate(scores_arr):
        start = i * ALSTM_STRIDE
        end = min(start + ALSTM_WINDOW, n_rows)
        row_scores[start:end] = np.maximum(row_scores[start:end], s)

    s_min, s_max = row_scores.min(), row_scores.max()
    full_probas = (row_scores - s_min) / (s_max - s_min + 1e-9)
    full_preds = (row_scores > threshold).astype(int)
    return full_preds, full_probas
