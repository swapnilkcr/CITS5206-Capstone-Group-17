#!/usr/bin/env python3
"""
Pump Failure Detection — Flask backend (load-only).

Checkpoints live under pre_trained_model/artifacts/ (train offline, never on startup).
- Single file: LOGO evaluation (train on all other SKAB files, test on uploaded)
- ZIP dataset: Group K-Fold (exactly mirrors the notebook evaluation)
"""
from __future__ import annotations

# macOS: XGBoost + PyTorch share OpenMP — limit threads before numpy/sklearn/torch import
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import hashlib
import io
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
UI_DIR = REPO_ROOT / "ui"
FRONTEND_DIR = UI_DIR / "frontend"
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
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

from constants import ALSTM_WINDOW, EXTENDED_COLS, FEATURE_COLS, LSTM_SEQ_LEN, TRANSFORMER_SEQ_LEN
from data_pipeline import (
    has_skab_schema,
    numeric_column_candidates,
    parse_column_map,
    prepare_inference_frame,
    suggest_column_map,
    validate_column_map,
)
from model_pipeline import DATA_DIR, PipelineState, load_sklearn_artifacts, load_torch_artifacts

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


def _upload_context(raw_bytes: bytes, column_map_raw: str | None = None) -> dict:
    """Parse upload and classify SKAB-known vs unseen client datasets."""
    df = parse_uploaded_csv(raw_bytes)
    uploaded_md5 = file_hash(raw_bytes)
    is_known_skab = _skab_index.get(uploaded_md5) is not None
    column_map = parse_column_map(column_map_raw) if column_map_raw else None

    if column_map:
        errs = validate_column_map(column_map)
        if errs:
            raise ValueError("; ".join(errs))

    prep_meta: dict = {}
    ready = df
    if not has_skab_schema(df):
        if column_map:
            ready, prep_meta = prepare_inference_frame(
                df, column_map=column_map, is_known_skab=is_known_skab
            )
        else:
            prep_meta = {
                "needs_column_pick": True,
                "is_unseen_dataset": not is_known_skab,
                "has_skab_schema": False,
            }
    else:
        ready, prep_meta = prepare_inference_frame(
            df, column_map=None, is_known_skab=is_known_skab
        )

    return {
        "df_raw": df,
        "df": ready,
        "is_known_skab": is_known_skab,
        "is_unseen_dataset": not is_known_skab,
        "has_skab_schema": has_skab_schema(df),
        "needs_column_pick": prep_meta.get("needs_column_pick", False),
        "column_map": column_map,
        "prep_meta": prep_meta,
    }


def make_model(model_choice: str, y_train: np.ndarray):
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos = neg / pos if pos > 0 else 1.0
    if model_choice == "xgboost":
        return XGBClassifier(
            n_estimators=150, max_depth=6,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=42, n_jobs=1,
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
# Checkpoint load (sklearn before torch — avoids macOS XGBoost segfault)
# ---------------------------------------------------------------------------
_pipe = PipelineState()
print("=" * 60)
print("Pump Failure Detection — loading checkpoints…")
load_sklearn_artifacts(_pipe)
_skab_index = build_skab_index()

from torch_models import (  # noqa: E402 — after sklearn joblib load
    adapted_lstm_predict as _alstm_predict_impl,
    lstm_predict as _lstm_predict_impl,
    transformer_predict as _transformer_predict_impl,
)

load_torch_artifacts(_pipe)

_scaler = _pipe.scaler
_full_models = _pipe.full_models
_if_model = _pipe.if_model
_if_threshold = _pipe.if_threshold

print("Ready.")
print("=" * 60)


def if_predict(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = -_pipe.if_model.decision_function(X)
    preds  = (scores > _pipe.if_threshold).astype(int)
    s_min, s_max = scores.min(), scores.max()
    probas = (scores - s_min) / (s_max - s_min + 1e-9)
    return preds, probas


def lstm_predict(X_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _lstm_predict_impl(_pipe.lstm_model, _pipe.lstm_threshold, X_scaled)


def transformer_predict(df_raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return _transformer_predict_impl(
        _pipe.transformer_model, _pipe.transformer_scaler, _pipe.transformer_threshold, df_raw
    )


def adapted_lstm_predict(df_raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return _alstm_predict_impl(
        _pipe.alstm_encoder,
        _pipe.alstm_center,
        _pipe.alstm_threshold,
        _pipe.alstm_scaler,
        df_raw,
    )


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
# ZIP / unsupervised eval runners (torch predictors in torch_models.py)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# LOGO / Group K-Fold (on-demand training for evaluation — not startup)
# ---------------------------------------------------------------------------
def train_logo_model(model_choice: str, exclude_path: Path):
    """Leave-one-file-out: train on all labelled SKAB CSVs except the test file."""
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

    if not train_dfs:
        raise ValueError("No training files available for LOGO evaluation.")

    combined = pd.concat(train_dfs, ignore_index=True)
    feat_df  = add_rolling_features(combined)
    feat_df  = feat_df.dropna(subset=["anomaly"] + EXTENDED_COLS)
    feat_df["anomaly"] = feat_df["anomaly"].astype(int)
    X = _scaler.transform(feat_df[EXTENDED_COLS])
    y = feat_df["anomaly"].values

    clf = make_model(model_choice, y)
    if model_choice == "xgboost":
        clf.set_params(n_jobs=1)
    clf.fit(X, y)
    return clf


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
    for _, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        clf = make_model(model_choice, y[train_idx])
        if model_choice == "xgboost":
            clf.set_params(n_jobs=1)
        clf.fit(X[train_idx], y[train_idx])
        all_preds[test_idx]  = clf.predict(X[test_idx])
        all_probas[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
        fold_f1 = float(f1_score(y[test_idx], all_preds[test_idx], zero_division=0))
        fold_f1s.append(round(fold_f1, 4))

    overall = compute_metrics(y, all_preds, all_probas)

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

    cm = confusion_matrix(y, all_preds).tolist()
    step = max(1, len(all_probas) // 1000)
    timeline = [
        {"i": int(i), "prob": round(float(all_probas[i]), 4), "pred": int(all_preds[i])}
        for i in range(0, len(all_probas), step)
    ]

    return {
        "overall":       overall,
        "fold_f1s":      fold_f1s,
        "per_file":      per_file,
        "confusion":     cm,
        "timeline":      timeline,
        "n_folds":       actual_splits,
        "total_rows":    len(y),
        "anomaly_count": int(all_preds.sum()),
        "normal_count":  int((all_preds == 0).sum()),
        "anomaly_rate":  round(float(all_preds.mean()) * 100, 1),
    }


# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "skab_files": len(_skab_index),
        "models_loaded": {
            "scaler": _pipe.scaler is not None,
            "xgboost": "xgboost" in _pipe.full_models,
            "random_forest": "random_forest" in _pipe.full_models,
            "isolation_forest": _pipe.if_model is not None,
            "lstm_autoencoder": _pipe.lstm_model is not None,
            "transformer": _pipe.transformer_model is not None,
            "adapted_lstm": _pipe.alstm_encoder is not None,
        },
    })


@app.route("/api/preview", methods=["POST"])
def preview():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    raw_bytes = request.files["file"].read()
    column_map_raw = request.form.get("column_map")
    try:
        ctx = _upload_context(raw_bytes, column_map_raw)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    df = ctx["df_raw"]
    has_labels = "anomaly" in df.columns
    feature_present = [c for c in FEATURE_COLS if c in df.columns]
    preview_rows = df.head(20).copy()
    for col in preview_rows.select_dtypes(include="datetime").columns:
        preview_rows[col] = preview_rows[col].astype(str)

    numeric_cols = numeric_column_candidates(df)
    if not numeric_cols:
        numeric_cols = [c for c in df.columns if c not in ("anomaly", "changepoint")]

    return jsonify({
        "rows":                len(df),
        "cols":                len(df.columns),
        "columns":             df.columns.tolist(),
        "has_labels":          has_labels,
        "feature_cols":        feature_present,
        "anomaly_rate":        round(float(df["anomaly"].mean()) * 100, 1) if has_labels else None,
        "preview":             preview_rows.fillna("").to_dict(orient="records"),
        "is_known_skab":       ctx["is_known_skab"],
        "is_unseen_dataset":   ctx["is_unseen_dataset"],
        "has_skab_schema":     ctx["has_skab_schema"],
        "needs_column_pick":   ctx["needs_column_pick"],
        "skab_feature_cols":   FEATURE_COLS,
        "numeric_columns":     numeric_cols,
        "suggested_column_map": suggest_column_map(df),
        "column_map_valid": (
            True
            if ctx["has_skab_schema"] and not ctx["needs_column_pick"]
            else bool(
                column_map_raw
                and not validate_column_map(parse_column_map(column_map_raw) or {})
            )
        ),
        "migration_risk": (
            "Models were trained on SKAB valve data. Scores on unseen client data are "
            "indicative only (domain shift). For reliable results, retrain or fine-tune on "
            "client baseline data — see Retrain guide in Docs."
        ),
    })


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

    raw_bytes = request.files["file"].read()
    filename = request.files["file"].filename or "uploaded.csv"
    column_map_raw = request.form.get("column_map")

    try:
        ctx = _upload_context(raw_bytes, column_map_raw)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    df = ctx["df"]
    is_known = ctx["is_known_skab"]
    is_unseen = ctx["is_unseen_dataset"]
    matched_path = _skab_index.get(file_hash(raw_bytes))

    if ctx["needs_column_pick"]:
        return jsonify({
            "error": "Unseen dataset — please map all 8 SKAB sensor columns in the preview panel first.",
            "needs_column_pick": True,
        }), 400

    has_labels = "anomaly" in ctx["df_raw"].columns
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

    load_note = (
        f"Recognised SKAB file <strong>{matched_path.parent.name}/{matched_path.name}</strong>."
        if is_known
        else (
            "<strong>Unseen client dataset</strong> — columns mapped to SKAB sensor contract. "
            "<em>Migration risk: scores are indicative only.</em>"
            if is_unseen
            else "Ground-truth labels found." if has_labels else "No labels — anomaly scores only."
        )
    )
    step1 = {
        "id": 1, "name": "File Loaded", "icon": "📁",
        "detail": (
            f"<strong>{len(df):,} rows</strong> · "
            f"<strong>{len(FEATURE_COLS)} sensor channels</strong> ready. {load_note}"
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

    try:
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
    except Exception as exc:
        return jsonify({"error": f"Prediction failed ({model_label}): {exc}"}), 500

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
        "steps":             [step1, step2, step3, step4, step5],
        "metrics":           metrics,
        "has_labels":        has_labels,
        "is_known":          is_known,
        "is_unseen_dataset": is_unseen,
        "migration_risk": (
            "Models were trained on SKAB valve data. Scores on this unseen client dataset "
            "are indicative screening only (domain shift). For reliable deployment, retrain "
            "or fine-tune on your pump's baseline — see Retrain guide in Docs."
            if is_unseen
            else None
        ),
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
