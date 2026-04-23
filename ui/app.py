#!/usr/bin/env python3
"""
Pump Failure Detection — Flask backend
Trains XGBoost & Random Forest on SKAB data at startup,
then serves predictions via /api/predict.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_and_save() -> None:
    print("=" * 60)
    print("Training models on SKAB data (first run — please wait)…")
    print("=" * 60)

    baseline_path = DATA_DIR / "anomaly-free" / "anomaly-free.csv"
    baseline_df   = read_skab_csv(baseline_path)
    baseline_feat = add_rolling_features(baseline_df)

    scaler = StandardScaler()
    scaler.fit(baseline_feat[EXTENDED_COLS])

    # Hold out the last 20% of files per dataset so users can test on unseen data.
    # Held-out files: valve1/13-15, valve2/3, other/12-14
    train_dfs: list[pd.DataFrame] = []
    held_out: list[str] = []
    for dataset in ("valve1", "valve2", "other"):
        d = DATA_DIR / dataset
        if not d.exists():
            continue
        files = sorted(d.glob("*.csv"))
        cutoff = max(1, int(len(files) * 0.8))
        for i, csv_file in enumerate(files):
            df = read_skab_csv(csv_file)
            df["source_file"] = csv_file.name
            if i < cutoff:
                train_dfs.append(df)
            else:
                held_out.append(str(csv_file.relative_to(DATA_DIR.parent)))

    print(f"  Held-out files (for testing): {held_out}")

    combined = pd.concat(train_dfs, ignore_index=True)
    feat_df  = add_rolling_features(combined)

    X = scaler.transform(feat_df[EXTENDED_COLS])
    y = feat_df["anomaly"].values

    neg = (y == 0).sum()
    pos = (y == 1).sum()
    scale_pos = neg / pos if pos > 0 else 1.0

    print(f"  Total rows: {len(y)} | Anomaly rate: {y.mean():.1%}")

    print("  Training XGBoost…")
    xgb = XGBClassifier(
        n_estimators=150, max_depth=6,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    xgb.fit(X, y)

    print("  Training Random Forest…")
    rf = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    rf.fit(X, y)

    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(xgb,    MODELS_DIR / "xgboost.joblib")
    joblib.dump(rf,     MODELS_DIR / "random_forest.joblib")
    print("  Models saved to models_cache/")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Load (or train) models at startup
# ---------------------------------------------------------------------------
if not (MODELS_DIR / "xgboost.joblib").exists():
    train_and_save()

_scaler  = joblib.load(MODELS_DIR / "scaler.joblib")
_models  = {
    "xgboost":       joblib.load(MODELS_DIR / "xgboost.joblib"),
    "random_forest": joblib.load(MODELS_DIR / "random_forest.joblib"),
}
print("Models ready.")

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
    return jsonify({"status": "ok", "models": list(_models.keys())})


@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    model_choice = request.form.get("model", "xgboost")
    if model_choice not in _models:
        return jsonify({"error": f"Unknown model: {model_choice}"}), 400

    raw_bytes = request.files["file"].read()
    filename  = request.files["file"].filename or "uploaded.csv"

    # ── Step 1 : Load ────────────────────────────────────────────────────────
    try:
        df = parse_uploaded_csv(raw_bytes)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        return jsonify({
            "error": (
                f"Missing required sensor columns: {missing}. "
                "Make sure your CSV has the standard SKAB column names."
            )
        }), 400

    has_labels = "anomaly" in df.columns

    step1 = {
        "id": 1, "name": "File Loaded",
        "icon": "📁",
        "detail": (
            f"<strong>{len(df):,} rows</strong> and "
            f"<strong>{len(FEATURE_COLS)} sensor columns</strong> detected. "
            + ("Ground-truth labels found — full metrics will be computed."
               if has_labels else
               "No 'anomaly' column — anomaly scores only (no F1/Recall).")
        ),
    }

    # ── Step 2 : Feature engineering ─────────────────────────────────────────
    feat_df = add_rolling_features(df)

    step2 = {
        "id": 2, "name": "Feature Engineering",
        "icon": "⚙️",
        "detail": (
            f"Added a <strong>5-point rolling average and variability score</strong> for each of "
            f"the {len(FEATURE_COLS)} sensors — giving the model short-term memory "
            "so it can spot gradual drifts, not just sudden spikes. "
            f"Feature count: <strong>{len(EXTENDED_COLS)}</strong>."
        ),
    }

    # ── Step 3 : Normalisation ────────────────────────────────────────────────
    try:
        X = _scaler.transform(feat_df[EXTENDED_COLS])
    except Exception as exc:
        return jsonify({"error": f"Scaling failed: {exc}"}), 500

    step3 = {
        "id": 3, "name": "Normalisation",
        "icon": "⚖️",
        "detail": (
            "Each sensor reading was <strong>scaled to the same range</strong> using the pump's "
            "known-normal baseline, so a high-voltage reading doesn't overshadow a tiny pressure "
            "fluctuation. Think of it as converting all units to the same language."
        ),
    }

    # ── Step 4 : Inference ────────────────────────────────────────────────────
    clf    = _models[model_choice]
    preds  = clf.predict(X)
    probas = clf.predict_proba(X)[:, 1]

    model_label = "XGBoost" if model_choice == "xgboost" else "Random Forest"
    step4 = {
        "id": 4, "name": "Model Running",
        "icon": "🤖",
        "detail": (
            f"<strong>{model_label}</strong> examined each of the {len(df):,} readings "
            "and assigned an <strong>anomaly probability score</strong> between 0 and 1. "
            "Scores above 0.5 are flagged as anomalies."
        ),
    }

    # ── Step 5 : Results ──────────────────────────────────────────────────────
    anomaly_count = int(preds.sum())
    normal_count  = len(preds) - anomaly_count
    anomaly_rate  = round(float(preds.mean()) * 100, 1)

    metrics: dict = {}
    confusion: list | None = None

    if has_labels:
        y_true = df["anomaly"].values.astype(int)
        unique_labels = np.unique(y_true)

        metrics = {
            "f1":        round(float(f1_score(y_true, preds, zero_division=0)), 4),
            "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_true, preds, zero_division=0)), 4),
            "roc_auc":   (
                round(float(roc_auc_score(y_true, probas)), 4)
                if len(unique_labels) > 1 else None
            ),
        }
        cm = confusion_matrix(y_true, preds)
        confusion = cm.tolist()

    step5 = {
        "id": 5, "name": "Results Ready",
        "icon": "📊",
        "detail": (
            f"<strong>{anomaly_count}</strong> anomalies detected out of "
            f"<strong>{len(df):,}</strong> readings "
            f"(<strong>{anomaly_rate}%</strong> anomaly rate)."
            + (f" F1 score: <strong>{metrics['f1']}</strong>." if metrics else "")
        ),
    }

    # Downsample predictions for the timeline chart (max 1000 points)
    step = max(1, len(probas) // 1000)
    timeline = [
        {"i": int(i), "prob": round(float(probas[i]), 4), "pred": int(preds[i])}
        for i in range(0, len(probas), step)
    ]

    return jsonify({
        "steps":          [step1, step2, step3, step4, step5],
        "metrics":        metrics,
        "has_labels":     has_labels,
        "timeline":       timeline,
        "total_rows":     len(df),
        "anomaly_count":  anomaly_count,
        "normal_count":   normal_count,
        "anomaly_rate":   anomaly_rate,
        "model":          model_label,
        "confusion":      confusion,
        "filename":       filename,
    })


if __name__ == "__main__":
    app.run(debug=False, port=5001)
