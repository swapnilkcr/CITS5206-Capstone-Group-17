#!/usr/bin/env python3
"""
Pump Failure Detection — Flask backend
Uses Leave-One-Out (LOGO) evaluation to match the Group K-Fold results
from the notebook: when a known SKAB file is uploaded, the model trains
on all OTHER files and predicts on the uploaded one — giving honest metrics.
"""
from __future__ import annotations

import hashlib
import io
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


def file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# ---------------------------------------------------------------------------
# Index all SKAB files by MD5 hash at startup
# ---------------------------------------------------------------------------
def build_skab_index() -> dict[str, Path]:
    """Returns {md5_hash: path} for every SKAB CSV file."""
    index: dict[str, Path] = {}
    for dataset in ("valve1", "valve2", "other"):
        d = DATA_DIR / dataset
        if d.exists():
            for f in sorted(d.glob("*.csv")):
                index[file_hash(f.read_bytes())] = f
    print(f"  SKAB index: {len(index)} files indexed.")
    return index


# ---------------------------------------------------------------------------
# Scaler (fitted on anomaly-free baseline, shared across all evaluations)
# ---------------------------------------------------------------------------
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
# Build a model trained on all files EXCEPT `exclude_path`  (LOGO logic)
# ---------------------------------------------------------------------------
def train_logo_model(model_choice: str, exclude_path: Path) -> RandomForestClassifier | XGBClassifier:
    train_dfs: list[pd.DataFrame] = []
    for dataset in ("valve1", "valve2", "other"):
        d = DATA_DIR / dataset
        if not d.exists():
            continue
        for csv_file in sorted(d.glob("*.csv")):
            if csv_file.resolve() == exclude_path.resolve():
                continue
            df = read_skab_csv(csv_file)
            df["source_file"] = csv_file.name
            train_dfs.append(df)

    combined = pd.concat(train_dfs, ignore_index=True)
    feat_df  = add_rolling_features(combined)
    X = _scaler.transform(feat_df[EXTENDED_COLS])
    y = feat_df["anomaly"].values

    neg = (y == 0).sum()
    pos = (y == 1).sum()
    scale_pos = neg / pos if pos > 0 else 1.0

    if model_choice == "xgboost":
        clf = XGBClassifier(
            n_estimators=150, max_depth=6,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
    clf.fit(X, y)
    return clf


# ---------------------------------------------------------------------------
# Fallback full model (for truly unknown files with no ground truth)
# ---------------------------------------------------------------------------
def train_and_save_full() -> None:
    print("Training fallback full model…")
    all_dfs: list[pd.DataFrame] = []
    for dataset in ("valve1", "valve2", "other"):
        d = DATA_DIR / dataset
        if d.exists():
            for csv_file in sorted(d.glob("*.csv")):
                df = read_skab_csv(csv_file)
                df["source_file"] = csv_file.name
                all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    feat_df  = add_rolling_features(combined)
    X = _scaler.transform(feat_df[EXTENDED_COLS])
    y = feat_df["anomaly"].values

    neg, pos = (y == 0).sum(), (y == 1).sum()
    scale_pos = neg / pos if pos > 0 else 1.0

    xgb = XGBClassifier(n_estimators=150, max_depth=6, scale_pos_weight=scale_pos,
                         eval_metric="logloss", random_state=42, n_jobs=-1)
    xgb.fit(X, y)

    rf = RandomForestClassifier(n_estimators=150, class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    rf.fit(X, y)

    joblib.dump(xgb, MODELS_DIR / "xgboost_full.joblib")
    joblib.dump(rf,  MODELS_DIR / "random_forest_full.joblib")
    print("  Fallback models saved.")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
print("=" * 60)
print("Pump Failure Detection — starting up…")
_scaler     = fit_or_load_scaler()
_skab_index = build_skab_index()

if not (MODELS_DIR / "xgboost_full.joblib").exists():
    train_and_save_full()

_full_models = {
    "xgboost":       joblib.load(MODELS_DIR / "xgboost_full.joblib"),
    "random_forest": joblib.load(MODELS_DIR / "random_forest_full.joblib"),
}
print("Ready — LOGO evaluation active for known SKAB files.")
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


@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    model_choice = request.form.get("model", "xgboost")
    if model_choice not in _full_models:
        return jsonify({"error": f"Unknown model: {model_choice}"}), 400

    raw_bytes = request.files["file"].read()
    filename  = request.files["file"].filename or "uploaded.csv"

    # ── Step 1 : Load ─────────────────────────────────────────────────────────
    try:
        df = parse_uploaded_csv(raw_bytes)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        return jsonify({"error": f"Missing required columns: {missing}"}), 400

    has_labels   = "anomaly" in df.columns
    uploaded_md5 = file_hash(raw_bytes)
    matched_path = _skab_index.get(uploaded_md5)       # None if unknown file
    is_known     = matched_path is not None

    step1 = {
        "id": 1, "name": "File Loaded", "icon": "📁",
        "detail": (
            f"<strong>{len(df):,} rows</strong> · "
            f"<strong>{len(FEATURE_COLS)} sensor columns</strong> detected. "
            + (f"Recognised as <strong>{matched_path.parent.name}/{matched_path.name}</strong> — "
               "Leave-One-Out evaluation will be used for honest metrics."
               if is_known else
               ("Ground-truth labels found." if has_labels else "No labels — anomaly scores only."))
        ),
    }

    # ── Step 2 : Feature engineering ──────────────────────────────────────────
    feat_df = add_rolling_features(df)
    step2 = {
        "id": 2, "name": "Feature Engineering", "icon": "⚙️",
        "detail": (
            f"Computed <strong>5-point rolling mean &amp; std</strong> for each of the "
            f"{len(FEATURE_COLS)} sensors — expanding to "
            f"<strong>{len(EXTENDED_COLS)} features</strong> total. "
            "This gives the model short-term memory to catch gradual drifts, not just sudden spikes."
        ),
    }

    # ── Step 3 : Normalisation ─────────────────────────────────────────────────
    try:
        X = _scaler.transform(feat_df[EXTENDED_COLS])
    except Exception as exc:
        return jsonify({"error": f"Scaling failed: {exc}"}), 500

    step3 = {
        "id": 3, "name": "Normalisation", "icon": "⚖️",
        "detail": (
            "All sensor readings scaled to a common range using the pump's "
            "<strong>anomaly-free baseline</strong>, so no single sensor dominates. "
            "Like converting miles and km to the same unit before comparing."
        ),
    }

    # ── Step 4 : Train + infer ────────────────────────────────────────────────
    model_label = "XGBoost" if model_choice == "xgboost" else "Random Forest"

    if is_known and has_labels:
        # LOGO: fresh model trained on all SKAB files except this one
        step4 = {
            "id": 4, "name": "Leave-One-Out Training", "icon": "🤖",
            "detail": (
                f"Training <strong>{model_label}</strong> on all SKAB files "
                f"<em>except</em> this one — then predicting on this file. "
                "This is the same evaluation strategy used in the research notebook "
                "and gives <strong>honest, unbiased metrics</strong>."
            ),
        }
        clf = train_logo_model(model_choice, matched_path)
    else:
        step4 = {
            "id": 4, "name": "Model Running", "icon": "🤖",
            "detail": (
                f"<strong>{model_label}</strong> (trained on all SKAB data) examined "
                f"each of the {len(df):,} readings and assigned an "
                "<strong>anomaly probability score</strong> (0 = normal, 1 = anomaly)."
            ),
        }
        clf = _full_models[model_choice]

    preds  = clf.predict(X)
    probas = clf.predict_proba(X)[:, 1]

    # ── Step 5 : Results ───────────────────────────────────────────────────────
    anomaly_count = int(preds.sum())
    normal_count  = len(preds) - anomaly_count
    anomaly_rate  = round(float(preds.mean()) * 100, 1)

    metrics: dict = {}
    confusion: list | None = None

    if has_labels:
        y_true = df["anomaly"].values.astype(int)
        metrics = {
            "f1":        round(float(f1_score(y_true, preds, zero_division=0)), 4),
            "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_true, preds, zero_division=0)), 4),
            "roc_auc":   (
                round(float(roc_auc_score(y_true, probas)), 4)
                if len(np.unique(y_true)) > 1 else None
            ),
        }
        confusion = confusion_matrix(y_true, preds).tolist()

    step5 = {
        "id": 5, "name": "Results Ready", "icon": "📊",
        "detail": (
            f"<strong>{anomaly_count:,}</strong> anomalies out of "
            f"<strong>{len(df):,}</strong> readings "
            f"(<strong>{anomaly_rate}%</strong>)."
            + (f" F1: <strong>{metrics['f1']}</strong>." if metrics else "")
        ),
    }

    step_size = max(1, len(probas) // 1000)
    timeline = [
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
        "confusion":     confusion,
        "filename":      filename,
    })


if __name__ == "__main__":
    app.run(debug=False, port=5001)
