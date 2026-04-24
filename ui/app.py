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
            df["source_file"] = csv_file.name
            train_dfs.append(df)

    combined = pd.concat(train_dfs, ignore_index=True)
    feat_df  = add_rolling_features(combined)
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
                df["source_file"] = csv_file.name
                all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    feat_df  = add_rolling_features(combined)
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
    if model_choice not in _full_models:
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

    has_labels  = "anomaly" in df.columns
    model_label = "XGBoost" if model_choice == "xgboost" else "Random Forest"

    step1 = {
        "id": 1, "name": "File Loaded", "icon": "📁",
        "detail": (
            f"<strong>{len(df):,} rows</strong> · "
            f"<strong>{len(FEATURE_COLS)} sensor columns</strong> detected. "
            + (f"Recognised as <strong>{matched_path.parent.name}/{matched_path.name}</strong> — "
               "Leave-One-Out evaluation will give honest metrics."
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

    if is_known and has_labels:
        step4 = {
            "id": 4, "name": "Leave-One-Out Training", "icon": "🤖",
            "detail": (
                f"Training <strong>{model_label}</strong> on all SKAB files "
                "<em>except</em> this one — then predicting on this file for honest metrics."
            ),
        }
        clf = train_logo_model(model_choice, matched_path)
    else:
        step4 = {
            "id": 4, "name": "Model Running", "icon": "🤖",
            "detail": (
                f"<strong>{model_label}</strong> examined each of the {len(df):,} readings "
                "and assigned an anomaly probability score."
            ),
        }
        clf = _full_models[model_choice]

    preds  = clf.predict(X)
    probas = clf.predict_proba(X)[:, 1]

    anomaly_count = int(preds.sum())
    normal_count  = len(preds) - anomaly_count
    anomaly_rate  = round(float(preds.mean()) * 100, 1)

    metrics: dict  = {}
    confusion_mat: list | None = None

    if has_labels:
        y_true     = df["anomaly"].values.astype(int)
        metrics    = compute_metrics(y_true, preds, probas)
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
    if model_choice not in _full_models:
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

    model_label  = "XGBoost" if model_choice == "xgboost" else "Random Forest"
    n_files      = combined["source_file"].nunique()
    n_rows       = len(combined)
    anomaly_rate = round(float(combined["anomaly"].mean()) * 100, 1)

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
            "id": 4, "name": "Group K-Fold Running", "icon": "🔄",
            "detail": (
                f"<strong>{model_label}</strong> evaluated with <strong>Group K-Fold CV</strong> "
                f"(grouped by file) — same method as the research notebook. "
                "Each file is tested only on models that never saw it."
            ),
        },
        {
            "id": 5, "name": "Results Ready", "icon": "📊",
            "detail": "Per-file and overall metrics computed.",
        },
    ]

    try:
        result = run_group_kfold(combined, model_choice)
    except Exception as exc:
        return jsonify({"error": f"Evaluation failed: {exc}"}), 500

    # Update step 5 detail with real numbers
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
    app.run(debug=False, port=5001)
