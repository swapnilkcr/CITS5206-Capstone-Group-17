"""
Registry-driven checkpoint loader for the Flask UI.

macOS note: sklearn/xgboost joblib files must be loaded BEFORE ``import torch``
or loading XGBoost can segfault. Call ``load_sklearn_artifacts()`` first.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
PRETRAINED_DIR = REPO_ROOT / "pre_trained_model"
REGISTRY_PATH = PRETRAINED_DIR / "registry.yaml"
ARTIFACTS_DIR = Path(os.environ.get("PUMP_ARTIFACTS_DIR", PRETRAINED_DIR / "artifacts"))
DATA_DIR = REPO_ROOT / "data"

# registry model_id -> UI / API model key
UI_KEYS: dict[str, str] = {
    "feature_scaler_skab_v1": "scaler",
    "xgboost_full_skab_v1": "xgboost",
    "random_forest_full_skab_v1": "random_forest",
    "isolation_forest_skab_v1": "isolation_forest",
    "lstm_ae_skab_v1": "lstm_autoencoder",
    "transformer_ae_skab_v1": "transformer",
    "adapted_lstm_valve1_v1": "adapted_lstm",
}

SKLEARN_IDS = (
    "feature_scaler_skab_v1",
    "xgboost_full_skab_v1",
    "random_forest_full_skab_v1",
    "isolation_forest_skab_v1",
)
TORCH_BUNDLE_IDS = (
    "lstm_ae_skab_v1",
    "transformer_ae_skab_v1",
    "adapted_lstm_valve1_v1",
)


@dataclass
class RegistryEntry:
    model_id: str
    owner: str
    created_at: str
    task: str
    framework: str
    artifact_uri: str
    dataset: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    notes: str | None = None


@dataclass
class PipelineState:
    registry: dict[str, RegistryEntry] = field(default_factory=dict)
    scaler: Any = None
    full_models: dict[str, Any] = field(default_factory=dict)
    if_model: Any = None
    if_threshold: float = 0.0
    lstm_model: Any = None
    lstm_threshold: float = 0.0
    transformer_model: Any = None
    transformer_scaler: Any = None
    transformer_threshold: float = 0.0
    alstm_encoder: Any = None
    alstm_center: Any = None
    alstm_threshold: float = 0.0
    alstm_scaler: Any = None


def _load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        _fail("PyYAML required: pip install pyyaml")
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        _fail(f"{path} must be a YAML mapping")
    return data


def _fail(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def resolve_artifact_path(uri: str) -> Path:
    p = Path(uri)
    if p.is_absolute():
        return p
    if uri.startswith("pre_trained_model/"):
        return REPO_ROOT / uri
    if uri.startswith("artifacts/"):
        return PRETRAINED_DIR / uri
    return REPO_ROOT / uri


def load_registry(state: PipelineState) -> None:
    if not REGISTRY_PATH.exists():
        _fail(f"Registry not found: {REGISTRY_PATH}")
    raw = _load_yaml(REGISTRY_PATH).get("models", [])
    if not isinstance(raw, list):
        _fail("registry.yaml 'models' must be a list")

    state.registry.clear()
    for item in raw:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("model_id", "")).strip()
        if not mid:
            continue
        state.registry[mid] = RegistryEntry(
            model_id=mid,
            owner=str(item.get("owner", "")),
            created_at=str(item.get("created_at", "")),
            task=str(item.get("task", "")),
            framework=str(item.get("framework", "")),
            artifact_uri=str(item.get("artifact_uri", "")),
            dataset=item.get("dataset") if isinstance(item.get("dataset"), dict) else None,
            metrics=item.get("metrics") if isinstance(item.get("metrics"), dict) else None,
            notes=item.get("notes") if isinstance(item.get("notes"), str) else None,
        )

    missing_ui = [mid for mid in UI_KEYS if mid not in state.registry]
    if missing_ui:
        _fail(f"registry.yaml missing entries: {', '.join(missing_ui)}")


def _require_artifact(state: PipelineState, model_id: str) -> Path:
    entry = state.registry.get(model_id)
    if entry is None:
        _fail(f"Unknown model_id in registry: {model_id}")
    path = resolve_artifact_path(entry.artifact_uri)
    if not path.exists():
        _fail(
            f"Missing checkpoint for {model_id} ({UI_KEYS.get(model_id, model_id)}):\n"
            f"  {path}\n"
            f"Train offline and register in {REGISTRY_PATH}"
        )
    return path


def _load_one(state: PipelineState, model_id: str) -> None:
    path = _require_artifact(state, model_id)
    print(f"  [ok] {model_id}  ← {path.name}")


def load_sklearn_artifacts(state: PipelineState) -> None:
    """Load joblib/sklearn checkpoints — call before ``import torch``."""
    load_registry(state)
    print(f"  Registry: {len(state.registry)} models")

    _load_one(state, "feature_scaler_skab_v1")
    state.scaler = joblib.load(_require_artifact(state, "feature_scaler_skab_v1"))

    _load_one(state, "xgboost_full_skab_v1")
    state.full_models["xgboost"] = joblib.load(_require_artifact(state, "xgboost_full_skab_v1"))

    _load_one(state, "random_forest_full_skab_v1")
    state.full_models["random_forest"] = joblib.load(
        _require_artifact(state, "random_forest_full_skab_v1")
    )

    _load_one(state, "isolation_forest_skab_v1")
    if_data = joblib.load(_require_artifact(state, "isolation_forest_skab_v1"))
    state.if_model = if_data["model"]
    state.if_threshold = float(if_data["threshold"])


def load_torch_artifacts(state: PipelineState) -> None:
    """Load PyTorch bundles — call after ``import torch`` and torch_models."""
    from torch_models import (  # noqa: WPS433 — intentional late import
        load_adapted_lstm_bundle,
        load_lstm_bundle,
        load_transformer_bundle,
    )

    mid = "lstm_ae_skab_v1"
    _load_one(state, mid)
    state.lstm_model, state.lstm_threshold = load_lstm_bundle(_require_artifact(state, mid))

    mid = "transformer_ae_skab_v1"
    _load_one(state, mid)
    state.transformer_model, state.transformer_scaler, state.transformer_threshold = (
        load_transformer_bundle(_require_artifact(state, mid))
    )

    mid = "adapted_lstm_valve1_v1"
    _load_one(state, mid)
    state.alstm_encoder, state.alstm_center, state.alstm_threshold, state.alstm_scaler = (
        load_adapted_lstm_bundle(_require_artifact(state, mid), PRETRAINED_DIR)
    )


def bootstrap(state: PipelineState | None = None) -> PipelineState:
    """Full load sequence with console progress."""
    st = state or PipelineState()
    print("=" * 60)
    print("Pump Failure Detection — loading checkpoints…")
    print(f"  Registry: {REGISTRY_PATH.resolve()}")
    print(f"  Artifacts: {ARTIFACTS_DIR.resolve()}")

    load_sklearn_artifacts(st)
    print("  Scaler + XGBoost + Random Forest + Isolation Forest loaded.")

    load_torch_artifacts(st)
    print("  LSTM + Transformer + Adapted LSTM loaded.")
    print("Ready.")
    print("=" * 60)
    return st
