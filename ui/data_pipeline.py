""" unseen CSV → SKAB feature contract for inference."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from constants import FEATURE_COLS

META_COLS = frozenset({"anomaly", "changepoint", "datetime", "timestamp", "time", "date"})


def has_skab_schema(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in FEATURE_COLS)


def numeric_column_candidates(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for col in df.columns:
        if col in META_COLS or col in FEATURE_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)
    return out


def all_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in META_COLS]


def parse_column_map(raw: str | None) -> dict[str, str] | None:
    if not raw or not raw.strip():
        return None
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("column_map must be a JSON object")
    return {str(k): str(v) for k, v in data.items()}


def validate_column_map(mapping: dict[str, str]) -> list[str]:
    """Return list of validation errors (empty if ok)."""
    errors: list[str] = []
    missing_targets = [c for c in FEATURE_COLS if c not in mapping or not str(mapping[c]).strip()]
    if missing_targets:
        errors.append(f"Map all 8 SKAB sensors; missing: {', '.join(missing_targets)}")
    sources = [mapping[c] for c in FEATURE_COLS if c in mapping and mapping[c]]
    if len(sources) != len(set(sources)):
        errors.append("Each source column can only be used once.")
    return errors


def apply_column_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    errors = validate_column_map(mapping)
    if errors:
        raise ValueError("; ".join(errors))

    out = pd.DataFrame()
    for skab_col in FEATURE_COLS:
        src = mapping[skab_col]
        if src not in df.columns:
            raise ValueError(f"Source column not found in file: {src}")
        out[skab_col] = pd.to_numeric(df[src], errors="coerce")

    for time_col in ("datetime", "timestamp", "time"):
        if time_col in df.columns:
            out["datetime"] = pd.to_datetime(df[time_col], errors="coerce")
            break

    out = out.dropna(subset=FEATURE_COLS, how="any")
    if "datetime" in out.columns:
        out = out.sort_values("datetime").reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)

    return out


def prepare_inference_frame(
    df: pd.DataFrame,
    *,
    column_map: dict[str, str] | None,
    is_known_skab: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Returns (df_ready, meta).
    Raises ValueError if unseen schema without a complete column_map.
    """
    meta: dict[str, Any] = {
        "is_known_skab": is_known_skab,
        "is_unseen_dataset": not is_known_skab,
        "has_skab_schema": has_skab_schema(df),
        "needs_column_pick": False,
    }

    if has_skab_schema(df):
        ready = df.copy()
        for c in FEATURE_COLS:
            ready[c] = pd.to_numeric(ready[c], errors="coerce")
        meta["needs_column_pick"] = False
    elif column_map:
        ready = apply_column_mapping(df, column_map)
        meta["needs_column_pick"] = False
        meta["column_map_applied"] = True
    else:
        meta["needs_column_pick"] = True
        raise ValueError(
            "Unseen dataset: map your CSV columns to the 8 SKAB sensor channels before prediction."
        )

    if len(ready) < 30:
        raise ValueError(f"Not enough valid rows after cleaning ({len(ready)}). Need at least 30.")

    return ready, meta


def suggest_column_map(df: pd.DataFrame) -> dict[str, str]:
    """Best-effort auto-fill from similar column names (user can override in UI)."""
    candidates = all_numeric_columns(df) + [c for c in df.columns if c in FEATURE_COLS]
    norm = {c.lower().replace(" ", "").replace("_", ""): c for c in candidates}
    hints: dict[str, str] = {}
    aliases: dict[str, list[str]] = {
        "Accelerometer1RMS": ["accelerometer1", "accel1", "vibration1", "rms1"],
        "Accelerometer2RMS": ["accelerometer2", "accel2", "vibration2", "rms2"],
        "Current": ["current", "amp", "amps", "amperage"],
        "Pressure": ["pressure", "press"],
        "Temperature": ["temperature", "temp"],
        "Thermocouple": ["thermocouple", "thermo"],
        "Voltage": ["voltage", "volt"],
        "Volume Flow RateRMS": ["volumeflow", "flowrate", "flow", "volumeflowraterms"],
    }
    used: set[str] = set()
    for skab, keys in aliases.items():
        for key in keys:
            for nk, orig in norm.items():
                if key in nk and orig not in used:
                    hints[skab] = orig
                    used.add(orig)
                    break
            if skab in hints:
                break
    return hints
