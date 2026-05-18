#!/usr/bin/env python3
"""
Merge Zenodo NeWater Pump 1 CSVs into one wide raw dataset.

Place downloads in:  unseen_data/zenodo_newater_pump1_raw/
Writes:              unseen_data/processed/newater_pump1_merged.csv
                     unseen_data/processed/manifest.json

Usage:
    python scripts/prepare_newater_pump1.py
    python scripts/prepare_newater_pump1.py --resample 1min --raw-dir path/to/csvs
"""
from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = REPO_ROOT / "unseen_data" / "zenodo_newater_pump1_raw"
DEFAULT_OUT = REPO_ROOT / "unseen_data" / "processed"

# Exact Zenodo filenames → short column prefix (only files for Pump 1 bundle)
PUMP1_FILES: dict[str, str] = {
    "Current Sensor - NeWater Pump 1.csv": "current",
    "Power Sensor - NeWater Pump 1.csv": "power",
    "Energy Sensor - NeWater Pump 1.csv": "energy",
    "Vibration Sensor - NeWater Pump 1 Temperature.csv": "vib_temperature",
    "Vibration Sensor - NeWater Pump 1 X-Axis Speed.csv": "vib_x_speed",
    "Vibration Sensor - NeWater Pump 1 Y-Axis Speed.csv": "vib_y_speed",
    "Vibration Sensor - NeWater Pump 1 Z-Axis Speed.csv": "vib_z_speed",
    "Vibration Sensor - NeWater Pump 1 X-Axis Displacement.csv": "vib_x_disp",
    "Vibration Sensor - NeWater Pump 1 Y-Axis Displacement.csv": "vib_y_disp",
    "Vibration Sensor - NeWater Pump 1 Z-Axis Displacement.csv": "vib_z_disp",
    "Pressure Sensor - NeWater Incoming Pump 1.csv": "pressure_incoming_p1",
}

OPTIONAL_FILES: dict[str, str] = {
    "Pressure Sensor - NeWater Outgoing Pump.csv": "pressure_outgoing",
    "Water Level Sensor - NeWater Tank.csv": "tank_level",
}

TIME_CANDIDATES = (
    "datetime", "timestamp", "time", "Time", "date", "t", "recorded_at", "Recorded At",
)


def _normalize_datetime(series: pd.Series) -> pd.Series:
    """Parse to UTC; drop out-of-range values (avoids nanosecond overflow on bad rows)."""
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    lo = pd.Timestamp("2020-01-01", tz="UTC")
    hi = pd.Timestamp("2030-12-31", tz="UTC")
    return dt.where((dt >= lo) & (dt <= hi))
VALUE_SKIP = frozenset(TIME_CANDIDATES) | frozenset(
    c.lower() for c in TIME_CANDIDATES
)


def _strip_excel_sep_line(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip().lower().startswith("sep="):
        lines = lines[1:]
    return "\n".join(lines)


def _detect_sep(header_line: str) -> str:
    if ";" in header_line and header_line.count(";") >= header_line.count(","):
        return ";"
    if "\t" in header_line:
        return "\t"
    return ","


def _to_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    cleaned = series.astype(str).str.replace(r"[^\d.\-+eE]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _read_sensor_csv(path: Path) -> pd.DataFrame:
    text = _strip_excel_sep_line(path.read_text(encoding="utf-8-sig", errors="replace"))
    header_line = text.splitlines()[0] if text.splitlines() else ""
    sep = _detect_sep(header_line)
    df = pd.read_csv(io.StringIO(text), sep=sep, low_memory=False)
    df.columns = [str(c).strip().strip('"') for c in df.columns]

    time_col = next(
        (c for c in df.columns if c in TIME_CANDIDATES or c.lower() in VALUE_SKIP),
        df.columns[0],
    )
    df[time_col] = _normalize_datetime(df[time_col])
    df = df.dropna(subset=[time_col])
    # Zenodo pump data is 2023–2024; drop parse garbage
    df = df[(df[time_col] >= "2020-01-01") & (df[time_col] <= "2030-12-31")]

    value_cols = [c for c in df.columns if c != time_col]
    for c in value_cols:
        df[c] = _to_numeric_series(df[c])
    value_cols = [c for c in value_cols if df[c].notna().any()]

    if len(value_cols) == 1:
        out = df[[time_col, value_cols[0]]].copy()
    else:
        # Multiple numeric cols: keep all, prefixed later
        out = df[[time_col] + value_cols].copy()

    out = out.rename(columns={time_col: "datetime"})
    out["datetime"] = pd.DatetimeIndex(out["datetime"]).as_unit("ns")
    out = out.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
    return out


def _load_one(path: Path, prefix: str) -> pd.DataFrame:
    df = _read_sensor_csv(path)
    value_cols = [c for c in df.columns if c != "datetime"]
    if len(value_cols) == 1:
        return df.rename(columns={value_cols[0]: prefix})
    renamed = {c: f"{prefix}_{_slug(c)}" for c in value_cols}
    return df.rename(columns=renamed)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "value"


def merge_pump1(
    raw_dir: Path,
    *,
    resample: str | None = "1min",
    include_optional: bool = True,
) -> pd.DataFrame:
    files = dict(PUMP1_FILES)
    if include_optional:
        files.update(OPTIONAL_FILES)

    present: list[tuple[Path, str]] = []
    missing: list[str] = []
    for fname, prefix in files.items():
        p = raw_dir / fname
        if p.is_file():
            present.append((p, prefix))
        else:
            missing.append(fname)

    if not present:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. "
            f"Expected at least: {next(iter(PUMP1_FILES))}"
        )

    print(f"Found {len(present)} files, missing {len(missing)}:")
    for m in missing:
        print(f"  - (optional/missing) {m}")

    merged: pd.DataFrame | None = None
    for path, prefix in present:
        part = _load_one(path, prefix)
        if merged is None:
            merged = part
        else:
            merged = pd.merge_asof(
                merged.sort_values("datetime"),
                part.sort_values("datetime"),
                on="datetime",
                direction="nearest",
                tolerance=pd.Timedelta("2min"),
            )

    assert merged is not None
    merged = merged.dropna(subset=["datetime"]).reset_index(drop=True)

    # Rolling RMS proxies for SKAB-like accelerometer channels
    for axis in ("x", "y", "z"):
        col = f"vib_{axis}_speed"
        if col in merged.columns:
            merged[f"vib_{axis}_speed_rms"] = (
                merged[col]
                .rolling(60, min_periods=10)
                .apply(lambda s: float(np.sqrt((s.astype(float) ** 2).mean())), raw=False)
            )

    if resample:
        merged = merged.set_index("datetime")
        num = merged.select_dtypes(include="number").columns
        merged = merged[num].resample(resample).mean()
        merged = merged.reset_index()

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge NeWater Pump 1 Zenodo CSVs")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resample", default="1min", help="Pandas offset alias, or 'none'")
    parser.add_argument("--no-optional", action="store_true")
    args = parser.parse_args()

    resample = None if str(args.resample).lower() in ("none", "") else args.resample
    args.out_dir.mkdir(parents=True, exist_ok=True)

    merged = merge_pump1(
        args.raw_dir,
        resample=resample,
        include_optional=not args.no_optional,
    )

    out_csv = args.out_dir / "newater_pump1_merged.csv"
    merged.to_csv(out_csv, index=False)

    manifest = {
        "source": "https://zenodo.org/records/13808085",
        "pump": "NeWater Pump 1",
        "raw_dir": str(args.raw_dir),
        "output": str(out_csv),
        "rows": len(merged),
        "columns": merged.columns.tolist(),
        "resample": resample,
        "required_files": list(PUMP1_FILES.keys()),
        "optional_files": list(OPTIONAL_FILES.keys()),
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {out_csv} ({len(merged):,} rows, {len(merged.columns)} columns)")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
