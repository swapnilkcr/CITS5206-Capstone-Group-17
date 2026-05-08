#!/usr/bin/env python3
from __future__ import annotations

import csv
import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from config import Config
from skab_dataset import SKABSeries, concat_by_time, read_skab_csv


@dataclass(frozen=True)
class SplitSpec:
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15

    def __post_init__(self) -> None:
        if not (0 < self.train_frac < 1 and 0 < self.val_frac < 1 and 0 < self.test_frac < 1):
            raise ValueError("Split fractions must be in (0, 1)")
        s = self.train_frac + self.val_frac + self.test_frac
        if abs(s - 1.0) > 1e-9:
            raise ValueError(f"Split fractions must sum to 1.0 (got {s})")


def _ensure_dirs(base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)


def _write_series_csv(
    out_path: Path,
    series: SKABSeries,
    *,
    include_changepoint: bool,
    delimiter: str = ";",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [series.time_col, *series.feature_cols, series.label_col]
    if include_changepoint:
        fieldnames.append("changepoint")

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        w.writeheader()
        for ti, xi, yi in zip(series.t, series.x, series.y):
            row: dict[str, str] = {series.time_col: ti.strftime("%Y-%m-%d %H:%M:%S")}
            for col, val in zip(series.feature_cols, xi):
                row[col] = f"{val:.10g}"
            row[series.label_col] = f"{float(yi):.10g}"
            if include_changepoint:
                row["changepoint"] = "0"
            w.writerow(row)


def _slice_by_time(series: SKABSeries, start: int, end: int) -> SKABSeries:
    return SKABSeries(
        x=series.x[start:end],
        y=series.y[start:end],
        t=series.t[start:end],
        feature_cols=series.feature_cols,
        label_col=series.label_col,
        time_col=series.time_col,
    )


def split_series_timewise(series: SKABSeries, spec: SplitSpec) -> dict[str, SKABSeries]:
    n = len(series.t)
    if n < 10:
        raise ValueError(f"Series too small to split (n={n})")

    n_train = int(n * spec.train_frac)
    n_val = int(n * spec.val_frac)
    n_test = n - n_train - n_val
    if n_test <= 0:
        raise ValueError(f"Bad split resulting in empty test (n={n}, spec={spec})")

    train = _slice_by_time(series, 0, n_train)
    val = _slice_by_time(series, n_train, n_train + n_val)
    test = _slice_by_time(series, n_train + n_val, n)
    return {"train": train, "val": val, "test": test}


def _make_anomaly_free_like(series: SKABSeries, *, cfg: Config) -> SKABSeries:
    # anomaly-free.csv has no label column; read_skab_csv will fail if label missing.
    # We create a "label" vector of zeros after reading features/time.
    return SKABSeries(
        x=series.x,
        y=[0.0 for _ in series.x],
        t=series.t,
        feature_cols=list(cfg.feature_cols),
        label_col=cfg.label_col,
        time_col=cfg.time_col,
    )


def read_anomaly_free_csv(path: str | Path, *, cfg: Config) -> SKABSeries:
    path = Path(path)
    # Reuse read_skab_csv by temporarily treating label_col as a feature-less read,
    # then overwrite labels. We'll do a small custom read here to avoid assumptions.
    x: list[list[float]] = []
    t: list[dt.datetime] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        missing = [c for c in [cfg.time_col, *cfg.feature_cols] if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns: {missing}. Found: {reader.fieldnames}")
        for row in reader:
            t.append(dt.datetime.strptime(row[cfg.time_col], "%Y-%m-%d %H:%M:%S"))
            x.append([float(row[c]) for c in cfg.feature_cols])

    return SKABSeries(
        x=x,
        y=[0.0 for _ in x],
        t=t,
        feature_cols=list(cfg.feature_cols),
        label_col=cfg.label_col,
        time_col=cfg.time_col,
    )


def _collect_csvs(dir_path: Path) -> list[Path]:
    return sorted([p for p in dir_path.glob("*.csv") if p.is_file()])


def prepare(
    *,
    repo_root: Path,
    split: SplitSpec,
) -> dict[str, str]:
    cfg = Config()

    raw_data = repo_root / "data"
    # Intentionally write into Adapted-LSTM/dataset/ (keep raw data/ untouched).
    out_base = repo_root / "Adapted-LSTM" / "dataset"
    _ensure_dirs(out_base)

    # ----------------
    # anomaly-free
    # ----------------
    anomaly_free_raw = raw_data / "anomaly-free" / "anomaly-free.csv"
    anomaly_free = read_anomaly_free_csv(anomaly_free_raw, cfg=cfg)
    _write_series_csv(out_base / "anomaly_free.csv", anomaly_free, include_changepoint=True)

    # ----------------
    # valve1 (concat 16 files)
    # ----------------
    valve1_dir = raw_data / "valve1"
    valve1_parts = [
        read_skab_csv(p, feature_cols=cfg.feature_cols, label_col=cfg.label_col, time_col=cfg.time_col)
        for p in _collect_csvs(valve1_dir)
    ]
    valve1 = concat_by_time(valve1_parts)
    _write_series_csv(out_base / "valve1_all.csv", valve1, include_changepoint=True)
    v1_splits = split_series_timewise(valve1, split)
    for k, s in v1_splits.items():
        _write_series_csv(out_base / f"valve1_{k}.csv", s, include_changepoint=True)

    # ----------------
    # valve2 (concat 4 files)
    # ----------------
    valve2_dir = raw_data / "valve2"
    valve2_parts = [
        read_skab_csv(p, feature_cols=cfg.feature_cols, label_col=cfg.label_col, time_col=cfg.time_col)
        for p in _collect_csvs(valve2_dir)
    ]
    valve2 = concat_by_time(valve2_parts)
    _write_series_csv(out_base / "valve2_all.csv", valve2, include_changepoint=True)
    # per our protocol: valve2 is test-only (domain shift)
    _write_series_csv(out_base / "valve2_test.csv", valve2, include_changepoint=True)

    manifest = {
        "out_dir": str(out_base.relative_to(repo_root)),
        "split": asdict(split),
        "files": {
            "anomaly_free": "anomaly_free.csv",
            "valve1_all": "valve1_all.csv",
            "valve1_train": "valve1_train.csv",
            "valve1_val": "valve1_val.csv",
            "valve1_test": "valve1_test.csv",
            "valve2_all": "valve2_all.csv",
            "valve2_test": "valve2_test.csv",
        },
        "columns": {
            "time_col": cfg.time_col,
            "feature_cols": list(cfg.feature_cols),
            "label_col": cfg.label_col,
            "delimiter": ";",
        },
    }

    manifest_path = out_base / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"manifest": str(manifest_path)}


def main() -> int:
    # repo_root is the parent of Adapted-LSTM/
    repo_root = Path(__file__).resolve().parent.parent
    split = SplitSpec()
    out = prepare(repo_root=repo_root, split=split)
    print(f"Wrote dataset manifest: {out['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

