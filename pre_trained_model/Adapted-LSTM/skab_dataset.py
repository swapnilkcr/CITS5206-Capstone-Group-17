from __future__ import annotations

try:
    import torch  # type: ignore
    from torch.utils.data import Dataset  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    Dataset = object  # type: ignore

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


class WindowDataset(Dataset):
    def __init__(self, data, labels, window_size, stride):
        if torch is None:
            raise RuntimeError("WindowDataset requires torch, but torch is not installed in this environment.")
        self.x, self.y = [], []

        # Include the last possible full window.
        for i in range(0, len(data) - window_size + 1, stride):
            x = data[i : i + window_size]
            y = labels[i : i + window_size]

            self.x.append(torch.tensor(x, dtype=torch.float32))
            self.y.append(1 if max(y) > 0 else 0)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], torch.tensor(self.y[idx])  # type: ignore[attr-defined]


@dataclass(frozen=True)
class SKABSeries:
    """
    A single concatenated time series.
    - x: (N, D) float features
    - y: (N,) float labels (0/1) from anomaly column
    - t: (N,) python datetimes
    """

    x: list[list[float]]
    y: list[float]
    t: list[dt.datetime]
    feature_cols: list[str]
    label_col: str
    time_col: str


def _parse_dt(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def read_skab_csv(
    path: str | Path,
    *,
    feature_cols: Sequence[str],
    label_col: str = "anomaly",
    time_col: str = "datetime",
    delimiter: str = ";",
) -> SKABSeries:
    path = Path(path)
    x: list[list[float]] = []
    y: list[float] = []
    t: list[dt.datetime] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        missing = [c for c in [time_col, *feature_cols, label_col] if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns: {missing}. Found: {reader.fieldnames}")

        for row in reader:
            t.append(_parse_dt(row[time_col]))
            x.append([float(row[c]) for c in feature_cols])
            y.append(float(row[label_col]))

    return SKABSeries(
        x=x,
        y=y,
        t=t,
        feature_cols=list(feature_cols),
        label_col=label_col,
        time_col=time_col,
    )


def concat_by_time(parts: Iterable[SKABSeries]) -> SKABSeries:
    parts = list(parts)
    if not parts:
        raise ValueError("concat_by_time: no parts provided")

    feature_cols = parts[0].feature_cols
    label_col = parts[0].label_col
    time_col = parts[0].time_col

    for p in parts[1:]:
        if p.feature_cols != feature_cols:
            raise ValueError("concat_by_time: feature_cols mismatch across parts")
        if p.label_col != label_col or p.time_col != time_col:
            raise ValueError("concat_by_time: label/time column mismatch across parts")

    rows = []
    for p in parts:
        rows.extend(zip(p.t, p.x, p.y))

    rows.sort(key=lambda r: r[0])

    t: list[dt.datetime] = []
    x: list[list[float]] = []
    y: list[float] = []

    last_t: dt.datetime | None = None
    for ti, xi, yi in rows:
        if last_t is not None and ti == last_t:
            t[-1] = ti
            x[-1] = xi
            y[-1] = yi
        else:
            t.append(ti)
            x.append(xi)
            y.append(yi)
            last_t = ti

    return SKABSeries(
        x=x, y=y, t=t, feature_cols=list(feature_cols), label_col=label_col, time_col=time_col
    )


def make_window_dataset(
    series: SKABSeries,
    *,
    window_size: int,
    stride: int,
) -> WindowDataset:
    return WindowDataset(series.x, series.y, window_size, stride)

