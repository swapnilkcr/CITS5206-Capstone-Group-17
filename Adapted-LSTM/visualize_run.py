#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # stable column order: epoch first, then rest sorted
    keys = list(rows[0].keys())
    if "epoch" in keys:
        keys = ["epoch"] + [k for k in keys if k != "epoch"]
    else:
        keys = sorted(keys)
    # add any missing keys across rows
    all_keys = set(keys)
    for r in rows[1:]:
        all_keys.update(r.keys())
    keys = [k for k in keys if k in all_keys] + sorted([k for k in all_keys if k not in keys])

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _try_plot_png(out_dir: Path, train_hist: list[dict[str, Any]], metrics_hist: list[dict[str, Any]]) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    out_dir.mkdir(parents=True, exist_ok=True)

    def series(rows: list[dict[str, Any]], key: str) -> list[float]:
        return [float(r[key]) for r in rows]

    epochs = series(train_hist, "epoch")

    # ---- loss curves ----
    plt.figure(figsize=(10, 5))
    for k in ["loss", "svdd", "anom", "contrast"]:
        if k in train_hist[0]:
            plt.plot(epochs, series(train_hist, k), label=k)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training loss curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=150)
    plt.close()

    # ---- threshold ----
    mepochs = series(metrics_hist, "epoch")
    if "thr" in metrics_hist[0]:
        plt.figure(figsize=(10, 4))
        plt.plot(mepochs, series(metrics_hist, "thr"))
        plt.xlabel("epoch")
        plt.ylabel("threshold")
        plt.title("Threshold per epoch")
        plt.tight_layout()
        plt.savefig(out_dir / "threshold.png", dpi=150)
        plt.close()

    # ---- metrics ----
    for metric in ["f1", "precision", "recall", "fpr"]:
        keys = [f"val_{metric}", f"test_{metric}", f"valve2_{metric}"]
        if any(k in metrics_hist[0] for k in keys):
            plt.figure(figsize=(10, 5))
            for k in keys:
                if k in metrics_hist[0]:
                    plt.plot(mepochs, series(metrics_hist, k), label=k)
            plt.xlabel("epoch")
            plt.ylabel(metric)
            plt.title(f"{metric.upper()} curves")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"{metric}_curves.png", dpi=150)
            plt.close()

    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Visualize an Adapted-LSTM training run")
    ap.add_argument(
        "--run-dir",
        required=True,
        help="Path to outputs/<run_id> directory (contains train_history.json etc.)",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    train_path = run_dir / "train_history.json"
    metrics_path = run_dir / "metrics_history.json"
    summary_path = run_dir / "summary.json"

    if not train_path.exists() or not metrics_path.exists():
        raise SystemExit(f"Missing history files in {run_dir}. Need train_history.json and metrics_history.json.")

    train_hist: list[dict[str, Any]] = _load_json(train_path)
    metrics_hist: list[dict[str, Any]] = _load_json(metrics_path)
    summary: dict[str, Any] = _load_json(summary_path) if summary_path.exists() else {}

    plots_dir = run_dir / "plots"
    _write_csv(plots_dir / "train_history.csv", train_hist)
    _write_csv(plots_dir / "metrics_history.csv", metrics_hist)

    best_epoch = summary.get("best_epoch", None)
    best_val_f1 = summary.get("best_val_f1", None)
    q = summary.get("threshold_quantile", None)

    print(f"Run: {run_dir}")
    if best_epoch is not None:
        print(f"- best_epoch: {best_epoch}")
    if best_val_f1 is not None:
        print(f"- best_val_f1: {best_val_f1}")
    if q is not None:
        print(f"- threshold_quantile: {q}")
    print(f"Wrote CSVs to: {plots_dir}")

    ok_png = _try_plot_png(plots_dir, train_hist, metrics_hist)
    if ok_png:
        print(f"Wrote PNG plots to: {plots_dir}")
        return 0

    print(
        "PNG plots skipped (matplotlib not installed). To enable, run:\n"
        "  python -m pip install matplotlib\n"
        "then rerun this script."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

