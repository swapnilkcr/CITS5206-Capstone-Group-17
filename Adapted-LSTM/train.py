from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import Config
from model import ALSS_SVDD_CR
from utils import svdd_loss, anomaly_loss, contrastive_loss
from skab_dataset import make_window_dataset, read_skab_csv

cfg = Config()

# Always anchor paths at Adapted-LSTM/ (script directory),
# so `cd Adapted-LSTM && python train.py` works as expected.
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
OUTPUTS_DIR = PROJECT_DIR / "outputs"


def _new_run_dir(prefix: str = "run") -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_dir = OUTPUTS_DIR / f"{prefix}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _binary_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
    y_true = y_true.to(dtype=torch.long)
    y_pred = y_pred.to(dtype=torch.long)
    tp = int(((y_true == 1) & (y_pred == 1)).sum().item())
    tn = int(((y_true == 0) & (y_pred == 0)).sum().item())
    fp = int(((y_true == 0) & (y_pred == 1)).sum().item())
    fn = int(((y_true == 1) & (y_pred == 0)).sum().item())

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    fpr = fp / (fp + tn + 1e-12)

    return {
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
    }


@torch.no_grad()
def _scores_and_labels(model: ALSS_SVDD_CR, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    scores = []
    ys = []
    for x, y in loader:
        x = x.float()
        y = y.to(dtype=torch.long)
        z = model(x)
        s = ((z - model.center) ** 2).sum(dim=1)
        scores.append(s.detach().cpu())
        ys.append(y.detach().cpu())
    return torch.cat(scores, dim=0), torch.cat(ys, dim=0)


def _threshold_from_val_normal(scores: torch.Tensor, y: torch.Tensor, q: float) -> float:
    normal_scores = scores[y == 0]
    if normal_scores.numel() == 0:
        raise RuntimeError("No normal windows found in validation split; cannot compute threshold.")
    thr = torch.quantile(normal_scores, q).item()
    return float(thr)

# ---------------------------
# init center (only normal)
# ---------------------------
def init_center(model: ALSS_SVDD_CR, loader: DataLoader) -> torch.Tensor:
    zs = []
    for x, y in loader:
        x = x.float()
        # Only use normal windows to estimate the SVDD center.
        y = y.to(dtype=torch.long)
        x = x[y == 0]
        if x.numel() == 0:
            continue
        z = model(x)
        zs.append(z.detach())

    return torch.cat(zs).mean(dim=0)


# ---------------------------
# training loop
# ---------------------------
def train(
    *,
    model: ALSS_SVDD_CR,
    loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    valve2_loader: DataLoader,
    run_dir: Path,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.center = init_center(model, loader)

    history: list[dict[str, float]] = []
    metrics_history: list[dict[str, float]] = []
    best_f1 = -1.0
    best_epoch = -1
    no_improve = 0

    for epoch in range(cfg.epochs):
        total_loss = 0.0
        total_l1 = 0.0
        total_l2 = 0.0
        total_l3 = 0.0
        n_batches = 0

        model.train()
        for x, y in loader:
            x = x.float()
            y = y.to(dtype=torch.long)  # window-level 0/1

            z = model(x)

            z_normal = z[y == 0]
            z_anom = z[y == 1]

            loss1 = svdd_loss(z_normal, model.center)
            loss2 = torch.tensor(0.0, device=loss1.device)
            if len(z_anom) > 0:
                loss2 = anomaly_loss(z_anom, model.center, cfg.margin)
            loss3 = contrastive_loss(z, y.float(), cfg.temperature)

            loss = loss1 + cfg.lambda_anomaly * loss2 + cfg.lambda_contrast * loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().item())
            total_l1 += float(loss1.detach().item())
            total_l2 += float(loss2.detach().item())
            total_l3 += float(loss3.detach().item())
            n_batches += 1

        denom = max(n_batches, 1)
        row = {
            "epoch": float(epoch),
            "loss": total_loss / denom,
            "svdd": total_l1 / denom,
            "anom": total_l2 / denom,
            "contrast": total_l3 / denom,
        }
        history.append(row)

        # ----- evaluation -----
        val_scores, val_y = _scores_and_labels(model, val_loader)
        thr = _threshold_from_val_normal(val_scores, val_y, cfg.threshold_quantile)
        val_pred = (val_scores > thr).to(dtype=torch.long)
        val_m = _binary_metrics(val_y, val_pred)

        test_scores, test_y = _scores_and_labels(model, test_loader)
        test_pred = (test_scores > thr).to(dtype=torch.long)
        test_m = _binary_metrics(test_y, test_pred)

        v2_scores, v2_y = _scores_and_labels(model, valve2_loader)
        v2_pred = (v2_scores > thr).to(dtype=torch.long)
        v2_m = _binary_metrics(v2_y, v2_pred)

        mrow: dict[str, float] = {
            "epoch": float(epoch),
            "thr": float(thr),
            "val_f1": float(val_m["f1"]),
            "val_fpr": float(val_m["fpr"]),
            "val_precision": float(val_m["precision"]),
            "val_recall": float(val_m["recall"]),
            "test_f1": float(test_m["f1"]),
            "test_fpr": float(test_m["fpr"]),
            "test_precision": float(test_m["precision"]),
            "test_recall": float(test_m["recall"]),
            "valve2_f1": float(v2_m["f1"]),
            "valve2_fpr": float(v2_m["fpr"]),
            "valve2_precision": float(v2_m["precision"]),
            "valve2_recall": float(v2_m["recall"]),
        }
        metrics_history.append(mrow)

        print(
            f"Epoch {epoch} | loss={row['loss']:.6f} | thr(q={cfg.threshold_quantile:.2f})={thr:.6g} "
            f"| val_f1={mrow['val_f1']:.3f} val_fpr={mrow['val_fpr']:.3f} "
            f"| test_f1={mrow['test_f1']:.3f} "
            f"| valve2_f1={mrow['valve2_f1']:.3f}"
        )

        # ----- early stop on valve1_val F1 -----
        if mrow["val_f1"] > best_f1 + 1e-6:
            best_f1 = float(mrow["val_f1"])
            best_epoch = int(epoch)
            no_improve = 0
            torch.save(
                {"state_dict": model.state_dict(), "center": model.center, "thr": thr, "best_epoch": best_epoch},
                run_dir / "best_model.pt",
            )
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop_patience:
                break

    (run_dir / "train_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (run_dir / "metrics_history.json").write_text(
        json.dumps(metrics_history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {"best_epoch": best_epoch, "best_val_f1": best_f1, "threshold_quantile": cfg.threshold_quantile},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def build_loader_from_dataset(split_name: str, *, shuffle: bool) -> DataLoader:
    path = DATASET_DIR / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {path}. Run: python Adapted-LSTM/prepare_dataset.py"
        )
    series = read_skab_csv(
        path,
        feature_cols=cfg.feature_cols,
        label_col=cfg.label_col,
        time_col=cfg.time_col,
        delimiter=";",
    )
    ds = make_window_dataset(series, window_size=cfg.window_size, stride=cfg.stride)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, drop_last=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-prefix", default="adapted_lstm", help="Outputs run directory prefix")
    args = ap.parse_args()

    run_dir = _new_run_dir(args.run_prefix)
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "splits": {
                    "train": "valve1_train",
                    "val": "valve1_val",
                    "test": "valve1_test",
                    "valve2_test": "valve2_test",
                },
                "dataset_dir": str(DATASET_DIR),
                "config": {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("__")},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    model = ALSS_SVDD_CR(cfg)
    train_loader = build_loader_from_dataset("valve1_train", shuffle=True)
    val_loader = build_loader_from_dataset("valve1_val", shuffle=False)
    test_loader = build_loader_from_dataset("valve1_test", shuffle=False)
    valve2_loader = build_loader_from_dataset("valve2_test", shuffle=False)
    train(
        model=model,
        loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        valve2_loader=valve2_loader,
        run_dir=run_dir,
    )

    torch.save(
        {"state_dict": model.state_dict(), "center": model.center},
        run_dir / "model.pt",
    )
    print(f"Wrote outputs to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())