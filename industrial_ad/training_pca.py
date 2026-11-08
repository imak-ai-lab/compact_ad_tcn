"""One-step PCA fitting with the same local artifacts as neural training."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from industrial_ad.training import _build_summary, _close_figures, _save_figures
from industrial_ad.utils import dump_json


@torch.no_grad()
def _fit_pca_model(model, train_loader, *, max_batches: int | None, seed: int) -> None:
    """Collect normal train windows, flatten them and fit the PCA basis once."""
    flat_windows = []
    for batch_index, (x, _, _) in enumerate(train_loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        flat_windows.append(x.reshape(x.shape[0], -1).cpu())

    if not flat_windows:
        raise RuntimeError("No training samples were available to fit PCA.")
    model.fit(torch.cat(flat_windows, dim=0), seed=seed)


def train_pca_anomaly_detector(
    detector,
    train_loader,
    val_loader,
    test_loader,
    trainer_config: dict[str, Any],
    debug_trainer_config: dict[str, Any],
    save_dir: str | Path,
    *,
    config_snapshot: dict[str, Any] | None = None,
    save_last: bool = False,
):
    """Fit PCA once, then reuse the standard detector evaluation and artifact pipeline."""
    save_dir = Path(save_dir)
    checkpoints_dir = save_dir / "checkpoints"
    plots_dir = save_dir / "plots"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    if config_snapshot is not None:
        dump_json(save_dir / "config.json", config_snapshot)

    max_train_batches = debug_trainer_config["max_train_batches"]
    max_eval_batches = debug_trainer_config["max_eval_batches"]

    history = {
        "train_epochs": [],
        "evaluations": [],
        "best_epoch": None,
        "best_metric_name": trainer_config["checkpoint"]["metric"],
        "best_metric_value": None,
        "stopped_early": False,
        "total_time_seconds": 0.0,
    }
    print(f"Fitting {save_dir.name}...")

    fit_start = time.time()
    detector.to("cpu")
    seed = int(config_snapshot["run"]["seed"]) if config_snapshot is not None else 0
    _fit_pca_model(detector.model, train_loader, max_batches=max_train_batches, seed=seed)
    fit_time = time.time() - fit_start

    train_record = {
        "epoch": 1,
        "loss": float("nan"),
        "grad_norm": float("nan"),
        "lr": float("nan"),
        "epoch_time_seconds": fit_time,
    }
    history["train_epochs"].append(train_record)
    history["total_time_seconds"] = fit_time

    detector.to(torch.device(trainer_config["device"]))
    eval_start = time.time()
    detector.fit_score_estimator(val_loader, max_batches=max_eval_batches)
    detector.fit_threshold(val_loader, max_batches=max_eval_batches)
    val_metrics, val_figures = detector.evaluate(val_loader, prefix="val", max_batches=max_eval_batches)
    test_metrics, test_figures = detector.evaluate(test_loader, prefix="test", max_batches=max_eval_batches)

    evaluation_record = {
        "epoch": 1,
        "metrics": {**val_metrics, **test_metrics},
        "is_best": True,
    }
    history["evaluations"].append(evaluation_record)
    history["best_epoch"] = 1
    history["best_metric_value"] = float(evaluation_record["metrics"][trainer_config["checkpoint"]["metric"]])
    history["total_time_seconds"] += time.time() - eval_start

    torch.save(detector.state_dict(), checkpoints_dir / "best.pt")
    if save_last:
        torch.save(detector.state_dict(), checkpoints_dir / "last.pt")

    _save_figures(val_figures, plots_dir, "val", 1)
    _save_figures(test_figures, plots_dir, "test", 1)
    dump_json(save_dir / "history.json", history)
    _close_figures(val_figures)
    _close_figures(test_figures)

    summary = _build_summary(history, trainer_config["checkpoint"]["metric"], evaluation_record)
    dump_json(save_dir / "history.json", history)
    dump_json(save_dir / "summary.json", summary)
    return history, summary
