"""
M3 ablation: Mahalanobis with vs without mean subtraction.

Goal
----
The paper's §III.E uses an anomaly score of the form

    s_i = e_i^T (Σ + εI)^{-1} e_i

(see industrial_ad.scoring.MahalanobisScorer), justified textually by «нулевая
ошибка — естественная reference точка идеальной реконструкции/прогноза».

This script compares it numerically with the classical Mahalanobis
distance with mean subtraction:

    s_i^{(μ)} = (e_i - μ)^T (Σ + εI)^{-1} (e_i - μ)

where μ = E_normal[e_i] is estimated on normal validation windows alongside Σ.

This script runs that ablation on a handful of representative MLP
reconstruction models. It uses the SAME pipeline as the main paper
(industrial_ad.experiments / scoring / dataset) and only swaps the score
estimator. Existing FP32 checkpoints are loaded if available; otherwise a
small subset of MLP-reconstruction configs is re-trained from the runner
template in ``runners/MLP-reconstruction.py``.

Usage on Home Cluster (preferred — fast 5080 path)
--------------------------------------------------

Files are transferred to the cluster via rsync (no git):

    ssh homecluster 'mkdir -p ~/projects/industrial_ad'
    rsync -avz --delete \
        --exclude '__pycache__' --exclude '.git' --exclude 'runs/' --exclude 'data/' \
        "<local>/industrial_ad-main/" homecluster:~/projects/industrial_ad/

Then on the server::

    ssh homecluster
    cd ~/projects/industrial_ad
    source .venv/bin/activate           # PyTorch 2.11+cu128 already installed
    export PYTHONPATH=$PWD
    python scripts/m3_mahalanobis_ablation.py \
        --runs-root ./runs/MLP-reconstruction \
        --output ./m3_ablation_result.csv

If no trained checkpoints exist under ``./runs/MLP-reconstruction/``, train
the first 15 sweep elements (5 configs × 3 seeds) first::

    python runners/MLP-reconstruction.py 0 15
    # ~30 min on RTX 5080 (210 epochs, batch=1024, FP32 CUDA)

Expected output
---------------
A CSV with one row per (config, seed, scorer_variant) — columns:

    family, run_name, seed, scorer_variant,
    val_roc_auc, test_roc_auc, mu_norm_l2, n_normal_val_windows

…followed by a printed summary table showing per-config medians and a
paired-test (Wilcoxon signed-rank) on test ROC-AUC between the two
variants. If the paired test fails to reject H0 (p > 0.05) for the
majority of configs, that supports the paper's choice of mean-less
Mahalanobis. If it rejects with WITH-MEAN winning, we need to switch the
paper to mean-subtracted Mahalanobis (and most likely re-run all 3717
runs).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

# Import the project's own pipeline so the ablation is a drop-in replacement
# at the scorer level only (everything else — data, model, error reducer —
# is identical to the main experiment).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from industrial_ad.config import make_default_config            # noqa: E402
from industrial_ad.datasets.PU.dataset import build_pu_dataloaders  # noqa: E402
from industrial_ad.experiments import load_detector_from_run    # noqa: E402
from industrial_ad.scoring import (                              # noqa: E402
    AnomalyDetectorWrapper,
    MahalanobisScorer,
    build_error_reducer,
)
from sklearn.metrics import roc_auc_score                        # noqa: E402


# ---------------------------------------------------------------------------
# Two scorer variants — only the score_estimator changes.
# ---------------------------------------------------------------------------


class MahalanobisScorerWithMean(nn.Module):
    """Classical Mahalanobis: subtract the mean of normal errors before quadratic form.

    fit() stores both μ and Σ⁻¹; forward() applies (e - μ)^T Σ⁻¹ (e - μ).
    """

    def __init__(self, feature_dim: int, eps: float) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(self.feature_dim))
        self.register_buffer("inv_cov", torch.eye(self.feature_dim))

    @torch.no_grad()
    def fit(self, normal_features: torch.Tensor) -> None:
        if normal_features.shape[0] < 2:
            self.mean.zero_()
            self.inv_cov.copy_(torch.eye(self.feature_dim, device=normal_features.device))
            return
        mu = normal_features.mean(dim=0)
        centered = normal_features - mu
        cov = centered.T @ centered / (centered.shape[0] - 1)
        cov = cov + torch.eye(self.feature_dim, device=normal_features.device) * self.eps
        self.mean.copy_(mu)
        self.inv_cov.copy_(torch.linalg.pinv(cov))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        centered = features - self.mean
        return (centered @ self.inv_cov * centered).sum(dim=1)


# ---------------------------------------------------------------------------
# Per-run evaluation
# ---------------------------------------------------------------------------


def evaluate_one_run(run_dir: Path, device: torch.device) -> list[dict[str, Any]]:
    """Evaluate one trained run with both scorer variants; return one row per variant."""
    # load_detector_from_run returns a TUPLE (detector, config); unpack it.
    detector, config = load_detector_from_run(run_dir)
    detector = detector.to(device).eval()

    bundle = build_pu_dataloaders(config)
    val_loader = bundle["loaders"]["val"]
    test_loader = bundle["loaders"]["test"]

    error_reducer = build_error_reducer(config["scoring"]["error_reducer"])

    # 1) Compute error features on normal val windows once and reuse.
    normal_features = []
    val_features = []
    val_labels = []
    with torch.no_grad():
        for x, y, is_anomaly in val_loader:
            x = x.to(device)
            y = y.to(device)
            pred = detector(x)
            feats = error_reducer(pred, y)
            val_features.append(feats.cpu())
            val_labels.append(is_anomaly.cpu())
            normal_mask = ~is_anomaly.bool()
            if normal_mask.any():
                normal_features.append(feats[normal_mask].cpu())
    normal_features = torch.cat(normal_features, dim=0)
    val_features = torch.cat(val_features, dim=0)
    val_labels = torch.cat(val_labels, dim=0).numpy().astype(int)

    # 2) Same for test (we just need ROC-AUC, threshold not relevant).
    test_features = []
    test_labels = []
    with torch.no_grad():
        for x, y, is_anomaly in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = detector(x)
            feats = error_reducer(pred, y)
            test_features.append(feats.cpu())
            test_labels.append(is_anomaly.cpu())
    test_features = torch.cat(test_features, dim=0)
    test_labels = torch.cat(test_labels, dim=0).numpy().astype(int)

    feature_dim = normal_features.shape[1]
    eps = float(config["scoring"]["score_estimator"]["params"]["eps"])

    rows = []
    for variant, scorer_cls in [
        ("no_mean", MahalanobisScorer),
        ("with_mean", MahalanobisScorerWithMean),
    ]:
        scorer = scorer_cls(feature_dim=feature_dim, eps=eps)
        scorer.fit(normal_features)
        with torch.no_grad():
            val_scores = scorer(val_features).numpy()
            test_scores = scorer(test_features).numpy()
        val_auc = float(roc_auc_score(val_labels, val_scores))
        test_auc = float(roc_auc_score(test_labels, test_scores))
        mu_l2 = float(normal_features.mean(dim=0).norm().item())
        rows.append({
            "run_name": run_dir.name,
            "scorer_variant": variant,
            "val_roc_auc": val_auc,
            "test_roc_auc": test_auc,
            "mu_norm_l2": mu_l2,
            "n_normal_val_windows": int(normal_features.shape[0]),
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="M3 ablation: Mahalanobis with vs without mean")
    parser.add_argument("--runs-root", type=Path, required=True,
                        help="Directory containing trained MLP-reconstruction runs (each with config.json + checkpoints/best.pt)")
    parser.add_argument("--n-configs", type=int, default=5,
                        help="Number of distinct configs to sample (default: 5)")
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44],
                        help="Seeds to include from each config (default: 42 43 44)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    args = parser.parse_args()

    device = torch.device(args.device)
    runs_root = args.runs_root.resolve()
    if not runs_root.is_dir():
        print(f"ERROR: runs root not found: {runs_root}", file=sys.stderr)
        return 1

    # Discover all subdirs that look like completed runs.
    candidate_runs = sorted(
        d for d in runs_root.iterdir()
        if d.is_dir() and (d / "config.json").exists() and (d / "checkpoints" / "best.pt").exists()
    )
    if not candidate_runs:
        print(f"ERROR: no completed runs found under {runs_root}", file=sys.stderr)
        print("Re-train a subset of MLP-reconstruction first:", file=sys.stderr)
        print(f"  cd {runs_root.parent.parent.parent}  # repo root", file=sys.stderr)
        print("  python runners/MLP-reconstruction.py 0 15   # first 5 configs × 3 seeds", file=sys.stderr)
        return 1

    # Group by config (run_name without `-s<seed>-` part) and keep N distinct configs.
    by_config: dict[str, list[Path]] = {}
    for run_dir in candidate_runs:
        # MLP runner naming: "001-s42-dims-3240-1728" → config_id = "001-dims-3240-1728"
        parts = run_dir.name.split("-")
        config_id = parts[0] + "-" + "-".join(p for p in parts[2:] if not p.startswith("s"))
        by_config.setdefault(config_id, []).append(run_dir)

    selected = []
    for config_id, runs in list(by_config.items())[: args.n_configs]:
        for run_dir in runs:
            seed_part = next((p for p in run_dir.name.split("-") if p.startswith("s")), "s?")
            try:
                seed = int(seed_part[1:])
            except ValueError:
                continue
            if seed in args.seeds:
                selected.append((config_id, seed, run_dir))

    if not selected:
        print("ERROR: no runs matched the requested (n_configs × seeds) selection", file=sys.stderr)
        return 1

    print(f"Evaluating {len(selected)} runs on {device}")

    all_rows: list[dict[str, Any]] = []
    for config_id, seed, run_dir in selected:
        print(f"  {run_dir.name} ...", flush=True)
        try:
            for row in evaluate_one_run(run_dir, device):
                row["family"] = "MLP-reconstruction"
                row["config_id"] = config_id
                row["seed"] = seed
                all_rows.append(row)
        except Exception as exc:  # pragma: no cover
            print(f"    FAILED: {exc}", file=sys.stderr)

    df = pd.DataFrame(all_rows)
    df = df[[
        "family", "config_id", "run_name", "seed", "scorer_variant",
        "val_roc_auc", "test_roc_auc", "mu_norm_l2", "n_normal_val_windows",
    ]]
    df.to_csv(args.output, index=False)
    print(f"Wrote: {args.output}")

    # Summary: paired test ROC-AUC, no_mean vs with_mean, per config
    print()
    print("=== Per-config summary (mean over seeds) ===")
    pivot = df.groupby(["config_id", "scorer_variant"])[["test_roc_auc", "val_roc_auc", "mu_norm_l2"]].mean().unstack("scorer_variant")
    print(pivot.to_string())

    print()
    print("=== Paired Wilcoxon (test ROC-AUC, no_mean − with_mean across all (config, seed)) ===")
    paired = df.pivot_table(index=["config_id", "seed"], columns="scorer_variant", values="test_roc_auc")
    if {"no_mean", "with_mean"} <= set(paired.columns):
        deltas = (paired["no_mean"] - paired["with_mean"]).dropna()
        print(f"n_pairs = {len(deltas)}")
        print(f"median Δ (no_mean − with_mean) = {deltas.median():+.4f}")
        print(f"mean Δ = {deltas.mean():+.4f}")
        wins_no_mean = int((deltas > 0).sum())
        wins_with_mean = int((deltas < 0).sum())
        ties = int((deltas == 0).sum())
        print(f"wins: no_mean = {wins_no_mean}, with_mean = {wins_with_mean}, ties = {ties}")
        try:
            from scipy.stats import wilcoxon
            stat, pvalue = wilcoxon(deltas)
            print(f"Wilcoxon signed-rank: statistic = {stat:.4f}, p-value = {pvalue:.4g}")
            print()
            if pvalue > 0.05:
                print("INTERPRETATION: p > 0.05 — no significant difference between scorers.")
                print("  → paper's choice of mean-less Mahalanobis is supported by data.")
                print("  → add this ablation to §III.E as Table VI / Appendix.")
            elif deltas.median() > 0:
                print("INTERPRETATION: no_mean wins by a significant margin.")
                print("  → paper's choice is empirically optimal. Strong support for §III.E claim.")
            else:
                print("INTERPRETATION: with_mean wins by a significant margin.")
                print("  → paper's claim is contradicted by data — REVISIT scoring choice")
                print("  → likely need to re-run all 3717 experiments with mean-subtracted variant.")
        except ImportError:
            print("(scipy not installed; skipping Wilcoxon test)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
