#!/usr/bin/env python3
"""Why does a covariance scorer give TranAD 0.92 for one person and 0.77 for us?

Context. A previous in-house TranAD implementation (code lost) reported ~0.92
ROC-AUC on this data with a covariance-based anomaly score, and ~0.68 with the
author's own score. Our runs reproduce the second number exactly (TA001 0.7056,
TA010 0.6617 with the author's windowing), so the architecture and the training
recipe are confirmed by an independent implementation. The entire 0.68 -> 0.92
gap therefore lives in the SCORING path — which is this paper's own central
claim, now corroborated from outside.

What is left to explain is our 0.77 against their 0.92 under "a covariance
scorer". A covariance scorer is not one thing; this script enumerates the
plausible readings of that phrase and reports what each one produces, using the
checkpoints we already have. No retraining.

Axes swept (all post-hoc, on the same error maps):
  * error map      — the author's own `anomaly_error`, |x_hat - x|, (x_hat - x)^2
  * time reduction — mean / max / std over the predicted frames
  * scorer         — Mahalanobis without mean (this paper), classic Mahalanobis
                     with mean, diagonal (per-channel z), plain mean (author-style)
  * covariance fit — normal validation windows only (this paper), or all of them
  * evaluation unit— per window (this paper, 23 076 test windows) or per recording
                     (396 test files), aggregating window scores by mean or max

The per-recording unit is the leading hypothesis for the missing 0.15: averaging
30-340 window scores per file removes noise, and "is this bearing damaged?" is a
per-recording question, which is what a student would naturally implement.

    python3.11 scripts/phase1/scoring_sweep.py --runs runs_phase1/TranAD/017-s42-author-w10-slide
    python3.11 scripts/phase1/scoring_sweep.py --codes TA001,TA010,TA011 --seeds 42

Output: scoring_sweep.csv next to the run tree, plus a printed table sorted by
test ROC-AUC, with the paper's own variant marked so the delta is readable.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def _ensure_deps() -> None:
    need = {"pywt": "PyWavelets", "pandas": "pandas", "sklearn": "scikit-learn",
            "scipy": "scipy", "pyarrow": "pyarrow", "tqdm": "tqdm", "matplotlib": "matplotlib"}
    missing = [pkg for mod, pkg in need.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print(f"[sweep] installing: {missing}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "--no-input", *missing])


_ensure_deps()

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from industrial_ad.datasets.PU.dataset import (  # noqa: E402
    TimeSeriesDataset, discover_file_paths,
)
from industrial_ad.datasets.PU.features import build_pu_feature_pipeline, build_target_builder  # noqa: E402
from industrial_ad.experiments import load_detector_from_run  # noqa: E402

EPS = 1e-5


# --------------------------------------------------------------------------- #
# 1. error maps, extracted once per run
# --------------------------------------------------------------------------- #
def extract_features(run_dir: Path, device: str, use_cache: bool = True) -> dict:
    """Return per-window error features, labels and file ids for val and test.

    Windows are produced by the project's own TimeSeriesDataset, one file at a
    time, so the windowing is bit-identical to training while still telling us
    which recording each window came from (the shared dataset does not expose
    that, and this script must not change it).

    The error maps are cached next to the run: extraction is the slow part
    (the feature pipeline runs over every file), while sweeping variants on
    top of them is seconds. Adding a variant should not cost another pass.
    """
    cache_path = run_dir / "sweep_features.npz"
    if use_cache and cache_path.exists():
        blob = np.load(cache_path)
        data = {split: {"features": {k: blob[f"{split}__{k}"] for k in FEATURES},
                        "labels": blob[f"{split}__labels"],
                        "files": blob[f"{split}__files"]}
                for split in ("val", "test")}
        print(f"[sweep] {run_dir.name}: error maps from cache ({cache_path.name})", flush=True)
        return {"data": data, "config": json.loads((run_dir / "config.json").read_text())}

    detector, config = load_detector_from_run(run_dir, checkpoint="best")
    model = detector.model.to(torch.device(device)).eval()

    params = config["dataset"]["params"]
    kwargs = {
        "window_size": int(params["window_size"]),
        "horizon_size": int(params["horizon_size"]),
        "window_overlap": int(params["window_overlap"]),
        "files_per_chunk": 1,
        "bytes_cache_limit": float(params["bytes_cache_limit"]),
        "file_transform": build_pu_feature_pipeline(params["feature_pipeline"]),
        "target_builder": build_target_builder(config["task"]["type"]),
        "holdout_bearings": list(params.get("holdout_bearings", []) or []),
    }

    out: dict[str, dict] = {}
    for split in ("val", "test"):
        paths = discover_file_paths(params["root"], params[f"{split}_patterns"], None)
        rows: dict[str, list] = {k: [] for k in
                                 ("native", "abs_mean", "abs_max", "abs_std", "sq_mean")}
        labels, files = [], []
        for file_index, path in enumerate(paths):
            dataset = TimeSeriesDataset([path], split=split, **kwargs)
            loader = DataLoader(dataset, batch_size=256)
            for batch in loader:
                x, y = batch[0].float().to(device), batch[1].float().to(device)
                is_anomaly = batch[2]
                with torch.no_grad():
                    native = model.anomaly_error(x, y)               # (B, C)
                    prediction = model(x)                            # (B, P, C)
                    target = (y[:, -prediction.shape[1]:, :]
                              if prediction.shape[1] != y.shape[1] else y)
                    error = (prediction - target).abs()              # (B, P, C)
                    squared = error.pow(2)
                rows["native"].append(native.reshape(native.shape[0], -1).cpu().numpy())
                rows["abs_mean"].append(error.mean(dim=1).cpu().numpy())
                rows["abs_max"].append(error.amax(dim=1).cpu().numpy())
                rows["abs_std"].append(error.std(dim=1).cpu().numpy() if error.shape[1] > 1
                                       else torch.zeros_like(error[:, 0]).cpu().numpy())
                rows["sq_mean"].append(squared.mean(dim=1).cpu().numpy())
                labels.append(np.asarray(is_anomaly).astype(int).reshape(-1))
                files.append(np.full(labels[-1].shape, file_index))
            if (file_index + 1) % 50 == 0:
                print(f"[sweep] {run_dir.name} {split}: {file_index + 1}/{len(paths)} files", flush=True)
        out[split] = {
            "features": {k: np.concatenate(v) for k, v in rows.items()},
            "labels": np.concatenate(labels),
            "files": np.concatenate(files),
        }
        labels_array, files_array = out[split]["labels"], out[split]["files"]
        counts = np.array([(files_array == i).sum() for i in np.unique(files_array)])
        file_labels = np.array([labels_array[files_array == i].max() for i in np.unique(files_array)])
        normal_windows = counts[file_labels == 0]
        anomalous_windows = counts[file_labels == 1]
        print(f"[sweep] {run_dir.name} {split}: {len(labels_array)} windows from {len(paths)} files, "
              f"{int(labels_array.sum())} anomalous", flush=True)
        if len(normal_windows) and len(anomalous_windows):
            ratio = anomalous_windows.mean() / max(1e-9, normal_windows.mean())
            print(f"[sweep]   windows per file: normal ~{normal_windows.mean():.0f}, "
                  f"anomalous ~{anomalous_windows.mean():.0f} ({ratio:.1f}x) — this asymmetry is "
                  f"exactly what inflates file_max; compare it against file_max_eq", flush=True)

    if use_cache:
        blob = {}
        for split, payload in out.items():
            for name, array in payload["features"].items():
                blob[f"{split}__{name}"] = array
            blob[f"{split}__labels"] = payload["labels"]
            blob[f"{split}__files"] = payload["files"]
        np.savez_compressed(cache_path, **blob)
        print(f"[sweep] {run_dir.name}: error maps cached -> {cache_path.name}", flush=True)
    return {"data": out, "config": config}


# --------------------------------------------------------------------------- #
# 2. scorers
# --------------------------------------------------------------------------- #
def fit_scorer(train_features: np.ndarray, scorer: str) -> dict:
    mean = train_features.mean(axis=0)
    if scorer in {"maha_nomean", "maha_mean"}:
        covariance = np.cov(train_features.T) + np.eye(train_features.shape[1]) * EPS
        return {"inv": np.linalg.pinv(covariance), "mu": mean}
    if scorer == "diag":
        return {"std": train_features.std(axis=0) + EPS, "mu": mean}
    return {}


def apply_scorer(features: np.ndarray, scorer: str, state: dict) -> np.ndarray:
    if scorer == "maha_nomean":
        return np.einsum("ij,jk,ik->i", features, state["inv"], features)
    if scorer == "maha_mean":
        centered = features - state["mu"]
        return np.einsum("ij,jk,ik->i", centered, state["inv"], centered)
    if scorer == "diag":
        return (((features - state["mu"]) / state["std"]) ** 2).sum(axis=1)
    return features.mean(axis=1)  # author-style


def per_file(scores: np.ndarray, labels: np.ndarray, files: np.ndarray, how: str):
    """Aggregate window scores to one score per recording.

    `max` is NOT a neutral choice on this split. Normal test files contribute
    ~30 windows (the last 5 % of the recording) while anomalous ones contribute
    ~341 (the second half), and the maximum of 341 draws sits above the maximum
    of 30 even when the two distributions are identical. So part of what looks
    like detection under `file_max` is a counting artefact. `q90` is less
    sensitive to the count, and `max_eq` removes it outright by taking the same
    number of evenly spaced windows from every file.
    """
    ids = np.unique(files)
    if how == "max_eq":
        counts = [int((files == i).sum()) for i in ids]
        k = max(1, min(counts))
        file_scores = []
        for i in ids:
            values = scores[files == i]
            # Evenly spaced, not the first k: the first windows of a recording
            # are not representative of the whole.
            picks = np.linspace(0, len(values) - 1, k).round().astype(int)
            file_scores.append(values[picks].max())
        file_scores = np.array(file_scores)
    else:
        reduce = {"mean": np.mean, "max": np.max,
                  "q90": lambda v: np.quantile(v, 0.90)}[how]
        file_scores = np.array([reduce(scores[files == i]) for i in ids])
    file_labels = np.array([labels[files == i].max() for i in ids])
    return file_scores, file_labels


# --------------------------------------------------------------------------- #
# 3. sweep
# --------------------------------------------------------------------------- #
FEATURES = ["native", "abs_mean", "abs_max", "abs_std", "sq_mean"]
SCORERS = ["maha_nomean", "maha_mean", "diag", "mean"]
FIT_POOLS = ["val_normal", "val_all"]
UNITS = ["window", "file_mean", "file_max", "file_q90", "file_max_eq"]

# What the paper itself does, so the table can show a delta rather than a level.
PAPER_UNIFIED = ("abs_mean", "maha_nomean", "val_normal", "window")
PAPER_NATIVE = ("native", "mean", "val_normal", "window")


def sweep_run(run_dir: Path, device: str, use_cache: bool = True) -> list[dict]:
    extracted = extract_features(run_dir, device, use_cache=use_cache)
    val, test = extracted["data"]["val"], extracted["data"]["test"]
    code = extracted["config"].get("phase1", {}).get("code", run_dir.name)
    results = []

    for feature in FEATURES:
        val_features, test_features = val["features"][feature], test["features"][feature]
        if not np.isfinite(val_features).all() or not np.isfinite(test_features).all():
            print(f"[sweep] {code}/{feature}: non-finite features, skipped", flush=True)
            continue
        for pool in FIT_POOLS:
            fit_data = val_features[val["labels"] == 0] if pool == "val_normal" else val_features
            for scorer in SCORERS:
                try:
                    state = fit_scorer(fit_data, scorer)
                    scores = apply_scorer(test_features, scorer, state)
                except np.linalg.LinAlgError as exc:
                    print(f"[sweep] {code}/{feature}/{scorer}/{pool}: {exc}", flush=True)
                    continue
                for unit in UNITS:
                    if unit == "window":
                        s, y = scores, test["labels"]
                    else:
                        s, y = per_file(scores, test["labels"], test["files"],
                                        unit[len("file_"):])
                    if len(np.unique(y)) < 2:
                        continue
                    results.append({
                        "run": run_dir.name, "code": code, "feature": feature,
                        "scorer": scorer, "cov_fit": pool, "unit": unit,
                        "n": len(y), "roc_auc": float(roc_auc_score(y, s)),
                    })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="", help="explicit run dirs, comma separated")
    parser.add_argument("--runs-root", default="runs_phase1")
    parser.add_argument("--codes", default="TA001,TA010,TA011")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--device", default=os.environ.get(
        "PHASE1_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--out", default="scoring_sweep.csv")
    parser.add_argument("--no-cache", action="store_true",
                        help="recompute error maps instead of reading sweep_features.npz")
    args, extra = parser.parse_known_args()
    if extra:
        print(f"[sweep] ignoring launcher args: {extra}", flush=True)
    os.chdir(REPO)

    if args.runs:
        run_dirs = [Path(p.strip()) for p in args.runs.split(",") if p.strip()]
    else:
        codes = {c.strip().upper() for c in args.codes.split(",") if c.strip()}
        seeds = {s.strip() for s in args.seeds.split(",") if s.strip()}
        run_dirs = []
        for summary in sorted(Path(args.runs_root).glob("*/*/summary.json")):
            config = json.loads((summary.parent / "config.json").read_text())
            code = config.get("phase1", {}).get("code", "").upper()
            if code in codes and str(config["run"]["seed"]) in seeds:
                run_dirs.append(summary.parent)
    assert run_dirs, "no runs matched"

    results = []
    for run_dir in run_dirs:
        print(f"\n[sweep] === {run_dir}", flush=True)
        results.extend(sweep_run(run_dir, args.device, use_cache=not args.no_cache))

    out_path = Path(args.runs_root) / args.out
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    for code in sorted({r["code"] for r in results}):
        rows = [r for r in results if r["code"] == code]
        rows.sort(key=lambda r: -r["roc_auc"])
        base = {(r["feature"], r["scorer"], r["cov_fit"], r["unit"]): r["roc_auc"] for r in rows}
        paper = base.get(PAPER_UNIFIED)
        print(f"\n=== {code}   (paper unified variant = "
              f"{paper:.4f})" if paper else f"\n=== {code}", flush=True)
        print(f"{'feature':<10}{'scorer':<14}{'cov fit':<12}{'unit':<11}{'n':>7}{'ROC-AUC':>10}", flush=True)
        for r in rows[:15]:
            mark = ""
            if (r["feature"], r["scorer"], r["cov_fit"], r["unit"]) == PAPER_UNIFIED:
                mark = "  <- paper unified"
            elif (r["feature"], r["scorer"], r["cov_fit"], r["unit"]) == PAPER_NATIVE:
                mark = "  <- author score"
            print(f"{r['feature']:<10}{r['scorer']:<14}{r['cov_fit']:<12}{r['unit']:<11}"
                  f"{r['n']:>7}{r['roc_auc']:>10.4f}{mark}", flush=True)

    print(f"\n[sweep] written: {out_path}", flush=True)
    print("[sweep] READ THIS AS A SENSITIVITY ANALYSIS, NOT A LEADERBOARD. The best row is the best "
          "of ~600 variants picked ON THE TEST SET, so it is optimistically biased and cannot be "
          "reported as a method's score. What is reportable is the spread: how far one checkpoint "
          "moves when only the scoring rule and the evaluation unit change.", flush=True)
    print("[sweep] compare file_max against file_max_eq: the difference is the counting artefact "
          "(anomalous recordings contribute ~11x more windows than normal ones on this split).",
          flush=True)


if __name__ == "__main__":
    main()
