"""
Phase 1 diagnostics: why are the baselines so far below our own families?

Two hypotheses, both testable on the checkpoints that already exist — no training.

H1 — IDENTITY COLLAPSE. A reconstruction detector that reproduces its input
perfectly also reproduces anomalies perfectly, so the score stops separating.
Train losses of 0.0000 (TranAD, USAD) and 0.0004 (Anomaly Transformer) point
straight at it, as does "more epochs made it worse" (TA004 < TA001).
Test: per split, the mean author-score on ANOMALOUS windows divided by the mean
on NORMAL windows. A working detector gives a ratio clearly above 1; a collapsed
one sits at ~1 no matter how small the absolute error is.

H2 — SCORING SUPPORT. TranAD reconstructs only the LAST timestep of the window,
so its error map covers 1 of 120 timesteps, while our families average the error
over the whole window. Averaging over 120 steps suppresses noise, so part of the
gap may be the scoring support rather than the model.
Test: take OUR phase-0 checkpoints and rescore them using the last timestep only.
If their ROC-AUC drops towards the baselines', the gap is largely support, and
the paper must say so instead of claiming the methods are weaker.

Usage (from repo root, on the cluster; GPU recommended):
    python scripts/phase1/phase1_diagnose.py                    # both parts
    python scripts/phase1/phase1_diagnose.py --skip-ours        # H1 only

Writes runs_phase1/phase1_diagnosis.json and prints a readable table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _attach_nfs_env() -> None:
    if os.environ.get("AD_REEXEC"):
        return
    envjson = os.environ.get("AD_ENV_JSON", "")
    if not envjson or not os.path.exists(envjson):
        return
    spec = json.loads(open(envjson).read())
    py = spec.get("python") or ""
    pp = spec.get("pythonpath") or ""
    if pp:
        os.environ["PYTHONPATH"] = pp + (
            os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""
        )
    if py and os.path.exists(py) and os.path.realpath(py) != os.path.realpath(sys.executable):
        os.environ["AD_REEXEC"] = "1"
        os.execve(py, [py, os.path.abspath(__file__)] + sys.argv[1:], os.environ)


_attach_nfs_env()


def _ensure_deps() -> None:
    import importlib.util
    import subprocess

    need = {"pywt": "PyWavelets", "pandas": "pandas", "sklearn": "scikit-learn",
            "scipy": "scipy", "pyarrow": "pyarrow", "tqdm": "tqdm", "matplotlib": "matplotlib"}
    missing = [pkg for mod, pkg in need.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print(f"[diag] installing missing deps: {missing}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "--no-input", *missing])


_ensure_deps()

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from industrial_ad.datasets.PU.dataset import build_pu_dataloaders  # noqa: E402
from industrial_ad.experiments import load_detector_from_run  # noqa: E402

PHASE1 = REPO / "runs_phase1"
PHASE0 = REPO / "runs_phase0"
OUT = PHASE1 / "phase1_diagnosis.json"

# Our own headline configs (phase 0), used for the scoring-support test.
OURS = [
    ("TR138", "TCN-reconstruction", "138-s42-sep0-hid64-lat4-bl5-ker3"),
    ("CR048", "Conv-reconstruction", "048-s42-hid32.48.64.96-lat24-ker9"),
    ("MR064", "MLP-reconstruction", "064-s42-dims-3240-27"),
]


@torch.no_grad()
def collect(detector, loader, device, reducer=None):
    """Return (per-window score, label) using the run's own error map, or a custom one."""
    detector.model.eval()
    scores, labels = [], []
    for x, y, is_anomaly in loader:
        x, y = x.to(device), y.to(device)
        errors = reducer(detector, x, y) if reducer else detector.get_errors(x, y)
        scores.append(errors.mean(dim=1).cpu())
        labels.append(is_anomaly.cpu())
    return torch.cat(scores).numpy(), torch.cat(labels).numpy().astype(int)


def separation(scores, labels) -> dict:
    normal = scores[labels == 0]
    anomalous = scores[labels == 1]
    mean_normal = float(normal.mean()) if normal.size else float("nan")
    mean_anomalous = float(anomalous.mean()) if anomalous.size else float("nan")
    return {
        "mean_error_normal": mean_normal,
        "mean_error_anomalous": mean_anomalous,
        "ratio_anomalous_over_normal": mean_anomalous / mean_normal if mean_normal else float("nan"),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) > 1 else float("nan"),
    }


def h1_baselines(device: str, bundle_cache: dict) -> list[dict]:
    """Identity collapse: how much larger is the error on anomalies than on normals?"""
    results = []
    for summary_path in sorted(PHASE1.glob("*/*/summary.json")):
        run_dir = summary_path.parent
        detector, config = load_detector_from_run(run_dir, checkpoint="best")
        detector.to(torch.device(device))
        key = json.dumps(config["dataset"], sort_keys=True)
        if key not in bundle_cache:
            bundle_cache.clear()
            bundle_cache[key] = build_pu_dataloaders(config)
        scores, labels = collect(detector, bundle_cache[key]["loaders"]["test"], torch.device(device))
        record = {"code": config.get("phase1", {}).get("code", "?"),
                  "run": config["run"]["name"], **separation(scores, labels)}
        results.append(record)
        print(f"[H1] {record['code']} {record['run']:<40} "
              f"err_norm={record['mean_error_normal']:.3e} err_anom={record['mean_error_anomalous']:.3e} "
              f"ratio={record['ratio_anomalous_over_normal']:.3f} roc={record['roc_auc']:.4f}", flush=True)
    return results


def h2_ours(device: str, bundle_cache: dict) -> list[dict]:
    """Scoring support: our own models, scored on the full window vs the last timestep only."""
    def full_window(detector, x, y):
        return (detector.model(x) - y).abs().mean(dim=1)

    def last_step(detector, x, y):
        return (detector.model(x)[:, -1:, :] - y[:, -1:, :]).abs().mean(dim=1)

    results = []
    for code, family, run_name in OURS:
        run_dir = PHASE0 / family / run_name
        if not (run_dir / "summary.json").exists():
            print(f"[H2] skip {code}: {run_dir} not found", flush=True)
            continue
        detector, config = load_detector_from_run(run_dir, checkpoint="best")
        detector.to(torch.device(device))
        key = json.dumps(config["dataset"], sort_keys=True)
        if key not in bundle_cache:
            bundle_cache.clear()
            bundle_cache[key] = build_pu_dataloaders(config)
        loader = bundle_cache[key]["loaders"]["test"]

        full_scores, labels = collect(detector, loader, torch.device(device), reducer=full_window)
        last_scores, _ = collect(detector, loader, torch.device(device), reducer=last_step)
        record = {
            "code": code,
            "roc_auc_full_window": float(roc_auc_score(labels, full_scores)),
            "roc_auc_last_step_only": float(roc_auc_score(labels, last_scores)),
        }
        record["drop"] = record["roc_auc_full_window"] - record["roc_auc_last_step_only"]
        results.append(record)
        print(f"[H2] {code}: full window {record['roc_auc_full_window']:.4f} -> "
              f"last step only {record['roc_auc_last_step_only']:.4f} "
              f"(drop {record['drop']:+.4f})", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.environ.get(
        "PHASE1_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--skip-ours", action="store_true")
    args, _ = parser.parse_known_args()

    os.chdir(REPO)
    bundle_cache: dict = {}
    print("=== H1: identity collapse (ratio ~1 means the score cannot separate) ===", flush=True)
    baselines = h1_baselines(args.device, bundle_cache)

    ours = []
    if not args.skip_ours:
        print("\n=== H2: scoring support (our models, full window vs last timestep) ===", flush=True)
        ours = h2_ours(args.device, bundle_cache)

    OUT.write_text(json.dumps({"h1_baselines": baselines, "h2_ours": ours}, indent=2))
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
