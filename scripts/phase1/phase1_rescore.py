"""
Phase 1: second scoring column — this project's Mahalanobis scorer on the
baselines' own error maps.

Each phase-1 run is trained and checkpointed with the author's own anomaly score.
This script reloads every best checkpoint, swaps only the score estimator for the
project's Mahalanobis scorer (fitted on normal validation windows, exactly as in
the paper's protocol), re-fits the threshold and re-evaluates the test split.

It answers the question — "would these baselines look better
with your scoring scheme?" — without retraining anything.

Usage (from repo root, on the cluster, after phase1_run.py):
    python scripts/phase1/phase1_submit.py --task rescore     # as a GPU job (recommended)
    python scripts/phase1/phase1_rescore.py                   # in-place; CPU-only JS is slow

Building the PU feature pipeline takes ~5 min, and each run needs four passes
over val/test — on the CPU Jupyter Server that is hours for 18 runs, on one A100
it is minutes. Submit it as a job.

Two scoring modes:
  --scoring native-maha (default) — Mahalanobis on the method's OWN error map.
      Answers "would the baseline look better with our scorer?".
  --scoring unified — |x_hat - x| averaged over time into 27 channel features, then
      Mahalanobis: exactly the paper's own scoring pipeline (SS III.E), applied to
      every model identically. Needed because the diagnosis showed our headline
      numbers carry a large scorer contribution (TR138 0.9431 with Mahalanobis vs
      0.8501 with plain averaging), so a table that pairs "our model + our scorer"
      against "their model + their scorer" overstates the architectural gap. It
      also fixes an asymmetry: the Anomaly Transformer's native error map is
      indexed by TIME (120), so its Mahalanobis covariance was over timesteps
      while ours is over channels.

Writes runs_phase1/<family>/<run>/rescore_{mahalanobis,unified}.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


# Same cluster bootstrap as phase1_run.py — this script is also submitted as a
# job (rescoring 18+ runs on the JS CPU takes hours; on one A100 it is minutes).
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
        print(f"[rescore] exec {py} (from {sys.version.split()[0]})", flush=True)
        os.execve(py, [py, os.path.abspath(__file__)] + sys.argv[1:], os.environ)


_attach_nfs_env()


def _ensure_deps() -> None:
    import importlib.util
    import subprocess

    need = {
        "pywt": "PyWavelets",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "scipy": "scipy",
        "pyarrow": "pyarrow",
        "tqdm": "tqdm",
        "matplotlib": "matplotlib",
    }
    missing = [pkg for mod, pkg in need.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print(f"[rescore] installing missing deps for {sys.executable}: {missing}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "--no-input", *missing])


_ensure_deps()

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from industrial_ad.datasets.PU.dataset import build_pu_dataloaders  # noqa: E402
from industrial_ad.experiments import load_detector_from_run  # noqa: E402
from industrial_ad.scoring import build_score_estimator  # noqa: E402
from industrial_ad.utils import dump_json  # noqa: E402

OUT_ROOT = REPO / "runs_phase1"


class _UnifiedReducer:
    """Error reducer that ignores the model's native score and uses ours."""

    needs_model = True

    def __call__(self, model, x, y):
        prediction = model(x)
        target = y[:, -prediction.shape[1]:, :] if prediction.shape[1] != y.shape[1] else y
        return (prediction - target).abs().mean(dim=1)


def rescore_run(run_dir: Path, device: str, bundle_cache: dict, scoring: str = "native-maha") -> dict:
    detector, config = load_detector_from_run(run_dir, checkpoint="best")
    detector.to(torch.device(device))

    if scoring != "unified" and config["scoring"]["error_reducer"]["name"] != "model_native":
        raise ValueError(f"{run_dir.name}: expected a model_native reducer, got {config['scoring']['error_reducer']}")

    key = json.dumps({"d": config["dataset"], "t": config["task"],
                      "b": config["debug"]["dataset"]}, sort_keys=True)
    if key not in bundle_cache:
        bundle_cache.clear()  # one dataset bundle in memory at a time
        bundle_cache[key] = build_pu_dataloaders(config)
    loaders = bundle_cache[key]["loaders"]

    if scoring == "unified":
        # Swap the error map for the project's own; feature_dim becomes the channel count.
        detector.error_reducer = _UnifiedReducer()
        feature_dim = int(config["runtime"]["input_shape"][1])
    else:
        feature_dim = int(config["runtime"]["score_feature_dim"])

    detector.score_estimator = build_score_estimator(
        {"name": "mahalanobis", "params": {"eps": 1e-5}}, feature_dim=feature_dim
    ).to(torch.device(device))
    detector.fit_score_estimator(loaders["val"], max_batches=None)
    detector.fit_threshold(loaders["val"], max_batches=None)
    val_metrics, _ = detector.evaluate(loaders["val"], prefix="val", max_batches=None)
    test_metrics, _ = detector.evaluate(loaders["test"], prefix="test", max_batches=None)

    record = {
        "run_name": config["run"]["name"],
        "phase1": config.get("phase1", {}),
        "score_estimator": "mahalanobis",
        "scoring": scoring,
        "score_feature_dim": feature_dim,
        "metrics": {**val_metrics, **test_metrics},
    }
    name = "rescore_unified.json" if scoring == "unified" else "rescore_mahalanobis.json"
    dump_json(run_dir / name, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.environ.get(
        "PHASE1_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--scoring", choices=["native-maha", "unified"],
                        default=os.environ.get("PHASE1_SCORING", "native-maha"))
    parser.add_argument("--runs-root", default=os.environ.get("PHASE1_RUNS_ROOT", ""),
                        help="rescore a different tree, e.g. runs_phase0. Used as a CONTROL: "
                             "for our own models `unified` IS their production scoring, so the "
                             "rescored number must reproduce the phase-0 value. If it does not, "
                             "the bug is in the rescoring code, not in the baselines.")
    args, launcher_args = parser.parse_known_args()
    if launcher_args:
        print(f"[rescore] ignoring launcher args: {launcher_args}", flush=True)

    os.chdir(REPO)
    root = Path(args.runs_root) if args.runs_root else OUT_ROOT
    if not root.is_absolute():
        root = REPO / root
    run_dirs = sorted(p.parent for p in root.glob("*/*/summary.json"))
    if not run_dirs:
        sys.exit(f"no runs found under {root} — run phase1_run.py first")
    print(f"[rescore] root={root} scoring={args.scoring} runs={len(run_dirs)}", flush=True)

    bundle_cache: dict = {}
    failed = []
    for run_dir in run_dirs:
        target = run_dir / ("rescore_unified.json" if args.scoring == "unified"
                            else "rescore_mahalanobis.json")
        if target.exists() and not args.overwrite:
            print(f"[rescore] skip (exists): {run_dir.name}", flush=True)
            continue
        # A run trained under the project's own scoring (TA011, TA006, US006, AT004)
        # has no author error map to re-score, so "native-maha" simply does not
        # apply to it. That is a skip, not a failure: letting it reach the failure
        # list makes the whole job exit 1, and the platform then restarts the job
        # in a loop (caught 2026-08-15 — a rescore job cycling running/pending).
        if args.scoring != "unified":
            config_path = run_dir / "config.json"
            if config_path.exists():
                reducer = json.loads(config_path.read_text()).get(
                    "scoring", {}).get("error_reducer", {}).get("name")
                if reducer != "model_native":
                    print(f"[rescore] skip (incompatible reducer '{reducer}', "
                          f"trained under project scoring): {run_dir.name}", flush=True)
                    continue
        try:
            record = rescore_run(run_dir, args.device, bundle_cache, args.scoring)
            print(f"[rescore:{args.scoring}] {run_dir.parent.name}/{run_dir.name}: "
                  f"test/roc_auc={record['metrics']['test/roc_auc']:.4f}", flush=True)
        except Exception:
            failed.append(run_dir)
            print(f"[rescore] FAILED {run_dir}:\n{traceback.format_exc()}", flush=True)

    if failed:
        print(f"[rescore] {len(failed)} runs failed", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
