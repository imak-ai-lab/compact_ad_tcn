"""
Phase 1: train the SOTA baselines on the A100 cluster.

Same cluster contract as phase 0:
  * parse_known_args — the pytorch2 launcher appends --local-rank=N;
  * shard passed via CLI *and* the PHASE1_SHARD env variable;
  * absolute data path + os.chdir(REPO) — the launcher does not start in repo root;
  * loud errors with a non-zero exit, so a failed job never reports green;
  * TF32 off, matching phase 0 and the original FP32 numbers.

Usage (from repo root, on the cluster):
    python scripts/phase1/phase1_run.py --list-only        # print the plan
    python scripts/phase1/phase1_run.py --smoke            # 2 epochs, few batches
    python scripts/phase1/phase1_run.py --shard 0/9        # one of 9 shards

After all shards finish:
    python scripts/phase1/phase1_rescore.py                # Mahalanobis column
    python scripts/phase1/phase1_report.py                 # tables
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path


def _attach_nfs_env() -> None:
    """Re-exec into the NFS python if bootstrap_ad_env.sh wrote a spec (phase-0 pattern)."""
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
        print(f"[phase1] exec {py} (from {sys.version.split()[0]})", flush=True)
        os.execve(py, [py, os.path.abspath(__file__)] + sys.argv[1:], os.environ)


_attach_nfs_env()


def _ensure_deps() -> None:
    """Install the light dependencies missing from the base image (no venv, as in phase 0)."""
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
        print(f"[phase1] installing missing deps for {sys.executable}: {missing}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "--no-input", *missing])


_ensure_deps()

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import industrial_ad  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_configs import build_phase1_configs  # noqa: E402

OUT_ROOT = REPO / "runs_phase1"
DATA_ROOT = REPO / "data" / "PU"


def write_env_report() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "phase": "1-sota",
    }
    (OUT_ROOT / "env_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=os.environ.get("PHASE1_SEEDS", ""),
                        help="comma-separated seeds, e.g. 45,46 to top a config up to 5 seeds "
                             "(default: the SEEDS constant in phase1_configs.py)")
    parser.add_argument("--shard", default=os.environ.get("PHASE1_SHARD", "0/1"),
                        help="i/N contiguous shard of the run list")
    # NB: boolean job flags arrive as `--smoke ''` (Cloud.ru renders flags as
    # key/value pairs), which argparse's store_true rejects with exit 2. So every
    # switch takes an optional value and also has an env fallback — the same
    # env-first contract as the shard.
    parser.add_argument("--list-only", nargs="?", const="1",
                        default=os.environ.get("PHASE1_LIST_ONLY", ""))
    parser.add_argument("--smoke", nargs="?", const="1",
                        default=os.environ.get("PHASE1_SMOKE", ""),
                        help="2 epochs, 3 train batches, 2 eval batches — timing sanity check")
    parser.add_argument("--epochs", default=os.environ.get("PHASE1_EPOCHS", ""),
                        help="override trainer.epochs")
    parser.add_argument("--codes", default=os.environ.get("PHASE1_CODES", ""),
                        help="comma-separated config codes to run (e.g. TA004,US004). "
                             "Sharding then applies to the filtered list, so adding configs "
                             "never renumbers the shards of the ones already finished.")
    parser.add_argument("--count-batches", nargs="?", const="1",
                        default=os.environ.get("PHASE1_COUNT_BATCHES", ""),
                        help="build the loaders, print batches per split and exit "
                             "(turns a per-batch time into a per-epoch estimate)")
    parser.add_argument("--train-batches", default=os.environ.get("PHASE1_TRAIN_BATCHES", ""),
                        help="cap train batches per epoch (smoke default: 3)")
    parser.add_argument("--eval-batches", default=os.environ.get("PHASE1_EVAL_BATCHES", ""),
                        help="cap eval batches (default: no cap — a capped val split is "
                             "single-class at the start and yields NaN ROC-AUC)")
    # NB: parse_known_args — the pytorch2 launcher adds --local-rank=N.
    args, launcher_args = parser.parse_known_args()
    if launcher_args:
        print(f"[phase1] ignoring launcher args: {launcher_args}", flush=True)

    def _switch(value) -> bool:
        return str(value).strip().lower() not in {"", "0", "false", "none", "no"}

    def _int_or_none(value):
        text = str(value).strip()
        return int(text) if text and text.lower() != "none" else None

    list_only = _switch(args.list_only)
    smoke = _switch(args.smoke)
    epochs_override = _int_or_none(args.epochs)
    train_batches = _int_or_none(args.train_batches)
    eval_batches = _int_or_none(args.eval_batches)
    print(f"[phase1] shard={args.shard} smoke={smoke} list_only={list_only} "
          f"epochs_override={epochs_override} train_batches={train_batches} "
          f"eval_batches={eval_batches}", flush=True)

    mls_region = os.environ.get("MLS_JOB_REGION_NAME")
    if mls_region is not None:
        assert mls_region == "SR006", f"wrong region: {mls_region}"
    tmpdir = os.environ.get("TMPDIR")
    if tmpdir:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32

    # Extra seeds for a single config are additive: run names carry the seed, and
    # skip_existing leaves the finished ones alone. Used to lift the best baseline
    # to the 5 seeds our own headline numbers carry (TA011: std 0.0465 over 3 seeds
    # against 0.0090 for TR138 over 5 — a 3-seed mean is too loose to compare against).
    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()] or None
    configs = build_phase1_configs(OUT_ROOT, data_root=DATA_ROOT.resolve(), seeds=seeds)
    codes = [c.strip().upper() for c in str(args.codes).split(",") if c.strip()]
    if codes:
        known = {c["phase1"]["code"] for c in configs}
        unknown = [c for c in codes if c not in known]
        assert not unknown, f"unknown config codes: {unknown} (known: {sorted(known)})"
        configs = [c for c in configs if c["phase1"]["code"] in codes]
        print(f"[phase1] codes filter {codes} -> {len(configs)} runs", flush=True)
    # Stable order that keeps identical dataset configs adjacent (data-bundle reuse):
    # the Anomaly Transformer uses a different loader batch size, so its runs group together.
    configs.sort(key=lambda c: (
        json.dumps({"d": c["dataset"], "t": c["task"]}, sort_keys=True),
        c["run"]["dir"],
    ))

    if smoke:
        # Train is capped (we only need timing), evaluation is NOT: the val split
        # starts with normal-only files, so a capped eval is single-class and
        # ROC-AUC comes out NaN — a smoke that cannot produce the metric proves
        # nothing about the scoring path.
        for config in configs:
            config["trainer"]["epochs"] = epochs_override or 2
            config["trainer"]["eval_every_n_epochs"] = 1
            config["debug"]["trainer"] = {
                "max_train_batches": train_batches if train_batches is not None else 3,
                "max_eval_batches": eval_batches,
            }
            config["run"]["dir"] = str(Path(config["run"]["dir"]).parent.parent / "_smoke" /
                                       Path(config["run"]["dir"]).parent.name / config["run"]["name"])
    else:
        for config in configs:
            if epochs_override is not None:
                config["trainer"]["epochs"] = epochs_override
            if train_batches is not None or eval_batches is not None:
                config["debug"]["trainer"] = {
                    "max_train_batches": train_batches,
                    "max_eval_batches": eval_batches,
                }

    shard_index, shard_total = map(int, args.shard.split("/"))
    assert 0 <= shard_index < shard_total, f"bad --shard {args.shard}"
    chunk = (len(configs) + shard_total - 1) // shard_total
    shard_configs = configs[shard_index * chunk : (shard_index + 1) * chunk]

    print(f"phase1: {len(configs)} total runs, shard {args.shard} -> {len(shard_configs)} runs", flush=True)
    for config in shard_configs:
        print("  ", config["phase1"]["code"], config["phase1"]["family"], config["run"]["name"], flush=True)
    if list_only:
        return

    os.chdir(REPO)
    parsed = [p for p in DATA_ROOT.glob("K*") if p.is_dir()]
    assert parsed, f"PU dataset not found/empty at {DATA_ROOT} — upload it first"
    print(f"[phase1] data: {len(parsed)} bearing dirs under {DATA_ROOT}", flush=True)

    if _switch(args.count_batches):
        # The PU dataset is a streaming IterableDataset — no __len__, so the only
        # honest count is one pass. That pass also measures data-loading
        # throughput and confirms both classes reach val/test (a single-class
        # split silently turns ROC-AUC into NaN).
        import time

        from industrial_ad.datasets.PU.dataset import build_pu_dataloaders

        batch_size = min(c["dataset"]["loader"]["batch_size"] for c in configs)
        config = next(c for c in configs if c["dataset"]["loader"]["batch_size"] == batch_size)
        print(f"[phase1] counting with batch_size={batch_size} "
              f"(one full pass per split, no GPU needed)", flush=True)
        bundle = build_pu_dataloaders(config)

        for split, loader in bundle["loaders"].items():
            started = time.time()
            batches = windows = anomalies = 0
            for x, _, is_anomaly in loader:
                batches += 1
                windows += int(x.shape[0])
                anomalies += int(is_anomaly.sum())
            elapsed = time.time() - started
            print(f"[phase1] {split:>5}: batches={batches} windows={windows} "
                  f"anomalous={anomalies} ({anomalies / max(windows, 1):.1%}) "
                  f"load={elapsed:.1f}s ({elapsed / max(batches, 1) * 1000:.0f} ms/batch)", flush=True)
            for other in sorted({c["dataset"]["loader"]["batch_size"] for c in configs} - {batch_size}):
                print(f"[phase1] {split:>5}: ~{-(-windows // other)} batches at batch_size={other}", flush=True)
        return

    write_env_report()
    # A smoke run must always re-run: its whole point is to exercise the current
    # code. With skip_existing it would silently no-op on the previous smoke's
    # summary.json and report success without running anything.
    results = industrial_ad.run_experiments(
        shard_configs, skip_existing=not smoke, overwrite=smoke, stop_on_error=False
    )

    # Loud failure summary: a batch job must never report green on a total wipe-out.
    failed = [r for r in results if r.get("error")]
    if failed:
        print(f"\n[phase1] {len(failed)}/{len(results)} runs FAILED:", flush=True)
        for record in failed:
            print(f"  {record['run_name']}: {record['exception']}", flush=True)
        sys.exit(1)
    print(f"[phase1] shard {args.shard} finished: {len(results)} runs ok", flush=True)


if __name__ == "__main__":
    main()
