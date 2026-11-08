"""
Phase 2: train the CWRU slice on the A100 cluster.

Same cluster contract as phases 0/1:
  * parse_known_args — the pytorch2 launcher appends --local-rank=N;
  * every switch has an env fallback (PHASE2_*) — boolean flags cannot be passed
    through the platform's key/value flags;
  * absolute data path + os.chdir(REPO);
  * loud errors with a non-zero exit;
  * TF32 off.

What is new in phase 2: after every finished run the per-window test/val scores
are dumped (`scores_test.csv`, `scores_val.csv`, with recording / bearing / fault
metadata) from the `best` AND the `last` checkpoint, so recording-level and
per-fault-type AUCs, and the "checkpoint chosen without anomalies" column, are
computed by the report without another GPU job.

Usage (from repo root, on the cluster):
    python scripts/phase2/phase2_run.py --list-only
    python scripts/phase2/phase2_run.py --count-batches --folds 0 --tier A
    python scripts/phase2/phase2_run.py --smoke --folds 0 --codes TR138
    python scripts/phase2/phase2_run.py --folds 0 --tier A --shard 0/2
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
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
        os.environ["PYTHONPATH"] = pp + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    if py and os.path.exists(py) and os.path.realpath(py) != os.path.realpath(sys.executable):
        os.environ["AD_REEXEC"] = "1"
        print(f"[phase2] exec {py} (from {sys.version.split()[0]})", flush=True)
        os.execve(py, [py, os.path.abspath(__file__)] + sys.argv[1:], os.environ)


_attach_nfs_env()


def _ensure_deps() -> None:
    import importlib.util
    import subprocess

    need = {"pywt": "PyWavelets", "pandas": "pandas", "sklearn": "scikit-learn", "scipy": "scipy",
            "pyarrow": "pyarrow", "tqdm": "tqdm", "matplotlib": "matplotlib"}
    missing = [pkg for mod, pkg in need.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print(f"[phase2] installing missing deps for {sys.executable}: {missing}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "--no-input", *missing])


_ensure_deps()

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import industrial_ad  # noqa: E402
from industrial_ad.datasets.CWRU.dataset import build_cwru_datasets  # noqa: E402
from industrial_ad.datasets.CWRU.download import check_present  # noqa: E402
from industrial_ad.experiments import load_detector_from_run  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_configs import FOLDS, VARIANTS, build_phase2_configs, fill_usad_input_range, tier_codes  # noqa: E402

OUT_ROOT = REPO / "runs_phase2"
DATA_ROOT = REPO / "data" / "CWRU"


def write_env_report(out_root: Path, variant: str = "v1") -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    report = {
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "phase": "2-cwru", "protocol": VARIANTS[variant]["protocol"], "variant": variant,
    }
    (out_root / "env_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


@torch.no_grad()
def dump_window_scores(run_dir: Path, bundle: dict, device: str) -> None:
    """Per-window scores with recording metadata for best and last checkpoints."""
    for checkpoint in ("best", "last"):
        ckpt_path = run_dir / "checkpoints" / f"{checkpoint}.pt"
        if not ckpt_path.exists():
            continue
        detector, _ = load_detector_from_run(run_dir, checkpoint=checkpoint)
        detector = detector.to(device).eval()
        for split in ("val", "test"):
            rows = []
            dataset = bundle["datasets"][split]
            batch_x, batch_y, batch_meta = [], [], []

            def _flush():
                if not batch_x:
                    return
                x = torch.as_tensor(__import__("numpy").stack(batch_x)).to(device)
                y = torch.as_tensor(__import__("numpy").stack(batch_y)).to(device)
                scores = detector.get_anomaly_scores(x, y).float().cpu().numpy()
                for meta, score in zip(batch_meta, scores):
                    rows.append({**meta, "score": float(score)})
                batch_x.clear(); batch_y.clear(); batch_meta.clear()

            for window, target, is_anomaly, meta in dataset.iter_with_meta():
                batch_x.append(window); batch_y.append(target)
                batch_meta.append({**meta, "is_anomaly": int(is_anomaly)})
                if len(batch_x) == 512:
                    _flush()
            _flush()
            out = run_dir / f"scores_{split}_{checkpoint}.csv"
            with open(out, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        del detector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", default=os.environ.get("PHASE2_SHARD", "0/1"))
    parser.add_argument("--folds", default=os.environ.get("PHASE2_FOLDS", ""), help="comma-separated folds (default all 4)")
    parser.add_argument("--tier", default=os.environ.get("PHASE2_TIER", "A"), help="A | B | AB | V1B")
    parser.add_argument("--variant", default=os.environ.get("PHASE2_VARIANT", "v1"),
                        help="v1 (frozen protocol) | v1b (common 5 kHz low-pass control; runs_phase2_v1b)")
    parser.add_argument("--codes", default=os.environ.get("PHASE2_CODES", ""), help="comma-separated codes; overrides --tier")
    parser.add_argument("--seeds", default=os.environ.get("PHASE2_SEEDS", ""))
    parser.add_argument("--list-only", nargs="?", const="1", default=os.environ.get("PHASE2_LIST_ONLY", ""))
    parser.add_argument("--smoke", nargs="?", const="1", default=os.environ.get("PHASE2_SMOKE", ""))
    parser.add_argument("--epochs", default=os.environ.get("PHASE2_EPOCHS", ""))
    parser.add_argument("--count-batches", nargs="?", const="1", default=os.environ.get("PHASE2_COUNT_BATCHES", ""))
    parser.add_argument("--train-batches", default=os.environ.get("PHASE2_TRAIN_BATCHES", ""))
    parser.add_argument("--eval-batches", default=os.environ.get("PHASE2_EVAL_BATCHES", ""))
    parser.add_argument("--data-root", default=os.environ.get("PHASE2_DATA_ROOT", str(DATA_ROOT)))
    parser.add_argument("--out-root", default=os.environ.get("PHASE2_OUT_ROOT", ""),
                        help="default: runs_phase2 for v1, runs_phase2_v1b for v1b")
    parser.add_argument("--device", default=os.environ.get("PHASE2_DEVICE", ""), help="override trainer.device (local CPU tests)")
    args, launcher_args = parser.parse_known_args()
    if launcher_args:
        print(f"[phase2] ignoring launcher args: {launcher_args}", flush=True)

    def _switch(value) -> bool:
        return str(value).strip().lower() not in {"", "0", "false", "none", "no"}

    def _int_or_none(value):
        text = str(value).strip()
        return int(text) if text and text.lower() != "none" else None

    def _ints(value) -> list[int]:
        return [int(x) for x in str(value).split(",") if x.strip()]

    list_only, smoke, count_batches = _switch(args.list_only), _switch(args.smoke), _switch(args.count_batches)
    epochs_override, train_batches, eval_batches = _int_or_none(args.epochs), _int_or_none(args.train_batches), _int_or_none(args.eval_batches)
    folds = _ints(args.folds) or list(FOLDS)
    seeds = _ints(args.seeds) or None
    variant = str(args.variant).lower()
    assert variant in VARIANTS, f"unknown variant {variant}"
    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve() if args.out_root else (REPO / VARIANTS[variant]["out_root"]).resolve()
    if smoke:
        out_root = out_root / "_smoke"
    print(f"[phase2] variant={variant} shard={args.shard} folds={folds} tier={args.tier} codes={args.codes!r} seeds={seeds} smoke={smoke} "
          f"epochs_override={epochs_override} data_root={data_root} out_root={out_root}", flush=True)

    mls_region = os.environ.get("MLS_JOB_REGION_NAME")
    if mls_region is not None:
        assert mls_region == "SR006", f"wrong region: {mls_region}"
    tmpdir = os.environ.get("TMPDIR")
    if tmpdir:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32

    codes = [c.strip().upper() for c in str(args.codes).split(",") if c.strip()] or tier_codes(args.tier)
    configs = build_phase2_configs(out_root, data_root, codes=codes, folds=folds, seeds=seeds, variant=variant)
    if args.device:
        for cfg in configs:
            cfg["trainer"]["device"] = args.device
            if args.device == "cpu":
                cfg["dataset"]["loader"]["pin_memory"] = False
    # identical dataset+task configs adjacent -> the data bundle (features + scaler) is built once per fold/task
    configs.sort(key=lambda c: (json.dumps({"d": c["dataset"], "t": c["task"]}, sort_keys=True), c["run"]["dir"]))

    if smoke:
        for cfg in configs:
            cfg["trainer"]["epochs"] = epochs_override or 2
            cfg["trainer"]["eval_every_n_epochs"] = 1
            cfg["debug"]["trainer"] = {"max_train_batches": train_batches if train_batches is not None else 3,
                                       "max_eval_batches": eval_batches}
    elif epochs_override:
        for cfg in configs:
            cfg["trainer"]["epochs"] = epochs_override

    shard_index, shard_total = map(int, args.shard.split("/"))
    assert 0 <= shard_index < shard_total, f"bad --shard {args.shard}"
    chunk = (len(configs) + shard_total - 1) // shard_total
    shard_configs = configs[shard_index * chunk:(shard_index + 1) * chunk]
    print(f"[phase2] {len(configs)} total runs ({len(codes)} codes x {len(folds)} folds x "
          f"{len(seeds) if seeds else 3} seeds), shard {args.shard} -> {len(shard_configs)} runs", flush=True)
    if tier_codes(args.tier) == tier_codes("S") or any(c in {"TA011", "AT004", "US006"} for c in codes):
        assert variant == "v1", "tier S runs under the headline protocol only"
    for cfg in shard_configs:
        print(f"   fold{cfg['phase2']['fold']} {cfg['phase2']['code']:6s} {cfg['run']['name']}", flush=True)
    if list_only:
        return

    os.chdir(REPO)
    missing = check_present(data_root)
    assert not missing, f"CWRU data incomplete under {data_root}: {len(missing)} files missing, e.g. {missing[:5]}"

    if count_batches:
        seen = set()
        for cfg in shard_configs:
            key = (cfg["phase2"]["fold"], cfg["task"]["type"], cfg["dataset"]["params"]["horizon_size"])
            if key in seen:
                continue
            seen.add(key)
            bundle = build_cwru_datasets(cfg)
            bs = int(cfg["dataset"]["loader"]["batch_size"])
            counts = bundle["metadata"]["window_counts"]
            print(f"[phase2] fold{key[0]} {key[1]} h{key[2]}: windows {counts} -> batches@{bs} "
                  f"{ {k: -(-v // bs) for k, v in counts.items()} }; steps/110 epochs = {110 * -(-counts['train'] // bs)}",
                  flush=True)
        return

    write_env_report(out_root, variant)
    started = time.time()
    # USAD (tier S): its own min-max input scaling, fitted on THIS fold's train frames.
    bundle_cache: dict[str, dict] = {}
    for cfg in shard_configs:
        if cfg["model"]["params"].get("input_range") == "AUTO_CWRU":
            key = json.dumps({"d": cfg["dataset"], "t": cfg["task"]}, sort_keys=True)
            if key not in bundle_cache:
                bundle_cache[key] = build_cwru_datasets(cfg)
            fill_usad_input_range(cfg, bundle_cache[key])
            print(f"[phase2] {cfg['phase2']['code']} fold{cfg['phase2']['fold']}: USAD input_range fitted on "
                  f"{cfg['phase2']['usad_input_range_provenance']['n_frames']} train frames", flush=True)
    results = industrial_ad.run_experiments(shard_configs, skip_existing=True, stop_on_error=False)
    errors = [r for r in results if r.get("error")]

    # Score dumps (best + last checkpoints). Cheap; one bundle per fold/task.
    dump_errors = 0
    for cfg, result in zip(shard_configs, results):
        if result.get("error"):
            continue
        run_dir = Path(cfg["run"]["dir"])
        if (run_dir / "scores_test_best.csv").exists():
            continue
        key = json.dumps({"d": cfg["dataset"], "t": cfg["task"]}, sort_keys=True)
        if key not in bundle_cache:
            bundle_cache[key] = build_cwru_datasets(cfg)
        try:
            dump_window_scores(run_dir, bundle_cache[key], cfg["trainer"]["device"])
        except Exception as exc:  # noqa: BLE001
            dump_errors += 1
            print(f"[phase2]   score dump FAILED {run_dir.name}: {exc!r}", flush=True)

    print(f"[phase2] finished: {len(results) - len(errors)} ok, {len(errors)} failed, {dump_errors} dump failures, "
          f"{(time.time() - started) / 60:.1f} min", flush=True)
    for r in errors[:5]:
        print(f"[phase2]   FAILED {r.get('run_name')}: {r.get('exception')}", flush=True)
    if errors or dump_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
