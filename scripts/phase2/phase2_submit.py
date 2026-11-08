"""
Phase 2: submit the CWRU slice to Cloud.ru SR006.

Same wrapper as phase1_submit.py (client_lib 0.6.5: enums, checkpoint_dir,
allocation_name, NO queue_name). Switches travel as PHASE2_* env variables
(boolean flags cannot be passed through the platform's key/value flags); the
shard is passed both as a flag and as PHASE2_SHARD.

Run from the workspace Jupyter Server (NOT from a job), from the repo root:
    python3.11 scripts/phase2/phase2_submit.py --dry-run
    python3.11 scripts/phase2/phase2_submit.py --folds 0 --codes TR138 --smoke        # smoke first!
    python3.11 scripts/phase2/phase2_submit.py --folds 0 --tier A --count-batches
    python3.11 scripts/phase2/phase2_submit.py --tier A                                # 4 jobs, one per fold
    python3.11 scripts/phase2/phase2_submit.py --tier B --n-shards 2                   # 8 jobs

One job per (fold, shard): a fold shares one data bundle (features + the fold's
scaler), so splitting by fold costs nothing and keeps `runs_phase2/fold<L>/`
self-contained. Jobs are named `<prefix>-compact-ad-phase2-cwru-f<L>-<i>of<N>`.
"""
from __future__ import annotations

import argparse
import inspect
import os
import time

import client_lib
from client_lib import Job, JobHealthAction, JobHealthParams, JobHealthSubAction, RegionMT

USER_NFS = "/home/jovyan/kovalenko"
REPO_ROOT = f"{USER_NFS}/industrial_ad-main"
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/cuda12.3-torch2-py310:0.0.37"
AD_ENV_JSON = f"{USER_NFS}/venvs/ad_env.json"
INSTANCE_TYPE = "a100.1gpu.8C.243G"
ALLOCATION = "alloc-airi-job-master-a100-sr006"
JOB_PREFIX = os.environ.get("PHASE2_JOB_PREFIX", "kovalenko")
FOLDS = [0, 1, 2, 3]

_sig = inspect.signature(client_lib.get_instance_types)
assert "regions" in _sig.parameters, f"client_lib too old: {_sig}"


def build_job_config(fold: int, shard_index: int, n_shards: int, extra_env: dict | None = None,
                     job_prefix: str = JOB_PREFIX) -> dict:
    flags = {"shard": f"{shard_index}/{n_shards}", "folds": str(fold)}
    tier = (extra_env or {}).get("PHASE2_TIER", "A")
    variant = (extra_env or {}).get("PHASE2_VARIANT", "v1")
    smoke = "-smoke" if (extra_env or {}).get("PHASE2_SMOKE") else ""
    desc = f"{job_prefix}-compact-ad-phase2-cwru-{variant}-{tier}-f{fold}-{shard_index}of{n_shards}{smoke}"
    return dict(
        base_image=BASE_IMAGE,
        script=f"{REPO_ROOT}/scripts/phase2/phase2_run.py",
        instance_type=INSTANCE_TYPE,
        type=Job.Type.pytorch2,
        region=RegionMT.SR006,
        n_workers=1,
        processes_per_worker=1,
        allocation_name=ALLOCATION,
        job_desc=desc,
        flags=flags,
        env_variables={
            "PYTHONUNBUFFERED": "1",
            "TMPDIR": "/home/jovyan/tmp",
            "AD_ENV_JSON": AD_ENV_JSON,
            "PHASE2_SHARD": f"{shard_index}/{n_shards}",
            "PHASE2_FOLDS": str(fold),
            **(extra_env or {}),
        },
        max_retry=10,
        checkpoint_dir=f"{REPO_ROOT}/runs_phase2" + ("_v1b" if variant == "v1b" else ""),
        health_params=JobHealthParams(log_period=60, action=JobHealthAction.restart,
                                      sub_actions=[JobHealthSubAction.notify]),
    )


def make_job(fold: int, shard_index: int, n_shards: int, extra_env: dict | None, job_prefix: str) -> Job:
    config = build_job_config(fold, shard_index, n_shards, extra_env, job_prefix)
    accepted = set(inspect.signature(Job.__init__).parameters) - {"self"}
    return Job(**{k: v for k, v in config.items() if k in accepted})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--folds", default=None, help="comma-separated folds (default: all four)")
    parser.add_argument("--tier", default="A", help="A | B | AB | V1B | S (ignored when --codes is given)")
    parser.add_argument("--variant", default="v1", help="v1 | v1b (common low-pass control)")
    parser.add_argument("--codes", default=None, help="comma-separated config codes, e.g. TR138,TL082")
    parser.add_argument("--seeds", default=None, help="comma-separated seeds (default 42,43,44)")
    parser.add_argument("--n-shards", type=int, default=1, help="shards per fold (default 1: one job per fold)")
    parser.add_argument("--shards", default=None, help="comma-separated shard indices within each fold")
    parser.add_argument("--job-prefix", default=JOB_PREFIX)
    parser.add_argument("--smoke", action="store_true", help="2 epochs, 3 train batches, full eval; runs land in runs_phase2/_smoke")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--count-batches", action="store_true", help="build the loaders, print windows/batches per split, exit")
    args = parser.parse_args()

    folds = [int(x) for x in args.folds.split(",")] if args.folds else list(FOLDS)
    shards = [int(x) for x in args.shards.split(",")] if args.shards else list(range(args.n_shards))
    extra = {"PHASE2_TIER": args.tier.upper(), "PHASE2_VARIANT": args.variant.lower()}
    if args.codes:
        extra["PHASE2_CODES"] = args.codes
    if args.seeds:
        extra["PHASE2_SEEDS"] = args.seeds
    if args.smoke:
        extra["PHASE2_SMOKE"] = "1"
    if args.epochs is not None:
        extra["PHASE2_EPOCHS"] = str(args.epochs)
    if args.eval_batches is not None:
        extra["PHASE2_EVAL_BATCHES"] = str(args.eval_batches)
    if args.count_batches:
        extra["PHASE2_COUNT_BATCHES"] = "1"

    jobs = []
    for fold in folds:
        for shard in shards:
            if args.dry_run:
                config = build_job_config(fold, shard, args.n_shards, extra, args.job_prefix)
                print(f"[dry-run] fold {fold} shard {shard}/{args.n_shards}")
                for key in ("job_desc", "script", "instance_type", "flags", "env_variables", "checkpoint_dir"):
                    print(f"    {key}: {config[key]}")
                continue
            job = make_job(fold, shard, args.n_shards, extra, args.job_prefix)
            result = job.submit()
            if isinstance(result, str) and "Error" in result:
                raise RuntimeError(f"fold {fold} shard {shard} submit failed: {result}")
            print(f"submitted fold {fold} shard {shard}/{args.n_shards}: {job.job_name}")
            jobs.append(job)
            time.sleep(3)

    if jobs:
        print("\npoll with (>=30 s interval):  python3 ~/jobs.py")
        print("count results:  find runs_phase2 -name summary.json -not -path '*/_smoke/*' | wc -l")
        print("               tier A = 15 codes x 4 folds x 3 seeds = 180; tier B = 19 x 4 x 3 = 228")


if __name__ == "__main__":
    main()
