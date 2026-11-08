"""
Phase 1: submit the SOTA baseline shards to Cloud.ru SR006.

Same wrapper as phase0_submit.py (client_lib 0.6.5-ready: enums for region/type,
checkpoint_dir, explicit allocation_name, NO queue_name), with the shard passed
both as a flag and as the PHASE1_SHARD env variable.

Run from the workspace Jupyter Server (NOT from a job):
    python scripts/phase1/phase1_submit.py --dry-run
    python scripts/phase1/phase1_submit.py --shards 0        # smoke first!
    python scripts/phase1/phase1_submit.py

Jobs are named `<prefix>-compact-ad-phase1-sota-<i>of<N>` (prefix defaults to
`kovalenko`, override with --job-prefix or PHASE1_JOB_PREFIX) so they are easy to
spot in the shared workspace queue.

Default: 9 shards x 3 runs — exactly one config per shard, 3 seeds each:
    0-2  TranAD (TA001 author, TA002 ff64, TA003 sigmoid ablation)   ~0.25 MiB
    3-5  USAD   (US001 author, US002 z40,  US003 sigmoid ablation)   ~75-186 MiB
    6-8  Anomaly Transformer (AT001 author d512, AT002 d128, AT003 d64)

Measured on PU (2026-08-14): 873 train batches/epoch at batch 256, 218 at 1024.
AT001 (author d512/l3) runs ~4.4 min/epoch on one A100, so 110 epochs is ~8 h per
run and ~24 h for its 3-seed shard. TranAD/USAD shards are far cheaper.

Shards 6-8 are the heavy ones (association matrices are batch x heads x 120 x 120
per layer). Submit `--shards 6 --smoke` first: it now runs a full evaluation, so
it also proves the scoring path produces a real ROC-AUC before a long job starts.

If a long shard is killed by a platform time limit, resubmit it — `skip_existing`
keeps finished runs — or split it with `--n-shards 27 --shards 18,19,20` (one job
per seed).
"""
from __future__ import annotations

import argparse
import inspect
import os
import time

import client_lib
from client_lib import (
    Job,
    JobHealthAction,
    JobHealthParams,
    JobHealthSubAction,
    RegionMT,
)

USER_NFS = "/home/jovyan/kovalenko"
REPO_ROOT = f"{USER_NFS}/industrial_ad-main"
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/cuda12.3-torch2-py310:0.0.37"
AD_ENV_JSON = f"{USER_NFS}/venvs/ad_env.json"
INSTANCE_TYPE = "a100.1gpu.8C.243G"
ALLOCATION = "alloc-airi-job-master-a100-sr006"
N_SHARDS = 9
# Prefix for job_desc so our jobs are identifiable in the shared workspace queue.
# Override per invocation with --job-prefix or the PHASE1_JOB_PREFIX env variable.
JOB_PREFIX = os.environ.get("PHASE1_JOB_PREFIX", "kovalenko")

_sig = inspect.signature(client_lib.get_instance_types)
assert "regions" in _sig.parameters, f"client_lib too old: {_sig}"


SCRIPTS = {
    "train": "phase1_run.py",
    "rescore": "phase1_rescore.py",   # Mahalanobis column; no shard, runs over everything
    "diagnose": "phase1_diagnose.py",  # identity collapse + scoring support checks
    "leakage": "leakage_probe.py",     # phase-0 config re-run with healthy bearings held out
    "sweep": "scoring_sweep.py",       # post-hoc scoring variants on existing checkpoints
}


def build_job_config(shard_index: int, extra_env: dict | None = None, n_shards: int = N_SHARDS,
                     job_prefix: str = JOB_PREFIX, task: str = "train") -> dict:
    """Return the job kwargs. Kept separate so --dry-run can print real values:
    `repr(Job)` is just an object address and hides exactly what you want to check."""
    # Only key=value flags here. A boolean flag would be rendered as `--smoke ''`
    # and argparse would exit 2 before the run starts — switches go through env.
    flags = {"shard": f"{shard_index}/{n_shards}"} if task == "train" else {}
    suffix = (extra_env or {}).get("PHASE1_SCORING", "")
    desc = (f"{job_prefix}-compact-ad-phase1-sota-{shard_index}of{n_shards}" if task == "train"
            else f"{job_prefix}-compact-ad-phase1-{task}" + (f"-{suffix}" if suffix else ""))
    return dict(
        base_image=BASE_IMAGE,
        script=f"{REPO_ROOT}/scripts/phase1/{SCRIPTS[task]}",
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
            # env duplicate of the shard: the proven channel on this platform
            "PHASE1_SHARD": f"{shard_index}/{n_shards}",
            **(extra_env or {}),
        },
        max_retry=10,
        checkpoint_dir=f"{REPO_ROOT}/runs_phase1",
        health_params=JobHealthParams(
            log_period=60,
            action=JobHealthAction.restart,
            sub_actions=[JobHealthSubAction.notify],
        ),
    )


def make_job(shard_index: int, extra_env: dict | None = None, n_shards: int = N_SHARDS,
             job_prefix: str = JOB_PREFIX, task: str = "train") -> Job:
    job_config = build_job_config(shard_index, extra_env, n_shards, job_prefix, task)
    accepted = set(inspect.signature(Job.__init__).parameters) - {"self"}
    return Job(**{k: v for k, v in job_config.items() if k in accepted})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task", choices=sorted(SCRIPTS), default="train",
                        help="train = the sharded training runs; rescore = one job that adds the "
                             "Mahalanobis column to every finished run (needs a GPU, ignores --shards)")
    parser.add_argument("--shards", default=None, help="comma-separated shard indices (default: all)")
    parser.add_argument("--job-prefix", default=JOB_PREFIX,
                        help="prefix for job_desc, so jobs are identifiable in the shared queue "
                             f"(default: {JOB_PREFIX!r} -> {JOB_PREFIX}-compact-ad-phase1-sota-0of9)")
    parser.add_argument("--n-shards", type=int, default=N_SHARDS,
                        help="how many shards to split the 27 runs into. 9 = one config per job "
                             "(3 seeds sequential); 27 = one run per job, so wall-clock equals the "
                             "slowest single run instead of 3x it")
    parser.add_argument("--holdout", default=None,
                        help="healthy bearings held out of training (task=leakage), e.g. K001,K005")
    parser.add_argument("--seeds", default=None,
                        help="comma-separated seeds passed through to the runner (e.g. 45,46)")
    parser.add_argument("--scoring", choices=["native-maha", "unified"], default=None,
                        help="for --task rescore: native-maha = our scorer on the method's own "
                             "error map; unified = our full pipeline (|x_hat-x| over time -> 27 "
                             "channels -> Mahalanobis), the like-for-like column")
    parser.add_argument("--runs-root", default=None,
                        help="for --task rescore: which run tree to rescore (default runs_phase1). "
                             "Pass runs_phase0 to run the control check on our own models")
    parser.add_argument("--codes", default=None,
                        help="comma-separated config codes (e.g. TA004,US004); shards apply to "
                             "the filtered list only")
    parser.add_argument("--smoke", action="store_true", help="submit in smoke mode (2 epochs, few batches)")
    parser.add_argument("--epochs", type=int, default=None, help="override trainer.epochs for these jobs")
    parser.add_argument("--eval-batches", type=int, default=None, help="cap eval batches (smoke speed-up)")
    parser.add_argument("--count-batches", action="store_true",
                        help="just build the loaders and print batches per split, then exit")
    args = parser.parse_args()

    n_shards = int(args.n_shards)
    if args.task in {"rescore", "diagnose"}:
        indices = [0]  # single job over all finished runs
    else:
        indices = [int(x) for x in args.shards.split(",")] if args.shards else list(range(n_shards))
    extra = {}
    if args.smoke:
        extra["PHASE1_SMOKE"] = "1"
    if args.epochs is not None:
        extra["PHASE1_EPOCHS"] = str(args.epochs)
    if args.eval_batches is not None:
        extra["PHASE1_EVAL_BATCHES"] = str(args.eval_batches)
    if args.count_batches:
        extra["PHASE1_COUNT_BATCHES"] = "1"
    if args.codes:
        extra["PHASE1_CODES"] = args.codes
    if args.seeds:
        extra["PHASE1_SEEDS"] = args.seeds
    if args.holdout:
        extra["PHASE1_HOLDOUT"] = args.holdout
    if args.scoring:
        extra["PHASE1_SCORING"] = args.scoring
    if args.runs_root:
        extra["PHASE1_RUNS_ROOT"] = args.runs_root

    jobs = []
    for i in indices:
        if args.dry_run:
            config = build_job_config(i, extra, n_shards, args.job_prefix, args.task)
            print(f"[dry-run] shard {i}/{n_shards}")
            for key in ("job_desc", "script", "instance_type", "flags", "env_variables", "checkpoint_dir"):
                print(f"    {key}: {config[key]}")
            continue
        job = make_job(i, extra, n_shards=n_shards, job_prefix=args.job_prefix, task=args.task)
        result = job.submit()
        if isinstance(result, str) and "Error" in result:
            raise RuntimeError(f"shard {i} submit failed: {result}")
        print(f"submitted shard {i}/{n_shards}: {job.job_name}")
        jobs.append(job)
        time.sleep(3)

    if jobs:
        print("\npoll with (>=30 s interval):  python3 ~/jobs.py")
        print("count results:  find runs_phase1 -name summary.json | wc -l   # expect 27")


if __name__ == "__main__":
    main()
