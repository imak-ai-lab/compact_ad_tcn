"""
Phase 0: submit repro shards to the Cloud.ru SR006 A100 cluster.

client_lib 0.6.5 wrapper: enums for region/type, checkpoint_dir (no 's'), pydantic
JobHealthParams, explicit allocation_name, NO queue_name.

Run from the workspace Jupyter Server (NOT from a job):
    python scripts/phase0/phase0_submit.py             # submit all shards
    python scripts/phase0/phase0_submit.py --dry-run   # print configs only

Before submitting, edit REPO_ROOT below to the actual NFS path of the repo.
Each shard is a single-GPU job; N_SHARDS=6 x ~18 runs each. The tiny models
don't need more than 1 GPU; parallelism comes from independent jobs.

Polling: not more often than every 30 s (API limit: 100 GET/min, 10k/day).
"""
from __future__ import annotations

import argparse
import inspect
import time

import client_lib
from client_lib import (
    Job,
    JobHealthAction,
    JobHealthParams,
    JobHealthSubAction,
    RegionMT,
)

# --- adjust to the actual NFS layout -----------------------------------------
# 2026-08-13: py3.11-torch-образов в реестре платформы НЕ оказалось
# (get_docker_images -> только legacy cuda10/11 jupyter-образы). Рабочая схема —
# L2D-паттерн: проверенный py310-образ + NFS-env c python3.11 и torch==2.6.0
# из bootstrap_ad_env.sh; раннер сам exec-ается в него по AD_ENV_JSON.
USER_NFS = "/home/jovyan/kovalenko"
REPO_ROOT = f"{USER_NFS}/industrial_ad-main"
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/cuda12.3-torch2-py310:0.0.37"
AD_ENV_JSON = f"{USER_NFS}/venvs/ad_env.json"  # пишется bootstrap_ad_env.sh; без файла guard no-op
INSTANCE_TYPE = "a100.1gpu.8C.243G"  # A100! (a100plus.* = H100, в нашей allocation их 0)
ALLOCATION = "alloc-airi-job-master-a100-sr006"
N_SHARDS = 6

# Functional version check: modern client_lib has get_instance_types(regions=)
_sig = inspect.signature(client_lib.get_instance_types)
assert "regions" in _sig.parameters, f"client_lib too old: {_sig}"


def make_job(shard_index: int) -> Job:
    job_config = dict(
        base_image=BASE_IMAGE,
        script=f"{REPO_ROOT}/scripts/phase0/phase0_repro.py",
        instance_type=INSTANCE_TYPE,
        type=Job.Type.pytorch2,          # enum (default 'horovod' is a footgun)
        region=RegionMT.SR006,           # enum (default DGX2-MT is a footgun)
        n_workers=1,                     # SR006: multi-worker unsupported
        processes_per_worker=1,          # plain script, not DDP
        allocation_name=ALLOCATION,      # explicit, otherwise Error 400
        # queue_name intentionally NOT set (Cloud.ru picks the workspace queue)
        job_desc=f"compact-ad-phase0-repro-{shard_index}of{N_SHARDS}",
        flags={"shard": f"{shard_index}/{N_SHARDS}"},
        env_variables={
            "PYTHONUNBUFFERED": "1",
            # литеральный путь: TMPDIR="" (незаданная project variable) роняет
            # OMPI session_dir с Permission denied
            "TMPDIR": "/home/jovyan/tmp",
            # дубль flags через env: L2D параметризовал job-ы только env-переменными,
            # flags на этой платформе не проверялись — страхуемся обоими каналами
            "PHASE0_SHARD": f"{shard_index}/{N_SHARDS}",
            # если bootstrap_ad_env.sh собрал NFS-окружение — раннер в него exec-нется
            "AD_ENV_JSON": AD_ENV_JSON,
        },
        max_retry=10,
        checkpoint_dir=f"{REPO_ROOT}/runs_phase0",  # refund on tech errors
        health_params=JobHealthParams(
            log_period=60,
            action=JobHealthAction.restart,
            sub_actions=[JobHealthSubAction.notify],
        ),
    )
    # keep only kwargs the installed Job actually accepts
    accepted = set(inspect.signature(Job.__init__).parameters) - {"self"}
    return Job(**{k: v for k, v in job_config.items() if k in accepted})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--shards", default=None,
                        help="comma-separated shard indices (default: all)")
    args = parser.parse_args()

    indices = (
        [int(x) for x in args.shards.split(",")]
        if args.shards
        else list(range(N_SHARDS))
    )

    jobs = []
    for i in indices:
        job = make_job(i)
        if args.dry_run:
            print(f"[dry-run] shard {i}/{N_SHARDS}: {job!r}")
            continue
        result = job.submit()
        if isinstance(result, str) and "Error" in result:
            raise RuntimeError(f"shard {i} submit failed: {result}")
        print(f"submitted shard {i}/{N_SHARDS}: {job.job_name}")
        jobs.append(job)
        time.sleep(3)  # POST limit: 30/min

    if jobs:
        print("\npoll with (>=30 s interval):")
        print("  from rich.console import Console; c = Console()")
        print("  c.print(client_lib.jobs(region=RegionMT.SR006))")


if __name__ == "__main__":
    main()
