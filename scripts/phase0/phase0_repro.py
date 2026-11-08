"""
Phase 0: reproducibility run of headline configs on the A100 cluster.

What it does:
  1. Re-enumerates the ORIGINAL sweep configs by executing the family runner
     scripts (runners/<family>.py) with `industrial_ad.run_experiments`
     monkey-patched to capture configs instead of training. This guarantees the
     phase-0 configs are bit-identical to the submitted-paper sweep configs
     (no manual re-construction from tags).
  2. Filters the captured configs down to the headline set from
     phase0_manifest.json (produced by select_phase0_configs.py).
  3. Expands seeds: original 42/43/44 (repro comparison) + 45/46 (new, for
     5-seed headline stats). Extra seeds are cloned from the seed-42 config.
  4. Disables TF32 (A100 defaults) so the arithmetic matches the original FP32
     runs, writes an environment report, and trains into ./runs_phase0/.

Usage (from repo root, on the cluster):
    python scripts/phase0/phase0_repro.py --list-only          # sanity: print plan
    python scripts/phase0/phase0_repro.py --shard 0/6          # one of 6 shards
    python scripts/phase0/phase0_repro.py                      # everything (slow)

Shards split the run list into N contiguous chunks (grouping by dataset config
is preserved, so the shared data bundle keeps working). Submit one job per
shard: see phase0_submit.py.

After all shards finish: python scripts/phase0/phase0_compare.py
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import re
import runpy
import sys
from pathlib import Path


# --- optional: re-exec into an NFS env (проверенный L2D-паттерн) --------------
# Если воркер должен работать на python из NFS-окружения (например, 3.11 при
# базовом образе py310), bootstrap-скрипт пишет ad_env.json с путём к python;
# раннер сам в него пере-запускается. Если файла нет — no-op, работаем на
# python-е образа.
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
        print(f"[phase0] exec {py} (from {sys.version.split()[0]})", flush=True)
        os.execve(py, [py, os.path.abspath(__file__)] + sys.argv[1:], os.environ)


_attach_nfs_env()


# --- self-service deps (путь «как в edge-gnn»: без venv) ----------------------
# torch есть в базовом образе; недостающие лёгкие пакеты раннер доставляет сам
# в ~/.local текущего питона (NFS -> один раз на все job-ы). У job-ов internet
# включён по умолчанию. Гонки между шардами избегаем порядком запуска: сначала
# шард 0 (smoke) ставит всё, остальные шарды стартуют позже и видят готовое.
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
        print(f"[phase0] installing missing deps for {sys.executable}: {missing}", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--user", "--no-input", *missing]
        )


_ensure_deps()

# --- repo root on sys.path (script lives in scripts/phase0/) -----------------
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# --- TF32 OFF: must happen before any model/training code runs ---------------
# A100 enables TF32 for conv/matmul by default; the paper numbers are FP32
# (consumer GPU). Keep the arithmetic comparable.
import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import industrial_ad  # noqa: E402  (also forces mp start method 'fork')

MANIFEST = Path(__file__).resolve().parent / "phase0_manifest.json"
RUNNERS_DIR = REPO / "runners"
OUT_ROOT = REPO / "runs_phase0"

REPRO_SEEDS = [42, 43, 44]
EXTRA_SEEDS = [45, 46]

SEED_RE = re.compile(r"^(\d{3})-s(\d+)-")


def seedless(name: str) -> str:
    """'138-s42-sep0-...' -> '138-sep0-...' (matches summary_table run_name)."""
    return SEED_RE.sub(r"\1-", name)


def seed_of(name: str) -> int:
    m = SEED_RE.match(name)
    if not m:
        raise ValueError(f"run name without seed part: {name}")
    return int(m.group(2))


def capture_family_configs(family: str) -> list[dict]:
    """Execute runners/<family>.py, capturing configs instead of training."""
    runner_path = RUNNERS_DIR / f"{family}.py"
    if not runner_path.exists():
        raise FileNotFoundError(f"runner not found: {runner_path}")

    captured: list[dict] = []

    def _capture(configs, **kwargs):
        captured.extend(configs)

    original = industrial_ad.run_experiments
    original_argv = sys.argv
    try:
        industrial_ad.run_experiments = _capture
        sys.argv = [str(runner_path)]  # no slice args -> full sweep enumerated
        runpy.run_path(str(runner_path), run_name="__main__")
    finally:
        industrial_ad.run_experiments = original
        sys.argv = original_argv
    return captured


def build_phase0_configs(manifest: list[dict]) -> list[dict]:
    wanted: dict[str, set[str]] = {}
    for entry in manifest:
        wanted.setdefault(entry["family"], set()).add(entry["run_name"])

    phase0: list[dict] = []
    for family, run_names in sorted(wanted.items()):
        captured = capture_family_configs(family)
        by_name: dict[tuple[str, int], dict] = {}
        for config in captured:
            name = config["run"]["name"]
            by_name[(seedless(name), seed_of(name))] = config

        for run_name in sorted(run_names):
            base_seeds = [s for (n, s) in by_name if n == run_name]
            if not base_seeds:
                raise KeyError(
                    f"{family}/{run_name}: not found in runner enumeration — "
                    f"runner script changed since the sweep?"
                )
            # original seeds, as enumerated by the runner
            for seed in sorted(base_seeds):
                phase0.append(_relocate(by_name[(run_name, seed)], family))
            # extra seeds cloned from the lowest original seed
            template = by_name[(run_name, min(base_seeds))]
            for seed in EXTRA_SEEDS:
                clone = copy.deepcopy(template)
                old_name = clone["run"]["name"]
                new_name = SEED_RE.sub(rf"\1-s{seed}-", old_name)
                clone["run"]["name"] = new_name
                clone["run"]["seed"] = seed
                phase0.append(_relocate(clone, family, name_override=new_name))
    return phase0


def _relocate(config: dict, family: str, name_override: str | None = None) -> dict:
    """Point the run dir at runs_phase0/<family>/<name> (family = parent dir)."""
    config = copy.deepcopy(config)
    name = name_override or config["run"]["name"]
    config["run"]["name"] = name
    config["run"]["dir"] = str(OUT_ROOT / family / name)
    # Датасет в исходных конфигах задан ОТНОСИТЕЛЬНО ("./data/PU") — launcher на
    # кластере запускает скрипт не из корня репо, glob находил 0 файлов и все
    # runs мгновенно падали. Делаем путь абсолютным.
    data_root = Path(config["dataset"]["params"]["root"])
    if not data_root.is_absolute():
        config["dataset"]["params"]["root"] = str((REPO / data_root).resolve())
    return config


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
        "repro_seeds": REPRO_SEEDS,
        "extra_seeds": EXTRA_SEEDS,
    }
    (OUT_ROOT / "env_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # шард приходит либо CLI-флагом, либо env-переменной PHASE0_SHARD (submit
    # передаёт оба; env — проверенный на этой платформе канал, как в L2D)
    parser.add_argument(
        "--shard",
        default=os.environ.get("PHASE0_SHARD", "0/1"),
        help="i/N contiguous shard of the run list",
    )
    # NB: parse_known_args, не parse_args — pytorch2-launcher (torch.distributed.
    # launch) добавляет свой --local-rank=N, argparse не должен на нём падать.
    # (Именно поэтому L2D-раннер вообще не использовал argparse.)
    parser.add_argument("--list-only", action="store_true", help="print the plan and exit")
    parser.add_argument("--manifest", default=str(MANIFEST))
    args, _launcher_args = parser.parse_known_args()
    if _launcher_args:
        print(f"[phase0] ignoring launcher args: {_launcher_args}", flush=True)

    assert (REPO / "summary_table.csv").exists(), (
        f"run from repo root layout; summary_table.csv not found under {REPO}"
    )
    # cluster guards: right region; TMPDIR must exist (empty/missing dir breaks OMPI)
    mls_region = os.environ.get("MLS_JOB_REGION_NAME")
    if mls_region is not None:
        assert mls_region == "SR006", f"wrong region: {mls_region}"
    tmpdir = os.environ.get("TMPDIR")
    if tmpdir:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32

    manifest = json.loads(Path(args.manifest).read_text())
    configs = build_phase0_configs(manifest)

    # stable order: keeps identical dataset configs adjacent (data-bundle reuse)
    configs.sort(key=lambda c: (
        json.dumps({"d": c["dataset"], "t": c["task"]}, sort_keys=True),
        c["run"]["dir"],
    ))

    shard_index, shard_total = map(int, args.shard.split("/"))
    assert 0 <= shard_index < shard_total, f"bad --shard {args.shard}"
    chunk = (len(configs) + shard_total - 1) // shard_total
    shard_configs = configs[shard_index * chunk : (shard_index + 1) * chunk]

    print(f"phase0: {len(configs)} total runs, shard {args.shard} -> {len(shard_configs)} runs")
    for config in shard_configs:
        print("  ", Path(config["run"]["dir"]).parent.name, config["run"]["name"])
    if args.list_only:
        return

    os.chdir(REPO)  # страховка для любых оставшихся относительных путей

    data_root = REPO / "data" / "PU"
    parsed = [p for p in data_root.glob("K*") if p.is_dir()]
    assert parsed, f"PU dataset not found/empty at {data_root} — upload it first"
    print(f"[phase0] data: {len(parsed)} bearing dirs under {data_root}", flush=True)

    write_env_report()
    results = industrial_ad.run_experiments(
        shard_configs, skip_existing=True, stop_on_error=False
    )

    # Никаких молчаливых провалов: сводка + ненулевой exit, чтобы job не
    # отчитывался Completed, когда всё упало (случилось 2026-08-13).
    errors = [r for r in results if r.get("error")]
    print(f"[phase0] finished: {len(results) - len(errors)} ok, {len(errors)} failed", flush=True)
    for r in errors[:5]:
        print(f"[phase0]   FAILED {r.get('run_name')}: {r.get('exception')}", flush=True)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
