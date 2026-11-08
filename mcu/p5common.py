"""Shared helpers for phase 5 (MCU) host-side scripts.

Every number goes to a CSV row together with date / host / seed / conditions.
The windowing here mirrors
industrial_ad.datasets.PU.dataset.TimeSeriesDataset.__iter__ exactly.
"""
from __future__ import annotations

import csv
import os
import datetime as _dt
import json
import platform
import sys
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parent.parent
# P5_WORK: alternative work dir (synthetic smoke tests); default = this mcu/ folder
MCU = Path(os.environ.get("P5_WORK") or Path(__file__).resolve().parent)
# published repository: mcu/ sits next to industrial_ad/; local paper folder: code/industrial_ad-main
REPO = PAPER if (PAPER / "industrial_ad").is_dir() else PAPER / "code" / "industrial_ad-main"
sys.path.insert(0, str(REPO))

# The repo forces multiprocessing 'fork' on import (Linux DataLoader workers).
# Windows has no 'fork'; phase-5 scripts use no workers, so make that one call a
# no-op here instead of editing the published repo copy.
if sys.platform == "win32":
    import multiprocessing as _mp
    _orig_ssm = _mp.set_start_method
    _mp.set_start_method = lambda method, force=False: None if method == "fork" else _orig_ssm(method, force)

# code -> (family dir, seedless run_name, paper reference)
MODELS = {
    "TR047": ("TCN-reconstruction", "047-sep1-hid48-lat4-bl4-ker5",
              {"params": 45251, "mmac": 5.2416, "roc": 0.9381}),
    "TL082": ("TCN-light-forecasting", "082-h240-sep1-hid64-bl4-ker5-fs60-tb2-mix128",
              {"params": 87355, "mmac": 11.05152, "roc": 0.9290}),
    "TR138": ("TCN-reconstruction", "138-sep0-hid64-lat4-bl5-ker3",
              {"params": 252323, "mmac": 29.96736, "roc": 0.9431}),
}
SEEDS_REPRO = [42, 43, 44]
SEEDS_ALL = [42, 43, 44, 45, 46]

RAW_DIR = MCU / "checkpoints_raw"      # tarball from SR006 extracted here
CKPT_DIR = MCU / "checkpoints"
SPLITS = MCU / "data" / "phase5_splits.npz"
SPLITS_META = MCU / "data" / "phase5_splits_meta.json"


def seeded_name(run_name: str, seed: int) -> str:
    """'047-sep1-...' -> '047-s42-sep1-...' (phase-0 directory naming)."""
    head, rest = run_name.split("-", 1)
    return f"{head}-s{seed}-{rest}"


def run_dir(code: str, seed: int) -> Path:
    family, name, _ = MODELS[code]
    return RAW_DIR / family / seeded_name(name, seed)


def now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def host_tag() -> str:
    return f"{platform.node()}|{platform.processor() or platform.machine()}|py{platform.python_version()}"


def append_csv(path: Path, rows: list[dict], *, fresh: bool = False) -> None:
    """Write rows (dicts with identical keys). fresh=True rewrites the file."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if fresh or not path.exists() else "a"
    with path.open(mode, newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if mode == "w":
            w.writeheader()
        w.writerows(rows)


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# ------------------------------------------------------------- Cube.AI ----
STEDGEAI = r"C:\ST\stedgeai-10.2.0\Utilities\windows\stedgeai.exe"
ARM_GCC_BIN = r"C:\ST\arm-gnu-14.2\bin"
CUBEPROG = r"C:\ST\STMProgrammer\bin\STM32_Programmer_CLI.exe"   # STM32CubeProgrammer v2.23.0 (installed 2026-11-05)
ST_VID = 0x0483                                                   # ST-LINK V3 USB vendor id
# Same options for analyze / validate / generate. int8 models keep float I/O so the
# firmware feeds every model the same float window and the measured latency
# includes input quantisation / output dequantisation.
CUBEAI_OPTS = {
    "fp32": ["--target", "stm32h7"],
    "int8": ["--target", "stm32h7", "--input-data-type", "float32", "--output-data-type", "float32"],
}


def cubeai_env() -> dict:
    env = dict(os.environ)
    env["PATH"] = ARM_GCC_BIN + os.pathsep + env["PATH"]
    return env


# ---------------------------------------------------------------- windows ----
def load_split_matrices(split: str):
    """Return [(matrix[T_rows, 27], is_anomaly, key)] in loader file order."""
    meta = json.loads(SPLITS_META.read_text())
    data = np.load(SPLITS)
    return [(data[e["key"]], bool(e["is_anomaly"]), e["key"]) for e in meta["splits"][split]]


def build_windows(config: dict, split: str, *, normal_only: bool = False):
    """Rebuild (x[N,T,C], y[N,T',C], label[N]) for one config, as the loader does."""
    p = config["dataset"]["params"]
    win, hor = int(p["window_size"]), int(p["horizon_size"])
    step = win - int(p["window_overlap"])
    task = config["task"]["type"].lower()
    xs, ys, ls = [], [], []
    for mat, is_anom, _ in load_split_matrices(split):
        if normal_only and is_anom:
            continue
        if len(mat) < win + hor:
            continue
        for s in range(0, len(mat) - win - hor + 1, step):
            w = mat[s:s + win]
            xs.append(w)
            ys.append(w if task == "reconstruction" else mat[s + win:s + win + hor])
            ls.append(int(is_anom))
    return (np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32),
            np.asarray(ls, dtype=np.int64))
