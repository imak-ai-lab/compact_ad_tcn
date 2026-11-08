#!/usr/bin/env python3
"""Write a synthetic CWRU look-alike (same file ids, variable names, sampling
rates, lengths and known quirks) so the phase-2 code path can be exercised on
any CPU without the real data.

    python scripts/phase2/make_synthetic_cwru.py --dst /tmp/cwru_synth

The signals are NOT bearing physics — a broadband noise floor plus periodic
impulse trains at the nominal defect frequencies, with a fault-size-dependent
amplitude. Good enough for: loader keys, decimation, protocol invariants,
window counts, a ROC-AUC that is above chance. Never for a number in the paper.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from industrial_ad.datasets.CWRU.catalog import RECORDINGS  # noqa: E402

# nominal SKF 6205 (drive end) characteristic orders
ORDERS = {"IR": 5.415, "B": 4.7135, "OR": 3.585}
# real lengths (samples) of the published files
LEN_NORMAL_48K = {0: 243_938, 1: 483_903, 2: 485_643, 3: 485_643}
LEN_FAULT_12K = 121_265


def synth(rec, rng: np.random.Generator) -> np.ndarray:
    fs = rec.fs_hz
    n = LEN_NORMAL_48K[rec.load_hp] if rec.fault == "N" else LEN_FAULT_12K
    t = np.arange(n) / fs
    shaft_hz = rec.rpm / 60.0
    x = 0.05 * rng.standard_normal(n) + 0.02 * np.sin(2 * np.pi * shaft_hz * t)
    if rec.fault != "N":
        f_defect = ORDERS[rec.fault] * shaft_hz
        amp = 0.15 + 8.0 * float(rec.size_in)          # 0.007" -> 0.21, 0.028" -> 0.37
        period = int(fs / f_defect)
        impulses = np.zeros(n)
        impulses[::period] = 1.0
        ring = np.exp(-np.arange(0, int(fs * 0.002)) / (fs * 0.0004)) * np.sin(2 * np.pi * 3000 * np.arange(0, int(fs * 0.002)) / fs)
        x = x + amp * np.convolve(impulses, ring)[:n] * (1 + 0.3 * rng.standard_normal(n))
    return x.astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    raw = Path(args.dst) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for rec in RECORDINGS:
        x = synth(rec, rng)[:, None]
        payload = {rec.de_key: x, f"X{rec.file_id:03d}_FE_time": 0.5 * x, f"X{rec.file_id:03d}RPM": np.array([[rec.rpm]])}
        if rec.file_id >= 3001:
            # published 0.028" files name their series after another X-number
            payload = {f"X{rec.file_id - 2945:03d}_DE_time": x}
        if rec.file_id == 99:
            payload["X098_DE_time"] = 0.9 * x  # foreign variable quirk of the real 99.mat
        savemat(str(raw / rec.mat_name), payload)
    print(f"[synthetic cwru] {len(RECORDINGS)} files -> {raw}")


if __name__ == "__main__":
    main()
