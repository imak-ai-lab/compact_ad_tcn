"""Phase 5 / stage 5a (host only): how far are block-wise features from the whole-file features the models saw?

The repo's pipeline (features.py) denoises a WHOLE file (4 s, ~256 000 samples): wavelet coefficients, MAD and the
universal threshold all come from the entire file, and 640 samples are trimmed at the file ends. A device only ever
holds a block, so the on-device pipeline cannot be the same function of the data. This script quantifies the gap with
the repo's own functions, on the raw test files dumped from SR006 (data/phase5_raw.npz: 1 normal, 1 damaged file):

  0. reproduce: local pywt/scipy on the full raw file vs the SR006 feature dump (tol 1e-4 abs, scaled units) PASS/FAIL
  1. for block lengths N: denoise + features on the block alone, trim as stated, take the T = 120 rows,
     compare with the same 120 rows of the whole-file feature matrix, in scaled units (scaler std = 1).
Rows -> features_block_vs_file.csv. Nothing here runs on the board.
"""
from __future__ import annotations

import json

import numpy as np

from p5common import CKPT_DIR, MCU, append_csv, host_tag, now, verdict
from industrial_ad.datasets.PU.features import (CLASSIC_27D_FEATURE_STATS, build_pu_feature_pipeline,
                                                process_bearing_signals_to_features)

OUT = MCU / "features_block_vs_file.csv"
T, WIN, HOP = 120, 64, 16
LEN_RAW = WIN + (T - 1) * HOP            # 1968 samples under the 120 rows
TOL_REPRO = 1e-4
N_BLOCKS, RNG = 200, 2028

selection = json.loads((CKPT_DIR / "selection.json").read_text())
cfg = json.loads((CKPT_DIR / f"TR047_s{selection['TR047']}.config.json").read_text())
fp = cfg["dataset"]["params"]["feature_pipeline"]
pipe = build_pu_feature_pipeline(fp)
print("feature_pipeline params:", fp["params"])
FS, DROP = int(pipe.sampling_rate), int(pipe.drop_edges)
mean = np.asarray(CLASSIC_27D_FEATURE_STATS["mean"], dtype=np.float64)
std = np.asarray(CLASSIC_27D_FEATURE_STATS["std"], dtype=np.float64)
NAMES = ["Mean", "Variance", "P2P", "LineInt", "Skew", "SpecEnergy", "SpecSpread", "WavD2", "WavD1"]

d = np.load(MCU / "data" / "phase5_raw.npz")
rows = []
for kind in ("norm", "anom"):
    raw, ref = d[f"raw_{kind}"], d[f"feat_{kind}"]
    f, _ = process_bearing_signals_to_features(raw, WIN, HOP, FS, DROP)
    full = ((f.T.astype(np.float32) - mean.astype(np.float32)) / std.astype(np.float32)).astype(np.float32)
    err = float(np.abs(full - ref).max()) if full.shape == ref.shape else float("inf")
    rows.append(dict(date=now(), host=host_tag(), file=kind, block_N="file", trim="file ends", check="reproduce_sr006_dump",
                     value=f"{err:.3e}", tol=TOL_REPRO, status=verdict(err < TOL_REPRO), detail=f"shape {full.shape}",
                     fs=FS, drop_edges=DROP))
    print(kind, "reproduce max abs diff (scaled):", err, full.shape)

    rng = np.random.default_rng(RNG)
    for n, trim in ((LEN_RAW + 2 * DROP, DROP), (LEN_RAW, 0)):
        # the block starts so that its 120 rows coincide with rows r..r+119 of the whole-file matrix
        r_max = (raw.shape[1] - 2 * DROP - LEN_RAW) // HOP
        rs = rng.integers(DROP // HOP + 1, r_max - DROP // HOP - 1, N_BLOCKS)
        diffs = []
        for r in rs:
            s = DROP + r * HOP - trim
            fb, _ = process_bearing_signals_to_features(raw[:, s:s + n], WIN, HOP, FS, trim)
            b = (fb.T - mean) / std
            assert b.shape == (T, 27), b.shape
            diffs.append(np.abs(b - full[r:r + T].astype(np.float64)))
        diffs = np.stack(diffs)                                   # [blocks, 120, 27], scaled units
        per_feat = diffs.reshape(-1, 27)
        worst = int(per_feat.mean(axis=0).argmax())
        rows.append(dict(date=now(), host=host_tag(), file=kind, block_N=n, trim=trim, check="block_vs_file_scaled_abs",
                         value=f"{per_feat.mean():.4f}", tol="", status="INFO",
                         detail=f"median {np.median(per_feat):.4f} p99 {np.quantile(per_feat, 0.99):.4f} max {per_feat.max():.3f}; "
                                f"worst feature ch{worst // 9} {NAMES[worst % 9]} mean {per_feat.mean(axis=0)[worst]:.4f}; {N_BLOCKS} blocks",
                         fs=FS, drop_edges=DROP))
        print(f"{kind} N={n} trim={trim}: mean {per_feat.mean():.4f} median {np.median(per_feat):.4f} "
              f"p99 {np.quantile(per_feat, 0.99):.4f} max {per_feat.max():.3f}; worst ch{worst // 9} {NAMES[worst % 9]}")
append_csv(OUT, rows, fresh=True)
print("->", OUT)
