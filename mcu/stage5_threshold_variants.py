"""Phase 5 / stage 5a (host only): what causes the block-vs-file feature gap, and which on-device threshold closes it.

Block N = 3248 (120 rows + 640 trimmed samples per edge). Same blocks as a fair comparison for every variant:
  V0  repo function on the block: MAD of the block's finest details, N = block length in sigma*sqrt(2 ln N)
  V1  block MAD, N = whole-file length (only the sqrt(2 ln N) factor is corrected)
  V2  threshold of the whole file (not available on a device: separates threshold from boundary effects)
  V3  fixed per-channel threshold calibrated on the NORMAL file only, used for both files (deployable: a constant)
Compared with the whole-file matrix in scaled units. Rows -> features_threshold_variants.csv. Host only.
"""
from __future__ import annotations

import json

import numpy as np
import pywt

from p5common import CKPT_DIR, MCU, append_csv, host_tag, now
from industrial_ad.datasets.PU.features import (CLASSIC_27D_FEATURE_STATS, build_pu_feature_pipeline,
                                                extract_features_vectorized)

OUT = MCU / "features_threshold_variants.csv"
T, WIN, HOP, N_BLOCKS, RNG = 120, 64, 16, 200, 2028
SPECS = [("sym8", 6), ("sym8", 6), ("db4", 4)]
selection = json.loads((CKPT_DIR / "selection.json").read_text())
cfg = json.loads((CKPT_DIR / f"TR047_s{selection['TR047']}.config.json").read_text())
pipe = build_pu_feature_pipeline(cfg["dataset"]["params"]["feature_pipeline"])
FS, DROP = int(pipe.sampling_rate), int(pipe.drop_edges)
LEN_RAW = WIN + (T - 1) * HOP
N = LEN_RAW + 2 * DROP
mean = np.asarray(CLASSIC_27D_FEATURE_STATS["mean"], dtype=np.float64)
std = np.asarray(CLASSIC_27D_FEATURE_STATS["std"], dtype=np.float64)
NAMES = ["Mean", "Variance", "P2P", "LineInt", "Skew", "SpecEnergy", "SpecSpread", "WavD2", "WavD1"]


def sigma_of(x, wavelet, level):
    d = pywt.wavedec(x, wavelet, mode="per", level=level)[-1]
    return float(np.median(np.abs(d - np.median(d))) / 0.6745)


def denoise(x, wavelet, level, thr):
    c = pywt.wavedec(x, wavelet, mode="per", level=level)
    c = [c[0]] + [pywt.threshold(d, value=thr, mode="soft") for d in c[1:]]
    return pywt.waverec(c, wavelet, mode="per")[: len(x)]


def feats(block, thrs, trim):
    out = []
    for ch, (w, lv) in enumerate(SPECS):
        y = denoise(block[ch], w, lv, thrs[ch])
        y = y[trim:-trim] if trim else y
        out.append(np.asarray(list(extract_features_vectorized(y, WIN, HOP, FS).values())))
    return (np.vstack(out).T - mean) / std


def file_matrix(raw, thrs):
    return feats(raw, thrs, DROP)


d = np.load(MCU / "data" / "phase5_raw.npz")
raws = {k: d[f"raw_{k}"] for k in ("norm", "anom")}
n_file = raws["norm"].shape[1]
sig_file = {k: [sigma_of(r[c], *SPECS[c]) for c in range(3)] for k, r in raws.items()}
thr_file = {k: [sig_file[k][c] * np.sqrt(2 * np.log(raws[k].shape[1])) for c in range(3)] for k in raws}
thr_cal = thr_file["norm"]                                  # V3: calibrated on the normal file only
print("file thresholds:", {k: np.round(v, 4).tolist() for k, v in thr_file.items()})

rows = []
for kind, raw in raws.items():
    full = file_matrix(raw, thr_file[kind])
    rng = np.random.default_rng(RNG)
    r_max = (raw.shape[1] - 2 * DROP - LEN_RAW) // HOP
    rs = rng.integers(DROP // HOP + 1, r_max - DROP // HOP - 1, N_BLOCKS)
    res = {v: [] for v in ("V0", "V1", "V2", "V3")}
    thr_blocks = []
    for r in rs:
        s = DROP + r * HOP - DROP
        b = raw[:, s:s + N]
        sig = [sigma_of(b[c], *SPECS[c]) for c in range(3)]
        t0 = [sig[c] * np.sqrt(2 * np.log(N)) for c in range(3)]
        thr_blocks.append(t0)
        variants = {"V0": t0, "V1": [sig[c] * np.sqrt(2 * np.log(n_file)) for c in range(3)],
                    "V2": thr_file[kind], "V3": thr_cal}
        for v, t in variants.items():
            res[v].append(np.abs(feats(b, t, DROP) - full[r:r + T]))
    tb = np.asarray(thr_blocks)
    print(f"{kind}: block threshold mean {np.round(tb.mean(0), 4).tolist()} std {np.round(tb.std(0), 4).tolist()}")
    for v, a in res.items():
        a = np.stack(a).reshape(-1, 27)
        worst = int(a.mean(axis=0).argmax())
        print(f"  {kind} {v}: mean {a.mean():.4f} median {np.median(a):.4f} p99 {np.quantile(a, .99):.4f} max {a.max():.3f}; "
              f"worst ch{worst // 9} {NAMES[worst % 9]} {a.mean(axis=0)[worst]:.3f}")
        rows.append(dict(date=now(), host=host_tag(), file=kind, block_N=N, trim=DROP, variant=v, mean_abs=f"{a.mean():.4f}",
                         median_abs=f"{np.median(a):.4f}", p99_abs=f"{np.quantile(a, .99):.4f}", max_abs=f"{a.max():.3f}",
                         worst_feature=f"ch{worst // 9} {NAMES[worst % 9]}", n_blocks=N_BLOCKS, units="scaler std",
                         thr_file=" ".join(f"{t:.5f}" for t in thr_file[kind]), thr_calibrated=" ".join(f"{t:.5f}" for t in thr_cal)))
append_csv(OUT, rows, fresh=True)
print("->", OUT)
