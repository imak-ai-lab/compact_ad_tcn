"""Run ON SR006 (Jupyter terminal) from the repo root, one trip for three things:

    cd /home/jovyan/kovalenko/industrial_ad-main && python sr006_dump_bundle.py

Reads only, writes three files to the home directory and packs them into ~/phase5_sr006_bundle.tgz:
  A. phase5_block_features.npz + .meta.json: BLOCK-WISE features, what the microcontroller computes (one block, threshold
     V1 = block MAD * sqrt(2 ln 256000)), for every 4th window of the PU test split and of the normal val split, so that
     the laptop can compute ROC-AUC of the whole chain with the deployed models. For TL082 also the block-wise forecast
     targets. The same windows of the whole-file features are already on the laptop (phase5_splits.npz), for a paired difference.
  B. phase5_n09_features.npz + .meta.json: whole-file features of the N09_M07_F10 condition (900 rpm), which no model has
     seen: val and test regions cut exactly as for N15, for the operating-condition test.
  C. phase0_summaries.tgz: every runs_phase0/**/summary.json, for the label-free checkpoint comparison (E1).
Expected: well under an hour on CPU, bundle about 300 MB. Nothing in the repo or in runs_phase0 is modified.

On the laptop `python sr006_dump_bundle.py --selftest` checks the block code against the firmware reference (needs no data root).
"""
from __future__ import annotations

import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pywt

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
if "--selftest" in sys.argv:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code" / "industrial_ad-main"))
    if sys.platform == "win32":                      # the repo forces 'fork' on import; no workers are used here
        import multiprocessing as _mp
        _orig = _mp.set_start_method
        _mp.set_start_method = lambda method, force=False: None if method == "fork" else _orig(method, force)
from industrial_ad.datasets.PU.features import (CLASSIC_27D_FEATURE_STATS, build_pu_feature_pipeline,  # noqa: E402
                                                extract_features_vectorized)

WIN, HOP, DROP, T, STEP, HORIZON_TL = 64, 16, 640, 120, 23, 240
STRIDE = 4                               # every 4th window
K_V1 = float(np.sqrt(2 * np.log(256000)))
SPECS = [("sym8", 6), ("sym8", 6), ("db4", 4)]
MEAN = np.asarray(CLASSIC_27D_FEATURE_STATS["mean"], dtype=np.float32)
STD = np.asarray(CLASSIC_27D_FEATURE_STATS["std"], dtype=np.float32)


def block_features(raw3: np.ndarray, first_row: int, rows: int, fs: int) -> np.ndarray:
    """Scaled features of whole-file rows first_row .. first_row + rows - 1, computed from their block alone.

    Whole-file row r starts at raw sample DROP + r * HOP, so the block with DROP samples of margin on each side starts
    at r * HOP. Same steps as the firmware (firmware/features/main.c, threshold variant V1).
    """
    n = WIN + (rows - 1) * HOP + 2 * DROP
    b = raw3[:, first_row * HOP: first_row * HOP + n]
    if b.shape[1] != n:
        raise ValueError(f"block out of file: row {first_row}, {rows} rows, file {raw3.shape[1]}")
    out = []
    for ch, (w, lv) in enumerate(SPECS):
        c = pywt.wavedec(b[ch], w, mode="per", level=lv)
        d = c[-1]
        thr = np.median(np.abs(d - np.median(d))) / 0.6745 * K_V1
        y = pywt.waverec([c[0]] + [pywt.threshold(q, value=thr, mode="soft") for q in c[1:]], w, mode="per")[:n]
        out.append(np.asarray(list(extract_features_vectorized(y[DROP:-DROP], WIN, HOP, fs).values())))
    return ((np.vstack(out).T.astype(np.float32) - MEAN) / STD).astype(np.float32)


def selftest():
    """Against the firmware reference (build_features.py) and against the whole-file rows of the same file."""
    mcu = Path(__file__).resolve().parent
    w = np.load(mcu / "data" / "stage5_blocks.npz")
    x, y1 = w["x"].astype(np.float64), w["y_v1"]
    worst = 0.0
    for k in range(len(x)):
        got = block_features(x[k], 0, T, 40000)
        worst = max(worst, float(np.abs(got - y1[k]).max()))
    print(f"block_features vs firmware reference: max abs {worst:.3e} (must be 0)")
    assert worst == 0.0
    d = np.load(mcu / "data" / "phase5_raw.npz")
    for kind in ("norm", "anom"):
        raw, full = d[f"raw_{kind}"], d[f"feat_{kind}"]
        rows = [1001, 3333, 5002, 7777, 12003]     # not multiples of 4: a block on the file's dyadic grid would look too good
        right = [np.abs(block_features(raw, r, T, 40000) - full[r:r + T]) for r in rows]
        wrong = [np.abs(block_features(raw, r, T, 40000) - full[r + 1:r + T + 1]) for r in rows]
        med, med_wrong = float(np.median(np.stack(right))), float(np.median(np.stack(wrong)))
        print(f"{kind}: median |block - whole file| = {med:.4f} scaled units; with the rows shifted by one = {med_wrong:.4f}")
        assert med < 0.03 < 0.05 < med_wrong
    n = WIN + (HORIZON_TL - 1) * HOP + 2 * DROP
    tgt = block_features(d["raw_norm"], 1000, HORIZON_TL, 40000)
    assert tgt.shape == (HORIZON_TL, 27), tgt.shape
    print(f"target block length {n}, shape {tgt.shape}: selftest PASS")


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    from industrial_ad.datasets.PU.dataset import TimeSeriesDataset, discover_file_paths, load_file

    home = Path.home()
    run = REPO / "runs_phase0/TCN-reconstruction/047-s42-sep1-hid48-lat4-bl4-ker5"
    ref = json.loads((run / "config.json").read_text())["dataset"]["params"]
    tl = json.loads((REPO / "runs_phase0/TCN-light-forecasting/082-s42-h240-sep1-hid64-bl4-ker5-fs60-tb2-mix128/config.json").read_text())["dataset"]["params"]
    assert (int(ref["window_size"]), int(ref["window_overlap"]), int(ref["horizon_size"])) == (T, T - STEP, 0)
    assert (int(tl["window_size"]), int(tl["window_overlap"]), int(tl["horizon_size"])) == (T, T - STEP, HORIZON_TL)
    root = ref["root"] if Path(ref["root"]).is_absolute() else str((REPO / ref["root"]).resolve())
    pipe = build_pu_feature_pipeline(ref["feature_pipeline"])
    fs = int(pipe.sampling_rate)
    assert fs == 40000 and int(pipe.drop_edges) == DROP and int(pipe.aggregation_window) == WIN and int(pipe.aggregation_step) == HOP
    holdout = list(ref.get("holdout_bearings", []) or [])
    splitter = {s: TimeSeriesDataset([], split=s, window_size=1, horizon_size=0, window_overlap=0, files_per_chunk=1, bytes_cache_limit=0,
                                     file_transform=None, target_builder=None, holdout_bearings=holdout) for s in ("val", "test")}
    chans = list(pipe.raw_channel_indices)

    # ---------------------------------------------------------------------------- A: block-wise features ----
    t0 = time.time()
    out, meta = {}, {"stride": STRIDE, "threshold": "block MAD * sqrt(2 ln 256000)", "fs": fs, "splits": {}}
    for split in ("val", "test"):
        paths = discover_file_paths(root, ref[f"{split}_patterns"], None)
        meta["splits"][split] = []
        for i, path in enumerate(paths):
            anom = bool(TimeSeriesDataset.is_anomaly(path))
            if split == "val" and anom:
                continue                                  # only normal val windows are used, for the scorer
            bearing = TimeSeriesDataset.bearing_code(path)
            raw = load_file(path)[:, chans].T.astype(np.float64)
            n_rows = (raw.shape[1] - 2 * DROP - WIN) // HOP + 1
            region = splitter[split].split_file(np.arange(n_rows)[:, None], anom, bearing)
            if region is None or len(region) < T:
                continue
            r0, length = int(region[0, 0]), len(region)
            starts = np.arange(0, length - T + 1, STEP)[::STRIDE]
            key = f"{split}/{i:04d}"
            out[f"{key}/x"] = np.stack([block_features(raw, r0 + int(s), T, fs) for s in starts])
            tl_s = starts[starts <= length - T - HORIZON_TL]
            if len(tl_s):
                out[f"{key}/y_tl"] = np.stack([block_features(raw, r0 + int(s) + T, HORIZON_TL, fs) for s in tl_s])
            meta["splits"][split].append({"key": key, "path": Path(path).name, "bearing": bearing, "is_anomaly": anom, "region_first_row": r0,
                                          "region_rows": length, "starts": starts.tolist(), "n_tl": int(len(tl_s))})
            print(f"A {split} {i + 1}/{len(paths)} {Path(path).name} windows={len(starts)} tl={len(tl_s)} {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(home / "phase5_block_features.npz", **out)
    (home / "phase5_block_features.meta.json").write_text(json.dumps(meta), encoding="utf-8")

    # ---------------------------------------------------------------------------- B: N09 whole-file features ----
    out, meta = {}, {"feature_pipeline": ref["feature_pipeline"], "splits": {}}
    for split in ("val", "test"):
        patterns = [p.replace("N15", "N09") for p in ref[f"{split}_patterns"]]
        paths = discover_file_paths(root, patterns, None)
        print(f"B {split}: {len(paths)} N09 files, patterns {patterns}", flush=True)
        meta["splits"][split] = []
        for i, path in enumerate(paths):
            anom = bool(TimeSeriesDataset.is_anomaly(path))
            part = splitter[split].split_file(pipe(load_file(path)), anom, TimeSeriesDataset.bearing_code(path))
            if part is None:
                continue
            key = f"{split}/{i:04d}"
            out[key] = np.ascontiguousarray(part, dtype=np.float32)
            meta["splits"][split].append({"key": key, "path": Path(path).name, "bearing": TimeSeriesDataset.bearing_code(path),
                                          "is_anomaly": anom, "rows": int(len(part))})
    if out:
        np.savez_compressed(home / "phase5_n09_features.npz", **out)
        (home / "phase5_n09_features.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    else:
        print("B: no N09 files found, skipped", flush=True)

    # ---------------------------------------------------------------------------- C: summaries ----
    found = sorted((REPO / "runs_phase0").rglob("summary.json"))
    with tarfile.open(home / "phase0_summaries.tgz", "w:gz") as tf:
        for p in found:
            tf.add(p, arcname=str(p.relative_to(REPO)))
    print(f"C: {len(found)} summary.json", flush=True)

    names = ["phase5_block_features.npz", "phase5_block_features.meta.json", "phase5_n09_features.npz", "phase5_n09_features.meta.json", "phase0_summaries.tgz"]
    with tarfile.open(home / "phase5_sr006_bundle.tgz", "w") as tf:
        for n in names:
            if (home / n).exists():
                tf.add(home / n, arcname=n)
    for n in names + ["phase5_sr006_bundle.tgz"]:
        p = home / n
        if p.exists():
            print(n, p.stat().st_size, md5(p))
    print(f"done in {time.time() - t0:.0f} s: download ~/phase5_sr006_bundle.tgz")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
