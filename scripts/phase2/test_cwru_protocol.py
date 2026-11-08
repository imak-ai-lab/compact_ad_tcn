#!/usr/bin/env python3
"""Protocol invariants for CWRU lolo_v1 — run before anything goes to the cluster.

    python scripts/phase2/test_cwru_protocol.py --root /tmp/cwru_synth     # synthetic
    python scripts/phase2/test_cwru_protocol.py --root ./data/CWRU          # real data

Checks, for every fold 0..3:
  (a) no recording appears in two splits (train/val/test), except the documented
      normal-val tail of the train recordings, which must be time-disjoint;
  (b) the healthy test recording is exactly the held-out load, and is not in train/val;
  (c) validation and test faulty bearings are disjoint sets of physical bearings;
  (d) the scaler was fitted on train frames only (recomputed here independently);
  (e) every evaluated recording contributes the same number of windows;
  (f) input windows are (120, 9) and normalised train frames have ~0 mean / ~1 std;
  (g) train is healthy-only and train windows never reach into the val tail.
Exit 1 on the first failed check.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import industrial_ad  # noqa: E402
from industrial_ad.datasets.CWRU import DEFAULT_CWRU_DATASET_CONFIG, build_cwru_datasets, protocol_segments  # noqa: E402
from industrial_ad.datasets.CWRU.catalog import BY_ID  # noqa: E402
from industrial_ad.datasets.CWRU.dataset import load_recording_signal  # noqa: E402
from industrial_ad.datasets.CWRU.features import build_cwru_feature_pipeline  # noqa: E402
from industrial_ad.utils import clone_config  # noqa: E402


def check(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    print(f"  ok: {msg}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    for fold in range(4):
        print(f"fold {fold}")
        config = industrial_ad.make_default_config()
        config["dataset"] = clone_config(DEFAULT_CWRU_DATASET_CONFIG)
        config["dataset"]["params"]["root"] = args.root
        config["dataset"]["params"]["fold_load_hp"] = fold
        config["task"] = {"type": "reconstruction"}
        params = config["dataset"]["params"]

        segments, desc = protocol_segments(params)
        ids = {s: {seg.recording.file_id for seg in segments[s]} for s in segments}
        # (a)
        check(not (ids["train"] & ids["test"]), "train ∩ test = ∅ (recordings)")
        check(not (ids["val"] & ids["test"]), "val ∩ test = ∅ (recordings)")
        shared = ids["train"] & ids["val"]
        check(shared == ids["train"], "train recordings reappear in val only as their normal tail")
        for seg in segments["val"]:
            if seg.recording.file_id in shared:
                tr = next(t for t in segments["train"] if t.recording.file_id == seg.recording.file_id)
                check(tr.end_frac <= seg.start_frac, f"{seg.recording.mat_name}: val tail starts where train ends")
        # (b)
        healthy_test = [seg for seg in segments["test"] if not seg.is_anomaly]
        check(len(healthy_test) == 1 and healthy_test[0].recording.load_hp == fold, "one healthy test recording = held-out load")
        check(all(seg.recording.load_hp != fold for seg in segments["train"]), "held-out load absent from train")
        # (c)
        vb = {seg.recording.bearing_id for seg in segments["val"] if seg.is_anomaly}
        tb = {seg.recording.bearing_id for seg in segments["test"] if seg.is_anomaly}
        check(bool(vb) and bool(tb) and not (vb & tb), f"val bearings {sorted(vb)} disjoint from test bearings ({len(tb)})")
        # (g)
        check(all(not seg.is_anomaly for seg in segments["train"]), "train is healthy-only")

        bundle = build_cwru_datasets(config)
        meta = bundle["metadata"]
        # (d) recompute the scaler from scratch
        pipe = build_cwru_feature_pipeline(params["feature_pipeline"])
        mats = []
        for seg in segments["train"]:
            sig, _ = load_recording_signal(args.root, BY_ID[seg.recording.file_id])
            f = pipe.raw_features(sig)
            mats.append(f[int(seg.start_frac * len(f)):int(seg.end_frac * len(f))])
        frames = np.concatenate(mats).astype(np.float64)
        prov = meta["scaler_provenance"]
        check(np.allclose(frames.mean(0), prov["mean"], rtol=1e-4, atol=1e-5), "scaler mean == train-frame mean")
        check(np.allclose(np.where(frames.std(0) < 1e-12, 1, frames.std(0)), prov["std"], rtol=1e-4, atol=1e-5), "scaler std == train-frame std")
        check(prov["n_frames"] == frames.shape[0], f"scaler fitted on {prov['n_frames']} train frames only")
        check(all("[0.00,0.80)" in r for r in prov["fitted_on_recordings"]), "scaler provenance lists train slices only")
        # (e)
        for split in ("val", "test"):
            counts = {w["windows"] for w in meta["windows_per_segment"][split]}
            check(counts == {params["eval_windows_per_recording"]}, f"{split}: every evaluated segment has {counts} windows")
        # (f)
        check(meta["input_shape"] == [120, 9], f"input shape {meta['input_shape']}")
        train_ds = bundle["datasets"]["train"]
        stacked = np.concatenate([sl for _, sl, _ in train_ds.plan])
        check(abs(stacked.mean()) < 1e-3 and abs(stacked.std() - 1) < 5e-2, "normalised train frames ~N(0,1)")
        # (g) train windows stay inside the train slice
        for seg, sliced, starts in train_ds.plan:
            check(starts[-1] + train_ds.window_size + train_ds.horizon_size <= len(sliced), f"{seg.recording.mat_name}: train windows inside train slice")
        print(f"  windows: {meta['window_counts']}  quirks: {len(meta['quirks_seen'])}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
