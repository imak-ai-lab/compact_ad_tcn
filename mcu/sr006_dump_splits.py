"""Phase 5, run ON SR006 (Jupyter terminal) from the repo root:

    cd /home/jovyan/kovalenko/industrial_ad-main && python sr006_dump_splits.py

Dumps, with the repo's own code (no re-implementation):
  * per-file val/test feature matrices (classic_27d_v1, preset scaler) exactly as
    TimeSeriesDataset.split_file cuts them -> ~/phase5_splits.npz
  * raw 3-channel signal of 2 test files (1 normal, 1 damaged) for the MCU
    feature-extraction check (stage 5) -> ~/phase5_raw.npz
  * the three seed-42 config.json are read to assert that all three models share
    the same feature pipeline and split patterns (window/horizon may differ).
Windowing is NOT done here; the host rebuilds windows per config from these
matrices, in the same file order and with the same start indices as the loader.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
from industrial_ad.datasets.PU.dataset import TimeSeriesDataset, discover_file_paths, load_file  # noqa: E402
from industrial_ad.datasets.PU.features import build_pu_feature_pipeline  # noqa: E402

RUNS = {
    "TR047": "runs_phase0/TCN-reconstruction/047-s42-sep1-hid48-lat4-bl4-ker5",
    "TL082": "runs_phase0/TCN-light-forecasting/082-s42-h240-sep1-hid64-bl4-ker5-fs60-tb2-mix128",
    "TR138": "runs_phase0/TCN-reconstruction/138-s42-sep0-hid64-lat4-bl5-ker3",
}
SHARED = ["root", "train_patterns", "val_patterns", "test_patterns", "feature_pipeline", "holdout_bearings"]

cfgs = {k: json.loads((REPO / p / "config.json").read_text()) for k, p in RUNS.items()}
ref = cfgs["TR047"]["dataset"]["params"]
for k, c in cfgs.items():
    p = c["dataset"]["params"]
    for key in SHARED:
        assert p.get(key) == ref.get(key), f"{k}: dataset.params.{key} differs from TR047"
    print(f"{k}: window={p['window_size']} overlap={p['window_overlap']} horizon={p['horizon_size']} task={c['task']['type']}")
    assert c["debug"]["dataset"] == {"train_file_limit": None, "val_file_limit": None, "test_file_limit": None}, k

root = ref["root"]
if not Path(root).is_absolute():
    root = str((REPO / root).resolve())
pipe = build_pu_feature_pipeline(ref["feature_pipeline"])
holdout = list(ref.get("holdout_bearings", []) or [])
splitter = {s: TimeSeriesDataset([], split=s, window_size=1, horizon_size=0, window_overlap=0,
                                 files_per_chunk=1, bytes_cache_limit=0, file_transform=None,
                                 target_builder=None, holdout_bearings=holdout) for s in ("val", "test")}

out: dict[str, np.ndarray] = {}
meta = {"root": root, "feature_pipeline": ref["feature_pipeline"], "holdout_bearings": holdout, "splits": {}}
raw_out: dict[str, np.ndarray] = {}
for split in ("val", "test"):
    paths = discover_file_paths(root, ref[f"{split}_patterns"], None)
    meta["splits"][split] = []
    for i, path in enumerate(paths):
        raw = load_file(path)
        feats = pipe(raw)
        part = splitter[split].split_file(feats, TimeSeriesDataset.is_anomaly(path), TimeSeriesDataset.bearing_code(path))
        if part is None:
            continue
        key = f"{split}/{i:04d}"
        out[key] = np.ascontiguousarray(part, dtype=np.float32)
        meta["splits"][split].append({"key": key, "path": Path(path).name, "bearing": TimeSeriesDataset.bearing_code(path),
                                      "is_anomaly": bool(TimeSeriesDataset.is_anomaly(path)), "rows": int(len(part))})
        if split == "test":
            kind = "anom" if TimeSeriesDataset.is_anomaly(path) else "norm"
            if f"raw_{kind}" not in raw_out:
                sel = raw[:, list(pipe.raw_channel_indices)].T.astype(np.float64)
                raw_out[f"raw_{kind}"] = sel
                raw_out[f"feat_{kind}"] = feats  # full-file host features of the same file
                meta[f"raw_{kind}_path"] = Path(path).name
        print(f"{split} {i+1}/{len(paths)} {Path(path).name} rows={0 if part is None else len(part)}", flush=True)

np.savez_compressed(Path.home() / "phase5_splits.npz", **out)
np.savez_compressed(Path.home() / "phase5_raw.npz", **raw_out)
(Path.home() / "phase5_splits_meta.json").write_text(json.dumps(meta, indent=1))
for f in ("phase5_splits.npz", "phase5_raw.npz", "phase5_splits_meta.json"):
    p = Path.home() / f
    print(f, p.stat().st_size, hashlib.md5(p.read_bytes()).hexdigest())
