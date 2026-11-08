"""CWRU loading, recording-level split (protocol lolo_v1), windowing and loaders.

The split is decided per RECORDING before any window is cut, from the
catalogue alone (`protocol_segments`). Windows never cross recordings and a
recording never feeds two splits, except the one documented concession:
the normal *validation* frames are the last `val_normal_fraction` of each
training recording (time-disjoint, not recording-disjoint) because CWRU has
four healthy recordings in total. The normal *test* recording is the held-out
load and is never seen in any form.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy.io import loadmat
from scipy.signal import decimate, filtfilt, firwin
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from industrial_ad.datasets.CWRU.catalog import BY_ID, FAULT_BEARING_IDS, Recording, quirks_of, recordings_for
from industrial_ad.datasets.CWRU.features import build_cwru_feature_pipeline, build_target_builder

PROTOCOL_NAME = "lolo_v1"
# v1b = identical split, but every recording (healthy AND faulty) goes through the
# same zero-phase FIR low-pass after being brought to 12 kHz. Control for the
# acquisition-chain confound: in v1 only the healthy class was decimated from 48 kHz.
PROTOCOL_VARIANTS = {"lolo_v1": {"common_lowpass": False}, "lolo_v1b": {"common_lowpass": True}}
TARGET_FS_HZ = 12_000
COMMON_LOWPASS_TAPS = 101


# --------------------------------------------------------------------------- raw
def load_recording_signal(root: str | Path, rec: Recording, target_fs: int = TARGET_FS_HZ,
                          common_lowpass_hz: float | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Read one .mat, pick the drive-end channel, bring it to `target_fs`.

    `common_lowpass_hz` (protocol v1b): after resampling, apply the SAME zero-phase FIR
    low-pass to every recording, so healthy (decimated) and faulty (native 12 kHz)
    signals share one band limit and features cannot read the acquisition chain."""
    path = Path(root) / "raw" / rec.mat_name
    if not path.exists():
        raise FileNotFoundError(f"CWRU file missing: {path} (run industrial_ad.datasets.CWRU.download)")
    mat = loadmat(str(path))
    keys = [k for k in mat if not k.startswith("__")]
    quirks = list(quirks_of(rec))

    if rec.de_key in mat:
        key = rec.de_key
    else:
        # 3001-3008 (0.028") store their DE series under a different X-number.
        de_keys = [k for k in keys if k.endswith("_DE_time")]
        if len(de_keys) != 1:
            raise KeyError(f"{path.name}: expected {rec.de_key} or exactly one *_DE_time variable, got {keys}")
        key = de_keys[0]
        quirks.append(f"de_key_fallback:{key}")
    foreign = [k for k in keys if k.endswith("_DE_time") and k != key]
    if foreign:
        quirks.append(f"foreign_de_variables:{','.join(foreign)}")

    signal = np.asarray(mat[key], dtype=np.float64).reshape(-1)
    n_raw = int(signal.shape[0])
    if rec.fs_hz != target_fs:
        if rec.fs_hz % target_fs:
            raise ValueError(f"{path.name}: cannot decimate {rec.fs_hz} -> {target_fs} by an integer factor")
        factor = rec.fs_hz // target_fs
        # FIR anti-alias + zero-phase: no group delay, so frame timing stays aligned.
        signal = decimate(signal, factor, ftype="fir", zero_phase=True)
        decimated_by = factor
    else:
        decimated_by = 1
    if common_lowpass_hz:
        if not 0 < common_lowpass_hz < target_fs / 2:
            raise ValueError(f"common_lowpass_hz must be in (0, {target_fs / 2})")
        taps = firwin(COMMON_LOWPASS_TAPS, common_lowpass_hz, fs=target_fs, window="hamming")
        signal = filtfilt(taps, [1.0], signal)
    meta = {
        "file": rec.mat_name,
        "variable": key,
        "bearing_id": rec.bearing_id,
        "load_hp": rec.load_hp,
        "fs_published_hz": rec.fs_hz,
        "fs_used_hz": target_fs,
        "decimated_by": decimated_by,
        "common_lowpass_hz": float(common_lowpass_hz) if common_lowpass_hz else None,
        "n_samples_raw": n_raw,
        "n_samples_used": int(signal.shape[0]),
        "duration_s": round(signal.shape[0] / target_fs, 3),
        "quirks": quirks,
    }
    return signal.astype(np.float32), meta


# ----------------------------------------------------------------------- split
@dataclass(frozen=True)
class Segment:
    recording: Recording
    split: str          # train | val | test
    start_frac: float   # time slice of the recording's feature frames
    end_frac: float

    @property
    def is_anomaly(self) -> bool:
        return self.recording.is_anomaly


def protocol_segments(params: dict[str, Any]) -> tuple[dict[str, list[Segment]], dict[str, Any]]:
    """Protocol lolo_v1: leave-one-load-out for the healthy class, bearing-wise for faults."""
    protocol_name = str(params.get("protocol", PROTOCOL_NAME))
    if protocol_name not in PROTOCOL_VARIANTS:
        raise ValueError(f"unknown CWRU protocol {protocol_name!r} (known: {sorted(PROTOCOL_VARIANTS)})")
    lowpass = params["feature_pipeline"]["params"].get("common_lowpass_hz")
    if PROTOCOL_VARIANTS[protocol_name]["common_lowpass"] != bool(lowpass):
        raise ValueError(f"protocol {protocol_name}: common_lowpass_hz must be "
                         f"{'set' if PROTOCOL_VARIANTS[protocol_name]['common_lowpass'] else 'absent'} (got {lowpass!r})")
    fold = int(params["fold_load_hp"])
    if fold not in (0, 1, 2, 3):
        raise ValueError("fold_load_hp must be 0..3")
    val_frac = float(params["val_normal_fraction"])
    if not 0.0 < val_frac < 0.5:
        raise ValueError("val_normal_fraction must be in (0, 0.5)")
    val_bearings = [str(b) for b in params["val_bearings"]]
    unknown = sorted(set(val_bearings) - set(FAULT_BEARING_IDS))
    if unknown:
        raise ValueError(f"val_bearings not in catalogue: {unknown}")
    sizes = params.get("include_sizes_in")
    sizes = [float(s) for s in sizes] if sizes else None
    test_bearings = [b for b in FAULT_BEARING_IDS if b not in val_bearings]
    train_loads = [l for l in (0, 1, 2, 3) if l != fold]

    segments: dict[str, list[Segment]] = {"train": [], "val": [], "test": []}
    for rec in recordings_for(["N"], loads=train_loads):
        segments["train"].append(Segment(rec, "train", 0.0, 1.0 - val_frac))
        segments["val"].append(Segment(rec, "val", 1.0 - val_frac, 1.0))
    for rec in recordings_for(["N"], loads=[fold]):
        segments["test"].append(Segment(rec, "test", 0.0, 1.0))
    for rec in recordings_for(val_bearings, sizes_in=sizes):
        segments["val"].append(Segment(rec, "val", 0.0, 1.0))
    for rec in recordings_for(test_bearings, loads=[fold], sizes_in=sizes):
        segments["test"].append(Segment(rec, "test", 0.0, 1.0))

    # Invariants the paper text relies on. Loud, because the whole point of the
    # phase is that these hold.
    test_ids = {s.recording.file_id for s in segments["test"]}
    train_ids = {s.recording.file_id for s in segments["train"]}
    val_anom_bearings = {s.recording.bearing_id for s in segments["val"] if s.is_anomaly}
    test_anom_bearings = {s.recording.bearing_id for s in segments["test"] if s.is_anomaly}
    assert not (test_ids & train_ids), "a recording is in both train and test"
    assert not (test_ids & {s.recording.file_id for s in segments["val"]}), "a recording is in both val and test"
    assert not (val_anom_bearings & test_anom_bearings), "a faulty bearing is in both val and test"
    assert all(s.recording.load_hp == fold for s in segments["test"]), "test must be the held-out load only"
    assert sum(1 for s in segments["test"] if not s.is_anomaly) == 1, "exactly one healthy test recording per fold"
    assert not any(s.is_anomaly for s in segments["train"]), "train must be healthy-only"

    description = {
        "protocol": protocol_name,
        "common_lowpass_hz": float(lowpass) if lowpass else None,
        "fold_load_hp": fold,
        "train_loads_hp": train_loads,
        "val_normal_fraction": val_frac,
        "val_bearings": val_bearings,
        "test_bearings": test_bearings,
        "include_sizes_in": sizes,
        "unit_of_split": "recording (physical bearing x load)",
        "healthy_class_holdout": "condition-wise (load); bearing-wise impossible: CWRU has one healthy bearing",
        "fault_class_holdout": "bearing-wise (val and test bearings disjoint)",
        "normal_val_caveat": "last val_normal_fraction of each TRAIN recording (time-disjoint, not recording-disjoint)",
    }
    return segments, description


# --------------------------------------------------------------------- windows
def _window_starts(n_frames: int, window: int, horizon: int, *, split: str, train_step: int, eval_count: int,
                   label: str) -> np.ndarray:
    n_starts = n_frames - window - horizon + 1
    if n_starts <= 0:
        raise ValueError(f"{label}: {n_frames} frames < window {window} + horizon {horizon}")
    if split == "train":
        return np.arange(0, n_starts, int(train_step), dtype=np.int64)
    if n_starts < eval_count:
        raise ValueError(f"{label}: only {n_starts} possible windows, eval_windows_per_recording={eval_count} — "
                         "lower the count so every evaluated recording contributes the same number of windows")
    # Fixed number of uniformly spaced windows per evaluated segment, so a long
    # recording cannot win a record-level max/quantile by sheer count.
    return np.unique(np.linspace(0, n_starts - 1, int(eval_count)).round().astype(np.int64))


class CWRUWindowDataset(IterableDataset):
    """Yield `(features, target, is_anomaly)` windows from pre-computed, normalised frames."""

    def __init__(self, segments: list[Segment], features: dict[int, np.ndarray], *, split: str, window_size: int,
                 horizon_size: int, train_window_step: int, eval_windows_per_recording: int, target_builder) -> None:
        super().__init__()
        self.segments = list(segments)
        self.features = features
        self.split = split
        self.window_size = int(window_size)
        self.horizon_size = int(horizon_size)
        self.train_window_step = int(train_window_step)
        self.eval_windows_per_recording = int(eval_windows_per_recording)
        self.target_builder = target_builder
        self.plan: list[tuple[Segment, np.ndarray, np.ndarray]] = []
        for seg in self.segments:
            frames = self.features[seg.recording.file_id]
            lo, hi = int(seg.start_frac * len(frames)), int(seg.end_frac * len(frames))
            sliced = frames[lo:hi]
            starts = _window_starts(len(sliced), self.window_size, self.horizon_size, split=split,
                                    train_step=self.train_window_step, eval_count=self.eval_windows_per_recording,
                                    label=f"{seg.recording.mat_name}[{seg.split}]")
            self.plan.append((seg, sliced, starts))

    def __len__(self) -> int:  # informative only (IterableDataset)
        return int(sum(len(starts) for _, _, starts in self.plan))

    def windows_per_segment(self) -> list[dict[str, Any]]:
        return [{"file": seg.recording.mat_name, "bearing_id": seg.recording.bearing_id, "load_hp": seg.recording.load_hp,
                 "is_anomaly": seg.is_anomaly, "frames": int(len(sliced)), "windows": int(len(starts))}
                for seg, sliced, starts in self.plan]

    def _emit(self, sliced: np.ndarray, start: int, is_anomaly: bool):
        window = np.asarray(sliced[start:start + self.window_size], dtype=np.float32)
        future = np.asarray(sliced[start + self.window_size:start + self.window_size + self.horizon_size], dtype=np.float32)
        target = np.asarray(self.target_builder(window, future), dtype=np.float32)
        return window, target, is_anomaly

    def __iter__(self):
        indices = list(range(len(self.plan)))
        worker = get_worker_info()
        if worker is not None:
            indices = indices[worker.id::worker.num_workers]
        if self.split == "train":
            random.shuffle(indices)
            buffer = []
            for i in indices:
                seg, sliced, starts = self.plan[i]
                buffer.extend((i, int(s)) for s in starts)
            random.shuffle(buffer)
            for i, start in buffer:
                seg, sliced, _ = self.plan[i]
                yield self._emit(sliced, start, seg.is_anomaly)
        else:
            for i in indices:
                seg, sliced, starts = self.plan[i]
                for start in starts:
                    yield self._emit(sliced, int(start), seg.is_anomaly)

    def iter_with_meta(self) -> Iterator[tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]]:
        """Deterministic pass (no workers) with recording metadata — for per-window score dumps."""
        for seg, sliced, starts in self.plan:
            rec = seg.recording
            for start in starts:
                window, target, is_anomaly = self._emit(sliced, int(start), seg.is_anomaly)
                yield window, target, is_anomaly, {"file_id": rec.file_id, "bearing_id": rec.bearing_id,
                                                   "fault": rec.fault, "size_in": rec.size_in, "load_hp": rec.load_hp,
                                                   "start_frame": int(start)}


# ---------------------------------------------------------------------- bundle
def build_cwru_datasets(config: dict[str, Any]) -> dict[str, Any]:
    """Load, split by recording, fit the fold's scaler on train frames only, window."""
    params = config["dataset"]["params"]
    root = Path(params["root"])
    pipeline = build_cwru_feature_pipeline(params["feature_pipeline"])
    target_builder = build_target_builder(config["task"]["type"])
    segments, protocol = protocol_segments(params)

    needed = {seg.recording.file_id for split in segments.values() for seg in split}
    raw_features: dict[int, np.ndarray] = {}
    recording_meta: dict[str, Any] = {}
    for file_id in sorted(needed):
        signal, meta = load_recording_signal(root, BY_ID[file_id], target_fs=int(params["feature_pipeline"]["params"]["sampling_rate"]),
                                             common_lowpass_hz=params["feature_pipeline"]["params"].get("common_lowpass_hz"))
        raw_features[file_id] = pipeline.raw_features(signal)
        meta["n_frames"] = int(raw_features[file_id].shape[0])
        recording_meta[str(file_id)] = meta

    # Scaler: train frames only — the train slice of the train recordings.
    train_mats, train_names = [], []
    for seg in segments["train"]:
        frames = raw_features[seg.recording.file_id]
        lo, hi = int(seg.start_frac * len(frames)), int(seg.end_frac * len(frames))
        train_mats.append(frames[lo:hi])
        train_names.append(f"{seg.recording.mat_name}[{seg.start_frac:.2f},{seg.end_frac:.2f})")
    provenance = pipeline.fit_train_scaler(train_mats, train_names, extra={"fold_load_hp": protocol["fold_load_hp"]})
    features = {fid: pipeline.normalise(mat) for fid, mat in raw_features.items()}

    common = {
        "window_size": int(params["window_size"]),
        "horizon_size": int(params["horizon_size"]),
        "train_window_step": int(params["train_window_step"]),
        "eval_windows_per_recording": int(params["eval_windows_per_recording"]),
        "target_builder": target_builder,
    }
    datasets = {split: CWRUWindowDataset(segments[split], features, split=split, **common) for split in ("train", "val", "test")}
    first = next(iter(datasets["train"]))

    def _count(split: str, anomalous: bool) -> int:
        return sum(1 for s in segments[split] if s.is_anomaly == anomalous)

    metadata = {
        "dataset": "cwru",
        "protocol": protocol,
        "scaler_provenance": provenance,
        "recordings": recording_meta,
        "train_file_count": len(segments["train"]),
        "val_file_count": len(segments["val"]),
        "test_file_count": len(segments["test"]),
        "recording_counts": {s: {"normal": _count(s, False), "anomalous": _count(s, True)} for s in segments},
        "window_counts": {s: len(datasets[s]) for s in datasets},
        "windows_per_segment": {s: datasets[s].windows_per_segment() for s in ("val", "test")},
        "input_shape": list(np.asarray(first[0]).shape),
        "target_shape": list(np.asarray(first[1]).shape),
        "quirks_seen": sorted({q for m in recording_meta.values() for q in m["quirks"]}),
    }
    return {"datasets": datasets, "metadata": metadata}


def build_cwru_dataloaders(config: dict[str, Any]) -> dict[str, Any]:
    bundle = build_cwru_datasets(config)
    loader_config = dict(config["dataset"]["loader"])
    bundle["loaders"] = {split: DataLoader(ds, **loader_config) for split, ds in bundle["datasets"].items()}
    return bundle


__all__ = [
    "CWRUWindowDataset",
    "PROTOCOL_NAME",
    "PROTOCOL_VARIANTS",
    "Segment",
    "build_cwru_dataloaders",
    "build_cwru_datasets",
    "load_recording_signal",
    "protocol_segments",
]
