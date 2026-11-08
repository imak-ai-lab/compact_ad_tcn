"""CWRU feature pipeline: the PU handcrafted family on ONE vibration channel.

`classic_9d_v1` = the 9 per-channel features of PU's `classic_27d_v1`
(`extract_features_vectorized`) applied to the drive-end accelerometer only,
with the PU vibration denoising recipe (db4, level 4). The functions are
imported from the PU module, not copied, so the two pipelines cannot drift.

What is deliberately different from PU:

* there is NO preset scaler. `CLASSIC_27D_FEATURE_STATS` was fitted on PU
  frames of unrecorded provenance; on CWRU the mean/std are fitted on the
  training frames of the current fold (`fit_train_scaler`) and written to the
  run's config.json together with the recordings they came from.
* `sampling_rate` is the true 12 kHz (it only scales spectral centroid/spread).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from industrial_ad.datasets.PU.features import (
    build_target_builder,
    denoise_signal_dwt,
    extract_features_vectorized,
)

FEATURE_NAMES = [
    "Mean", "Variance", "Peak-to-Peak", "Line Integral", "Skewness",
    "Spectral Energy", "Spectral Spread", "Wavelet Energy D2", "Wavelet Energy D1",
]
CHANNEL_SPEC = ("de_vibration", "db4", 4)  # same recipe as PU's vibration channel
PIPELINE_NAME = "classic_9d_v1"
SCALER_MODE = "train_fitted"


@dataclass
class CWRUFeaturePipeline:
    """Raw 1-D drive-end signal (already at 12 kHz) -> (frames, 9) feature matrix."""

    aggregation_window: int
    aggregation_step: int
    sampling_rate: int
    drop_edges: int
    scaler_mean: np.ndarray | None = None
    scaler_std: np.ndarray | None = None
    scaler_provenance: dict[str, Any] = field(default_factory=dict)

    def raw_features(self, signal: np.ndarray) -> np.ndarray:
        """Unnormalised (frames, 9) features — used both for scaler fitting and for windows."""
        _, wavelet, level = CHANNEL_SPEC
        clean = denoise_signal_dwt(np.asarray(signal, dtype=np.float64), wavelet=wavelet, level=level)
        if self.drop_edges:
            clean = clean[self.drop_edges:-self.drop_edges]
        feats = extract_features_vectorized(clean, window=self.aggregation_window,
                                            step=self.aggregation_step, fs=self.sampling_rate)
        matrix = np.stack([feats[name] for name in FEATURE_NAMES], axis=1)
        return matrix.astype(np.float32)

    def normalise(self, feature_matrix: np.ndarray) -> np.ndarray:
        if self.scaler_mean is None or self.scaler_std is None:
            raise RuntimeError("CWRU pipeline used before fit_train_scaler(): normalisation stats are "
                               "train-fitted per fold and must never be inherited from PU.")
        return ((feature_matrix - self.scaler_mean) / self.scaler_std).astype(np.float32)

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        return self.normalise(self.raw_features(signal))

    def fit_train_scaler(self, train_feature_matrices: list[np.ndarray], recordings: list[str],
                         extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fit mean/std on the given (train-only) frames and record where they came from."""
        if not train_feature_matrices:
            raise RuntimeError("fit_train_scaler: no training frames")
        frames = np.concatenate(train_feature_matrices, axis=0).astype(np.float64)
        mean = frames.mean(axis=0)
        std = frames.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        self.scaler_mean = mean.astype(np.float32)
        self.scaler_std = std.astype(np.float32)
        digest = hashlib.sha1(np.concatenate([self.scaler_mean, self.scaler_std]).tobytes()).hexdigest()
        self.scaler_provenance = {
            "mode": SCALER_MODE,
            "pipeline": PIPELINE_NAME,
            "fitted_on_recordings": list(recordings),
            "n_frames": int(frames.shape[0]),
            "feature_names": list(FEATURE_NAMES),
            "mean": [float(v) for v in self.scaler_mean],
            "std": [float(v) for v in self.scaler_std],
            "sha1": digest,
            **(extra or {}),
        }
        return self.scaler_provenance


def build_cwru_feature_pipeline(config: dict[str, Any]) -> CWRUFeaturePipeline:
    """Build the pipeline selected in `dataset.params.feature_pipeline` (scaler unfitted)."""
    if str(config["name"]).lower() != PIPELINE_NAME:
        raise ValueError(f"Unknown CWRU feature pipeline: {config['name']} (expected {PIPELINE_NAME})")
    params = dict(config["params"])
    if str(params.get("scaler", "")).lower() != SCALER_MODE:
        raise ValueError("CWRU feature pipeline requires params.scaler == 'train_fitted'; "
                         "a preset scaler is not allowed on this dataset (PU stats must not be inherited).")
    return CWRUFeaturePipeline(
        aggregation_window=int(params["aggregation_window"]),
        aggregation_step=int(params["aggregation_step"]),
        sampling_rate=int(params["sampling_rate"]),
        drop_edges=int(params["drop_edges"]),
    )


__all__ = [
    "CHANNEL_SPEC",
    "CWRUFeaturePipeline",
    "FEATURE_NAMES",
    "PIPELINE_NAME",
    "SCALER_MODE",
    "build_cwru_feature_pipeline",
    "build_target_builder",
]
