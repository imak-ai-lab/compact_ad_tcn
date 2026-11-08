"""Case Western Reserve University bearing dataset (12 kHz drive-end, protocol lolo_v1)."""

from industrial_ad.datasets.CWRU.catalog import BEARING_IDS, BY_ID, FAULT_BEARING_IDS, RECORDINGS, Recording, recordings_for
from industrial_ad.datasets.CWRU.dataset import (
    CWRUWindowDataset,
    PROTOCOL_NAME,
    PROTOCOL_VARIANTS,
    Segment,
    build_cwru_dataloaders,
    build_cwru_datasets,
    load_recording_signal,
    protocol_segments,
)
from industrial_ad.datasets.CWRU.features import CWRUFeaturePipeline, FEATURE_NAMES, build_cwru_feature_pipeline

# Dataset block for DEFAULT_EXPERIMENT_CONFIG["dataset"] — phase-2 configs swap
# the PU block for this one and change nothing else in the captured config.
DEFAULT_CWRU_DATASET_CONFIG = {
    "name": "cwru",
    "params": {
        "root": "./data/CWRU",
        "protocol": PROTOCOL_NAME,
        "fold_load_hp": 0,
        "val_bearings": ["IR014", "B014", "OR014@6"],
        "val_normal_fraction": 0.2,
        "include_sizes_in": [0.007, 0.014, 0.021, 0.028],
        "window_size": 120,
        "horizon_size": 0,
        "train_window_step": 2,
        "eval_windows_per_recording": 150,
        "feature_pipeline": {
            "name": "classic_9d_v1",
            "params": {
                "aggregation_window": 64,
                "aggregation_step": 16,
                "sampling_rate": 12000,
                "drop_edges": 640,
                "scaler": "train_fitted",
                "common_lowpass_hz": None,   # set (e.g. 5000) together with protocol "lolo_v1b"
            },
        },
    },
    "loader": {
        "batch_size": 256,
        "num_workers": 4,
        "persistent_workers": True,
        "pin_memory": True,
    },
}

__all__ = [
    "BEARING_IDS",
    "BY_ID",
    "CWRUFeaturePipeline",
    "CWRUWindowDataset",
    "DEFAULT_CWRU_DATASET_CONFIG",
    "FAULT_BEARING_IDS",
    "FEATURE_NAMES",
    "PROTOCOL_NAME",
    "PROTOCOL_VARIANTS",
    "RECORDINGS",
    "Recording",
    "Segment",
    "build_cwru_dataloaders",
    "build_cwru_datasets",
    "build_cwru_feature_pipeline",
    "load_recording_signal",
    "protocol_segments",
    "recordings_for",
]
