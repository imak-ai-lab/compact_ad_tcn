"""PU dataset discovery, parquet loading, window streaming and dataloader setup."""

from __future__ import annotations

import glob
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from industrial_ad.datasets.PU.features import build_pu_feature_pipeline, build_target_builder


def merge_interp(df: pd.DataFrame, other_dfs: tuple[pd.DataFrame, ...]) -> None:
    """Interpolate slow channels onto the HostService timestamp grid."""
    xs = df["ts"].values
    for other in other_dfs:
        xp = other["ts"].values
        for column in other.columns:
            if column == "ts":
                continue
            values = other[column].values
            if len(xp) == 0:
                df[column] = np.nan
            elif len(xp) == 1:
                df[column] = np.full(xs.shape, values[0])
            else:
                df[column] = np.interp(xs, xp, values)


def load_file(path: str | Path) -> np.ndarray:
    """Load one parsed PU run and align all channels to a single dataframe."""
    path = Path(path)
    host = pd.read_parquet(path / "HostService.parquet")
    medium = pd.read_parquet(path / "Mech_4kHz.parquet")
    slow = pd.read_parquet(path / "Temp_1Hz.parquet")
    merge_interp(host, (medium, slow))
    columns = [
        "phase_current_1",
        "phase_current_2",
        "vibration_1",
        "temp_2_bearing_module",
        "speed",
        "force",
        "torque",
    ]
    return host[columns].values


def discover_file_paths(root: str | Path, patterns: list[str], limit: int | None) -> list[str]:
    """Resolve the dataset glob patterns listed in the config."""
    root = Path(root)
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(str(root / pattern)))
    matches = sorted(dict.fromkeys(matches))
    if limit is not None:
        matches = matches[: int(limit)]
    return matches


class TimeSeriesDataset(IterableDataset):
    """Stream PU windows chunk by chunk so batches mix files without loading everything at once."""

    def __init__(
        self,
        file_paths: list[str],
        *,
        split: str,
        window_size: int,
        horizon_size: int,
        window_overlap: int,
        files_per_chunk: int,
        bytes_cache_limit: float,
        file_transform,
        target_builder,
        holdout_bearings: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.file_paths = list(file_paths)
        # Bearing-wise holdout. Empty (the paper protocol) keeps the original
        # behaviour: healthy recordings are cut in time, so the same physical
        # bearing AND the same 4-second recording feed both train and test.
        # That is a "segment-wise" split, the most optimistic level in the
        # taxonomy of arXiv:2509.22267. Listing bearings here moves them out of
        # training entirely, so the test normals come from bearings the model
        # has never seen.
        self.holdout_bearings = {str(b).upper() for b in (holdout_bearings or [])}
        self.split = split
        self.window_size = int(window_size)
        self.horizon_size = int(horizon_size)
        self.step = self.window_size - int(window_overlap)
        self.files_per_chunk = int(files_per_chunk)
        self.bytes_cache_limit = float(bytes_cache_limit)
        self.file_transform = file_transform
        self.target_builder = target_builder
        self.cache: dict[str, np.ndarray] = {}
        self.bytes_in_cache = 0

    @staticmethod
    def bearing_code(path: str | Path) -> str:
        """'N15_M07_F10_K001_1' -> 'K001'."""
        return Path(path).name.split("_")[-2].upper()

    @staticmethod
    def is_anomaly(path: str | Path) -> bool:
        """PU file names encode normal runs with the `K0*` prefix."""
        return not Path(path).name.split("_")[-2].startswith("K0")

    def split_file(self, data: np.ndarray, is_anomaly: bool, bearing: str = "") -> np.ndarray | None:
        """Slice one feature matrix into the requested train/val/test region."""
        num_rows = len(data)
        if (not is_anomaly) and bearing.upper() in self.holdout_bearings:
            # A held-out healthy bearing never trains and is never used to fit the
            # scorer or the threshold: the whole recording goes to test.
            split_slices = {"test": slice(None)}
        elif is_anomaly:
            split_slices = {
                "val": slice(None, int(0.5 * num_rows)),
                "test": slice(int(0.5 * num_rows), None),
            }
        else:
            split_slices = {
                "train": slice(None, int(0.9 * num_rows)),
                "val": slice(int(0.9 * num_rows), int(0.95 * num_rows)),
                "test": slice(int(0.95 * num_rows), None),
            }

        # Normal files feed all splits, while anomalous files are kept out of train.
        split_slice = split_slices.get(self.split)
        if split_slice is None:
            return None
        return data[split_slice]

    def _load_split_data(self, path: str, *, num_workers: int) -> np.ndarray | None:
        """Load one file, transform it once and keep it in a bounded in-memory cache."""
        if path in self.cache:
            return self.cache[path]

        transformed = self.file_transform(load_file(path))
        split_data = self.split_file(transformed, self.is_anomaly(path), self.bearing_code(path))
        if split_data is None:
            return None

        split_data = split_data.copy()
        next_size = split_data.nbytes
        if (self.bytes_in_cache + next_size) * num_workers <= self.bytes_cache_limit:
            self.cache[path] = split_data
            self.bytes_in_cache += next_size
        return split_data

    def __iter__(self):
        """Yield `(features, target, is_anomaly)` windows for one epoch."""
        if self.step <= 0:
            raise ValueError("window_overlap must be smaller than window_size.")

        # Split work across DataLoader workers before any expensive file processing.
        if self.split == "train":
            indices = [index for index, path in enumerate(self.file_paths) if not self.is_anomaly(path)]
        else:
            indices = list(range(len(self.file_paths)))

        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        if worker_info is not None:
            indices = indices[worker_info.id :: worker_info.num_workers]

        # Shuffle file order for train so chunk composition changes across epochs.
        if self.split == "train":
            random.shuffle(indices)

        # Process files in chunks so we can shuffle windows inside a small buffer.
        for chunk_start in range(0, len(indices), self.files_per_chunk):
            chunk_indices = indices[chunk_start : chunk_start + self.files_per_chunk]
            chunk_windows = []

            for index in chunk_indices:
                path = self.file_paths[index]
                split_data = self._load_split_data(path, num_workers=num_workers)
                if split_data is None or len(split_data) < self.window_size + self.horizon_size:
                    continue

                # Each file is expanded into overlapping windows first, then the
                # whole chunk is optionally shuffled before yielding to the loader.
                for start in range(0, len(split_data) - self.window_size - self.horizon_size + 1, self.step):
                    window = split_data[start : start + self.window_size]
                    future = split_data[start + self.window_size : start + self.window_size + self.horizon_size]
                    chunk_windows.append((window, future, self.is_anomaly(path)))

            # Shuffle train windows inside the chunk so a batch is not dominated by one file.
            if self.split == "train":
                random.shuffle(chunk_windows)

            for window, future, is_anomaly in chunk_windows:
                # Convert to float32 at the last moment so cached arrays stay in
                # their transformed form and targets can differ by task type.
                features = np.asarray(window, dtype=np.float32)
                target = np.asarray(self.target_builder(features, future), dtype=np.float32)
                yield features, target, is_anomaly


def build_pu_datasets(config: dict[str, Any]) -> dict[str, Any]:
    """Build the PU train/val/test datasets and record their tensor shapes."""
    dataset_params = config["dataset"]["params"]
    dataset_debug = config["debug"]["dataset"]
    feature_pipeline = build_pu_feature_pipeline(dataset_params["feature_pipeline"])
    target_builder = build_target_builder(config["task"]["type"])

    common_dataset_kwargs = {
        "window_size": int(dataset_params["window_size"]),
        "horizon_size": int(dataset_params["horizon_size"]),
        "window_overlap": int(dataset_params["window_overlap"]),
        "files_per_chunk": int(dataset_params["files_per_chunk"]),
        "bytes_cache_limit": float(dataset_params["bytes_cache_limit"]),
        "file_transform": feature_pipeline,
        "target_builder": target_builder,
        "holdout_bearings": list(dataset_params.get("holdout_bearings", []) or []),
    }

    train_paths = discover_file_paths(
        dataset_params["root"],
        dataset_params["train_patterns"],
        dataset_debug["train_file_limit"],
    )
    val_paths = discover_file_paths(
        dataset_params["root"],
        dataset_params["val_patterns"],
        dataset_debug["val_file_limit"],
    )
    test_paths = discover_file_paths(
        dataset_params["root"],
        dataset_params["test_patterns"],
        dataset_debug["test_file_limit"],
    )

    datasets = {
        "train": TimeSeriesDataset(train_paths, split="train", **common_dataset_kwargs),
        "val": TimeSeriesDataset(val_paths, split="val", **common_dataset_kwargs),
        "test": TimeSeriesDataset(test_paths, split="test", **common_dataset_kwargs),
    }

    # One sample from the train iterator is enough to store the runtime tensor shapes in config/history.
    first_sample = next(iter(datasets["train"]))
    metadata = {
        "train_file_count": len(train_paths),
        "val_file_count": len(val_paths),
        "test_file_count": len(test_paths),
        "input_shape": list(np.asarray(first_sample[0]).shape),
        "target_shape": list(np.asarray(first_sample[1]).shape),
    }
    return {"datasets": datasets, "metadata": metadata}


def build_pu_dataloaders(config: dict[str, Any]) -> dict[str, Any]:
    """Build the PU datasets together with their DataLoaders."""
    bundle = build_pu_datasets(config)
    loader_config = dict(config["dataset"]["loader"])
    dataloaders = {
        split: DataLoader(dataset, **loader_config)
        for split, dataset in bundle["datasets"].items()
    }
    bundle["loaders"] = dataloaders
    return bundle
