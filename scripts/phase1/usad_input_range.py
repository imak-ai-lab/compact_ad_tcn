"""
Phase 1: per-channel min/max of the training split, for USAD's own normalization.

USAD's phase-2 objective maximizes reconstruction error. That term is bounded
only when the decoder output is bounded — in the original pipeline the inputs are
min-max scaled to [0, 1] and the decoder ends in a sigmoid. Feeding raw z-scored
features with an identity head makes the term unbounded and training diverges
(observed on this data: loss -> 1e14 in 5 of 6 runs).

This script restores the author's setup: it walks the training split once and
writes the per-channel min/max that the USAD configs then pass to the model as
`input_range`. Train only — using val/test statistics would leak.

Usage (from repo root; no GPU needed, ~3-5 min):
    python scripts/phase1/usad_input_range.py

Writes scripts/phase1/usad_input_range.json (read by phase1_configs.py) and
prints the values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from industrial_ad import DEFAULT_EXPERIMENT_CONFIG, clone_config  # noqa: E402
from industrial_ad.datasets.PU.dataset import build_pu_dataloaders  # noqa: E402

OUT = Path(__file__).resolve().parent / "usad_input_range.json"


def dataset_config() -> dict:
    """The phase-1 dataset config, built without importing phase1_configs.

    Importing it here would be circular: the USAD entries resolve `input_range`
    from the very file this script writes. Only the dataset section matters for
    the statistics, and phase 1 leaves it at the project defaults (it only
    changes the loader batch size for the Anomaly Transformer).
    """
    config = clone_config(DEFAULT_EXPERIMENT_CONFIG)
    config["task"] = {"type": "reconstruction"}
    config["dataset"]["params"]["root"] = str((REPO / "data" / "PU").resolve())
    return config


def main() -> None:
    config = dataset_config()
    print(f"[usad-range] window={config['dataset']['params']['window_size']} "
          f"root={config['dataset']['params']['root']}", flush=True)
    bundle = build_pu_dataloaders(config)

    minimum = maximum = None
    windows = 0
    for x, _, is_anomaly in bundle["loaders"]["train"]:
        assert int(is_anomaly.sum()) == 0, "train split must be normal-only"
        batch_min = x.amin(dim=(0, 1))
        batch_max = x.amax(dim=(0, 1))
        minimum = batch_min if minimum is None else torch.minimum(minimum, batch_min)
        maximum = batch_max if maximum is None else torch.maximum(maximum, batch_max)
        windows += int(x.shape[0])

    if minimum is None:
        sys.exit("train split produced no batches")

    payload = {
        "note": "per-channel min/max of the 27 z-scored features over the TRAIN split only",
        "windows": windows,
        "channels": int(minimum.shape[0]),
        "min": [round(float(v), 6) for v in minimum],
        "max": [round(float(v), 6) for v in maximum],
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
