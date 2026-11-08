"""Phase 1: SOTA baseline configs.

Three published detectors — TranAD, Anomaly Transformer, USAD — under this
project's protocol: the same PU split, the same 120x27 feature windows, the same
optimizer/schedule/checkpoint rule as every model family in the paper. What stays
the authors' own: architecture, training objective and anomaly score.

Per method: the author configuration plus variants. 18 configs x 3 seeds = 54 runs
(the last four were added on 2026-08-14 after the first 18 runs exposed two problems:
USAD diverging without its own input scaling, and TranAD still improving at epoch 110).

Deliberate deviations (report them in the paper / Response):
  * window length is 120 for all baselines (author defaults: TranAD 10,
    USAD 12, Anomaly Transformer 100) — required for an identical protocol.
    For TranAD this turned out to matter: the model predicts only the last frame
    of its input window, so a 120-frame window means 1/120 of the evidence is
    scored. TA010/TA011 keep the 120-frame evaluation unit but restore the
    author's length-10 stride-1 scan inside it (see models/sota/tranad.py);
  * TranAD/USAD end in identity instead of sigmoid, because this project's
    features are z-scored and not min-max scaled; the sigmoid variant is run as
    an ablation ("-sig" configs) so the choice is backed by numbers;
  * the Anomaly Transformer runs at batch 256 (author default) instead of 1024:
    its association matrices are (batch, heads, 120, 120) per layer.

This module is imported by phase1_run.py; there is no runners/<family>.py
because phase 1 has no sweep to enumerate — the configs below are the source
of truth and are written verbatim into every run directory.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from industrial_ad import DEFAULT_EXPERIMENT_CONFIG, clone_config

SEEDS = [42, 43, 44]
EPOCHS = 110
EVAL_EVERY = 10

# USAD's own min-max normalization, computed on the training split by
# usad_input_range.py. Absent until that script has been run once.
INPUT_RANGE_PATH = Path(__file__).resolve().parent / "usad_input_range.json"


def usad_input_range() -> list[list[float]]:
    if not INPUT_RANGE_PATH.exists():
        raise SystemExit(
            f"{INPUT_RANGE_PATH.name} is missing — run `python scripts/phase1/usad_input_range.py` "
            "first (it walks the train split once and writes the per-channel min/max)."
        )
    payload = json.loads(INPUT_RANGE_PATH.read_text())
    return [payload["min"], payload["max"]]

# (code, family, model name, run tag, model params, overrides)
CONFIGS: list[dict[str, Any]] = [
    # --- TranAD (Tuli et al., VLDB 2022) -------------------------------------
    # Author config: 1 layer, dim_feedforward=16, dropout=0.1, nhead=channels.
    {"code": "TA001", "family": "TranAD", "model": "tranad", "tag": "author-ff16-l1",
     "params": {"dim_feedforward": 16, "num_layers": 1, "dropout": 0.1}},
    {"code": "TA002", "family": "TranAD", "model": "tranad", "tag": "ff64-l1",
     "params": {"dim_feedforward": 64, "num_layers": 1, "dropout": 0.1},
     "reason": "capacity check upward: TranAD already fits the compact budget"},
    {"code": "TA003", "family": "TranAD", "model": "tranad", "tag": "author-ff16-l1-sig",
     "params": {"dim_feedforward": 16, "num_layers": 1, "dropout": 0.1, "output_activation": "sigmoid"},
     "reason": "ablation: original sigmoid head on z-scored features"},

    # --- Anomaly Transformer (Xu et al., ICLR 2022) --------------------------
    # Author config: d_model=512, heads=8, layers=3, d_ff=512, k=3.
    {"code": "AT001", "family": "AnomalyTransformer", "model": "anomaly_transformer",
     "tag": "author-d512-l3-ff512",
     "params": {"d_model": 512, "n_heads": 8, "e_layers": 3, "d_ff": 512, "dropout": 0.0, "k": 3.0},
     "batch_size": 256},
    {"code": "AT002", "family": "AnomalyTransformer", "model": "anomaly_transformer",
     "tag": "d128-l2-ff128",
     "params": {"d_model": 128, "n_heads": 8, "e_layers": 2, "d_ff": 128, "dropout": 0.0, "k": 3.0},
     "batch_size": 256, "reason": "reduced towards the compact budget"},
    {"code": "AT003", "family": "AnomalyTransformer", "model": "anomaly_transformer",
     "tag": "d64-l1-ff64",
     "params": {"d_model": 64, "n_heads": 4, "e_layers": 1, "d_ff": 64, "dropout": 0.0, "k": 3.0},
     "batch_size": 256, "reason": "smallest variant that still trains"},

    # --- USAD (Audibert et al., KDD 2020) ------------------------------------
    # Author config: z_size = window_size * hidden_size, hidden_size = 100.
    {"code": "US001", "family": "USAD", "model": "usad", "tag": "author-hid100",
     "params": {"hidden_size": 100},
     "reason": "ablation: identity head on z-scored features — the phase-2 objective "
               "is unbounded and training diverges (loss -> 1e14 in 2 of 3 seeds)"},
    {"code": "US002", "family": "USAD", "model": "usad", "tag": "z40",
     "params": {"latent_size": 40},
     "reason": "ablation: same divergence at a small latent; also shows the footprint "
               "is dominated by window*channels, not by z"},
    {"code": "US003", "family": "USAD", "model": "usad", "tag": "author-hid100-sig",
     "params": {"hidden_size": 100, "output_activation": "sigmoid"},
     "reason": "ablation: original sigmoid head but unscaled z-scored inputs — stable, "
               "yet the reconstruction target is unreachable"},

    # --- author-faithful re-runs (2026-08-14) --------------------------------
    # The three entries above showed why: USAD needs its own min-max input scaling
    # for the adversarial term to stay bounded. These are the rows for the paper.
    {"code": "TA004", "family": "TranAD", "model": "tranad", "tag": "author-ff16-l1-e330",
     "params": {"dim_feedforward": 16, "num_layers": 1, "dropout": 0.1}, "epochs": 330,
     "reason": "val ROC-AUC was still rising at epoch 110 (0.639 -> 0.711), so the "
               "baseline gets 3x our own training budget"},
    {"code": "TA005", "family": "TranAD", "model": "tranad", "tag": "ff64-l1-e330",
     "params": {"dim_feedforward": 64, "num_layers": 1, "dropout": 0.1}, "epochs": 330,
     "reason": "same extended budget for the capacity check"},
    {"code": "US004", "family": "USAD", "model": "usad", "tag": "author-hid100-minmax",
     "params": {"hidden_size": 100, "output_activation": "sigmoid", "input_range": "AUTO"},
     "reason": "author configuration with the author's min-max input scaling"},
    {"code": "US005", "family": "USAD", "model": "usad", "tag": "z40-minmax",
     "params": {"latent_size": 40, "output_activation": "sigmoid", "input_range": "AUTO"},
     "reason": "reduced latent under the author-faithful setup"},

    # --- trained under OUR scoring (2026-08-15) -------------------------------
    # The unified rescoring showed the author scores badly underrate USAD and the
    # Anomaly Transformer here (US005: 0.5066 native -> 0.8916 unified). Those
    # numbers are post-hoc: the checkpoint was still selected by val ROC-AUC under
    # the AUTHOR's score. These configs close that gap by training with the
    # project's own scoring from the start, so the baseline also gets to pick its
    # best checkpoint under the metric it is finally judged by.
    {"code": "TA006", "family": "TranAD", "model": "tranad", "tag": "author-ff16-l1-ourscore",
     "params": {"dim_feedforward": 16, "num_layers": 1, "dropout": 0.1}, "scoring": "project",
     "reason": "author config, checkpoint selected under our scoring"},
    {"code": "US006", "family": "USAD", "model": "usad", "tag": "z40-minmax-ourscore",
     "params": {"latent_size": 40, "output_activation": "sigmoid", "input_range": "AUTO"},
     "scoring": "project",
     "reason": "best baseline so far (0.8916 post-hoc); now with a fair checkpoint"},
    {"code": "AT004", "family": "AnomalyTransformer", "model": "anomaly_transformer",
     "tag": "d128-l2-ff128-ourscore",
     "params": {"d_model": 128, "n_heads": 8, "e_layers": 2, "d_ff": 128, "dropout": 0.0, "k": 3.0},
     "batch_size": 256, "scoring": "project",
     "reason": "best baseline inside the <=1 MiB budget; fair checkpoint"},

    # --- author-faithful TranAD windowing (2026-08-15) ------------------------
    # TA001..TA006 hand TranAD a 120-frame window and score the ONE frame it
    # predicts (`tgt = src[-1:]`), i.e. 1/120 of the available evidence. The
    # original protocol uses a length-10 window with stride 1, so every frame is
    # scored. These two configs restore that: the 120-frame evaluation window is
    # unfolded into 111 sub-windows of length 10, each predicts its own last
    # frame, and the per-frame errors are averaged back to one score per window.
    # The unit of evaluation and the label vector are unchanged, so ROC-AUC stays
    # directly comparable with every other row of the table.
    # Motivation: a previous in-house TranAD implementation (code lost) reported
    # ~0.92 on this data. Publishing 0.66-0.71 for a baseline that may simply
    # have been handicapped by our windowing must be avoided.
    {"code": "TA010", "family": "TranAD", "model": "tranad", "tag": "author-w10-slide",
     "params": {"dim_feedforward": 16, "num_layers": 1, "dropout": 0.1, "inner_window": 10},
     "batch_size": 256,
     "reason": "author-faithful windowing: length-10 sub-windows at stride 1, "
               "per-frame scores aggregated to the same evaluation windows"},
    {"code": "TA011", "family": "TranAD", "model": "tranad", "tag": "author-w10-slide-ourscore",
     "params": {"dim_feedforward": 16, "num_layers": 1, "dropout": 0.1, "inner_window": 10},
     "batch_size": 256, "scoring": "project",
     "reason": "same author-faithful windowing, checkpoint selected under our scoring "
               "(TA010 is to TA011 what TA001 is to TA006)"},
]


def _base_config(entry: dict[str, Any], out_root: Path, data_root: Path | None) -> dict[str, Any]:
    config = clone_config(DEFAULT_EXPERIMENT_CONFIG)
    config["task"] = {"type": "reconstruction"}
    params = dict(entry["params"])
    if params.get("input_range") == "AUTO":
        params["input_range"] = usad_input_range()
    config["model"] = {"name": entry["model"], "params": params}
    config["optimizer"] = {"name": "adamw", "params": {"lr": 1e-3, "weight_decay": 1e-4}}
    config["scheduler"] = {
        "name": "warmup_cosine",
        "params": {"warmup_epochs": 10, "min_lr": 1e-6, "start_factor": 1e-4},
    }
    config["trainer"] = {
        **config["trainer"],
        "epochs": int(entry.get("epochs", EPOCHS)),
        "eval_every_n_epochs": EVAL_EVERY,
        "device": "cuda",
        "amp": False,
        "clip_grad_norm": 1.0,
    }
    if entry.get("scoring") == "project":
        # The project's own pipeline, identical to every family in the paper:
        # |prediction - target| averaged over time -> 27 channel features -> Mahalanobis.
        # Leaves DEFAULT_EXPERIMENT_CONFIG's scoring section untouched.
        pass
    else:
        # The authors' own anomaly score; "mean" reduces their per-element error map
        # exactly as the original code does. phase1_rescore.py adds the Mahalanobis
        # column on the same errors.
        config["scoring"]["error_reducer"] = {"name": "model_native", "params": {}}
        config["scoring"]["score_estimator"] = {"name": "mean", "params": {}}
    if "batch_size" in entry:
        config["dataset"]["loader"]["batch_size"] = int(entry["batch_size"])
    if data_root is not None:
        config["dataset"]["params"]["root"] = str(data_root)
    config["run"]["notes"] = f"phase1 SOTA baseline ({entry['code']}): {entry.get('reason', 'author configuration')}"
    config["run"]["tags"] = ["phase1", "sota", entry["family"].lower()]
    return config


def build_phase1_configs(out_root: Path, data_root: Path | None = None, seeds: list[int] | None = None) -> list[dict[str, Any]]:
    """Return the full phase-1 run list (config dicts ready for run_experiments)."""
    seeds = list(seeds or SEEDS)
    runs: list[dict[str, Any]] = []
    for index, entry in enumerate(CONFIGS, start=1):
        base = _base_config(entry, out_root, data_root)
        for seed in seeds:
            config = copy.deepcopy(base)
            name = f"{index:03d}-s{seed}-{entry['tag']}"
            config["run"] = {
                **config["run"],
                "name": name,
                "dir": str(out_root / entry["family"] / name),
                "seed": seed,
            }
            config["phase1"] = {"code": entry["code"], "family": entry["family"], "tag": entry["tag"]}
            runs.append(config)
    return runs
