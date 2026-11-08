"""Phase 2: the CWRU config slice.

Every config is the ORIGINAL sweep config of a Paderborn run (captured by
executing runners/<family>.py, exactly as phase 0 does) with the `dataset`
block swapped for CWRU protocol lolo_v1 and the run relocated to
runs_phase2/fold<L>/<family>/<name>. Model / optimizer / scheduler / trainer /
scoring blocks are untouched — "the same config transplanted to another
dataset", nothing re-tuned. The one CWRU-specific loader change is the batch
size (256 instead of 1024): the fold has ~9 k train windows, and 110 epochs at
1024 would be ~1 k optimizer steps against ~24 k on PU.

Tiers (selection rules are fixed here, without looking at any CWRU number):
  A  headline + paired TL/TF + best-per-family (Table I / Contribution 2 of the
     paper) — codes are read from scripts/phase0/phase0_manifest.json;
  B  "across size budgets": Table II budget winners of the TCN families that are
     not already in A, the K2/H16 TL/TF pair, and for every other family the
     best PU config at <= 0.10 MiB and <= 0.25 MiB (MLP families have nothing
     that small: best at <= 4 MiB) — read from summary_table.csv.

Usage:
    python scripts/phase2/phase2_configs.py            # print the slice
"""
from __future__ import annotations

import copy
import json
import re
import runpy
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import industrial_ad  # noqa: E402
from industrial_ad.datasets.CWRU import DEFAULT_CWRU_DATASET_CONFIG  # noqa: E402
from industrial_ad.utils import clone_config  # noqa: E402

MANIFEST = REPO / "scripts" / "phase0" / "phase0_manifest.json"
SUMMARY_CSV = REPO / "summary_table.csv"
RUNNERS_DIR = REPO / "runners"

FOLDS = [0, 1, 2, 3]
SEEDS = [42, 43, 44]
CWRU_BATCH_SIZE = 256
# Protocol variants: v1 = as frozen 2026-08-25; v1b = same split + common 5 kHz low-pass on
# every recording (acquisition-chain control, frozen 2026-08-27).
VARIANTS = {
    "v1": {"protocol": "lolo_v1", "common_lowpass_hz": None, "out_root": "runs_phase2"},
    "v1b": {"protocol": "lolo_v1b", "common_lowpass_hz": 5000.0, "out_root": "runs_phase2_v1b"},
}
# control slice for v1b: PCA at the ceiling, the headline TCNs, one TF and the weakest MF
V1B_CODES = ["PR001", "PR005", "TR047", "TR138", "TL082", "TF042", "MF043"]
# tier S (2026-08-27): the phase-1 SOTA baselines that §VI compares against, in their
# "trained under our scoring" form (fair checkpoint), transplanted to CWRU unchanged.
# USAD's min-max input range is re-fitted on the fold's TRAIN windows (never PU's).
SOTA_CODES = ["TA011", "AT004", "US006"]
SOTA_FAMILY_DIR = {"TranAD": "SOTA-TranAD", "AnomalyTransformer": "SOTA-AnomalyTransformer", "USAD": "SOTA-USAD"}
# PU reference for the report (phase 1, unified scoring column of the handoff 2026-08-15)
SOTA_PU_REF = {"TA011": (0.7712, 0.0465, "TA011 unified"), "AT004": (0.8225, None, "AT002 unified (post-hoc)"),
               "US006": (0.8916, None, "US005 unified (post-hoc)")}

CODE = {
    "PCA-reconstruction": "PR", "MLP-reconstruction": "MR", "MLP-forecasting": "MF",
    "Conv-reconstruction": "CR", "TCN-reconstruction": "TR", "TCN-forecasting": "TF",
    "TCN-light-forecasting": "TL", "GRU-repeated-reconstruction": "GR",
    "GRU-seq2seq-reconstruction": "GS", "GRU-seq2seq-forecasting": "GF",
    "Transformer-reconstruction": "XR",
}
FAMILY_OF = {v: k for k, v in CODE.items()}

TIER_A_CODES = [
    "TR138", "TR047", "TL082",                       # headline
    "TL101", "TF042", "TL069", "TF212", "TL072", "TF213",  # paired TL/TF cells (K1H16, K1H120, K2H120)
    "CR048", "XR133", "GR011", "MR064", "PR005", "MF088",  # best per family (Table I) + best MF
]
TIER_B_FIXED = ["TR025", "TR028", "TR065", "TL018", "TF051"]   # Table II budget winners + K2H16 pair
TIER_B_BUDGETS_MIB = {
    "Conv-reconstruction": [0.10, 0.25], "Transformer-reconstruction": [0.10, 0.25],
    "GRU-repeated-reconstruction": [0.10, 0.25], "GRU-seq2seq-reconstruction": [0.10, 0.25],
    "GRU-seq2seq-forecasting": [0.10, 0.25], "PCA-reconstruction": [0.10, 0.25],
    "MLP-reconstruction": [4.0], "MLP-forecasting": [4.0],
}
# TL/TF matched-body pairs (code_TL, code_TF, cell) — used by the report
PAIRS = [("TL101", "TF042", "K1H16"), ("TL018", "TF051", "K2H16"), ("TL069", "TF212", "K1H120"), ("TL072", "TF213", "K2H120")]

SEED_RE = re.compile(r"^(\d{3})-s(\d+)-")


def seedless(name: str) -> str:
    return SEED_RE.sub(r"\1-", name)


def code_of(family: str, run_name: str) -> str:
    return CODE[family] + run_name[:3]


def tier_b_codes() -> list[str]:
    """Deterministic 'across budgets' picks from the PU summary table."""
    df = pd.read_csv(SUMMARY_CSV)
    agg = df.groupby(["family", "run_name"], as_index=False).agg(roc=("best_metrics.test/roc_auc", "mean"),
                                                                 params=("parameter_count", "first"))
    agg["mib"] = agg["params"] * 4 / (1024 * 1024)
    codes: list[str] = list(TIER_B_FIXED)
    for family, budgets in TIER_B_BUDGETS_MIB.items():
        sub = agg[agg["family"] == family]
        for budget in budgets:
            cand = sub[sub["mib"] <= budget]
            if cand.empty:
                continue
            row = cand.loc[cand["roc"].idxmax()]
            code = code_of(family, row["run_name"])
            if code not in codes and code not in TIER_A_CODES:
                codes.append(code)
    return codes


def tier_codes(tier: str) -> list[str]:
    tier = tier.upper()
    if tier == "A":
        return list(TIER_A_CODES)
    if tier == "B":
        return tier_b_codes()
    if tier == "AB":
        return list(TIER_A_CODES) + tier_b_codes()
    if tier == "V1B":
        return list(V1B_CODES)
    if tier == "S":
        return list(SOTA_CODES)
    raise ValueError(f"unknown tier {tier!r} (A, B or AB)")


def capture_family_configs(family: str) -> list[dict]:
    """Execute runners/<family>.py with run_experiments patched to capture configs (phase-0 pattern)."""
    runner_path = RUNNERS_DIR / f"{family}.py"
    if not runner_path.exists():
        raise FileNotFoundError(runner_path)
    captured: list[dict] = []

    def _capture(configs, **kwargs):
        captured.extend(configs)

    original, original_argv = industrial_ad.run_experiments, sys.argv
    try:
        industrial_ad.run_experiments = _capture
        sys.argv = [str(runner_path)]
        runpy.run_path(str(runner_path), run_name="__main__")
    finally:
        industrial_ad.run_experiments, sys.argv = original, original_argv
    return captured


def _cwru_dataset_block(fold: int, data_root: Path, horizon_size: int, variant: str = "v1") -> dict:
    block = clone_config(DEFAULT_CWRU_DATASET_CONFIG)
    block["params"]["protocol"] = VARIANTS[variant]["protocol"]
    block["params"]["feature_pipeline"]["params"]["common_lowpass_hz"] = VARIANTS[variant]["common_lowpass_hz"]
    block["params"]["root"] = str(data_root)
    block["params"]["fold_load_hp"] = int(fold)
    block["params"]["horizon_size"] = int(horizon_size)   # forecasting configs carry their own horizon
    block["loader"]["batch_size"] = CWRU_BATCH_SIZE
    return block


def build_sota_configs(out_root: Path, data_root: Path, *, codes: list[str], folds: list[int], seeds: list[int],
                       variant: str) -> list[dict]:
    """Phase-1 baseline entries -> CWRU configs. `input_range: AUTO` becomes `AUTO_CWRU`,
    resolved per fold by phase2_run from the training windows (see fill_usad_input_range)."""
    sys.path.insert(0, str(REPO / "scripts" / "phase1"))
    from phase1_configs import CONFIGS as P1, _base_config  # noqa: E402

    configs: list[dict] = []
    for entry in P1:
        if entry["code"] not in codes:
            continue
        entry = copy.deepcopy(entry)
        if entry["params"].get("input_range") == "AUTO":
            entry["params"]["input_range"] = "AUTO_CWRU"
        base = _base_config(entry, out_root, None)
        for fold in folds:
            for seed in seeds:
                cfg = copy.deepcopy(base)
                name = f"{entry['code']}-s{seed}-{entry['tag']}"
                family_dir = SOTA_FAMILY_DIR[entry["family"]]
                cfg["run"] = {**cfg["run"], "name": name, "seed": seed,
                              "dir": str(out_root / f"fold{fold}" / family_dir / name),
                              "tags": ["phase2", "cwru", "sota", entry["family"].lower(), f"fold{fold}"],
                              "notes": f"phase 2 CWRU {VARIANTS[variant]['protocol']} fold {fold}; phase-1 baseline {entry['code']} transplanted unchanged"}
                batch = int(cfg["dataset"]["loader"]["batch_size"])
                cfg["dataset"] = _cwru_dataset_block(fold, data_root, 0, variant)
                cfg["dataset"]["loader"]["batch_size"] = min(batch, CWRU_BATCH_SIZE)
                cfg["trainer"]["save_last_checkpoint"] = True
                cfg["phase2"] = {"code": entry["code"], "family": family_dir, "fold": fold, "seed": seed, "variant": variant,
                                 "tier": "S", "phase1_tag": entry["tag"], "pu_batch_size": batch}
                configs.append(cfg)
    return configs


def fill_usad_input_range(config: dict, bundle: dict) -> None:
    """Resolve `input_range: AUTO_CWRU` from the fold's normalised TRAIN frames (train slices only)."""
    params = config["model"]["params"]
    if params.get("input_range") != "AUTO_CWRU":
        return
    import numpy as np
    frames = np.concatenate([sliced for _, sliced, _ in bundle["datasets"]["train"].plan], axis=0)
    params["input_range"] = [[round(float(v), 6) for v in frames.min(axis=0)], [round(float(v), 6) for v in frames.max(axis=0)]]
    config["phase2"]["usad_input_range_provenance"] = {
        "source": "per-channel min/max of normalised TRAIN frames of this fold",
        "fold": config["phase2"]["fold"], "n_frames": int(frames.shape[0]),
        "recordings": bundle["metadata"]["scaler_provenance"]["fitted_on_recordings"]}


def build_phase2_configs(out_root: Path, data_root: Path, *, codes: list[str], folds: list[int] | None = None,
                         seeds: list[int] | None = None, variant: str = "v1") -> list[dict]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r} (known: {sorted(VARIANTS)})")
    folds = list(folds or FOLDS)
    seeds = list(seeds or SEEDS)
    sota = [c for c in codes if c in SOTA_CODES]
    codes = [c for c in codes if c not in SOTA_CODES]
    configs_sota = build_sota_configs(out_root, data_root, codes=sota, folds=folds, seeds=seeds, variant=variant) if sota else []
    if not codes:
        return configs_sota
    wanted: dict[str, set[str]] = {}
    for code in codes:
        wanted.setdefault(FAMILY_OF[code[:2]], set()).add(code)

    configs: list[dict] = []
    for family, family_codes in sorted(wanted.items()):
        captured = capture_family_configs(family)
        templates: dict[str, dict] = {}
        for cfg in captured:
            name = cfg["run"]["name"]
            code = code_of(family, name)
            if code in family_codes and code not in templates:
                templates[code] = cfg            # lowest-seed enumeration = template
        missing = sorted(family_codes - set(templates))
        if missing:
            raise KeyError(f"{family}: codes {missing} not found in runner enumeration")
        for code in sorted(templates):
            template = templates[code]
            pu_dataset = template["dataset"]
            horizon = int(pu_dataset["params"]["horizon_size"])
            for fold in folds:
                for seed in seeds:
                    cfg = copy.deepcopy(template)
                    name = SEED_RE.sub(rf"\1-s{seed}-", cfg["run"]["name"])
                    cfg["run"]["name"] = name
                    cfg["run"]["seed"] = seed
                    cfg["run"]["dir"] = str(out_root / f"fold{fold}" / family / name)
                    cfg["run"]["tags"] = [t for t in cfg["run"].get("tags", []) if t != "pu"] + ["cwru", "phase2", f"fold{fold}"]
                    cfg["run"]["notes"] = f"phase 2 CWRU lolo_v1 fold {fold}; PU config {code} transplanted unchanged"
                    cfg["dataset"] = _cwru_dataset_block(fold, data_root, horizon, variant)
                    cfg["trainer"]["save_last_checkpoint"] = True   # last.pt -> "no-anomaly model selection" column
                    cfg["phase2"] = {"code": code, "family": family, "fold": fold, "seed": seed, "variant": variant,
                                     "pu_run_name": seedless(template["run"]["name"]),
                                     "pu_window_size": int(pu_dataset["params"]["window_size"]),
                                     "pu_batch_size": int(template["dataset"]["loader"]["batch_size"])}
                    if int(pu_dataset["params"]["window_size"]) != cfg["dataset"]["params"]["window_size"]:
                        raise ValueError(f"{code}: PU window {pu_dataset['params']['window_size']} != CWRU window")
                    configs.append(cfg)
    return configs + configs_sota


def main() -> None:
    for tier in ("A", "B"):
        codes = tier_codes(tier)
        print(f"tier {tier}: {len(codes)} configs -> {len(codes) * len(FOLDS) * len(SEEDS)} runs")
        print("  ", ", ".join(codes))
    configs = build_phase2_configs(REPO / "runs_phase2", REPO / "data" / "CWRU", codes=tier_codes("AB"))
    print(f"total runs (A+B, {len(FOLDS)} folds x {len(SEEDS)} seeds): {len(configs)}")
    families = sorted({c["phase2"]["family"] for c in configs})
    print("families:", ", ".join(families))
    example = next(c for c in configs if c["phase2"]["code"] == "TL082")
    print(json.dumps({k: example[k] for k in ("run", "dataset", "phase2")}, indent=1)[:1500])


if __name__ == "__main__":
    main()
