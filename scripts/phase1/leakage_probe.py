#!/usr/bin/env python3
"""Protocol probe: how much of our ROC-AUC comes from the split, not from the model?

The paper's PU protocol cuts each HEALTHY recording in time — train on the first
90 %, test on the last 5 % — and every healthy bearing K001..K006 is in training.
So the test normals come from the same physical bearings, and in fact from the
same 4-second recordings, that trained the model. In the taxonomy of
arXiv:2509.22267 that is a *segment-wise* split, the most optimistic of the four
levels; that paper reports such splits reaching almost 100 % accuracy on PU,
while an honest bearing-wise split lands at macro-AUROC 0.55-0.80. The best
published leakage-controlled unsupervised ROC-AUC on PU is 0.878 +/- 0.015
(Zaidi et al., Processes 2026, 14, 2452).

This script re-runs a config with healthy bearings held out of training entirely,
so the test normals come from bearings the model has never seen. Everything else
— architecture, optimizer, schedule, scoring, seeds — is unchanged, so the only
difference is the split.

Works for both config families:
  * phase-0 codes (TR138, TL082, PR005, ...) — enumerated from runners/ through
    phase0_repro, i.e. bit-identical to the paper sweep;
  * phase-1 codes (AT002, US005, TA011, ...) — from phase1_configs.

First result (2026-08-15, holdout K001,K005, 3 seeds):
    TR138  0.9431 -> 0.9094 +/- 0.0090   (delta -0.0337)
The headline number survives. What matters for section VI is comparative, so run
the baselines under the SAME holdout before drawing conclusions.

    python3.11 scripts/phase1/leakage_probe.py --list-only
    python3.11 scripts/phase1/leakage_probe.py --codes TR138,TL082,AT002,US005,TA011
    python3.11 scripts/phase1/leakage_probe.py --codes TR138 --holdout K002,K004

Run directories are namespaced by holdout (runs_leakage/ho-K001-K005/...), so
rotating the held-out pair never collides with an earlier round.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "phase0"))
sys.path.insert(0, str(HERE))

# phase0_repro does the NFS re-exec, dependency install and TF32-off at import,
# then exposes the runner-enumerated configs. Importing it keeps the phase-0 half
# of this probe bit-identical to the paper sweep instead of re-deriving it.
import phase0_repro  # noqa: E402
from phase0_repro import build_phase0_configs  # noqa: E402
from phase1_configs import build_phase1_configs  # noqa: E402

import industrial_ad  # noqa: E402

OUT_ROOT = REPO / "runs_leakage"
DATA_ROOT = REPO / "data" / "PU"
PHASE1_SUMMARY = REPO / "runs_phase1" / "phase1_summary.csv"
PUBLISHED_HONEST = 0.878  # Zaidi et al., Processes 2026, 14, 2452 (+/- 0.015)


def holdout_tag(holdout: list[str]) -> str:
    return "ho-" + "-".join(holdout)


def phase1_reference() -> dict[str, tuple[float, str]]:
    """Reference ROC-AUC per phase-1 code: the unified column, our like-for-like one."""
    if not PHASE1_SUMMARY.exists():
        return {}
    reference = {}
    with open(PHASE1_SUMMARY) as handle:
        for row in csv.DictReader(handle):
            for column, label in (("roc_unified_mean", "unified"), ("roc_native_mean", "native")):
                value = row.get(column)
                if value not in (None, "", "nan"):
                    reference[row["code"].upper()] = (float(value), label)
                    break
    return reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codes", default=os.environ.get("PHASE1_CODES", "TR138"),
                        help="phase-0 and/or phase-1 codes, e.g. TR138,TL082,AT002,US005,TA011")
    parser.add_argument("--holdout", default=os.environ.get("PHASE1_HOLDOUT", "K001,K005"),
                        help="healthy bearings kept out of training entirely")
    parser.add_argument("--seeds", default=os.environ.get("PHASE1_SEEDS", "42,43,44"))
    parser.add_argument("--shard", default=os.environ.get("PHASE1_SHARD", "0/1"))
    parser.add_argument("--list-only", nargs="?", const="1",
                        default=os.environ.get("PHASE1_LIST_ONLY", ""))
    parser.add_argument("--manifest", default=str(phase0_repro.MANIFEST))
    args, extra = parser.parse_known_args()
    if extra:
        print(f"[leakage] ignoring launcher args: {extra}", flush=True)

    os.chdir(REPO)
    codes = {c.strip().upper() for c in args.codes.split(",") if c.strip()}
    holdout = [b.strip().upper() for b in args.holdout.split(",") if b.strip()]
    seeds = {int(s) for s in args.seeds.split(",") if s.strip()}
    list_only = str(args.list_only).lower() not in {"", "0", "false", "no"}
    assert holdout, "--holdout must name at least one healthy bearing"
    root = OUT_ROOT / holdout_tag(holdout)

    manifest = json.loads(Path(args.manifest).read_text())
    phase0_entries = [e for e in manifest if e["code"].upper() in codes]
    phase0_codes = {e["code"].upper() for e in phase0_entries}
    reference: dict[str, tuple[float, str]] = {
        e["code"].upper(): (float(e["paper_roc_auc_mean"]), "paper") for e in phase0_entries
    }
    reference.update({k: v for k, v in phase1_reference().items() if k in codes - phase0_codes})

    configs: list[dict] = []
    code_of: dict[str, str] = {}

    for config in build_phase0_configs(phase0_entries):
        family = Path(config["run"]["dir"]).parent.name
        seedless = phase0_repro.seedless(config["run"]["name"])
        entry = next((e for e in phase0_entries
                      if e["family"] == family and e["run_name"] == seedless), None)
        if entry is not None:
            code_of[config["run"]["name"]] = entry["code"]
            configs.append(config)

    phase1_wanted = codes - phase0_codes
    if phase1_wanted:
        for config in build_phase1_configs(root, data_root=DATA_ROOT.resolve(),
                                           seeds=sorted(seeds)):
            code = config["phase1"]["code"].upper()
            if code in phase1_wanted:
                code_of[config["run"]["name"]] = code
                configs.append(config)
        missing = phase1_wanted - {c.upper() for c in code_of.values()}
        assert not missing, f"unknown codes: {sorted(missing)}"

    prepared = []
    for config in configs:
        if config["run"]["seed"] not in seeds:
            continue
        name = config["run"]["name"]
        family = Path(config["run"]["dir"]).parent.name
        config["dataset"]["params"]["holdout_bearings"] = holdout
        config["run"]["dir"] = str(root / family / name)
        config["run"]["notes"] = (f"leakage probe: healthy bearings {','.join(holdout)} held out "
                                  f"of training; everything else unchanged")
        config["run"]["tags"] = ["leakage-probe", "bearing-wise", holdout_tag(holdout)]
        prepared.append(config)

    # Keep identical dataset configs adjacent so the data bundle is reused.
    prepared.sort(key=lambda c: (json.dumps({"d": c["dataset"], "t": c["task"]}, sort_keys=True),
                                 c["run"]["dir"]))
    shard_index, shard_total = map(int, args.shard.split("/"))
    chunk = (len(prepared) + shard_total - 1) // shard_total
    shard_configs = prepared[shard_index * chunk: (shard_index + 1) * chunk]

    print(f"[leakage] holdout={holdout} codes={sorted(codes)} seeds={sorted(seeds)}", flush=True)
    print(f"[leakage] {len(prepared)} runs total, shard {args.shard} -> {len(shard_configs)}", flush=True)
    for config in shard_configs:
        print(f"[leakage]   {config['run']['dir']}", flush=True)
    if list_only:
        return

    root.mkdir(parents=True, exist_ok=True)
    results = industrial_ad.run_experiments(
        shard_configs, skip_existing=True, overwrite=False, stop_on_error=False
    )

    # --- the comparison, printed where you cannot miss it ---------------------
    per_code: dict[str, list[float]] = {}
    for config in shard_configs:
        summary_path = Path(config["run"]["dir"]) / "summary.json"
        if not summary_path.exists():
            continue
        auc = (json.loads(summary_path.read_text()).get("best_metrics") or {}).get("test/roc_auc")
        if auc is not None:
            per_code.setdefault(code_of.get(config["run"]["name"], "?"), []).append(float(auc))

    print(f"\n[leakage] ROC-AUC under bearing-wise holdout {','.join(holdout)}", flush=True)
    print(f"{'code':<8}{'reference':>11}{'src':>9}{'holdout':>10}{'std':>9}{'delta':>9}{'n':>4}", flush=True)
    for code, values in sorted(per_code.items()):
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        ref, src = reference.get(code, (None, "-"))
        ref_s = f"{ref:.4f}" if ref is not None else "?"
        delta = f"{mean - ref:+.4f}" if ref is not None else "?"
        print(f"{code:<8}{ref_s:>11}{src:>9}{mean:>10.4f}{std:>9.4f}{delta:>9}{len(values):>4}", flush=True)
    print(f"\n[leakage] published leakage-controlled reference on PU: {PUBLISHED_HONEST:.3f} +/- 0.015 "
          f"(Zaidi et al., Processes 2026, 14, 2452)", flush=True)
    print("[leakage] what matters for section VI is the ORDERING: compare the deltas, not the levels.",
          flush=True)

    errors = [r for r in results if r.get("error")]
    print(f"[leakage] finished: {len(results) - len(errors)} ok, {len(errors)} failed", flush=True)
    for r in errors[:5]:
        print(f"[leakage]   FAILED {r.get('run_name')}: {r.get('exception')}", flush=True)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
