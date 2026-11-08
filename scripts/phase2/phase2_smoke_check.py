#!/usr/bin/env python3
"""Read back a phase-2 smoke run and say whether the code path is sound.

    python3.11 scripts/phase2/phase2_smoke_check.py                    # all smokes
    python3.11 scripts/phase2/phase2_smoke_check.py --codes TR138,TL082

A smoke caps training at 3 batches; its ROC-AUC means nothing. What it proves:
the config reached the runner; the model built on (120, 9) input with the scorer
on 9 error channels; both val and test have both classes; the split metadata
says protocol lolo_v1 with the expected recording counts and 150 windows per
evaluated recording; the scaler is train-fitted; score dumps exist for best and
last checkpoints; and how long an epoch takes (-> full-run estimate).

Pure stdlib — safe on the JupyterServer instance.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

FULL_EPOCHS = 110
EVAL_WINDOWS = 150


def _num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isnan(value)


def check_run(run_dir: Path) -> list[str]:
    problems: list[str] = []
    summary = json.loads((run_dir / "summary.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    runtime = config.get("runtime", {})
    phase2 = config.get("phase2", {})
    best = summary.get("best_metrics") or {}

    if runtime.get("input_shape") != [120, 9]:
        problems.append(f"input_shape {runtime.get('input_shape')} != [120, 9]")
    if runtime.get("score_feature_dim") != 9:
        problems.append(f"score_feature_dim {runtime.get('score_feature_dim')} != 9 (mean_abs over time -> 9 channels)")
    if runtime.get("dataset") != "cwru" or runtime.get("protocol", {}).get("protocol") not in {"lolo_v1", "lolo_v1b"}:
        problems.append("runtime metadata is not CWRU lolo_v1/v1b")
    prot_lp = runtime.get("protocol", {}).get("common_lowpass_hz")
    recs = runtime.get("recordings", {})
    if recs and any(m.get("common_lowpass_hz") != prot_lp for m in recs.values()):
        problems.append("common_lowpass_hz differs between recordings and protocol")
    prot = runtime.get("protocol", {})
    if prot.get("fold_load_hp") != phase2.get("fold"):
        problems.append(f"fold mismatch: protocol {prot.get('fold_load_hp')} vs phase2 {phase2.get('fold')}")
    counts = runtime.get("recording_counts", {})
    if counts.get("train") != {"normal": 3, "anomalous": 0}:
        problems.append(f"train recordings {counts.get('train')} != 3 normal / 0 anomalous")
    if counts.get("test", {}).get("normal") != 1:
        problems.append(f"test normal recordings {counts.get('test')} != 1")
    scaler = runtime.get("scaler_provenance", {})
    if scaler.get("mode") != "train_fitted" or not scaler.get("fitted_on_recordings"):
        problems.append("scaler is not train-fitted with provenance")
    if any(not n.endswith(",0.80)") for n in scaler.get("fitted_on_recordings", [])):
        problems.append(f"scaler fitted on non-train slices: {scaler.get('fitted_on_recordings')}")
    for split in ("val", "test"):
        ws = {w["windows"] for w in runtime.get("windows_per_segment", {}).get(split, [])}
        if ws != {EVAL_WINDOWS}:
            problems.append(f"{split}: windows per recording {ws} != {{{EVAL_WINDOWS}}}")
    for key in ("val/roc_auc", "test/roc_auc"):
        if not _num(best.get(key)):
            problems.append(f"{key} missing or NaN: {best.get(key)!r}")
    for name in ("scores_test_best.csv", "scores_val_best.csv"):
        path = run_dir / name
        if not path.exists():
            problems.append(f"{name} missing")
            continue
        with open(path) as handle:
            rows = list(csv.DictReader(handle))
        labels = {r["is_anomaly"] for r in rows}
        if labels != {"0", "1"}:
            problems.append(f"{name}: labels {labels} (need both classes)")
        per_file = {}
        for r in rows:
            per_file[r["file_id"]] = per_file.get(r["file_id"], 0) + 1
        if set(per_file.values()) != {EVAL_WINDOWS}:
            problems.append(f"{name}: windows per file {sorted(set(per_file.values()))}")
    if summary.get("model_name", "").lower() != "pca" and not (run_dir / "scores_test_last.csv").exists():
        problems.append("scores_test_last.csv missing (save_last_checkpoint not honoured?)")
    return problems


def epoch_estimate(run_dir: Path) -> str:
    history = json.loads((run_dir / "history.json").read_text())
    epochs = history.get("train_epochs") or []
    if not epochs:
        return "no epoch timing"
    secs = [e.get("epoch_time_seconds") for e in epochs if _num(e.get("epoch_time_seconds"))]
    batches = json.loads((run_dir / "config.json").read_text())["debug"]["trainer"].get("max_train_batches")
    if not secs:
        return "no epoch timing"
    per_batch = max(secs) / (batches or 1)
    return f"{per_batch:.2f} s/batch (smoke, includes warm-up) -> ~{per_batch * 35 * FULL_EPOCHS / 60:.1f} min per full run @35 batches/epoch"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs_phase2/_smoke")
    parser.add_argument("--codes", default="")
    args = parser.parse_args()
    wanted = {c.strip().upper() for c in args.codes.split(",") if c.strip()}
    run_dirs = sorted(p.parent for p in Path(args.root).rglob("summary.json"))
    if not run_dirs:
        raise SystemExit(f"no smoke runs under {args.root}")
    bad = 0
    for run_dir in run_dirs:
        code = json.loads((run_dir / "config.json").read_text()).get("phase2", {}).get("code", "?")
        if wanted and code not in wanted:
            continue
        problems = check_run(run_dir)
        status = "OK  " if not problems else "FAIL"
        bad += bool(problems)
        print(f"{status} {code:6s} {run_dir.parent.parent.name} {run_dir.name}  {epoch_estimate(run_dir)}")
        for p in problems:
            print(f"      - {p}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
