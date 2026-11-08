#!/usr/bin/env python3
"""Read back a phase-1 smoke run and say whether the code path is sound.

    python3.11 scripts/phase1/phase1_smoke_check.py                      # TA010,TA011
    python3.11 scripts/phase1/phase1_smoke_check.py --codes AT001
    python3.11 scripts/phase1/phase1_smoke_check.py --codes ALL

A smoke caps training at 3 batches, so the ROC-AUC it produces is meaningless as
a result. What it does prove: the config reached the runner, the model built with
the intended shapes, the scoring path produced a real number instead of NaN, and
how long a full epoch is going to take. Those are the four things to check.

`runs_phase1/_smoke/` accumulates every smoke ever run, so this filters by config
code by default — otherwise last week's smoke shows up as this week's failure.

Pure stdlib, no torch — safe to run on the JupyterServer instance.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Measured on PU, 2026-08-14 (--count-batches): 223 050 train windows.
TRAIN_WINDOWS = 223_050
FULL_EPOCHS = 110
# Author TranAD ff16 / 1 layer on 120x27. Independent of inner_window: the sliding
# mode changes how the model is applied, not what it is made of.
TRANAD_AUTHOR_PARAMS = 66_279


def _num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isnan(value)


def check_run(run_dir: Path, expect_inner: bool) -> list[str]:
    problems: list[str] = []
    summary = json.loads((run_dir / "summary.json").read_text())
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}

    phase1 = config.get("phase1", {})
    code = phase1.get("code", "?")
    model_params = config.get("model", {}).get("params", {})
    batch_size = config.get("dataset", {}).get("loader", {}).get("batch_size")
    train_batches = config.get("debug", {}).get("trainer", {}).get("max_train_batches")

    # summary.json is FLAT — these keys sit at the top level, not under "model".
    input_shape = summary.get("input_shape") or config.get("runtime", {}).get("input_shape")
    feature_dim = summary.get("score_feature_dim")
    params = summary.get("parameter_count")
    state_bytes = summary.get("state_dict_size_bytes") or 0
    reducer = config.get("scoring", {}).get("error_reducer", {}).get("name", "?")
    estimator = config.get("scoring", {}).get("score_estimator", {}).get("name", "?")
    best = summary.get("best_metrics") or {}
    test_auc, val_auc = best.get("test/roc_auc"), best.get("val/roc_auc")

    inner = model_params.get("inner_window")
    n_positions = (int(input_shape[0]) - int(inner) + 1) if (inner and input_shape) else None

    epochs = summary.get("train_epochs") or []
    epoch_time = epochs[-1].get("epoch_time_seconds") if epochs else None
    if epoch_time is None:
        history_path = run_dir / "history.json"
        if history_path.exists():
            hist = json.loads(history_path.read_text()).get("train_epochs") or []
            epoch_time = hist[-1].get("epoch_time_seconds") if hist else None

    print(f"\n=== {run_dir.name}   [{code}]")
    print(f"  окно             : {input_shape}")
    print(f"  inner_window     : {inner}" + (f"  -> {n_positions} под-окон" if n_positions else "  (обычный режим)"))
    print(f"  скоринг          : {reducer} + {estimator}, feature_dim={feature_dim}")
    print(f"  параметры        : {params}  |  state_dict {state_bytes/1048576:.4f} MiB")
    print(f"  val/test ROC-AUC : {val_auc} / {test_auc}   (на смоуке это шум, важно что не NaN)")

    if list(input_shape or []) != [120, 27]:
        problems.append(f"{code}/{run_dir.name}: единица оценки не 120x27 ({input_shape}) — таблица перестанет быть сравнимой")
    if expect_inner:
        if inner is None:
            problems.append(f"{code}/{run_dir.name}: inner_window не доехал до модели — это старый, задушенный TranAD")
        elif n_positions != 111:
            problems.append(f"{code}/{run_dir.name}: под-окон {n_positions}, ждали 111")
        if feature_dim != 27:
            problems.append(f"{code}/{run_dir.name}: feature_dim={feature_dim}, ждали 27")
        if params != TRANAD_AUTHOR_PARAMS:
            problems.append(f"{code}/{run_dir.name}: параметров {params}, у авторского TranAD ff16/l1 — {TRANAD_AUTHOR_PARAMS}")
    if not _num(test_auc):
        problems.append(f"{code}/{run_dir.name}: test/roc_auc = {test_auc!r} — путь скоринга не дал числа")

    if epoch_time and train_batches and batch_size:
        per_batch = epoch_time / int(train_batches)
        full_epoch_min = per_batch * math.ceil(TRAIN_WINDOWS / int(batch_size)) / 60
        hours = full_epoch_min * FULL_EPOCHS / 60
        print(f"  время            : {per_batch:.3f} с/батч @{batch_size} -> {full_epoch_min:.1f} мин/эпоха "
              f"-> ~{hours:.1f} ч на прогон (~{hours*3:.1f} ч на шард из 3 seeds)")
        print(f"                     при --n-shards 6 (по прогону на job) стена ~{hours:.1f} ч")
    else:
        print("  время            : нет таймингов — оценить эпоху не из чего")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs_phase1/_smoke")
    parser.add_argument("--codes", default="TA010,TA011",
                        help="какие коды разбирать; ALL — все, что лежат в _smoke")
    args = parser.parse_args()

    wanted = None if args.codes.strip().upper() == "ALL" else {c.strip().upper() for c in args.codes.split(",")}
    run_dirs = sorted({p.parent for p in Path(args.root).rglob("summary.json")})

    selected, skipped = [], []
    for run_dir in run_dirs:
        config_path = run_dir / "config.json"
        code = ""
        if config_path.exists():
            code = json.loads(config_path.read_text()).get("phase1", {}).get("code", "")
        (selected if (wanted is None or code.upper() in wanted) else skipped).append((run_dir, code))

    if skipped:
        print(f"[smoke-check] пропущено {len(skipped)} прогонов от прошлых смоуков: "
              + ", ".join(sorted({c or d.name for d, c in skipped})))
    if not selected:
        print(f"[smoke-check] в {args.root} нет прогонов с кодами {args.codes} — job не дошёл до записи "
              f"результатов; смотреть joblog до конца")
        return 1

    problems: list[str] = []
    for run_dir, code in selected:
        problems.extend(check_run(run_dir, expect_inner=code.upper() in {"TA010", "TA011"}))

    print(f"\n{'='*64}\nразобрано прогонов: {len(selected)}")
    if problems:
        print("НЕ ГОТОВО К БОЕВОМУ ЗАПУСКУ:")
        for item in problems:
            print(f"  ✗ {item}")
        return 1
    print("✅ смоук зелёный: окно 120x27, 111 под-окон, скоринг даёт число, размер сошёлся")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
