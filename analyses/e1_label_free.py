"""Analysis E1: what does checkpoint selection without fault labels cost?

Every phase-0 run trains for a fixed budget; `best_metrics` is the epoch chosen by val ROC-AUC (uses fault labels), `last_metrics`
is the last epoch (uses none). Both are already in summary.json of each run (mcu/data/sr006_bundle/summaries, from SR006).
Limited to the 22 headline configurations of phase 0: `last_metrics` were not kept for the whole sweep.

Checks: 5 seeds per configuration; no run stopped early; 5-seed mean of best vs phase0_summary.csv, tol 1e-3 (the CSV keeps 4 decimals).
Output: e1_label_free.csv next to this script.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
MCU = PAPER / "mcu"
sys.path.insert(0, str(MCU))
from p5common import append_csv, host_tag, now, verdict  # noqa: E402

ROOT = MCU / "data" / "sr006_bundle" / "summaries" / "runs_phase0"
OUT = HERE / "e1_label_free.csv"
TOL = 1e-3
SEEDS_PAPER = (42, 43, 44)

with (PAPER / "phase0_summary.csv").open(encoding="utf-8") as fh:
    table = {(r["family"], r["run_name"]): r for r in csv.DictReader(fh)}

runs = defaultdict(dict)
stopped = 0
for p in ROOT.glob("*/*/summary.json"):
    s = json.loads(p.read_text(encoding="utf-8"))
    m = re.match(r"^(\d+)-s(\d+)-(.+)$", p.parent.name)
    run_name, seed = f"{m.group(1)}-{m.group(3)}", int(m.group(2))
    stopped += int(bool(s.get("stopped_early")))
    runs[(p.parent.parent.name, run_name)][seed] = (s["best_metrics"]["test/roc_auc"], s["last_metrics"]["test/roc_auc"], s["best_epoch"], s["last_epoch"])

rows, summ = [], []
for key, by_seed in sorted(runs.items()):
    family, run_name = key
    t = table.get(key)
    best = np.array([by_seed[s][0] for s in sorted(by_seed)])
    last = np.array([by_seed[s][1] for s in sorted(by_seed)])
    b3 = float(np.mean([by_seed[s][0] for s in SEEDS_PAPER]))
    l3 = float(np.mean([by_seed[s][1] for s in SEEDS_PAPER]))
    code = t["code"] if t else "?"
    base = dict(date=now(), host=host_tag(), code=code, family=family, run_name=run_name, n_seeds=len(by_seed))
    ref = float(t["roc_mean_5seed"]) if t else float("nan")
    rows.append({**base, "metric": "check_mean_best_5seed_vs_phase0_summary", "value": f"{best.mean():.5f}", "reference": f"{ref:.4f}", "tol": TOL,
                 "status": verdict(bool(t) and abs(best.mean() - ref) <= TOL and len(by_seed) == 5), "note": ""})
    rows.append({**base, "metric": "roc_best_mean_s42_44", "value": f"{b3:.5f}", "reference": "", "tol": "", "status": "MEASURED", "note": "epoch chosen by val ROC, uses fault labels"})
    rows.append({**base, "metric": "roc_last_mean_s42_44", "value": f"{l3:.5f}", "reference": "", "tol": "", "status": "MEASURED", "note": "last epoch, no fault labels"})
    rows.append({**base, "metric": "delta_last_minus_best_s42_44", "value": f"{l3 - b3:+.5f}", "reference": "", "tol": "", "status": "MEASURED",
                 "note": "per seed " + " ".join(f"{by_seed[s][1] - by_seed[s][0]:+.4f}" for s in sorted(by_seed))})
    rows.append({**base, "metric": "delta_last_minus_best_5seed", "value": f"{(last - best).mean():+.5f}", "reference": "", "tol": "", "status": "MEASURED", "note": ""})
    summ.append((code, family, b3, l3))

n = len(summ)
rows.append(dict(date=now(), host=host_tag(), code="all", family="all", run_name="all", n_seeds="", metric="check_no_early_stopping", value=stopped, reference=0, tol=0,
                 status=verdict(stopped == 0), note="last epoch is the end of the fixed budget"))
rho = float(spearmanr([s[2] for s in summ], [s[3] for s in summ]).statistic)
rows.append(dict(date=now(), host=host_tag(), code="all", family="all", run_name="all", n_seeds="", metric="spearman_best_vs_last_over_configs", value=f"{rho:.3f}", reference="", tol="",
                 status="MEASURED", note=f"{n} headline configurations, means over seeds 42-44"))
append_csv(OUT, rows, fresh=True)

print(f"{n} configurations, early-stopped runs: {stopped}, Spearman best vs last {rho:.3f}\n")
for code, fam, b, l in sorted(summ, key=lambda s: -s[2]):
    print(f"{code:7s} {fam:32s} best {b:.4f}  last {l:.4f}  {l - b:+.4f}")
fb = max(summ, key=lambda s: s[2])
fl = max(summ, key=lambda s: s[3])
print(f"\nleader by best: {fb[0]} {fb[2]:.4f}; leader by last: {fl[0]} {fl[3]:.4f}")
for r in rows:
    if r["status"] == "FAIL":
        print("FAIL", r["code"], r["metric"], r["value"], r["reference"])
if any(r["status"] == "FAIL" for r in rows):
    raise SystemExit(1)
