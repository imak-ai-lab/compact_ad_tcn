"""Phase 5 / stage 0: checkpoint acceptance.

Checks (each with explicit tolerance + PASS/FAIL, written to checkpoint_check.csv):
  A. summary.json test/roc_auc of seeds 42-44 -> mean equals phase0_summary.csv
     repro_roc_mean_s42_44 (tol 5e-5, the CSV is rounded to 4 dp); 5-seed mean
     equals roc_mean_5seed (same tol).
  B. |mean_s42_44 - paper reference| <= 0.01.
  C. host CPU re-inference of PU test with the repo's own loader + scorer reproduces
     summary.json test/roc_auc per seed within 1e-3 (weights loaded correctly).
Then picks, per model, the seed (42-44) whose ROC-AUC is closest to the 3-seed
mean and copies its best.pt/config.json/summary.json into mcu/checkpoints/.
"""
from __future__ import annotations

import csv
import json
import shutil

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from p5common import (CKPT_DIR, MCU, MODELS, PAPER, SEEDS_ALL, SEEDS_REPRO, append_csv, build_windows,
                      host_tag, now, run_dir, verdict)
from industrial_ad.experiments import load_detector_from_run
from industrial_ad.utils import count_parameters

TOL_CSV, TOL_PAPER, TOL_HOST = 5e-5, 0.01, 1e-3
OUT = MCU / "checkpoint_check.csv"

torch.set_num_threads(4)
phase0 = {r["code"]: r for r in csv.DictReader((PAPER / "phase0_summary.csv").open())}
rows, selection = [], {}

for code, (family, name, ref) in MODELS.items():
    roc = {}
    for seed in SEEDS_ALL:
        d = run_dir(code, seed)
        s = json.loads((d / "summary.json").read_text())
        roc[seed] = float(s["best_metrics"]["test/roc_auc"])
        has_pt = (d / "checkpoints" / "best.pt").exists()
        base = dict(date=now(), host=host_tag(), code=code, run_name=name, seed=seed)
        if seed not in SEEDS_REPRO:
            rows.append({**base, "check": "summary_roc", "value": roc[seed], "reference": "", "tol": "", "abs_diff": "",
                         "status": "INFO", "note": f"best.pt present={has_pt}"})
            continue
        if not has_pt:
            rows.append({**base, "check": "weights_present", "value": 0, "reference": 1, "tol": 0, "abs_diff": 1,
                         "status": "FAIL", "note": "best.pt missing"})
            continue
        # C. host re-inference
        det, cfg = load_detector_from_run(d)
        n_params = count_parameters(det.model)
        x, y, lab = build_windows(cfg, "test")
        with torch.no_grad():
            sc = torch.cat([det.get_anomaly_scores(torch.from_numpy(x[i:i + 512]), torch.from_numpy(y[i:i + 512]))
                            for i in range(0, len(x), 512)]).numpy()
        host_roc = float(roc_auc_score(lab, sc))
        rows.append({**base, "check": "param_count", "value": n_params, "reference": ref["params"], "tol": 0,
                     "abs_diff": abs(n_params - ref["params"]), "status": verdict(n_params == ref["params"]),
                     "note": "model params (scorer buffers excluded)"})
        rows.append({**base, "check": "host_reinference_roc", "value": round(host_roc, 6), "reference": round(roc[seed], 6),
                     "tol": TOL_HOST, "abs_diff": round(abs(host_roc - roc[seed]), 6),
                     "status": verdict(abs(host_roc - roc[seed]) <= TOL_HOST),
                     "note": f"CPU torch {torch.__version__}; test windows={len(lab)} (anom={int(lab.sum())})"})
    # A / B
    m3 = float(np.mean([roc[s] for s in SEEDS_REPRO]))
    m5 = float(np.mean([roc[s] for s in SEEDS_ALL]))
    p0 = phase0[code]
    base = dict(date=now(), host=host_tag(), code=code, run_name=name, seed="42-44")
    for check, val, refv, tol in [
        ("mean_s42_44_vs_phase0_csv", m3, float(p0["repro_roc_mean_s42_44"]), TOL_CSV),
        ("mean_5seed_vs_phase0_csv", m5, float(p0["roc_mean_5seed"]), TOL_CSV),
        ("mean_s42_44_vs_paper", m3, ref["roc"], TOL_PAPER),
    ]:
        rows.append({**base, "seed": "42-46" if "5seed" in check else "42-44", "check": check, "value": round(val, 6),
                     "reference": refv, "tol": tol, "abs_diff": round(abs(val - refv), 6),
                     "status": verdict(abs(val - refv) <= tol), "note": ""})
    pick = min(SEEDS_REPRO, key=lambda s: abs(roc[s] - m3))
    selection[code] = pick
    rows.append({**base, "seed": pick, "check": "deploy_seed", "value": round(roc[pick], 6), "reference": round(m3, 6),
                 "tol": "", "abs_diff": round(abs(roc[pick] - m3), 6), "status": "INFO",
                 "note": "seed in 42-44 closest to 3-seed mean"})

append_csv(OUT, rows, fresh=True)
fails = [r for r in rows if r["status"] == "FAIL"]
for r in rows:
    print(f"{r['status']:5} {r['code']} s{r['seed']} {r['check']}: {r['value']} vs {r['reference']} (tol {r['tol']})")

if fails:
    print(f"\n{len(fails)} FAIL -> checkpoints NOT copied")
    raise SystemExit(1)

CKPT_DIR.mkdir(exist_ok=True)
for code, seed in selection.items():
    d = run_dir(code, seed)
    shutil.copy2(d / "checkpoints" / "best.pt", CKPT_DIR / f"{code}_s{seed}.pt")
    shutil.copy2(d / "config.json", CKPT_DIR / f"{code}_s{seed}.config.json")
    shutil.copy2(d / "summary.json", CKPT_DIR / f"{code}_s{seed}.summary.json")
(CKPT_DIR / "selection.json").write_text(json.dumps(selection, indent=1))
print("\nall PASS; deploy seeds:", selection)
