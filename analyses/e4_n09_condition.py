"""Analysis E4: an operating condition no model has seen.

The protocol uses only the three 1500-rpm conditions; N09_M07_F10 (900 rpm) appears nowhere in training, validation or testing.
Models: TR047, TL082, TR138, seeds 42-44, best.pt, no retraining, as E2/E3. Data: N09 matrices cut exactly as N15 by the repo's own
code on SR006 (mcu/data/sr006_bundle/phase5_n09_features.npz). Windows are cut as in p5common.build_windows.

  as shipped: scorer and thresholds from N15 -> ROC-AUC on N09 test, FPR/TPR at the N15 thresholds
              (99 % and 99.5 % quantiles of normal N15 val scores; best-F1 on N15 val, which uses fault labels)
  recalibrated without fault labels: scorer and quantile thresholds refit on NORMAL N09 val windows, the model weights unchanged ->
              ROC-AUC, FPR, TPR on N09 test. This is what a deployment could do after moving to a new condition.
Check: N15 test and val ROC-AUC vs summary.json, tol 1e-4. Output: e4_n09_condition.csv next to this script.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MCU = HERE.parent / "mcu"
sys.path.insert(0, str(MCU))
from p5common import MODELS, SEEDS_REPRO, append_csv, build_windows, host_tag, now, run_dir, verdict  # noqa: E402

import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from industrial_ad.experiments import load_detector_from_run  # noqa: E402
from industrial_ad.scoring import fit_best_f1_threshold  # noqa: E402

BUNDLE = MCU / "data" / "sr006_bundle"
OUT = HERE / "e4_n09_condition.csv"
TOL_ROC, BATCH = 1e-4, 512
QUANT = {"q99_normal_val": 0.99, "q995_normal_val": 0.995}
torch.set_grad_enabled(False)
N09 = np.load(BUNDLE / "phase5_n09_features.npz")
N09_META = json.loads((BUNDLE / "phase5_n09_features.meta.json").read_text())


def n09_windows(cfg: dict, split: str):
    p = cfg["dataset"]["params"]
    win, hor = int(p["window_size"]), int(p["horizon_size"])
    step = win - int(p["window_overlap"])
    recon = cfg["task"]["type"].lower() == "reconstruction"
    xs, ys, ls = [], [], []
    for e in N09_META["splits"][split]:
        mat = N09[e["key"]]
        if len(mat) < win + hor:
            continue
        for s in range(0, len(mat) - win - hor + 1, step):
            w = mat[s:s + win]
            xs.append(w)
            ys.append(w if recon else mat[s + win:s + win + hor])
            ls.append(int(e["is_anomaly"]))
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32), np.asarray(ls, dtype=np.int64)


def errs(det, x, y) -> torch.Tensor:
    det.eval()
    return torch.cat([det.get_errors(torch.from_numpy(x[i:i + BATCH]), torch.from_numpy(y[i:i + BATCH])) for i in range(0, len(x), BATCH)])


def op(s, lab, thr):
    return float((s[lab == 0] >= thr).mean()), float((s[lab == 1] >= thr).mean())


rows = []
for code in MODELS:
    for seed in SEEDS_REPRO:
        rd = run_dir(code, seed)
        det, cfg = load_detector_from_run(rd)
        summ = json.loads((rd / "summary.json").read_text())["best_metrics"]
        base = dict(date=now(), host=host_tag(), code=code, seed=seed, run=rd.name, ckpt="best.pt (val ROC, N15)", torch=torch.__version__)

        xv, yv, lv = build_windows(cfg, "val")
        sv = det.score_estimator(errs(det, xv, yv)).numpy().astype(np.float64)
        del xv, yv
        xt, yt, lt = build_windows(cfg, "test")
        st = det.score_estimator(errs(det, xt, yt)).numpy().astype(np.float64)
        del xt, yt
        for split, (s, lab) in {"val": (sv, lv), "test": (st, lt)}.items():
            roc, ref = roc_auc_score(lab, s), summ[f"{split}/roc_auc"]
            rows.append({**base, "setting": "N15 reference", "metric": f"roc_auc_{split}_vs_summary", "rule": "", "value": f"{roc:.6f}", "reference": f"{ref:.6f}",
                         "tol": TOL_ROC, "status": verdict(abs(roc - ref) <= TOL_ROC), "note": f"N={len(lab)}"})
        thr15 = {"best_f1_val": float(fit_best_f1_threshold(sv, lv)), **{k: float(np.quantile(sv[lv == 0], q)) for k, q in QUANT.items()}}
        for rule, t in thr15.items():
            fpr, tpr = op(st, lt, t)
            rows.append({**base, "setting": "N15 reference", "metric": "fpr", "rule": rule, "value": f"{fpr:.4f}", "reference": "", "tol": "", "status": "INFO", "note": f"thr={t:.6g}"})
            rows.append({**base, "setting": "N15 reference", "metric": "tpr", "rule": rule, "value": f"{tpr:.4f}", "reference": "", "tol": "", "status": "INFO", "note": f"thr={t:.6g}"})

        xv9, yv9, lv9 = n09_windows(cfg, "val")
        xt9, yt9, lt9 = n09_windows(cfg, "test")
        et9, ev9 = errs(det, xt9, yt9), errs(det, xv9[lv9 == 0], yv9[lv9 == 0])
        del xt9, yt9, xv9, yv9
        s9 = det.score_estimator(et9).numpy().astype(np.float64)
        n_note = f"n_neg={int((lt9 == 0).sum())} n_pos={int((lt9 == 1).sum())}"
        rows.append({**base, "setting": "N09 as shipped", "metric": "roc_auc", "rule": "", "value": f"{roc_auc_score(lt9, s9):.6f}", "reference": "", "tol": "", "status": "MEASURED", "note": n_note})
        for rule, t in thr15.items():
            fpr, tpr = op(s9, lt9, t)
            rows.append({**base, "setting": "N09 as shipped", "metric": "fpr", "rule": rule, "value": f"{fpr:.4f}", "reference": "", "tol": "", "status": "MEASURED", "note": f"N15 threshold {t:.6g}; {n_note}"})
            rows.append({**base, "setting": "N09 as shipped", "metric": "tpr", "rule": rule, "value": f"{tpr:.4f}", "reference": "", "tol": "", "status": "MEASURED", "note": f"N15 threshold {t:.6g}; {n_note}"})

        det.score_estimator.fit(ev9)                      # normal N09 val only, no fault labels, weights unchanged
        s9r = det.score_estimator(et9).numpy().astype(np.float64)
        sv9r = det.score_estimator(ev9).numpy().astype(np.float64)
        rows.append({**base, "setting": "N09 recalibrated", "metric": "roc_auc", "rule": "", "value": f"{roc_auc_score(lt9, s9r):.6f}", "reference": "", "tol": "", "status": "MEASURED", "note": f"{n_note}; n_val_normal={len(ev9)}"})
        for rule, q in QUANT.items():
            t = float(np.quantile(sv9r, q))
            fpr, tpr = op(s9r, lt9, t)
            rows.append({**base, "setting": "N09 recalibrated", "metric": "fpr", "rule": rule, "value": f"{fpr:.4f}", "reference": "", "tol": "", "status": "MEASURED", "note": f"N09 threshold {t:.6g}"})
            rows.append({**base, "setting": "N09 recalibrated", "metric": "tpr", "rule": rule, "value": f"{tpr:.4f}", "reference": "", "tol": "", "status": "MEASURED", "note": f"N09 threshold {t:.6g}"})
        print(f"{code} s{seed}: N15 test {roc_auc_score(lt, st):.4f}; N09 shipped {roc_auc_score(lt9, s9):.4f}; recalibrated {roc_auc_score(lt9, s9r):.4f}", flush=True)
        del det, sv, st, s9, s9r, et9, ev9

append_csv(OUT, rows, fresh=True)
agg = defaultdict(list)
for r in rows:
    if r["status"] in ("MEASURED", "INFO"):
        agg[(r["code"], r["setting"], r["metric"], r["rule"])].append(float(r["value"]))
for k, v in sorted(agg.items()):
    print(f"{k[0]} {k[1]:17} {k[2]:8} {k[3]:16} mean={np.mean(v):.4f} seeds={' '.join(f'{x:.4f}' for x in v)}")
for r in rows:
    if r["status"] == "FAIL":
        print("FAIL", r["code"], r["seed"], r["metric"], r["value"], r["reference"])
if any(r["status"] == "FAIL" for r in rows):
    raise SystemExit(1)
