"""Analyses E2 and E3. Host CPU, no training.

Models: the three MCU configurations TR047, TL082, TR138, seeds 42-44 (best.pt, SR006 phase-0 runs
copied to mcu/checkpoints_raw). Data: the PU val/test feature matrices dumped by the repo's own
loader (mcu/data/phase5_splits.npz), windowed by p5common.build_windows (= loader windowing).

Consistency checks first (PASS/FAIL, explicit tol):
  * recomputed test / val ROC-AUC vs summary.json best_metrics, |diff| <= 1e-4
  * recomputed best-F1 threshold on val vs summary.json best_metrics/val/threshold, rel diff <= 1e-4

E2 (threshold without fault labels): thresholds
  * best_f1_val      - repo rule: F1-optimal on val, USES fault labels (reference)
  * q99_normal_val   - 99 % quantile of val scores of NORMAL windows only (no fault labels)
  * q995_normal_val  - 99.5 % quantile, same
  applied to test with the repo's rule (score >= thr): FPR, TPR, precision, F1, overall and per condition.

E3 (operating conditions): test ROC-AUC per PU operating condition present in the protocol.
  The protocol (train/val/test patterns K0*/N15_*/, KA*/N15_*_1/) contains only the three 1500-rpm
  conditions N15_M07_F10, N15_M01_F10, N15_M07_F04; N09_M07_F10 is not used anywhere. This is
  sensitivity to condition, not a condition-shift experiment.

Output: e23_threshold_conditions.csv (long format) next to this script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MCU = HERE.parent / "mcu"
sys.path.insert(0, str(MCU))
from p5common import (MODELS, SEEDS_REPRO, SPLITS_META, append_csv, build_windows, host_tag, now,  # noqa: E402
                      run_dir, verdict)

import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from industrial_ad.experiments import load_detector_from_run  # noqa: E402
from industrial_ad.scoring import fit_best_f1_threshold  # noqa: E402

OUT = HERE / "e23_threshold_conditions.csv"
TOL_ROC, TOL_THR = 1e-4, 1e-4
QUANTILES = {"q99_normal_val": 0.99, "q995_normal_val": 0.995}
BATCH = 512
torch.set_grad_enabled(False)


def window_conditions(cfg: dict, split: str) -> np.ndarray:
    """Condition string per window, in build_windows order (same skip rule and step)."""
    p = cfg["dataset"]["params"]
    win, hor = int(p["window_size"]), int(p["horizon_size"])
    step = win - int(p["window_overlap"])
    meta = json.loads(SPLITS_META.read_text())
    rows = []
    for e in meta["splits"][split]:
        n_rows = int(e["rows"])
        if n_rows < win + hor:
            continue
        n = len(range(0, n_rows - win - hor + 1, step))
        rows += [(e["path"][:11], int(e["is_anomaly"]))] * n
    return np.array(rows, dtype=object)


def scores(det, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    det.eval()
    out = []
    for i in range(0, len(x), BATCH):
        xb, yb = torch.from_numpy(x[i:i + BATCH]), torch.from_numpy(y[i:i + BATCH])
        out.append(det.get_anomaly_scores(xb, yb).numpy())
    return np.concatenate(out).astype(np.float64)


def op_point(s: np.ndarray, lab: np.ndarray, thr: float) -> dict:
    pred = s >= thr
    tp, fp = int((pred & (lab == 1)).sum()), int((pred & (lab == 0)).sum())
    n_pos, n_neg = int((lab == 1).sum()), int((lab == 0).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / n_pos if n_pos else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    return dict(fpr=fp / n_neg if n_neg else float("nan"), tpr=rec, precision=prec, f1=f1, n_neg=n_neg, n_pos=n_pos)


rows = []
for code, (family, name, _) in MODELS.items():
    for seed in SEEDS_REPRO:
        rd = run_dir(code, seed)
        det, cfg = load_detector_from_run(rd)
        summ = json.loads((rd / "summary.json").read_text())["best_metrics"]
        base = dict(date=now(), host=host_tag(), code=code, seed=seed, run=rd.name, ckpt="best.pt (val ROC)",
                    torch=torch.__version__)
        data = {}
        for split in ("val", "test"):
            x, y, lab = build_windows(cfg, split)
            cond = window_conditions(cfg, split)
            assert len(cond) == len(lab), (code, seed, split, len(cond), len(lab))
            assert np.array_equal(cond[:, 1].astype(int), lab), (code, seed, split, "label order")
            data[split] = (scores(det, x, y), lab, cond[:, 0].astype(str))
            del x, y
        sv, lv, _ = data["val"]
        st, lt, ct = data["test"]

        # ---- consistency with the training run
        for split, (s, lab, _) in data.items():
            roc = roc_auc_score(lab, s)
            ref = summ[f"{split}/roc_auc"]
            rows.append({**base, "analysis": "check", "split": split, "condition": "all", "rule": "",
                         "metric": "roc_auc_vs_summary", "value": round(roc, 6), "reference": round(ref, 6),
                         "tol": TOL_ROC, "status": verdict(abs(roc - ref) <= TOL_ROC), "note": f"N={len(lab)}"})
        thr_f1 = fit_best_f1_threshold(sv, lv)
        ref_thr = summ["val/threshold"]
        rel = abs(thr_f1 - ref_thr) / abs(ref_thr)
        rows.append({**base, "analysis": "check", "split": "val", "condition": "all", "rule": "best_f1_val",
                     "metric": "threshold_vs_summary", "value": f"{thr_f1:.6g}", "reference": f"{ref_thr:.6g}",
                     "tol": TOL_THR, "status": verdict(rel <= TOL_THR), "note": f"rel diff {rel:.2e}"})

        # ---- E2: operating points on test
        thr = {"best_f1_val": thr_f1, **{k: float(np.quantile(sv[lv == 0], q)) for k, q in QUANTILES.items()}}
        for rule, t in thr.items():
            uses = "uses fault labels (val)" if rule == "best_f1_val" else "normal val windows only"
            for c in ["all", *sorted(set(ct))]:
                m = (ct == c) if c != "all" else np.ones(len(ct), bool)
                op = op_point(st[m], lt[m], t)
                for k in ("fpr", "tpr", "precision", "f1"):
                    rows.append({**base, "analysis": "E2", "split": "test", "condition": c, "rule": rule,
                                 "metric": k, "value": round(op[k], 6), "reference": "", "tol": "", "status": "INFO",
                                 "note": f"thr={t:.6g}; {uses}; n_neg={op['n_neg']} n_pos={op['n_pos']}"})

        # ---- E3: ROC-AUC per operating condition on test
        for c in sorted(set(ct)):
            m = ct == c
            rows.append({**base, "analysis": "E3", "split": "test", "condition": c, "rule": "",
                         "metric": "roc_auc", "value": round(roc_auc_score(lt[m], st[m]), 6), "reference": "",
                         "tol": "", "status": "INFO",
                         "note": f"n_neg={int((lt[m] == 0).sum())} n_pos={int((lt[m] == 1).sum())}"})
        print(f"{code} s{seed}: done", flush=True)

append_csv(OUT, rows, fresh=True)

# ---- console summary: mean over seeds 42-44 (the CSV keeps per-seed rows)
import collections  # noqa: E402
agg = collections.defaultdict(list)
for r in rows:
    if r["analysis"] in ("E2", "E3"):
        agg[(r["code"], r["analysis"], r["rule"], r["condition"], r["metric"])].append(float(r["value"]))
for r in rows:
    if r["analysis"] == "check":
        print(f"{r['status']} {r['code']} s{r['seed']} {r['split']} {r['metric']}: {r['value']} ref={r['reference']}")
for k, v in sorted(agg.items()):
    if k[1] == "E3" or k[3] == "all":
        print(f"{k[0]} {k[1]} {k[2] or '-':16} {k[3]:12} {k[4]:9} mean={np.mean(v):.4f} "
              f"seeds={' '.join(f'{x:.4f}' for x in v)}")
if any(r["status"] == "FAIL" for r in rows):
    raise SystemExit(1)
