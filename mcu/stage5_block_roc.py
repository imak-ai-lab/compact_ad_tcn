"""Phase 5 / stage 5d: ROC-AUC of the deployed models on the features a device can compute (block-wise, threshold V1).

Usage (from mcu/):  .venv/Scripts/python stage5_block_roc.py [tr047_fp32 ...]     default: all six; env P5_TRIAL -> *_trial.csv
Input: data/sr006_bundle/phase5_block_features.npz (SR006, sr006_dump_bundle.py: every 4th window of PU test, every 4th
normal val window) and the whole-file matrices of the same regions (data/phase5_splits.npz).

Model outputs: Cube.AI x86 C-model of the deployed artifact (`stedgeai validate --mode host`, as stage 3).
Per artifact, on the SAME test windows:
  roc_file          whole-file features, scorer of stage 3 (trained for fp32, refit on whole-file normal val for int8)
  roc_block_same    block features, the same scorer: a device that keeps the shipped scorer
  roc_block_refit   block features, Mahalanobis scorer refit on block-feature normal val outputs (no fault labels)
  delta_*           difference to roc_file, 95 % percentile bootstrap over FILES, stratified by class (windows of one
                    file are correlated, so windows are not resampled)
  fpr/tpr           at the 99 % and 99.5 % quantile of normal val block scores, refit scorer
Checks PASS/FAIL: bundle files equal the local split files; median |block - file| < 0.05 scaled units (row alignment);
|roc_file on this subset - stage 3 roc on all test windows| <= 0.01 (the subset is representative); no NaN scores.
Rows -> features_block_roc.csv.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from p5common import CKPT_DIR, CUBEAI_OPTS, MCU, MODELS, SPLITS, SPLITS_META, STEDGEAI, append_csv, cubeai_env, host_tag, now, verdict
from industrial_ad.experiments import load_detector_from_run

TRIAL = bool(os.environ.get("P5_TRIAL"))
OUT = MCU / ("features_block_roc_trial.csv" if TRIAL else "features_block_roc.csv")
BUNDLE = MCU / "data" / "sr006_bundle"
TMP = MCU / "_tmp_block_roc"
LOGS = MCU / "cubeai" / "validate_block"
T, HOR_TL = 120, 240
N_BOOT, RNG = 1000, 2030
TOL_SUBSET, TOL_ALIGN = 0.01, 0.05
QUANT = {"q99": 0.99, "q995": 0.995}
torch.set_grad_enabled(False)
selection = json.loads((CKPT_DIR / "selection.json").read_text())
meta = json.loads((BUNDLE / "phase5_block_features.meta.json").read_text())
Z = np.load(BUNDLE / "phase5_block_features.npz")
S = np.load(SPLITS)
REF = {e["key"]: e for sp in json.loads(SPLITS_META.read_text())["splits"].values() for e in sp}


def load_det(code: str, seed: int):
    tmp = Path(tempfile.mkdtemp()) / f"{code}_s{seed}"
    (tmp / "checkpoints").mkdir(parents=True)
    shutil.copy2(CKPT_DIR / f"{code}_s{seed}.config.json", tmp / "config.json")
    shutil.copy2(CKPT_DIR / f"{code}_s{seed}.pt", tmp / "checkpoints" / "best.pt")
    return load_detector_from_run(tmp)


def gather(split: str, forecasting: bool):
    """Same windows in both feature versions; targets as the task needs them."""
    xf, yf, xb, yb, lab, fid, mism = [], [], [], [], [], [], 0
    for fi, e in enumerate(meta["splits"][split]):
        r = REF[e["key"]]
        mism += int(r["path"] != e["path"] or r["rows"] != e["region_rows"] or bool(r["is_anomaly"]) != e["is_anomaly"])
        M = S[e["key"]]
        st = np.asarray(e["starts"])
        bx = Z[f"{e['key']}/x"]
        if forecasting:
            n = e["n_tl"]
            if n == 0:
                continue
            st, bx = st[:n], bx[:n]
            by = Z[f"{e['key']}/y_tl"]
            fy = np.stack([M[s + T:s + T + HOR_TL] for s in st])
        fx = np.stack([M[s:s + T] for s in st])
        if not forecasting:
            by, fy = bx, fx
        xf.append(fx), yf.append(fy), xb.append(bx), yb.append(by)
        lab += [int(e["is_anomaly"])] * len(st)
        fid += [fi] * len(st)
    return (np.concatenate(xf), np.concatenate(yf), np.concatenate(xb), np.concatenate(yb), np.asarray(lab), np.asarray(fid), mism)


def c_model(code: str, art: str, x_tc: np.ndarray, label: str) -> np.ndarray:
    tag = f"{code.lower()}_{art}"
    d = TMP / f"{tag}_{label}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    x_cf = np.ascontiguousarray(x_tc.transpose(0, 2, 1))
    np.save(d / "in.npy", x_cf)
    cmd = [STEDGEAI, "validate", "--model", str(MCU / "models" / f"{code}_{art}.onnx"), *CUBEAI_OPTS[art], "--name", tag, "--mode", "host",
           "-vi", str(d / "in.npy"), "--no-exec-model", "--workspace", str(d / "ws"), "--output", str(d / "out")]
    res = subprocess.run(cmd, capture_output=True, text=True, env=cubeai_env())
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"validate_host_{tag}_{label}.log").write_text(res.stdout + res.stderr, encoding="utf-8")
    if res.returncode != 0:
        raise RuntimeError(f"stedgeai validate failed for {tag} {label} (rc={res.returncode}), see {LOGS}")
    out = np.load(d / "out" / f"{tag}_val_c_outputs_1.npy")
    assert out.dtype == np.float32, out.dtype
    out = out.reshape(len(x_cf), x_cf.shape[1], -1) if out.ndim != 3 else out
    shutil.rmtree(d, ignore_errors=True)
    return out


def errors(det, out_cf: np.ndarray, y_tc: np.ndarray) -> torch.Tensor:
    pred = torch.from_numpy(np.ascontiguousarray(out_cf.transpose(0, 2, 1)))
    return det.error_reducer(pred, torch.from_numpy(y_tc))


def auc(lab: np.ndarray, s: np.ndarray) -> float:
    r = rankdata(s)
    n1 = int(lab.sum())
    n0 = len(lab) - n1
    return float((r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_delta(lab, fid, a, b) -> tuple[float, float]:
    """95 % CI of auc(a) - auc(b), resampling files within class."""
    rng = np.random.default_rng(RNG)
    by_file = {f: np.flatnonzero(fid == f) for f in np.unique(fid)}
    fn = [f for f, ix in by_file.items() if lab[ix[0]] == 0]
    fa = [f for f, ix in by_file.items() if lab[ix[0]] == 1]
    out = []
    for _ in range(N_BOOT):
        pick = list(rng.choice(fn, len(fn))) + list(rng.choice(fa, len(fa)))
        ix = np.concatenate([by_file[f] for f in pick])
        out.append(auc(lab[ix], a[ix]) - auc(lab[ix], b[ix]))
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def stage3_roc(art_name: str) -> float:
    with (MCU / "cubeai_host_validate.csv").open(encoding="utf-8") as fh:
        return [float(r["value"]) for r in csv.DictReader(fh) if r["artifact"] == art_name and r["check"] == "roc_auc_pu_test"][0]


def run_tag(tag: str) -> list[dict]:
    code, art = tag.split("_")
    code = code.upper()
    seed = selection[code]
    name = f"{code}_{art}.onnx"
    det, cfg = load_det(code, seed)
    forecasting = cfg["task"]["type"].lower() == "forecasting"
    xf, yf, xb, yb, lab, fid, mism = gather("test", forecasting)
    _, _, xvb, yvb, labv, _, mism_v = gather("val", forecasting)
    assert (labv == 0).all()
    align = float(np.median(np.abs(xb - xf)))
    npz = np.load(MCU / "models" / f"{code}_{art}_cmodel_scorer.npz")
    det.score_estimator.load_state_dict({k: torch.from_numpy(npz[k]) for k in npz.files if k != "note"})
    scorer_note = str(npz["note"])

    s_file = det.score_estimator(errors(det, c_model(code, art, xf, "test_file"), yf)).numpy().astype(np.float64)
    out_block = c_model(code, art, xb, "test_block")
    err_block = errors(det, out_block, yb)
    s_same = det.score_estimator(err_block).numpy().astype(np.float64)
    ev = errors(det, c_model(code, art, xvb, "val_block"), yvb)
    s_val_same = det.score_estimator(ev).numpy().astype(np.float64)
    det.score_estimator.fit(ev)
    s_refit = det.score_estimator(err_block).numpy().astype(np.float64)
    s_val = det.score_estimator(ev).numpy().astype(np.float64)
    nan = int(np.isnan(s_file).sum() + np.isnan(s_same).sum() + np.isnan(s_refit).sum())

    roc_f, roc_s, roc_r = (float(roc_auc_score(lab, v)) for v in (s_file, s_same, s_refit))
    ci_s, ci_r = boot_delta(lab, fid, s_same, s_file), boot_delta(lab, fid, s_refit, s_file)
    ref = stage3_roc(name)
    base = dict(date=now(), host=host_tag(), code=code, seed=seed, artifact=name, deployed="Cube.AI C-model (x86 host)",
                features="block V1 (block MAD, sqrt(2 ln 256000)) from SR006 vs whole-file", stride=meta["stride"], n_test=len(lab),
                n_anom=int(lab.sum()), n_files=len(np.unique(fid)), n_val_normal=len(labv), scorer_stage3=scorer_note)
    rows = [
        {**base, "check": "bundle_files_equal_local", "value": mism + mism_v, "reference": 0, "tol": 0, "status": verdict(mism + mism_v == 0), "note": ""},
        {**base, "check": "median_abs_block_vs_file", "value": f"{align:.4f}", "reference": "", "tol": TOL_ALIGN, "status": verdict(align < TOL_ALIGN), "note": "scaled units; rows aligned"},
        {**base, "check": "no_nan_scores", "value": nan, "reference": 0, "tol": 0, "status": verdict(nan == 0), "note": ""},
        {**base, "check": "subset_roc_vs_stage3_full_test", "value": f"{roc_f:.6f}", "reference": f"{ref:.6f}", "tol": TOL_SUBSET, "status": verdict(abs(roc_f - ref) <= TOL_SUBSET), "note": "same model and features, every 4th window against all windows"},
        {**base, "check": "roc_file", "value": f"{roc_f:.6f}", "reference": "", "tol": "", "status": "MEASURED", "note": "whole-file features, stage-3 scorer"},
        {**base, "check": "roc_block_same", "value": f"{roc_s:.6f}", "reference": "", "tol": "", "status": "MEASURED", "note": "block features, stage-3 scorer"},
        {**base, "check": "roc_block_refit", "value": f"{roc_r:.6f}", "reference": "", "tol": "", "status": "MEASURED", "note": "block features, scorer refit on normal val block outputs, no fault labels"},
        {**base, "check": "delta_block_same", "value": f"{roc_s - roc_f:+.6f}", "reference": "", "tol": "", "status": "MEASURED", "note": f"95 % CI over files [{ci_s[0]:+.4f}, {ci_s[1]:+.4f}]"},
        {**base, "check": "delta_block_refit", "value": f"{roc_r - roc_f:+.6f}", "reference": "", "tol": "", "status": "MEASURED", "note": f"95 % CI over files [{ci_r[0]:+.4f}, {ci_r[1]:+.4f}]"},
    ]
    for qn, q in QUANT.items():
        thr = float(np.quantile(s_val, q))
        rows.append({**base, "check": f"fpr_block_refit_{qn}", "value": f"{float((s_refit[lab == 0] >= thr).mean()):.4f}", "reference": f"{1 - q:.4f}", "tol": "", "status": "MEASURED",
                     "note": f"threshold = {q} quantile of normal val block scores, no fault labels; reference = nominal rate"})
        rows.append({**base, "check": f"tpr_block_refit_{qn}", "value": f"{float((s_refit[lab == 1] >= thr).mean()):.4f}", "reference": "", "tol": "", "status": "MEASURED", "note": "same threshold"})
    print(f"{tag}: file {roc_f:.4f}; block same scorer {roc_s:.4f} ({roc_s - roc_f:+.4f} [{ci_s[0]:+.4f},{ci_s[1]:+.4f}]); block refit {roc_r:.4f} ({roc_r - roc_f:+.4f} [{ci_r[0]:+.4f},{ci_r[1]:+.4f}]); "
          f"stage3 full {ref:.4f}; align {align:.4f}; val same-scorer median {np.median(s_val_same):.1f}", flush=True)
    return rows


if __name__ == "__main__":
    tags = sys.argv[1:] or [f"{c.lower()}_{a}" for c in MODELS for a in ("fp32", "int8")]
    if len(tags) == 6 and not TRIAL:
        OUT.unlink(missing_ok=True)
    fail = 0
    for t in tags:
        rows = run_tag(t)
        append_csv(OUT, rows)
        fail += sum(r["status"] == "FAIL" for r in rows)
    shutil.rmtree(TMP, ignore_errors=True)
    print("FAIL rows:", fail)
    if fail:
        raise SystemExit(1)
