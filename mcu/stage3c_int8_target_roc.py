"""Phase 5 / stage 3c: ROC-AUC of mixed int8 computed from the BOARD's own outputs.

Why: the int8 ROC-AUC in Table MCU comes from the x86 C-model, and board outputs for mixed int8 are not
identical to it (cubeai_validate.csv: max rel diff 0.31-0.42, p99.99 0.035-0.19). Stage 3 kept only the
reports, not the board outputs. This stage repeats the stage-3 target validation on the same subset
(128 normal + 128 anomalous PU test windows, rng 2026, data/stage3_target_subset_<code>.npz), KEEPS the
board outputs, and scores board and C-model outputs with the same scorer of the deployed artifact
(models/<code>_<art>_cmodel_scorer.npz, stage 3a), so the ROC-AUC difference is paired.

Usage (from mcu/):
    .venv/Scripts/python stage3c_int8_target_roc.py host           # no board: C-model on the subset + scoring check
    .venv/Scripts/python stage3c_int8_target_roc.py run [tag ...]  # board: flash + validate + score
Default tags: tr047_fp32 (control), tr047_int8, tl082_int8, tr138_int8.

Checks -> int8_target_roc.csv (run only):
  roc_auc_board_vs_cmodel  value = ROC(board), reference = ROC(C-model), tol 0.01 (int8) / 0.001 (fp32 control)
  delta_roc_bootstrap95    INFO, window-level bootstrap of the paired difference (2000 resamples)
  spearman_scores          INFO, rank correlation of board vs C-model scores
  report_l2r_cos           INFO, stedgeai report: board vs reference, relative L2 and cosine
  latency_ms_mean          MEASURED, from the report (sanity vs stage 3)
Outputs kept: cubeai/validate_target_roc/<tag>_{target,host}/ and data/stage3c_<tag>_io.npz.
BCM4 must be 0 (checked, nothing is written to option bytes).
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from p5common import CKPT_DIR, CUBEAI_OPTS, MCU, STEDGEAI, append_csv, build_windows, cubeai_env, host_tag, now, verdict
import stage3_target_validate as s3                     # prog, sh, stlink_com, option_bytes, parse_timing, FW, BAUD

import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from industrial_ad.experiments import load_detector_from_run

OUT = MCU / "int8_target_roc.csv"
WORK = MCU / "cubeai" / "validate_target_roc"          # persistent on purpose (stage 3 deleted its outputs)
TAGS = ["tr047_fp32", "tr047_int8", "tl082_int8", "tr138_int8"]
TOL = {"int8": 0.01, "fp32": 0.001}
N_BOOT, BOOT_SEED = 2000, 2031
selection = json.loads((CKPT_DIR / "selection.json").read_text())


def load_det(code: str, seed: int):
    """Same as stage3_host_validate.load (that module runs stage 3a on import)."""
    import tempfile
    tmp = Path(tempfile.mkdtemp()) / f"{code}_s{seed}"
    (tmp / "checkpoints").mkdir(parents=True)
    shutil.copy2(CKPT_DIR / f"{code}_s{seed}.config.json", tmp / "config.json")
    shutil.copy2(CKPT_DIR / f"{code}_s{seed}.pt", tmp / "checkpoints" / "best.pt")
    return load_detector_from_run(tmp)


def scores(det, out_cf: np.ndarray, y_tc: np.ndarray) -> np.ndarray:
    pred = torch.from_numpy(np.ascontiguousarray(out_cf.transpose(0, 2, 1)))
    with torch.no_grad():
        return det.score_estimator(det.error_reducer(pred, torch.from_numpy(y_tc))).numpy().astype(np.float64)


def subset(code: str):
    """The stage-3 subset: indices are read back AND recomputed (same rng), they must agree."""
    seed = selection[code]
    cfg = json.loads((CKPT_DIR / f"{code}_s{seed}.config.json").read_text())
    x, y, lab = build_windows(cfg, "test")
    saved = np.load(MCU / "data" / f"stage3_target_subset_{code}.npz")
    rng = np.random.default_rng(int(saved["seed"]))
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(lab == c), s3.N_PER_CLASS, replace=False) for c in (0, 1)]))
    if not np.array_equal(idx, saved["idx"]) or not np.array_equal(lab[idx], saved["label"]):
        raise SystemExit(f"{code}: subset differs from stage 3 (data/stage3_target_subset_{code}.npz)")
    return np.ascontiguousarray(x[idx].transpose(0, 2, 1)), y[idx], lab[idx], seed


def validate(tag: str, art: str, x_cf: np.ndarray, mode: str, com: str | None):
    d = WORK / f"{tag}_{mode}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    np.save(d / "in.npy", x_cf)
    onnx = MCU / "models" / f"{tag.split('_')[0].upper()}_{art}.onnx"
    cmd = [STEDGEAI, "validate", "--model", onnx, *CUBEAI_OPTS[art], "--name", "network", "--mode", mode,
           "-vi", d / "in.npy", "--workspace", d / "ws", "--output", d / "out"]
    cmd += ["-d", f"serial:{com}:{s3.BAUD}"] if mode == "target" else ["--no-exec-model"]
    t0 = time.time()
    log = s3.sh(cmd, check=False, env=cubeai_env())
    (d / "validate.log").write_text(log, encoding="utf-8")
    rep = next(iter((d / "out").glob("*_validate_report.txt")), None)
    outs = sorted((d / "out").glob("*_val_c_outputs_1.npy"))
    if rep is None or not outs:
        raise RuntimeError(f"{tag} {mode}: no report or no outputs, see {d / 'validate.log'}")
    y = np.load(outs[0])
    y = y.reshape(len(x_cf), x_cf.shape[1], -1) if y.ndim != 3 else y
    return y, rep.read_text(encoding="utf-8", errors="replace"), time.time() - t0


def report_l2r_cos(txt: str):
    m = re.search(r"Cross accuracy report #1.*?rmse=([\d.eE+-]+).*?l2r=([\d.eE+-]+).*?cos=([\d.eE+-]+)", txt, re.S)
    return (float(m.group(2)), float(m.group(3))) if m else (None, None)


def boot_delta(lab, s_b, s_c):
    rng = np.random.default_rng(BOOT_SEED)
    d, n = [], len(lab)
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if lab[i].min() == lab[i].max():
            continue
        d.append(roc_auc_score(lab[i], s_b[i]) - roc_auc_score(lab[i], s_c[i]))
    return np.quantile(d, [0.025, 0.975])


def cmd_host():
    """No board: C-model outputs on the subset, scored; prints ROC on the subset (pipeline check only)."""
    for tag in TAGS:
        code, art = tag.split("_")
        code = code.upper()
        x_cf, y, lab, seed = subset(code)
        det, _ = load_det(code, seed)
        st = np.load(MCU / "models" / f"{code}_{art}_cmodel_scorer.npz")
        det.score_estimator.load_state_dict({k: torch.from_numpy(st[k]) for k in st.files if k != "note"})
        y_h, _, sec = validate(tag, art, x_cf, "host", None)
        s_h = scores(det, y_h, y)
        print(f"{tag}: subset N={len(lab)} anom={int(lab.sum())}, C-model ROC on subset = {roc_auc_score(lab, s_h):.4f} "
              f"({sec:.0f} s); scorer: {st['note']}", flush=True)


def cmd_run(tags):
    ob = s3.option_bytes()
    if ob["BCM4"] != 0:
        raise SystemExit(f"BCM4 = {ob['BCM4']} (must be 0). Nothing flashed.")
    com = s3.stlink_com()
    ver = " ".join(s3.sh([STEDGEAI, "--version"]).split())
    rows_all = []
    for tag in tags:
        code, art = tag.split("_")
        code = code.upper()
        x_cf, y, lab, seed = subset(code)
        det, _ = load_det(code, seed)
        st = np.load(MCU / "models" / f"{code}_{art}_cmodel_scorer.npz")
        det.score_estimator.load_state_dict({k: torch.from_numpy(st[k]) for k in st.files if k != "note"})
        binf = s3.FW / tag / f"validation_{tag}.bin"
        flash_log = s3.prog("-w", binf, "0x08000000", "-v", "-hardRst")
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / f"flash_{tag}.log").write_text(flash_log, encoding="utf-8")
        time.sleep(2.0)
        for attempt in (1, 2):
            try:
                y_t, rep_t, sec_t = validate(tag, art, x_cf, "target", com)
                break
            except RuntimeError as e:
                if attempt == 2:
                    raise
                print(f"{tag}: target attempt 1 failed ({str(e)[:120]}); reset and retry", flush=True)
                s3.prog("-hardRst")
                time.sleep(2.0)
        y_h, _, _ = validate(tag, art, x_cf, "host", None)
        s_b, s_c = scores(det, y_t, y), scores(det, y_h, y)
        np.savez(MCU / "data" / f"stage3c_{tag}_io.npz", out_board=y_t, out_cmodel=y_h, score_board=s_b,
                 score_cmodel=s_c, label=lab)
        roc_b, roc_c = roc_auc_score(lab, s_b), roc_auc_score(lab, s_c)
        lo, hi = boot_delta(lab, s_b, s_c)
        rho = float(spearmanr(s_b, s_c).correlation)
        l2r, cos = report_l2r_cos(rep_t)
        tm = s3.parse_timing(rep_t)
        base = dict(date=now(), host=host_tag(), stedgeai=ver, options=" ".join(CUBEAI_OPTS[art]), code=code, seed=seed,
                    artifact=f"{code}_{art}.onnx", firmware=binf.name, baud=s3.BAUD, target_attempt=attempt,
                    n=len(lab), subset=f"{s3.N_PER_CLASS}+{s3.N_PER_CLASS} test rng{s3.SUBSET_SEED} (stage 3)",
                    scorer=str(st["note"]), role="control" if art == "fp32" else "int8")
        rows = [
            {**base, "check": "roc_auc_board_vs_cmodel", "value": f"{roc_b:.6f}", "reference": f"{roc_c:.6f}",
             "tol": TOL[art], "status": verdict(abs(roc_b - roc_c) <= TOL[art]),
             "note": f"delta={roc_b - roc_c:+.6f}; same scorer for both"},
            {**base, "check": "delta_roc_bootstrap95", "value": f"{lo:+.6f}..{hi:+.6f}", "reference": "", "tol": "",
             "status": "INFO", "note": f"window-level paired bootstrap, {N_BOOT} resamples, seed {BOOT_SEED}"},
            {**base, "check": "spearman_scores", "value": f"{rho:.6f}", "reference": "", "tol": "", "status": "INFO",
             "note": "board vs C-model anomaly scores"},
            {**base, "check": "report_l2r_cos", "value": f"{l2r}/{cos}", "reference": "", "tol": "", "status": "INFO",
             "note": "stedgeai report #1, board vs reference model"},
            {**base, "check": "latency_ms_mean", "value": tm.get("dur_ms"), "reference": "", "tol": "",
             "status": "MEASURED", "note": f"validate run {sec_t:.0f} s for {len(lab)} windows"},
        ]
        append_csv(OUT, rows)
        rows_all += rows
        print(f"{tag}: ROC board {roc_b:.4f} vs C-model {roc_c:.4f} (delta {roc_b - roc_c:+.4f}, 95% {lo:+.4f}..{hi:+.4f}), "
              f"spearman {rho:.4f}, run {sec_t:.0f} s", flush=True)
    for r in rows_all:
        if r["check"] == "roc_auc_board_vs_cmodel":
            print(f"{r['status']:5} {r['artifact']:16} board={r['value']} cmodel={r['reference']} tol={r['tol']}")
    if any(r["status"] == "FAIL" for r in rows_all):
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("host", "run"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "host":
        cmd_host()
    else:
        cmd_run(sys.argv[2:] or TAGS)
