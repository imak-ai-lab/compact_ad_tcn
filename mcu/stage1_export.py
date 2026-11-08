"""Phase 5 / stage 1: PyTorch -> ONNX fp32 and ONNX Runtime static int8 (QDQ).

Deployed graph = the predictor network only, channels-first I/O:
    input  [1, 27, 120]  (B x C x T, as Conv1d sees it on the MCU)
    output [1, 27, T_out] (T_out = 120 for reconstruction, 240 for TL082)
Error reduction (mean |.| over time) and the Mahalanobis scorer stay on the host
(27-d, not part of the network). Scorer parameters used for every artifact are
saved next to it (models/<code>_<art>_scorer.npz) so device outputs can be scored
the same way later.

Checks -> export_check.csv:
  * fp32: max relative diff |ONNX - PyTorch| / (|PyTorch| + 1) over 100 test windows
    < 1e-4 (outputs reach +-290, absolute 1e-5 is below
    float32 resolution there; PyTorch batch-100 vs batch-1 differs by 2e-5 itself).
    Absolute max diff is logged as INFO.
  * fp32 ONNX ROC-AUC on PU test (ORT) vs PyTorch ROC-AUC, tol 1e-4 (trained scorer).
  * int8 (QDQ s8/s8, per-channel weights), calibration on NORMAL VALIDATION windows.
    As in the repo PTQ protocol (industrial_ad/quantization.py:229) the Mahalanobis
    scorer is refit on the quantized model's normal-val errors. Calibration method
    is chosen by VALIDATION ROC-AUC among CANDIDATES (all logged); test ROC-AUC is
    reported for the chosen file only as the deployed number, and for the others as INFO.
"""
from __future__ import annotations

import collections
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxslim
import torch
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process
from sklearn.metrics import roc_auc_score

from p5common import CKPT_DIR, MCU, append_csv, build_windows, host_tag, now, verdict
from industrial_ad.experiments import load_detector_from_run

OUT = MCU / "export_check.csv"
MODELS_DIR = MCU / "models"
OPSET = 17
N_CHECK, TOL_REL, TOL_ROC_FP32 = 100, 1e-4, 1e-4
N_CALIB = 2000          # normal val windows used for calibration (all if fewer)
SEED = 0
CANDIDATES = [
    ("minmax", CalibrationMethod.MinMax, {}),
    ("pct99.99", CalibrationMethod.Percentile, {"CalibPercentile": 99.99}),
    ("pct99.9", CalibrationMethod.Percentile, {"CalibPercentile": 99.9}),
    ("entropy", CalibrationMethod.Entropy, {}),
]

torch.set_num_threads(4)
MODELS_DIR.mkdir(exist_ok=True)
selection = json.loads((CKPT_DIR / "selection.json").read_text())


class ChannelsFirst(torch.nn.Module):
    """[B,C,T] -> net -> [B,C,T_out]; bypasses the (B,T,C) transposes of the repo models."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


def load(code: str, seed: int):
    """Reload via the repo loader from a temp run-dir layout built from mcu/checkpoints."""
    tmp = Path(tempfile.mkdtemp()) / f"{code}_s{seed}"
    (tmp / "checkpoints").mkdir(parents=True)
    shutil.copy2(CKPT_DIR / f"{code}_s{seed}.config.json", tmp / "config.json")
    shutil.copy2(CKPT_DIR / f"{code}_s{seed}.pt", tmp / "checkpoints" / "best.pt")
    return load_detector_from_run(tmp)


def errors(det, out_cf: np.ndarray, y_tc: np.ndarray) -> torch.Tensor:
    pred = torch.from_numpy(np.ascontiguousarray(out_cf.transpose(0, 2, 1)))
    with torch.no_grad():
        return det.error_reducer(pred, torch.from_numpy(y_tc))


def ort_run(path, x_cf: np.ndarray) -> np.ndarray:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return np.concatenate([sess.run(None, {name: x_cf[i:i + 1]})[0] for i in range(len(x_cf))])


def save_scorer(det, path: Path, note: str) -> None:
    sd = {k: v.cpu().numpy() for k, v in det.score_estimator.state_dict().items()}
    np.savez(path, note=np.array(note), **sd)


class Calib(CalibrationDataReader):
    def __init__(self, x_cf):
        self.it = iter([{"input": x_cf[i:i + 1]} for i in range(len(x_cf))])

    def get_next(self):
        return next(self.it, None)


rows = []
for code, seed in selection.items():
    det, cfg = load(code, seed)
    net = ChannelsFirst(det.model).eval()
    x, y, lab = build_windows(cfg, "test")
    x_cf = np.ascontiguousarray(x.transpose(0, 2, 1))
    xv, yv, labv = build_windows(cfg, "val")
    xv_cf = np.ascontiguousarray(xv.transpose(0, 2, 1))
    base = dict(date=now(), host=host_tag(), code=code, seed=seed, opset=OPSET,
                ort=ort.__version__, torch=torch.__version__)
    trained_scorer = {k: v.clone() for k, v in det.score_estimator.state_dict().items()}

    # ---- fp32 export (static graph via onnxslim; checks run on the slimmed file)
    fp32 = MODELS_DIR / f"{code}_fp32.onnx"
    torch.onnx.export(net, torch.from_numpy(x_cf[:1]), str(fp32), input_names=["input"], output_names=["output"],
                      opset_version=OPSET, do_constant_folding=True, dynamo=False)
    slim = onnxslim.slim(onnx.load(str(fp32)))
    onnx.save(slim, str(fp32))
    onnx.checker.check_model(onnx.load(str(fp32)))
    ops = dict(collections.Counter(n.op_type for n in slim.graph.node))
    idx = np.sort(np.random.default_rng(SEED).choice(len(x_cf), size=min(N_CHECK, len(x_cf)), replace=False))
    with torch.no_grad():   # batch-1 reference, same as ORT and the MCU
        ref = np.concatenate([net(torch.from_numpy(x_cf[i:i + 1])).numpy() for i in idx])
    got = ort_run(fp32, x_cf[idx])
    absd = float(np.max(np.abs(got - ref)))
    reld = float(np.max(np.abs(got - ref) / (np.abs(ref) + 1.0)))
    note = (f"{len(idx)} random test windows (rng {SEED}), batch 1; |ref|max={np.abs(ref).max():.1f}; "
            f"out {tuple(got.shape[1:])}; onnxslim {onnxslim.__version__}; ops {ops}")
    rows.append({**base, "artifact": fp32.name, "calib": "", "check": "max_rel_diff_onnx_vs_torch",
                 "value": f"{reld:.3e}", "reference": 0, "tol": TOL_REL, "status": verdict(reld < TOL_REL), "note": note})
    rows.append({**base, "artifact": fp32.name, "calib": "", "check": "max_abs_diff_onnx_vs_torch",
                 "value": f"{absd:.3e}", "reference": 0, "tol": "", "status": "INFO", "note": "absolute, info only"})

    # ---- ROC-AUC: torch vs fp32 onnx (full PU test, trained scorer)
    with torch.no_grad():
        out_t = np.concatenate([net(torch.from_numpy(x_cf[i:i + 512])).numpy() for i in range(0, len(x_cf), 512)])
        roc_t = float(roc_auc_score(lab, det.score_estimator(errors(det, out_t, y)).numpy()))
        out_o = ort_run(fp32, x_cf)
        roc_o = float(roc_auc_score(lab, det.score_estimator(errors(det, out_o, y)).numpy()))
    save_scorer(det, MODELS_DIR / f"{code}_fp32_scorer.npz", "trained scorer from best.pt")
    rows.append({**base, "artifact": "torch", "calib": "", "check": "roc_auc_pu_test", "value": round(roc_t, 6),
                 "reference": "", "tol": "", "status": "INFO", "note": f"N={len(lab)} anom={int(lab.sum())}"})
    rows.append({**base, "artifact": fp32.name, "calib": "", "check": "roc_auc_pu_test", "value": round(roc_o, 6),
                 "reference": round(roc_t, 6), "tol": TOL_ROC_FP32,
                 "status": verdict(abs(roc_o - roc_t) <= TOL_ROC_FP32), "note": "ORT CPU, trained scorer"})

    # ---- int8 candidates: calibrate on normal val, refit scorer on normal val, select by val ROC
    xn = xv_cf[labv == 0]
    if len(xn) > N_CALIB:
        xn = xn[np.sort(np.random.default_rng(SEED).choice(len(xn), N_CALIB, replace=False))]
    pre = MODELS_DIR / f"{code}_fp32_pre.onnx"
    quant_pre_process(str(fp32), str(pre))
    cands = []
    for name, meth, extra in CANDIDATES:
        q = MODELS_DIR / f"{code}_int8_{name}.onnx"
        quantize_static(str(pre), str(q), Calib(xn), quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
                        weight_type=QuantType.QInt8, per_channel=True, calibrate_method=meth, extra_options=extra)
        ev = errors(det, ort_run(q, xv_cf), yv)
        det.score_estimator.fit(ev[torch.from_numpy(labv == 0)])
        with torch.no_grad():
            roc_v = float(roc_auc_score(labv, det.score_estimator(ev).numpy()))
            roc_q = float(roc_auc_score(lab, det.score_estimator(errors(det, ort_run(q, x_cf), y)).numpy()))
        sc = {k: v.clone() for k, v in det.score_estimator.state_dict().items()}
        cands.append((roc_v, name, q, roc_q, sc))
    pre.unlink()
    best = max(cands, key=lambda c: c[0])
    for roc_v, name, q, roc_q, sc in cands:
        chosen = name == best[1]
        tag = f"calib {len(xn)} normal val windows; scorer refit on normal val (repo PTQ protocol)"
        rows.append({**base, "artifact": q.name, "calib": name, "check": "roc_auc_pu_val", "value": round(roc_v, 6),
                     "reference": "", "tol": "", "status": "SELECTED" if chosen else "INFO", "note": tag})
        rows.append({**base, "artifact": f"{code}_int8.onnx" if chosen else q.name, "calib": name,
                     "check": "roc_auc_pu_test", "value": round(roc_q, 6), "reference": round(roc_t, 6), "tol": "",
                     "status": "DEPLOYED" if chosen else "INFO", "note": f"delta vs torch fp32 = {roc_q - roc_t:+.4f}"})
    int8 = MODELS_DIR / f"{code}_int8.onnx"
    shutil.copy2(best[2], int8)
    det.score_estimator.load_state_dict(best[4])
    save_scorer(det, MODELS_DIR / f"{code}_int8_scorer.npz", f"refit on normal val, calib={best[1]}")
    det.score_estimator.load_state_dict(trained_scorer)
    mdq = float(np.max(np.abs(ort_run(int8, x_cf[idx]) - ref) / (np.abs(ref) + 1.0)))
    rows.append({**base, "artifact": int8.name, "calib": best[1], "check": "max_rel_diff_int8_vs_torch",
                 "value": f"{mdq:.3e}", "reference": "", "tol": "", "status": "INFO",
                 "note": "same 100 windows; quantisation error, not a pass gate"})

append_csv(OUT, rows, fresh=True)
for r in rows:
    print(f"{r['status']:8} {r['code']} {r['artifact']:24} {r['calib']:9} {r['check']}: {r['value']}  ref={r['reference']}")
if any(r["status"] == "FAIL" for r in rows):
    raise SystemExit(1)
