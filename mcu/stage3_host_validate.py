"""Phase 5 / stage 3a: ROC-AUC of the DEPLOYED Cube.AI artifact (x86 C-model).

`stedgeai validate --mode host` compiles the generated network C files and links
them with the reference C-kernels of the network runtime. Whole PU test (and, for
int8, PU val) is pushed through that C-model; outputs are scored on the host.

Why it is needed: in X-CUBE-AI 10.2 every Conv with dilation > 1 falls back to fp32
in an int8 model (supported_ops_*.html: "dilation factors different from 1 are not
supported for int8 model"), so the deployed int8 network is MIXED int8/fp32 and is
not the ORT int8 file of stage 1. The table uses the number from this script.

Scorer protocol: fp32 -> trained scorer (best.pt). int8 -> Mahalanobis refit on the
deployed C-model's normal-val errors (repo PTQ protocol, quantization.py:229).

Checks -> cubeai_host_validate.csv:
  * fp32: max rel diff C-model vs ORT fp32 (stage 1 file) on all test windows < 1e-4
  * fp32: |ROC(C-model) - ROC(ORT fp32)| <= 1e-4
  * int8 (mixed): ROC-AUC on PU test, plus rel diff vs ORT int8 as INFO
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import roc_auc_score

from p5common import (CKPT_DIR, CUBEAI_OPTS, MCU, STEDGEAI, append_csv, build_windows, cubeai_env, host_tag, now,
                      verdict)
from industrial_ad.experiments import load_detector_from_run


def load(code: str, seed: int):
    """Same as stage1_export.load (not imported: that module runs stage 1 on import)."""
    import tempfile
    tmp = Path(tempfile.mkdtemp()) / f"{code}_s{seed}"
    (tmp / "checkpoints").mkdir(parents=True)
    shutil.copy2(CKPT_DIR / f"{code}_s{seed}.config.json", tmp / "config.json")
    shutil.copy2(CKPT_DIR / f"{code}_s{seed}.pt", tmp / "checkpoints" / "best.pt")
    return load_detector_from_run(tmp)


def errors(det, out_cf: np.ndarray, y_tc: np.ndarray) -> torch.Tensor:
    pred = torch.from_numpy(np.ascontiguousarray(out_cf.transpose(0, 2, 1)))
    with torch.no_grad():
        return det.error_reducer(pred, torch.from_numpy(y_tc))

OUT = MCU / "cubeai_host_validate.csv"
TMP = MCU / "_tmp_validate"           # large npy files, deleted at the end
VAL = MCU / "cubeai" / "validate_host"
TOL_REL, TOL_ROC = 1e-4, 1e-4
ver = " ".join(subprocess.run([STEDGEAI, "--version"], capture_output=True, text=True).stdout.split())
selection = json.loads((CKPT_DIR / "selection.json").read_text())
VAL.mkdir(parents=True, exist_ok=True)


def c_model_outputs(code: str, art: str, x_cf: np.ndarray, split: str) -> np.ndarray:
    tag = f"{code.lower()}_{art}"
    d = TMP / f"{tag}_{split}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    np.save(d / "in.npy", x_cf)
    cmd = [STEDGEAI, "validate", "--model", str(MCU / "models" / f"{code}_{art}.onnx"), *CUBEAI_OPTS[art],
           "--name", tag, "--mode", "host", "-vi", str(d / "in.npy"), "--no-exec-model",
           "--workspace", str(d / "ws"), "--output", str(d / "out")]
    res = subprocess.run(cmd, capture_output=True, text=True, env=cubeai_env())
    rep = d / "out" / f"{tag}_validate_report.txt"
    if rep.exists():
        shutil.copy2(rep, VAL / f"validate_host_{code}_{art}_{split}.txt")
    if res.returncode != 0:
        (VAL / f"validate_host_{code}_{art}_{split}.log").write_text(res.stdout + res.stderr, encoding="utf-8")
        raise RuntimeError(f"stedgeai validate failed for {tag} {split} (rc={res.returncode})")
    out = np.load(d / "out" / f"{tag}_val_c_outputs_1.npy")
    assert out.dtype == np.float32, f"{tag}: C-model output dtype {out.dtype} (float I/O expected)"
    return out.reshape(len(x_cf), *x_cf.shape[1:2], -1) if out.ndim != 3 else out


def ort_out(path, x_cf):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    n = sess.get_inputs()[0].name
    return np.concatenate([sess.run(None, {n: x_cf[i:i + 1]})[0] for i in range(len(x_cf))])


rows = []
for code, seed in selection.items():
    det, cfg = load(code, seed)
    x, y, lab = build_windows(cfg, "test")
    x_cf = np.ascontiguousarray(x.transpose(0, 2, 1))
    xv, yv, labv = build_windows(cfg, "val")
    xv_cf = np.ascontiguousarray(xv.transpose(0, 2, 1))
    trained = {k: v.clone() for k, v in det.score_estimator.state_dict().items()}
    for art in ("fp32", "int8"):
        base = dict(date=now(), host=host_tag(), stedgeai=ver, options=" ".join(CUBEAI_OPTS[art]), code=code,
                    seed=seed, artifact=f"{code}_{art}.onnx", deployed="Cube.AI C-model (x86 host, reference kernels)")
        oc = c_model_outputs(code, art, x_cf, "test")
        assert oc.shape[0] == len(x_cf) and oc.shape[1] == 27, (code, art, oc.shape)
        oo = ort_out(MCU / "models" / f"{code}_{art}.onnx", x_cf)
        rel = float(np.max(np.abs(oc - oo) / (np.abs(oo) + 1.0)))
        if art == "fp32":
            det.score_estimator.load_state_dict(trained)
            note_sc = "trained scorer"
        else:
            ev = errors(det, c_model_outputs(code, art, xv_cf, "val"), yv)
            det.score_estimator.fit(ev[torch.from_numpy(labv == 0)])
            note_sc = "scorer refit on normal val outputs of this C-model"
        with torch.no_grad():
            roc_c = float(roc_auc_score(lab, det.score_estimator(errors(det, oc, y)).numpy()))
            roc_o = float(roc_auc_score(lab, det.score_estimator(errors(det, oo, y)).numpy()))
        n = f"N={len(lab)} anom={int(lab.sum())}; {note_sc}"
        if art == "fp32":
            rows.append({**base, "check": "max_rel_diff_cmodel_vs_ort", "value": f"{rel:.3e}", "reference": 0,
                         "tol": TOL_REL, "status": verdict(rel < TOL_REL), "note": "all test windows"})
            rows.append({**base, "check": "roc_auc_pu_test", "value": round(roc_c, 6), "reference": round(roc_o, 6),
                         "tol": TOL_ROC, "status": verdict(abs(roc_c - roc_o) <= TOL_ROC), "note": n})
        else:
            rows.append({**base, "check": "max_rel_diff_cmodel_vs_ort_int8", "value": f"{rel:.3e}", "reference": "",
                         "tol": "", "status": "INFO",
                         "note": "C-model is mixed int8/fp32 (dilated convs fp32), ORT file is all-int8"})
            rows.append({**base, "check": "roc_auc_pu_test", "value": round(roc_c, 6), "reference": round(roc_o, 6),
                         "tol": "", "status": "DEPLOYED",
                         "note": n + "; reference = ORT all-int8 file with the same refit scorer"})
        np.savez(MCU / "models" / f"{code}_{art}_cmodel_scorer.npz",
                 **{k: v.numpy() for k, v in det.score_estimator.state_dict().items()}, note=np.array(note_sc))
        print(f"{code} {art}: rel={rel:.2e} roc C-model={roc_c:.6f} ORT={roc_o:.6f}", flush=True)
    det.score_estimator.load_state_dict(trained)

append_csv(OUT, rows, fresh=True)
shutil.rmtree(TMP, ignore_errors=True)
for r in rows:
    print(f"{r['status']:8} {r['code']} {r['artifact']:16} {r['check']}: {r['value']} ref={r['reference']}")
if any(r["status"] == "FAIL" for r in rows):
    raise SystemExit(1)
