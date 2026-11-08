"""Phase 5 / stage 3a diagnostic: fp32 numerical floor (INFO only, does not change any verdict).

Why: cubeai_host_validate.csv has TL082 fp32 max_rel_diff_cmodel_vs_ort = 1.793e-04 > tol 1e-4 (FAIL),
while ROC-AUC is identical to 6 decimals. Stage 1 measured ORT vs PyTorch on only 100 windows, so the
max over 100 windows is not comparable with a max over the whole test split. This script puts the three
fp32 implementations side by side on ALL test windows, with the same metric |a-b| / (|b|+1):

    C-model vs ORT, C-model vs PyTorch, ORT vs PyTorch   (PyTorch: batch 1, as ORT and the MCU)

and reports max, quantiles, and how many elements / windows exceed 1e-4.
The FAIL row in cubeai_host_validate.csv stays as it is; this file only explains it.

Usage: .venv/Scripts/python stage3_diag_fp32.py [CODE ...]     (default: TL082)
"""
from __future__ import annotations

import shutil
import subprocess
import sys

import numpy as np
import onnxruntime as ort
import torch

from p5common import CKPT_DIR, CUBEAI_OPTS, MCU, STEDGEAI, append_csv, build_windows, cubeai_env, host_tag, now

import json
import tempfile
from pathlib import Path
from industrial_ad.experiments import load_detector_from_run

OUT = MCU / "cubeai_host_validate_diag.csv"
TMP = MCU / "_tmp_diag"
TOL_REL = 1e-4                       # the stage 3 tolerance, used here only to count exceedances
ver = " ".join(subprocess.run([STEDGEAI, "--version"], capture_output=True, text=True).stdout.split())
selection = json.loads((CKPT_DIR / "selection.json").read_text())
codes = sys.argv[1:] or ["TL082"]


class ChannelsFirst(torch.nn.Module):          # same as stage1_export.ChannelsFirst (that module runs on import)
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


def load(code: str, seed: int):
    tmp = Path(tempfile.mkdtemp()) / f"{code}_s{seed}"
    (tmp / "checkpoints").mkdir(parents=True)
    shutil.copy2(CKPT_DIR / f"{code}_s{seed}.config.json", tmp / "config.json")
    shutil.copy2(CKPT_DIR / f"{code}_s{seed}.pt", tmp / "checkpoints" / "best.pt")
    return load_detector_from_run(tmp)


def c_model(code: str, x_cf: np.ndarray) -> np.ndarray:
    tag = f"{code.lower()}_fp32"
    d = TMP / tag
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    np.save(d / "in.npy", x_cf)
    cmd = [STEDGEAI, "validate", "--model", str(MCU / "models" / f"{code}_fp32.onnx"), *CUBEAI_OPTS["fp32"],
           "--name", tag, "--mode", "host", "-vi", str(d / "in.npy"), "--no-exec-model",
           "--workspace", str(d / "ws"), "--output", str(d / "out")]
    res = subprocess.run(cmd, capture_output=True, text=True, env=cubeai_env())
    if res.returncode != 0:
        raise RuntimeError(f"stedgeai validate failed for {tag}: {res.stdout[-2000:]}{res.stderr[-2000:]}")
    out = np.load(d / "out" / f"{tag}_val_c_outputs_1.npy")
    return out.reshape(len(x_cf), x_cf.shape[1], -1) if out.ndim != 3 else out


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    r = np.abs(a - b) / (np.abs(b) + 1.0)
    win = r.reshape(len(r), -1).max(axis=1)
    i = np.unravel_index(int(np.argmax(r)), r.shape)
    return dict(max=float(r.max()), p99_99=float(np.quantile(r, 0.9999)), p99_9=float(np.quantile(r, 0.999)),
                median=float(np.median(r)), n_elem=int(r.size), n_elem_gt_tol=int((r > TOL_REL).sum()),
                n_win=int(len(win)), n_win_gt_tol=int((win > TOL_REL).sum()),
                max_abs=float(np.abs(a - b).max()), argmax=f"win={i[0]} ch={i[1]} t={i[2]} |b|={abs(float(b[i])):.3g}")


rows = []
for code in codes:
    seed = selection[code]
    det, cfg = load(code, seed)
    net = ChannelsFirst(det.model).eval()
    x, _, _ = build_windows(cfg, "test")
    x_cf = np.ascontiguousarray(x.transpose(0, 2, 1))
    sess = ort.InferenceSession(str(MCU / "models" / f"{code}_fp32.onnx"), providers=["CPUExecutionProvider"])
    n_in = sess.get_inputs()[0].name
    o_ort = np.concatenate([sess.run(None, {n_in: x_cf[i:i + 1]})[0] for i in range(len(x_cf))])
    with torch.no_grad():
        o_pt = np.concatenate([net(torch.from_numpy(x_cf[i:i + 1])).numpy() for i in range(len(x_cf))])
    print(f"{code}: ORT and PyTorch done, running C-model ({len(x_cf)} windows)", flush=True)
    o_c = c_model(code, x_cf)
    base = dict(date=now(), host=host_tag(), stedgeai=ver, ort=ort.__version__, torch=torch.__version__,
                code=code, seed=seed, split="test", metric="|a-b|/(|b|+1)", tol_counted=TOL_REL)
    for pair, a, b in (("cmodel_vs_ort", o_c, o_ort), ("cmodel_vs_torch", o_c, o_pt), ("ort_vs_torch", o_ort, o_pt)):
        s = stats(a, b)
        rows.append({**base, "pair": pair, **s, "status": "INFO"})
        print(f"{code} {pair:16} max={s['max']:.3e} p99.99={s['p99_99']:.3e} p99.9={s['p99_9']:.3e} "
              f"med={s['median']:.2e} elem>tol={s['n_elem_gt_tol']}/{s['n_elem']} "
              f"win>tol={s['n_win_gt_tol']}/{s['n_win']} maxabs={s['max_abs']:.3e} at {s['argmax']}", flush=True)

append_csv(OUT, rows, fresh=True)
shutil.rmtree(TMP, ignore_errors=True)
