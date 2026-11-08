"""Phase 5 / stage 2: ST Edge AI Core `analyze` for the six ONNX files (no board).

Usage:
    .venv/Scripts/python stage2_analyze.py [C:/ST/stedgeai-10.2.0/Utilities/windows/stedgeai.exe]

Per model: `stedgeai analyze --target stm32h7` (Cortex-M7 of the H755). Keeps the
console log (cubeai/analyze_<code>_<art>.txt), the analyze report and the
c_info JSON, and writes cubeai_summary.csv.

Conventions (made explicit, because Cube.AI and the paper count differently):
  * MACC: the paper (count_macs.py) counts Conv1d/Linear only; norms, activations,
    adds are excluded. Cube.AI MACC counts every node. The 5 % check compares the
    Conv2D + Dense/Gemm/MatMul part of Cube.AI MACC with macs_headline.csv; the full
    MACC and a per-type breakdown are logged too.
  * params: GroupNorm(1 group) exports as Reshape -> InstanceNormalization(1 ch,
    scale=1, bias=0) -> Reshape -> Mul/Add. Cube.AI counts the two 1-element
    constants of every InstanceNormalization node; they are subtracted
    (2 x n_InstanceNormalization) and the check is exact against the paper.
    For int8, `params #` is checked the same way.
Any field that cannot be read is empty and the row FAILs; nothing is guessed.
"""
from __future__ import annotations

import collections
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from p5common import CUBEAI_OPTS, MCU, MODELS, STEDGEAI, append_csv, cubeai_env, host_tag, now, verdict

OUT = MCU / "cubeai_summary.csv"
CUBE = MCU / "cubeai"
TOL_MACC = 0.05
DEFAULT_EXE = STEDGEAI
MAC_TYPES = {"Conv2D", "Dense", "Gemm", "MatMul", "Conv2dPool"}

ARM_GCC_BIN = os.environ.get("ARM_GCC_BIN", r"C:\ST\arm-gnu-14.2\bin")   # Cube.AI needs it for runtime code size
assert (Path(ARM_GCC_BIN) / "arm-none-eabi-gcc.exe").exists(), f"arm-none-eabi-gcc not in {ARM_GCC_BIN}"
os.environ["PATH"] = ARM_GCC_BIN + os.pathsep + os.environ["PATH"]
exe = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("STEDGEAI", DEFAULT_EXE)
assert Path(exe).exists(), f"stedgeai not found: {exe!r}"
ver = " ".join(subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.split())
CUBE.mkdir(exist_ok=True)
only = set(os.environ.get("P5_ONLY", "").split(",")) - {""}


def grab(pattern, text):
    m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    return int(m.group(1).replace(",", "")) if m else None


rows = []
for code, (_, name, ref) in MODELS.items():
    for art in ("fp32", "int8"):
        if only and f"{code}_{art}" not in only:
            continue
        model = MCU / "models" / f"{code}_{art}.onnx"
        tag = f"{code.lower()}_{art}"
        ws, out = CUBE / f"ws_{tag}", CUBE / f"out_{tag}"
        for d in (ws, out):
            shutil.rmtree(d, ignore_errors=True)
        cmd = [exe, "analyze", "--model", str(model), *CUBEAI_OPTS[art], "--name", tag,
               "--workspace", str(ws), "--output", str(out)]
        res = subprocess.run(cmd, capture_output=True, text=True, env=cubeai_env())
        log = re.sub(r"PASS:\s+\d+%\|[^\r\n]*?(?=\s{2,}|\n|$)", "", res.stdout + "\n" + res.stderr)
        (CUBE / f"analyze_{code}_{art}.txt").write_text(" ".join(cmd) + "\n\n" + log, encoding="utf-8")
        rep = out / f"{tag}_analyze_report.txt"
        cinfo = out / f"{tag}_c_info.json"
        txt = rep.read_text(encoding="utf-8", errors="replace") if rep.exists() else log
        if rep.exists():
            shutil.copy2(rep, CUBE / f"analyze_report_{code}_{art}.txt")
        params = grab(r"^\s*params #\s*:\s*([\d,]+)", txt)
        macc_total = grab(r"^\s*macc\s*:\s*([\d,]+)", txt)
        w_ro = grab(r"^\s*weights \(ro\)\s*:\s*([\d,]+)\s*B", txt)
        act = grab(r"^\s*activations \(rw\)\s*:\s*([\d,]+)\s*B", txt)
        by_type, n_by_type, kern_flash, kern_ram = collections.Counter(), collections.Counter(), None, None
        if cinfo.exists():
            shutil.copy2(cinfo, CUBE / f"c_info_{code}_{art}.json")
            j = json.loads(cinfo.read_text(encoding="utf-8"))
            for nd in j["graphs"][0]["nodes"]:
                by_type[nd.get("description") or "?"] += int(nd.get("macc") or 0)
                n_by_type[nd.get("description") or "?"] += 1
            kern_flash = j["memory_footprint"].get("kernel_flash")
            kern_ram = j["memory_footprint"].get("kernel_ram")
        macc_mac = sum(v for k, v in by_type.items() if k in MAC_TYPES) if by_type else None
        n_in = n_by_type.get("InstanceNormalization", 0)
        params_net = None if params is None else params - 2 * n_in
        ref_macc = ref["mmac"] * 1e6
        dm = None if macc_mac is None else abs(macc_mac - ref_macc) / ref_macc
        base = dict(date=now(), host=host_tag(), stedgeai=ver, options=" ".join(CUBEAI_OPTS[art]), code=code, artifact=model.name,
                    returncode=res.returncode, params_cubeai=params, n_instancenorm=n_in, params_net=params_net,
                    macc_total=macc_total, macc_conv_dense=macc_mac,
                    macc_by_type=json.dumps(dict(by_type.most_common())),
                    weights_ro_B=w_ro, activations_rw_B=act,
                    flash_weights_KiB=None if w_ro is None else round(w_ro / 1024, 2),
                    ram_activations_KiB=None if act is None else round(act / 1024, 2),
                    runtime_flash_B=kern_flash, runtime_ram_B=kern_ram)
        rows.append({**base, "check": "params_net_vs_paper", "reference": ref["params"],
                     "rel_diff": None if params_net is None else abs(params_net - ref["params"]) / ref["params"],
                     "tol": 0, "status": verdict(params_net == ref["params"])})
        rows.append({**base, "check": "macc_conv_dense_vs_macs_headline", "reference": int(ref_macc),
                     "rel_diff": None if dm is None else round(dm, 5), "tol": TOL_MACC,
                     "status": verdict(dm is not None and dm <= TOL_MACC)})
        print(f"{code} {art}: rc={res.returncode} params={params} (-2x{n_in}={params_net}) macc={macc_total} "
              f"conv/dense={macc_mac} w={w_ro} act={act} rt_flash={kern_flash}", flush=True)

append_csv(OUT, rows, fresh=not only)
for r in rows:
    print(f"{r['status']} {r['code']} {r['artifact']} {r['check']}: rel_diff={r['rel_diff']}")
if any(r["status"] == "FAIL" for r in rows):
    raise SystemExit(1)
