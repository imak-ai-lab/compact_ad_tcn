"""Phase 5 / stage 3c: Cube.AI on-target validation on the NUCLEO-H755ZI-Q (Cortex-M7, 400 MHz).

Usage (from mcu/):
    .venv/Scripts/python stage3_target_validate.py probe            # ST-LINK, COM port, option bytes (read only)
    .venv/Scripts/python stage3_target_validate.py run [tag ...]    # flash + validate; default: all six

`run` refuses to start unless option byte BCM4 = 0 (Cortex-M4 boot disabled; also
needed because TR138 fp32 spans flash bank 2). Changing BCM4 is NOT done here: it is a persistent
board setting and is done once, by explicit decision, with
    STM32_Programmer_CLI -c port=SWD mode=UR -ob BCM4=0

Per artifact (`run`):
  1. flash firmware/validation/build/<tag>/validation_<tag>.bin at 0x08000000 (verify, hardware reset);
  2. fixed subset of PU test windows: 128 normal + 128 anomalous, rng 2026, same per code for fp32 and int8
     (indices -> data/stage3_target_subset_<code>.npz);
  3. `stedgeai validate --mode target -d serial:<COM>:460800` (921600 lost data on long runs, 2026-11-05) with the stage-2/3a options -> report;
  4. the same subset through the x86 C-model (`--mode host`) and ORT, for the output checks.
Rows -> cubeai_validate.csv:
  latency  (device report: duration mean/min/max/std ms, cycles if reported, CPU clock)   MEASURED
  fp32:  max rel diff target vs ORT fp32, tol 1e-4                                      PASS/FAIL
  fp32/int8: max rel diff target vs host C-model, p99.99 of the same                    INFO
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import serial.tools.list_ports as list_ports

from p5common import (CKPT_DIR, CUBEAI_OPTS, CUBEPROG, MCU, MODELS, ST_VID, STEDGEAI, append_csv, build_windows,
                      cubeai_env, host_tag, now, verdict)

import json

TRIAL = bool(os.environ.get("P5_TRIAL"))       # trial runs go to a separate CSV, never to the official one
OUT = MCU / ("cubeai_validate_trial.csv" if TRIAL else "cubeai_validate.csv")
FW = MCU / "firmware" / "validation" / "build"
WORK = MCU / "_tmp_target"
REP = MCU / "cubeai" / "validate_target"
BAUD = int(os.environ.get("P5_BAUD", "460800"))
N_PER_CLASS, SUBSET_SEED = int(os.environ.get("P5_NPC", "128")), 2026
TOL_REL = 1e-4
selection = json.loads((CKPT_DIR / "selection.json").read_text())


def sh(cmd, check=True, **kw):
    res = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    out = res.stdout + res.stderr
    if check and res.returncode != 0:
        raise RuntimeError(f"rc={res.returncode}: {' '.join(map(str, cmd))[:200]}\n{out[-3000:]}")
    return out


SWD = ["-c", "port=SWD", "freq=4000", "mode=UR"]   # 8 MHz default gave DEV_USB_COMM_ERR / lost probe (2026-11-05)


def prog(*args, tries: int = 4) -> str:
    """STM32_Programmer_CLI with retries: the ST-LINK USB link drops now and then on this setup
    ("No debug probe detected", DEV_USB_COMM_ERR). Writes are verified (-v), so a retry cannot leave a
    silently wrong image; every failed attempt is printed."""
    for k in range(1, tries + 1):
        try:
            return sh([CUBEPROG, *SWD, *args])
        except RuntimeError as e:
            if k == tries:
                raise
            print(f"  CubeProgrammer attempt {k} failed ({str(e).splitlines()[-1][:80]}); retry in 8 s", flush=True)
            time.sleep(8.0)


def stlink_com() -> str:
    ports = [p.device for p in list_ports.comports() if p.vid == ST_VID]
    if len(ports) != 1:
        raise SystemExit(f"expected exactly one ST-LINK VCP (VID 0x{ST_VID:04X}), found {ports}")
    return ports[0]


def option_bytes() -> dict:
    """Decode FLASH_OPTSR_CUR (0x5200201C) directly. CubeProgrammer v2.23 `-ob displ` shows the H755 with the
    single-core layout and no BCM4/BCM7 (Data_Base/STM32_Prog_DB_0x450.xml), so it cannot be used here.
    Bit positions: cmsis-device-h7 v1.10.7 stm32h755xx.h."""
    out = prog("-r32", "0x5200201C", "4")
    m = re.search(r"^0x5200201C\s*:\s*([0-9A-Fa-f]{8})", out, re.M)
    if not m:
        raise SystemExit("could not read FLASH_OPTSR_CUR")
    v = int(m.group(1), 16)
    return {"OPTSR_CUR": f"0x{v:08X}", "BCM4": (v >> 22) & 1, "BCM7": (v >> 23) & 1, "RDP": (v >> 8) & 0xFF,
            "SECURITY": (v >> 21) & 1, "SWAP_BANK": (v >> 31) & 1, "_raw": out}


def subset(code: str):
    seed = selection[code]
    cfg = json.loads((CKPT_DIR / f"{code}_s{seed}.config.json").read_text())
    x, _, lab = build_windows(cfg, "test")
    rng = np.random.default_rng(SUBSET_SEED)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(lab == c), N_PER_CLASS, replace=False) for c in (0, 1)]))
    np.savez(MCU / "data" / f"stage3_target_subset_{code}.npz", idx=idx, label=lab[idx], seed=SUBSET_SEED)
    return np.ascontiguousarray(x[idx].transpose(0, 2, 1)), seed


def validate(tag: str, art: str, onnx: Path, x_cf: np.ndarray, mode: str, com: str | None) -> tuple[np.ndarray, str]:
    d = WORK / f"{tag}_{mode}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    np.save(d / "in.npy", x_cf)
    cmd = [STEDGEAI, "validate", "--model", onnx, *CUBEAI_OPTS[art], "--name", "network", "--mode", mode,
           "-vi", d / "in.npy", "--workspace", d / "ws", "--output", d / "out"]
    if mode == "target":
        cmd += ["-d", f"serial:{com}:{BAUD}"]
    else:
        cmd += ["--no-exec-model"]
    log = sh(cmd, check=False, env=cubeai_env())
    rep = next(iter((d / "out").glob("*_validate_report.txt")), None)
    REP.mkdir(parents=True, exist_ok=True)
    (REP / f"validate_{mode}_{tag}.log").write_text(log, encoding="utf-8")
    if rep is None:
        raise RuntimeError(f"{tag} {mode}: no validate report, see {REP / f'validate_{mode}_{tag}.log'}")
    shutil.copy2(rep, REP / f"validate_{mode}_{tag}.txt")
    outs = sorted((d / "out").glob("*_val_c_outputs_1.npy"))
    if not outs:
        raise RuntimeError(f"{tag} {mode}: no C outputs npy in {d / 'out'}")
    y = np.load(outs[0])
    return y.reshape(len(x_cf), x_cf.shape[1], -1) if y.ndim != 3 else y, rep.read_text(encoding="utf-8", errors="replace")


def ort_out(onnx: Path, x_cf: np.ndarray) -> np.ndarray:
    s = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    n = s.get_inputs()[0].name
    return np.concatenate([s.run(None, {n: x_cf[i:i + 1]})[0] for i in range(len(x_cf))])


def rel(a, b):
    r = np.abs(a - b) / (np.abs(b) + 1.0)
    return float(r.max()), float(np.quantile(r, 0.9999))


def parse_timing(txt: str) -> dict:
    t = {}
    m = re.search(r"duration\s*:\s*([\d.]+)\s*ms by sample\s*\(([\d.]+)/([\d.]+)/([\d.]+)\)", txt)
    if m:
        t.update(dur_ms=float(m.group(1)), dur_min_ms=float(m.group(2)), dur_max_ms=float(m.group(3)),
                 dur_std_ms=float(m.group(4)))
    m = re.search(r"nb sample\(s\)\s*:\s*(\d+)", txt)
    t["n_samples"] = int(m.group(1)) if m else None
    m = re.search(r"CPU cycles\s*:\s*\[?([\d,]+)", txt)        # report prints "CPU cycles : [54,056,764]"
    t["cycles"] = int(m.group(1).replace(",", "")) if m else None
    m = re.search(r"cycles/MACC\s*:\s*([\d.]+)", txt)
    t["cycles_per_macc"] = float(m.group(1)) if m else None
    m = re.search(r"device\.desc\s*:\s*(.+?)\s*$", txt, re.M)
    t["device"] = m.group(1) if m else None
    m = re.search(r"device\.attrs\s*:\s*(.+?)\s*$", txt, re.M)
    t["device_attrs"] = m.group(1) if m else None
    m = re.search(r"runtime lib\s*:\s*(.+?)\s*$", txt, re.M)
    t["runtime_reported"] = m.group(1) if m else None
    return t


def cmd_probe():
    print(sh([CUBEPROG, "-l", "st-link"], check=False)[-800:])
    print("ST-LINK VCP:", [(p.device, p.description) for p in list_ports.comports() if p.vid == ST_VID])
    ob = option_bytes()
    print({k: v for k, v in ob.items() if k != "_raw"})
    (MCU / "cubeai").mkdir(exist_ok=True)
    (MCU / "cubeai" / f"option_bytes_{time.strftime('%Y%m%d_%H%M%S')}.txt").write_text(ob["_raw"], encoding="utf-8")


def cmd_run(tags):
    ob = option_bytes()
    if ob["BCM4"] != 0:
        raise SystemExit(f"BCM4 = {ob['BCM4']} (must be 0: Cortex-M4 boot disabled). Nothing flashed.")
    com = stlink_com()
    ver = " ".join(sh([STEDGEAI, "--version"]).split())
    if len(tags) == 6 and not TRIAL and OUT.exists():
        OUT.unlink()                       # full official run: start the CSV fresh, then append per model
    all_rows = []
    for tag in tags:
        rows = []
        code, art = tag.split("_")
        code = code.upper()
        onnx = MCU / "models" / f"{code}_{art}.onnx"
        binf = FW / tag / f"validation_{tag}.bin"
        x_cf, seed = subset(code)
        flash_log = prog("-w", binf, "0x08000000", "-v", "-hardRst")
        REP.mkdir(parents=True, exist_ok=True)
        (REP / f"flash_{tag}.log").write_text(flash_log, encoding="utf-8")
        time.sleep(2.0)
        # A broken serial exchange (no report / unparsable message) is a link failure, not a measurement:
        # reset the board and retry once; the attempt number goes to the CSV.
        for attempt in (1, 2):
            try:
                y_t, rep_t = validate(tag, art, onnx, x_cf, "target", com)
                break
            except RuntimeError as e:
                if attempt == 2:
                    raise
                print(f"{tag}: target attempt 1 failed ({str(e)[:120]}); reset and retry", flush=True)
                shutil.copy2(REP / f"validate_target_{tag}.log", REP / f"validate_target_{tag}_attempt1.log")
                prog("-hardRst")
                time.sleep(2.0)
        y_h, _ = validate(tag, art, onnx, x_cf, "host", None)
        y_o = ort_out(onnx, x_cf)
        tm = parse_timing(rep_t)
        base = dict(date=now(), host=host_tag(), stedgeai=ver, options=" ".join(CUBEAI_OPTS[art]), code=code,
                    seed=seed, artifact=onnx.name, firmware=binf.name,
                    board="NUCLEO-H755ZI-Q CM7 400MHz VOS1 direct-SMPS, I/D-cache on, act in AXI SRAM, CM4 off (BCM4=0)",
                    device=tm.get("device"), device_attrs=tm.get("device_attrs"),
                    runtime_reported=tm.get("runtime_reported"), baud=BAUD, target_attempt=attempt,
                    n_samples=tm.get("n_samples"), subset=f"{N_PER_CLASS}+{N_PER_CLASS} test rng{SUBSET_SEED}")
        rows.append({**base, "check": "latency_ms_mean", "value": tm.get("dur_ms"), "reference": "", "tol": "",
                     "status": "MEASURED" if tm.get("dur_ms") is not None else "FAIL",
                     "note": f"min/max/std={tm.get('dur_min_ms')}/{tm.get('dur_max_ms')}/{tm.get('dur_std_ms')} ms; "
                             f"cycles={tm.get('cycles')}; cycles/MACC={tm.get('cycles_per_macc')}"})
        mx, q = rel(y_t, y_h)
        rows.append({**base, "check": "max_rel_diff_target_vs_host_cmodel", "value": f"{mx:.3e}", "reference": "",
                     "tol": "", "status": "INFO", "note": f"p99.99={q:.3e}; same generated code, x86 reference kernels"})
        mx, q = rel(y_t, y_o)
        if art == "fp32":
            rows.append({**base, "check": "max_rel_diff_target_vs_ort", "value": f"{mx:.3e}", "reference": 0,
                         "tol": TOL_REL, "status": verdict(mx < TOL_REL), "note": f"p99.99={q:.3e}"})
        else:
            rows.append({**base, "check": "max_rel_diff_target_vs_ort_int8", "value": f"{mx:.3e}", "reference": "",
                         "tol": "", "status": "INFO", "note": f"p99.99={q:.3e}; target is mixed int8/fp32, ORT file all-int8"})
        print(f"{tag}: {tm.get('dur_ms')} ms, device={tm.get('device')}", flush=True)
        append_csv(OUT, rows)
        all_rows += rows
    shutil.rmtree(WORK, ignore_errors=True)
    rows = all_rows
    for r in rows:
        print(f"{r['status']:8} {r['artifact']:16} {r['check']}: {r['value']}  {r['note']}")
    if any(r["status"] == "FAIL" for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("probe", "run"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "probe":
        cmd_probe()
    else:
        cmd_run(sys.argv[2:] or [f"{c.lower()}_{a}" for c in MODELS for a in ("fp32", "int8")])
