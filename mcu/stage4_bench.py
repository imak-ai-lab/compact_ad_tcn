"""Phase 5 / stage 4b: DWT latency with the own benchmark firmware (NUCLEO-H755ZI-Q, Cortex-M7, 400 MHz).

Usage (from mcu/):
    .venv/Scripts/python stage4_bench.py run [tag ...]     # default: all six run builds
Needs the board on the ST-LINK USB (flashing + UART). Energy (OWON / INA226) is a separate step.

Per tag: refuse unless option byte BCM4 = 0; flash firmware/bench/build/<tag>/bench_<tag>.bin
(verify, hardware reset) with the VCP already open; read the UART protocol of firmware/bench/main.c.
Checks -> latency_dwt.csv:
  header: K windows CRC32 == host CRC32 of the same windows (stored bit-exact)              PASS/FAIL
  per window: crc_in == host CRC32 of that window; crc_out == CRC32 of the received bytes      PASS/FAIL
  fp32: max rel diff (|a-b|/(|b|+1), as stage 3) device vs ORT fp32 on the K windows, tol 1e-4  PASS/FAIL
  int8: same vs ORT int8 file (target is mixed int8/fp32)                                    INFO
  DWT latency, N = 1000 (window i % K): mean/std/min/max ms at SYSCLK reported by the firmware MEASURED
  |DWT mean - Cube.AI validate mean| / Cube.AI <= 10 % (cubeai_validate.csv, stage 3)        PASS/FAIL
  clock check: host wall time RUN_START..RUN_END vs firmware loop time (SysTick)  tol 1 %    PASS/FAIL
  run errors (ai_network_run != 1) == 0                                                      PASS/FAIL
Raw per-inference cycles -> latency_dwt_cycles.csv; UART logs -> cubeai/bench/.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import onnxruntime as ort
import serial

from p5common import CKPT_DIR, MCU, MODELS, append_csv, host_tag, now, verdict
from stage3_target_validate import option_bytes, prog, stlink_com

FW = MCU / "firmware" / "bench" / "build"
LOG = MCU / "cubeai" / "bench"
TRIAL = bool(os.environ.get("P5_TRIAL"))          # trial runs never touch the official CSVs
OUT = MCU / ("latency_dwt_trial.csv" if TRIAL else "latency_dwt.csv")
OUT_CYC = MCU / ("latency_dwt_cycles_trial.csv" if TRIAL else "latency_dwt_cycles.csv")
BAUD = 460800
TOL_REL, TOL_LAT, TOL_CLK = 1e-4, 0.10, 0.01
selection = json.loads((CKPT_DIR / "selection.json").read_text())


def stage3_latency() -> dict:
    with (MCU / "cubeai_validate.csv").open(encoding="utf-8") as fh:
        return {r["artifact"]: float(r["value"]) for r in csv.DictReader(fh) if r["check"] == "latency_ms_mean"}


def kv(line: str) -> dict:
    return dict(re.findall(r"(\w+)=(\S+)", line))


class Reader:
    """Line / byte reader on the VCP with a host timestamp per protocol line. Reads in blocks (whatever the
    driver holds): byte-wise reads lost 130 cycle lines of a 1000-line dump while a second thread polled the
    supply (energy run, tr138_fp32, 2026-11-06)."""

    def __init__(self, ser: serial.Serial, log):
        self.ser, self.log, self.buf = ser, log, bytearray()

    def _fill(self):
        n = self.ser.in_waiting
        chunk = self.ser.read(n if n else 1)
        if chunk:
            self.buf += chunk

    def line(self, deadline: float) -> tuple[str, float]:
        while True:
            j = self.buf.find(b"\n")
            if j >= 0:
                s = bytes(self.buf[:j]).decode("ascii", errors="replace").strip()
                del self.buf[:j + 1]
                t = time.time()
                self.log.write(f"{t:.3f} {s}\n")
                if s:
                    return s, t
                continue
            if time.time() >= deadline:
                raise TimeoutError(f"no line before deadline; partial={bytes(self.buf[:80])!r}")
            self._fill()

    def exact(self, n: int, deadline: float) -> bytes:
        while len(self.buf) < n:
            if time.time() >= deadline:
                raise TimeoutError(f"got {len(self.buf)}/{n} bytes")
            self._fill()
        out = bytes(self.buf[:n])
        del self.buf[:n]
        return out


def rel(a, b):
    r = np.abs(a - b) / (np.abs(b) + 1.0)
    return float(r.max()), float(np.quantile(r, 0.9999))


def run_tag(tag: str, lat3: dict, com: str) -> tuple[list[dict], list[dict]]:
    code, art = tag.split("_")[:2]
    code = code.upper()
    onnx = MCU / "models" / f"{code}_{art}.onnx"
    binf = FW / tag / f"bench_{tag}.bin"
    w = np.load(MCU / "data" / f"stage4_windows_{code}.npz")
    xw, idx, lab = w["x"], w["idx"], w["label"]
    K = len(xw)
    LOG.mkdir(parents=True, exist_ok=True)
    flash_log = prog("-w", binf, "0x08000000", "-v", "-hardRst")
    (LOG / f"flash_{tag}.log").write_text(flash_log, encoding="utf-8")
    t_lat = lat3[onnx.name] / 1e3
    # The VCP is opened only after flashing: the ST-LINK USB link re-enumerates during programming on
    # this setup, which invalidates an open handle. A broken session (USB drop, timeout) is a link
    # failure, not a measurement: reset the board and repeat, at most 3 attempts; attempt -> CSV.
    for attempt in (1, 2, 3):
        if attempt > 1:
            prog("-hardRst")
        time.sleep(1.5)
        try:
            res = session(tag, vcp(), K, t_lat, LOG / f"uart_{tag}_a{attempt}.log")
            break
        except (serial.SerialException, TimeoutError, OSError, RuntimeError) as e:
            print(f"{tag}: attempt {attempt} failed: {str(e)[:120]}", flush=True)
            if attempt == 3:
                raise
    hdr, y, crc_in_raw, crc_out_ok, cyc_check, t_start, t_end, run_end, cyc, kk, done = res
    crc_in_ok = sum(int(c == f"{zlib.crc32(xw[k].tobytes()):08x}") for k, c in enumerate(crc_in_raw))
    sysclk = int(hdr["sysclk"])

    # host references on the same K windows
    sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    nin = sess.get_inputs()[0].name
    y_o = np.concatenate([sess.run(None, {nin: xw[i:i + 1]})[0].reshape(1, -1) for i in range(K)])
    mx, q = rel(y, y_o)
    ms = cyc / sysclk * 1e3
    lat_ref = lat3[onnx.name]
    dev = abs(ms.mean() - lat_ref) / lat_ref
    host_wall = t_end - t_start
    return finish(tag, code, art, onnx, binf, idx, lab, K, hdr, mx, q, ms, cyc, kk, cyc_check, lat_ref, dev,
                  host_wall, run_end, done, crc_in_ok, crc_out_ok, attempt, f"{zlib.crc32(xw.tobytes()):08x}")


def session(tag: str, com: str, K: int, t_lat: float, logpath: Path):
    """One UART session with the bench firmware: handshake, check pass, timed run, cycle dump."""
    with serial.Serial(com, BAUD, timeout=0.5) as ser, logpath.open("w", encoding="utf-8") as logf:
        ser.reset_input_buffer()
        rd = Reader(ser, logf)
        hdr = None
        while True:                                   # the firmware repeats header + init once per second
            s, _ = rd.line(time.time() + 40)
            if s.startswith("#BENCH"):
                hdr = kv(s)
                if hdr.get("tag") != tag or hdr.get("mode") != "run":
                    raise RuntimeError(f"{tag}: header from another firmware: {s}")
            elif s.startswith("#INIT") and hdr is not None:
                if kv(s).get("err_type") != "0":
                    raise RuntimeError(f"{tag}: init failed: {s}")
                ser.write(b"G")
            elif s.startswith("#GO"):
                if kv(s).get("host") != "1":
                    raise TimeoutError(f"{tag}: firmware started without the host: {s}")
                break
        y, crc_in, cyc_check = [], [], []
        crc_out_ok = 0
        for k in range(K):
            s, _ = rd.line(time.time() + 30 + 5 * t_lat)
            if not s.startswith("#OUT") or int(kv(s)["k"]) != k:
                raise RuntimeError(f"{tag}: expected #OUT {k}, got {s[:80]}")
            d = kv(s)
            raw = rd.exact(int(d["n"]), time.time() + 30)
            rd.exact(1, time.time() + 5)
            crc_in.append(d["crc_in"])
            crc_out_ok += int(d["crc_out"] == f"{zlib.crc32(raw):08x}")
            cyc_check.append(int(d["cyc"]))
            y.append(np.frombuffer(raw, dtype=np.float32).copy())
        while True:
            s, t_start = rd.line(time.time() + 30)
            if s.startswith("#RUN_START"):
                break
        s, t_end = rd.line(time.time() + 60 + 1000 * t_lat * 1.5)
        if not s.startswith("#RUN_END"):
            raise RuntimeError(f"{tag}: expected #RUN_END, got {s[:80]}")
        run_end = kv(s)
        cyc = np.zeros(int(hdr["n"]), dtype=np.int64)
        kk = np.zeros_like(cyc)
        for i in range(len(cyc)):
            d = kv(rd.line(time.time() + 30)[0])
            if int(d["i"]) != i:
                raise RuntimeError(f"{tag}: cycle line {i} out of order: {d}")
            cyc[i], kk[i] = int(d["cyc"]), int(d["k"])
        done = kv(rd.line(time.time() + 30)[0])
    return hdr, np.stack(y), crc_in, crc_out_ok, cyc_check, t_start, t_end, run_end, cyc, kk, done


def finish(tag, code, art, onnx, binf, idx, lab, K, hdr, mx, q, ms, cyc, kk, cyc_check, lat_ref, dev, host_wall,
           run_end, done, crc_in_ok, crc_out_ok, attempt, win_crc_host):
    sysclk = int(hdr["sysclk"])
    fw_loop = int(run_end["loop_ms"]) / 1e3
    clk_dev = abs(host_wall - fw_loop) / fw_loop
    seed = selection[code]
    base = dict(date=now(), host=host_tag(), code=code, seed=seed, artifact=onnx.name, firmware=binf.name,
                board="NUCLEO-H755ZI-Q CM7 VOS1 direct-SMPS, I/D-cache on, act in AXI SRAM, CM4 off (BCM4=0), USB ST-LINK power",
                clk=hdr.get("clk"), sysclk_hz=sysclk, hclk_hz=hdr.get("hclk"), n=len(cyc), k_windows=K,
                windows=f"data/stage4_windows_{code}.npz ({int(lab.sum())} anomalous / {K})", built=hdr.get("built"),
                session_attempt=attempt)
    rows = [
        {**base, "check": "windows_crc_device_eq_host", "value": hdr["windows_crc"], "reference": win_crc_host,
         "tol": 0, "status": verdict(hdr["windows_crc"] == win_crc_host), "note": "flash image holds the exact windows"},
        {**base, "check": "per_window_crc_in_ok", "value": crc_in_ok, "reference": K, "tol": 0,
         "status": verdict(crc_in_ok == K), "note": ""},
        {**base, "check": "per_window_crc_out_ok", "value": crc_out_ok, "reference": K, "tol": 0,
         "status": verdict(crc_out_ok == K), "note": "UART transfer of the outputs intact"},
        {**base, "check": "max_rel_diff_device_vs_ort" + ("" if art == "fp32" else "_int8"), "value": f"{mx:.3e}",
         "reference": 0 if art == "fp32" else "", "tol": TOL_REL if art == "fp32" else "",
         "status": verdict(mx < TOL_REL) if art == "fp32" else "INFO",
         "note": f"p99.99={q:.3e}" + ("" if art == "fp32" else "; target is mixed int8/fp32, ORT file all-int8")},
        {**base, "check": "latency_ms_mean", "value": f"{ms.mean():.4f}", "reference": "", "tol": "", "status": "MEASURED",
         "note": f"std={ms.std(ddof=1):.4f} min={ms.min():.4f} max={ms.max():.4f} ms; cycles mean={cyc.mean():.0f}; "
                 f"check-pass mean={np.mean(cyc_check) / sysclk * 1e3:.4f} ms"},
        {**base, "check": "latency_vs_cubeai_validate", "value": f"{dev:.4f}", "reference": lat_ref, "tol": TOL_LAT,
         "status": verdict(dev <= TOL_LAT), "note": f"DWT mean {ms.mean():.3f} ms vs Cube.AI validate {lat_ref} ms"},
        {**base, "check": "clock_host_wall_vs_fw_loop", "value": f"{clk_dev:.4f}", "reference": f"{fw_loop:.3f}",
         "tol": TOL_CLK, "status": verdict(clk_dev <= TOL_CLK),
         "note": f"host wall {host_wall:.3f} s; fw loop {fw_loop:.3f} s; sum DWT {cyc.sum() / sysclk:.3f} s"},
        {**base, "check": "run_errors", "value": int(done["errs"]), "reference": 0, "tol": 0,
         "status": verdict(int(done["errs"]) == 0), "note": f"sum={done['sum']} min={done['min']} max={done['max']}"},
    ]
    cyc_rows = [dict(date=base["date"], firmware=binf.name, artifact=onnx.name, seed=seed, i=i, k=int(kk[i]),
                     test_window_index=int(idx[kk[i]]), label=int(lab[kk[i]]), cycles=int(cyc[i]), sysclk_hz=sysclk)
                for i in range(len(cyc))]
    print(f"{tag}: DWT {ms.mean():.3f} ± {ms.std(ddof=1):.3f} ms (Cube.AI {lat_ref}); max rel vs ORT {mx:.2e}; "
          f"clock dev {clk_dev:.4f}", flush=True)
    return rows, cyc_rows


def vcp(tries: int = 6) -> str:
    """ST-LINK VCP with retries: the USB link on this setup re-enumerates now and then."""
    for k in range(1, tries + 1):
        try:
            return stlink_com()
        except SystemExit as e:
            if k == tries:
                raise
            print(f"  VCP attempt {k}: {e}; retry in 4 s", flush=True)
            time.sleep(4.0)


def cmd_run(tags):
    ob = option_bytes()
    if ob["BCM4"] != 0:
        raise SystemExit(f"BCM4 = {ob['BCM4']} (must be 0). Nothing flashed.")
    com = vcp()
    lat3 = stage3_latency()
    full = len(tags) == 6 and not TRIAL
    if full:
        for p in (OUT, OUT_CYC):
            p.unlink(missing_ok=True)
    rows_all = []
    for tag in tags:
        rows, cyc_rows = run_tag(tag, lat3, com)
        append_csv(OUT, rows)
        append_csv(OUT_CYC, cyc_rows)
        rows_all += rows
    for r in rows_all:
        print(f"{r['status']:8} {r['artifact']:16} {r['check']}: {r['value']}  {r['note']}")
    if any(r["status"] == "FAIL" for r in rows_all):
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "run":
        raise SystemExit(__doc__)
    cmd_run(sys.argv[2:] or [f"{c.lower()}_{a}" for c in MODELS for a in ("fp32", "int8")])
