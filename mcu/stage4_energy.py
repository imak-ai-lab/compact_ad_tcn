"""Phase 5 / stage 4c: board-level energy per inference with the OWON SPE3103 (SCPI over USB serial).

Usage (from mcu/):
    .venv/Scripts/python stage4_energy.py ports                       # list serial ports, no I/O
    .venv/Scripts/python stage4_energy.py probe COMx                  # read-only SCPI queries
    .venv/Scripts/python stage4_energy.py run COMx [segment ...]      # default: warm, idle_pre, six runs, idle_post
    .venv/Scripts/python stage4_energy.py analyze [tag ...]           # energy.csv from the raw logs, no hardware

This script NEVER sends a setting or output command to the supply: voltage, current limit and output
are set by hand at the front panel. It only sends queries (*IDN?, VOLT?, CURR?,
OUTP?, MEAS:VOLT?, MEAS:CURR?) and refuses to start unless the set and measured voltages are within the
range of the chosen input (P5_SUPPLY=EXT|VIN, see SUPPLIES) and the output is on.

Board set-up (UM2408 Rev 6 sections 7.4.2 / 7.4.3 / 7.4.6): JP2 and the OWON leads per SUPPLIES; supply on
first (LD5 green), then the ST-LINK USB into CN1 (flashing + UART markers). The measured power is board level
at the chosen input with the ST-LINK USB connected.

Per tag: flash the stage-4 run firmware; OWON polled in a thread for the whole UART session; the firmware's
#RUN_START / #RUN_END lines (host timestamps) delimit the timed loop. P_run = mean V*I over
[RUN_START + 1 s, RUN_END - 1 s]. Baseline: the idle firmware (same start-up, WFI, SysTick on, LEDs off),
logged for IDLE_S seconds before and after the six runs; P_idle = mean of the two.
Warm-up: WARM_S seconds under the heaviest load (TR138 fp32, logged, not used in any number) before the first
baseline, so that both baselines follow a heavy run in a similar thermal state. On 2026-11-06 the two baselines
differed by 11 mW (about 4 % of the TR047 dP) without it, in the trial session.
energy_per_window = (P_run - P_idle) * t_run / N, t_run = sum of the DWT cycles / SYSCLK of the same run.
Rows -> energy.csv; raw samples -> cubeai/bench/owon_<tag>.csv.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import serial
import serial.tools.list_ports as list_ports

from p5common import MCU, MODELS, append_csv, host_tag, now, verdict
import stage4_bench as sb

TRIAL = bool(os.environ.get("P5_TRIAL"))          # trial runs never touch the official CSV
OUT = MCU / ("energy_trial.csv" if TRIAL else "energy.csv")
LOG = MCU / "cubeai" / "bench"
OWON_BAUD = 115200
IDLE_S = 120.0
WARM_S = 600.0
OUT_BOOT = MCU / ("energy_boot_states_trial.csv" if TRIAL else "energy_boot_states.csv")
WAIT_LIGHT_MAX_MA = 157.0            # host-wait phase: about 137 mA in the light state, 177 mA with the extra +40 mA
TOL_LOCAL_BASELINE = 0.03            # energy from the run's own 2 s WFI step vs energy from the idle firmware
TOL_IDLE_DRIFT_MW = 15.0             # |P_idle_pre - P_idle_post|; also |2 s WFI step - P_idle| of every run
# Supply input, chosen explicitly with P5_SUPPLY (UM2408 Rev 6, section 7.4):
#   EXT: 5V_EXT, CN11 pin 6 / GND pin 8, JP2 5-6, 4.75-5.25 V (Table 6) - CN11 is an unfitted footprint here;
#   VIN: CN8 pin 15 / GND pin 13 (Zio, female), JP2 3-4, 7-11 V (Table 5); kept to 7-9 V, where the input
#        current capability is >= 450 mA. Board level then includes the on-board 5 V regulator.
SUPPLIES = {"EXT": (4.75, 5.25, "board level at 5V_EXT (JP2 5-6, CN11 pin 6/8)"),
            "VIN": (7.0, 9.0, "board level at VIN (JP2 3-4, CN8 pin 15/13), incl. on-board 5 V regulator")}
SUPPLY = os.environ.get("P5_SUPPLY", "").upper()
V_MIN, V_MAX, SUPPLY_DESC = SUPPLIES.get(SUPPLY, (None, None, None))
MIN_RUN_S = 60.0                     # >= 60 s of log per run
EDGE_S = 1.0


class Owon:
    def __init__(self, port: str, baud: int = OWON_BAUD):
        self.ser = serial.Serial(port, baud, timeout=1.0)
        self.lock = threading.Lock()

    def q(self, cmd: str) -> str:
        assert cmd.endswith("?"), "queries only"
        with self.lock:
            self.ser.reset_input_buffer()
            self.ser.write((cmd + "\n").encode("ascii"))
            return self.ser.readline().decode("ascii", errors="replace").strip()

    def close(self):
        self.ser.close()


class Logger(threading.Thread):
    """Polls MEAS:VOLT? and MEAS:CURR? as fast as the supply answers; host timestamps (time.time)."""

    def __init__(self, owon: Owon):
        super().__init__(daemon=True)
        self.owon, self.rows, self.stop_ev, self.errors = owon, [], threading.Event(), 0

    def run(self):
        while not self.stop_ev.is_set():
            t0 = time.time()
            try:
                v = float(self.owon.q("MEAS:VOLT?"))
                i = float(self.owon.q("MEAS:CURR?"))
            except (ValueError, serial.SerialException):
                self.errors += 1
                continue
            self.rows.append((t0, time.time(), v, i))

    def stop(self) -> np.ndarray:
        self.stop_ev.set()
        self.join(5)
        return np.asarray(self.rows, dtype=float).reshape(-1, 4)


def save_raw(name: str, a: np.ndarray, marks: dict):
    LOG.mkdir(parents=True, exist_ok=True)
    with (LOG / f"owon_{name}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["# marks"] + [f"{k}={v:.3f}" for k, v in marks.items()])
        w.writerow(["t_query_s", "t_reply_s", "v_V", "i_A", "p_W"])
        for t0, t1, v, i in a:
            w.writerow([f"{t0:.3f}", f"{t1:.3f}", f"{v:.4f}", f"{i:.4f}", f"{v * i:.5f}"])


def window(a: np.ndarray, t0: float, t1: float) -> np.ndarray:
    m = (a[:, 0] >= t0) & (a[:, 1] <= t1)
    return a[m]


def check_supply(o: Owon) -> dict:
    st = {"idn": o.q("*IDN?"), "set_v": o.q("VOLT?"), "set_i": o.q("CURR?"), "out": o.q("OUTP?"),
          "meas_v": o.q("MEAS:VOLT?"), "meas_i": o.q("MEAS:CURR?")}
    try:
        ok = (V_MIN <= float(st["set_v"]) <= V_MAX and V_MIN <= float(st["meas_v"]) <= V_MAX
              and st["out"].upper() in ("1", "ON"))
    except ValueError:
        ok = False
    st["ok"] = ok
    return st


def cmd_ports():
    for p in list_ports.comports():
        print(p.device, f"vid=0x{(p.vid or 0):04X} pid=0x{(p.pid or 0):04X}", p.description)


def cmd_probe(port: str):
    o = Owon(port)
    st = check_supply(o) if SUPPLY in SUPPLIES else {k: o.q(c) for k, c in (("idn", "*IDN?"), ("set_v", "VOLT?"), ("out", "OUTP?"))}
    o.close()
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / f"owon_probe_{time.strftime('%Y%m%d_%H%M%S')}.txt").write_text(repr(st), encoding="utf-8")
    print(st)



def save_meta(name: str, meta: dict):
    (LOG / f"owon_{name}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def light_boot(o: Owon, segment: str, tries: int = 15) -> int:
    """Reset until the board is in the light state; returns how many resets were discarded.

    After a hardware reset the board latches, at random, an extra constant +40 or +80 mA until the next reset
    (seen in the energy session of 2026-11-06; about 1 reset in 4). It does not depend on the VCP or on a debugger
    attach; RCC_GCR.BOOT_C2 = 0 and 132 RCC/PWR/DBGMCU/GPIO registers are identical in both states
    (stage4_power_diag.py, stage4_power_regs.py), so the Cortex-M4 is not started. The state is visible in the
    firmware's host-wait phase, before anything is measured: about 137 mA (light) vs 177 mA (at VIN 7.2 V).
    Every reading goes to energy_boot_states.csv.
    """
    for k in range(tries):
        if k:
            sb.prog("-hardRst")
        time.sleep(3.0)
        i = float(np.mean([float(o.q("MEAS:CURR?")) * 1e3 for _ in range(12)]))
        light = i < WAIT_LIGHT_MAX_MA
        append_csv(OUT_BOOT, [dict(date=now(), segment=segment, reset=k, wait_phase_mA=f"{i:.1f}",
                                   limit_mA=WAIT_LIGHT_MAX_MA, state="light" if light else "heavy", supply=SUPPLY)])
        print(f"{segment}: reset {k}: {i:.1f} mA, {'light' if light else 'heavy, reset again'}", flush=True)
        if light:
            return k
    raise RuntimeError(f"{segment}: no light boot in {tries} resets")


def warm_segment(o: Owon, tag: str = "tr138_fp32"):
    """Warm-up under load: the firmware starts by itself after BENCH_WAIT_S without a host. Log only."""
    binf = sb.FW / tag / f"bench_{tag}.bin"
    sb.prog("-w", binf, "0x08000000", "-v", "-hardRst")
    lg = Logger(o)
    lg.start()
    t0 = time.time()
    try:
        time.sleep(WARM_S)
    finally:
        a = lg.stop()
        save_raw("warm", a, {"WARM_START": t0})
        save_meta("warm", dict(supply=SUPPLY, poll_errors=lg.errors, firmware=binf.name, seconds=WARM_S, saved=now()))
    tail = window(a, t0 + WARM_S - 60.0, t0 + WARM_S)
    print(f"warm-up: last 60 s {float((tail[:, 2] * tail[:, 3]).mean()) * 1e3:.1f} mW", flush=True)


def idle_segment(o: Owon, label: str):
    """Idle firmware, IDLE_S seconds of WFI after #IDLE_START. Raw log saved even if the session breaks."""
    binf = sb.FW / "idle" / "bench_idle.bin"
    sb.prog("-w", binf, "0x08000000", "-v", "-hardRst")
    discarded = light_boot(o, f"idle_{label}")
    lg = Logger(o)
    lg.start()
    marks = {}
    try:
        with serial.Serial(sb.vcp(), sb.BAUD, timeout=0.5) as ser, \
                (LOG / f"uart_idle_{label}.log").open("w", encoding="utf-8") as lf:
            rd = sb.Reader(ser, lf)
            while True:
                s, t_line = rd.line(time.time() + 40)
                if s.startswith("#BENCH") and sb.kv(s).get("mode") != "idle":
                    raise RuntimeError(f"idle: wrong firmware: {s}")
                if s.startswith("#INIT"):
                    ser.write(b"G")
                if s.startswith("#IDLE_START"):
                    marks["IDLE_START"] = t_line
                    break
            time.sleep(IDLE_S + 2 * EDGE_S + 5)     # VCP stays open, as during the runs
    finally:
        a = lg.stop()
        save_raw(f"idle_{label}", a, marks)
        save_meta(f"idle_{label}", dict(supply=SUPPLY, poll_errors=lg.errors, firmware=binf.name, saved=now(),
                                        discarded_resets=discarded))


def run_segment(o: Owon, tag: str, lat3: dict):
    """Run firmware for one model; up to 2 sessions (reset between). Raw logs saved for every attempt."""
    code, art = tag.split("_")
    code = code.upper()
    binf = sb.FW / tag / f"bench_{tag}.bin"
    k = len(np.load(MCU / "data" / f"stage4_windows_{code}.npz")["x"])
    sb.prog("-w", binf, "0x08000000", "-v", "-hardRst")
    for attempt in (1, 2):
        if attempt > 1:
            sb.prog("-hardRst")
        discarded = light_boot(o, f"run_{tag}")
        lg = Logger(o)
        lg.start()
        marks = {}
        try:
            res = sb.session(tag, sb.vcp(), k, lat3[f"{code}_{art}.onnx"] / 1e3, LOG / f"uart_energy_{tag}.log")
            marks = {"RUN_START": res[5], "RUN_END": res[6]}
            return
        except (serial.SerialException, TimeoutError, OSError, RuntimeError) as e:
            print(f"{tag}: energy session {attempt} failed: {str(e)[:120]}", flush=True)
            if attempt == 2:
                raise
        finally:
            a = lg.stop()
            save_raw(f"run_{tag}" if marks else f"run_{tag}_failed{attempt}", a, marks)
            save_meta(f"run_{tag}" if marks else f"run_{tag}_failed{attempt}",
                      dict(supply=SUPPLY, poll_errors=lg.errors, firmware=binf.name, session_attempt=attempt, saved=now(),
                           discarded_resets=discarded))


# ------------------------------------------------------------------ analysis from the raw logs ----
def load_owon(name: str):
    lines = (LOG / f"owon_{name}.csv").read_text(encoding="utf-8").splitlines()
    marks = {k: float(v) for k, v in (kv.split("=") for kv in lines[0].split(",")[1:] if "=" in kv)}
    a = np.array([[float(x) for x in ln.split(",")[:4]] for ln in lines[2:]]).reshape(-1, 4)
    meta_p = LOG / f"owon_{name}.meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    return marks, a, meta


def load_uart(tag: str) -> dict:
    hdr, done, n_cyc, cyc_sum = None, None, 0, 0
    for ln in (LOG / f"uart_energy_{tag}.log").read_text(encoding="utf-8").splitlines():
        s = ln.split(" ", 1)[1] if " " in ln else ""
        if s.startswith("#BENCH") and hdr is None:
            hdr = sb.kv(s)
        elif s.startswith("#CYC"):
            n_cyc += 1
            cyc_sum += int(sb.kv(s)["cyc"])
        elif s.startswith("#DONE"):
            done = sb.kv(s)
    return dict(hdr=hdr, done=done, n_cyc=n_cyc, cyc_sum=cyc_sum)


def stats(a: np.ndarray) -> dict:
    p = a[:, 2] * a[:, 3]
    return dict(n=len(a), t=float(a[-1, 1] - a[0, 0]) if len(a) else 0.0, p=float(p.mean()) if len(a) else float("nan"),
                p_std=float(p.std(ddof=1)) if len(a) > 1 else float("nan"),
                v=float(a[:, 2].mean()) if len(a) else float("nan"), v_min=float(a[:, 2].min()) if len(a) else float("nan"),
                v_max=float(a[:, 2].max()) if len(a) else float("nan"), i=float(a[:, 3].mean()) if len(a) else float("nan"))


def supply_idn() -> str:
    probes = sorted(LOG.glob("owon_probe_*.txt"))
    if not probes:
        return ""
    txt = probes[-1].read_text(encoding="utf-8")
    m = re.search(r"'idn': '([^']*)'", txt)
    return m.group(1) if m else ""


def cmd_analyze(tags: list[str]) -> list[dict]:
    if SUPPLY not in SUPPLIES:
        raise SystemExit("set P5_SUPPLY=EXT or P5_SUPPLY=VIN (the input used for the logs)")
    idle = {}
    for lab in ("pre", "post"):
        marks, a, meta = load_owon(f"idle_{lab}")
        t0 = marks["IDLE_START"]
        idle[lab] = {**stats(window(a, t0 + 5.0, t0 + 5.0 + IDLE_S)), "t0": t0, "meta": meta}
    p_idle = 0.5 * (idle["pre"]["p"] + idle["post"]["p"])
    drift = abs(idle["pre"]["p"] - idle["post"]["p"])
    idn = supply_idn()
    cond = f"{SUPPLY_DESC}, ST-LINK USB connected, OWON readback; CM7 400 MHz VOS1 direct SMPS, CM4 off (BCM4=0)"
    rows = []
    for tag in tags:
        code, art = tag.split("_")
        code = code.upper()
        marks, a, meta = load_owon(f"run_{tag}")
        u = load_uart(tag)
        sysclk, n = int(u["hdr"]["sysclk"]), int(u["hdr"]["n"])
        t_run = u["cyc_sum"] / sysclk
        r = stats(window(a, marks["RUN_START"] + EDGE_S, marks["RUN_END"] - EDGE_S))
        pre = stats(window(a, marks["RUN_START"] - 1.8, marks["RUN_START"] - 0.2))
        e_mj = (r["p"] - p_idle) * t_run / n * 1e3
        e_loc = (r["p"] - pre["p"]) * t_run / n * 1e3
        in_range = V_MIN <= r["v_min"] and r["v_max"] <= V_MAX
        base = dict(date=_dt.datetime.fromtimestamp(marks["RUN_START"]).isoformat(timespec="seconds"), analysed=now(),
                    host=host_tag(), supply=idn, conditions=cond, clk=u["hdr"].get("clk"), code=code,
                    artifact=f"{code}_{art}.onnx", firmware=f"bench_{tag}.bin", idle_firmware="bench_idle.bin",
                    n_inferences=n, t_run_s=f"{t_run:.3f}", run_window_s=f"{r['t']:.1f}", samples_run=r["n"],
                    p_run_mW=f"{r['p'] * 1e3:.2f}", p_run_std_mW=f"{r['p_std'] * 1e3:.2f}", v_run_V=f"{r['v']:.4f}",
                    i_run_mA=f"{r['i'] * 1e3:.1f}",
                    p_idle_pre_mW=f"{idle['pre']['p'] * 1e3:.2f}", p_idle_post_mW=f"{idle['post']['p'] * 1e3:.2f}",
                    idle_pre_at=_dt.datetime.fromtimestamp(idle["pre"]["t0"]).isoformat(timespec="seconds"),
                    idle_post_at=_dt.datetime.fromtimestamp(idle["post"]["t0"]).isoformat(timespec="seconds"),
                    samples_idle=idle["pre"]["n"] + idle["post"]["n"], p_preidle_local_mW=f"{pre['p'] * 1e3:.2f}",
                    samples_preidle_local=pre["n"], lat_dwt_ms=f"{u['cyc_sum'] / n / sysclk * 1e3:.3f}",
                    poll_errors=meta.get("poll_errors", "n/a (not recorded)"),
                    session_attempt=meta.get("session_attempt", "n/a (not recorded)"),
                    raw=f"cubeai/bench/owon_run_{tag}.csv + uart_energy_{tag}.log")
        rows += [
            {**base, "check": "energy_per_window_mJ_board", "value": f"{e_mj:.4f}", "tol": "", "status": "MEASURED",
             "note": f"(P_run - P_idle) * t_run / N; dP = {(r['p'] - p_idle) * 1e3:.2f} mW; P_idle = mean of pre/post"},
            {**base, "check": "energy_per_window_mJ_board_local_baseline", "value": f"{e_loc:.4f}", "tol": "",
             "status": "INFO", "note": f"baseline = 2 s WFI step of the run firmware before #RUN_START ({pre['n']} samples); "
                                    f"diff vs idle-firmware baseline {(e_loc - e_mj) / e_mj * 100:+.1f} %"},
            {**base, "check": "run_window_ge_60s", "value": f"{r['t']:.1f}", "tol": MIN_RUN_S,
             "status": verdict(r["t"] >= MIN_RUN_S), "note": "s"},
            {**base, "check": "cycle_dump_complete", "value": u["n_cyc"], "tol": n, "status": verdict(u["n_cyc"] == n),
             "note": "t_run from the per-inference DWT cycles of this run"},
            {**base, "check": "p_run_gt_p_idle", "value": f"{(r['p'] - p_idle) * 1e3:.2f}", "tol": 0,
             "status": verdict(r["p"] > p_idle), "note": "mW"},
            {**base, "check": "run_errors", "value": u["done"]["errs"], "tol": 0,
             "status": verdict(u["done"]["errs"] == "0"), "note": ""},
            {**base, "check": f"supply_in_{SUPPLY}_range_run", "value": f"{r['v_min']:.3f}..{r['v_max']:.3f}",
             "tol": f"{V_MIN}-{V_MAX}", "status": verdict(in_range), "note": "OWON readback over the run window"},
            {**base, "check": "idle_drift_mW", "value": f"{drift * 1e3:.2f}", "tol": TOL_IDLE_DRIFT_MW,
             "status": verdict(drift * 1e3 <= TOL_IDLE_DRIFT_MW),
             "note": f"pre {idle['pre']['p'] * 1e3:.2f} / post {idle['post']['p'] * 1e3:.2f} mW"},
            {**base, "check": "boot_state_wfi_step_vs_idle_mW", "value": f"{(pre['p'] - p_idle) * 1e3:.2f}",
             "tol": TOL_IDLE_DRIFT_MW, "status": verdict(abs(pre["p"] - p_idle) * 1e3 <= TOL_IDLE_DRIFT_MW),
             "note": f"discarded resets before this run: {meta.get('discarded_resets', 'n/a (not guarded)')}; "
                     "an extra +40 mA state would show as +288 mW here"},
            {**base, "check": "energy_local_vs_idle_baseline", "value": f"{(e_loc - e_mj) / e_mj:+.4f}",
             "tol": TOL_LOCAL_BASELINE, "status": verdict(abs(e_loc - e_mj) <= TOL_LOCAL_BASELINE * abs(e_mj)),
             "note": "the two baselines must agree"},
        ]
    append_csv(OUT, rows, fresh=True)
    for row in rows:
        print(f"{row['status']:8} {row['artifact']:16} {row['check']}: {row['value']}  {row['note']}")
    return rows


ALL_TAGS = [f"{c.lower()}_{a}" for c in MODELS for a in ("fp32", "int8")]


def cmd_run(port: str, segs: list[str]):
    """segs: warm, idle_pre, model tags, idle_post (default: all, in that order)."""
    ob = sb.option_bytes()
    if ob["BCM4"] != 0:
        raise SystemExit(f"BCM4 = {ob['BCM4']} (must be 0). Nothing flashed.")
    if SUPPLY not in SUPPLIES:
        raise SystemExit("set P5_SUPPLY=EXT or P5_SUPPLY=VIN (how the board is powered)")
    o = Owon(port)
    st = check_supply(o)
    print("supply:", SUPPLY, st, flush=True)
    if not st["ok"]:
        raise SystemExit(f"supply not in {V_MIN}-{V_MAX} V for {SUPPLY} or output off: nothing flashed (set it at the front panel)")
    lat3 = sb.stage3_latency()
    for s in segs:
        if s == "warm":
            warm_segment(o)
        elif s.startswith("idle_"):
            idle_segment(o, s.removeprefix("idle_"))
        else:
            run_segment(o, s, lat3)
        print(f"{s}: done", flush=True)
    st_end = check_supply(o)
    print("supply at end:", st_end, flush=True)
    o.close()
    if not st_end["ok"]:
        raise SystemExit("supply left the allowed range during the session")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("ports", "probe", "run", "analyze"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "ports":
        cmd_ports()
    elif sys.argv[1] == "probe":
        cmd_probe(sys.argv[2])
    elif sys.argv[1] == "analyze":
        rows = cmd_analyze(sys.argv[2:] or ALL_TAGS)
        if any(r["status"] == "FAIL" for r in rows):
            raise SystemExit(1)
    else:
        cmd_run(sys.argv[2], sys.argv[3:] or ["warm", "idle_pre", *ALL_TAGS, "idle_post"])
