"""Phase 5 / stage 4: diagnose the discrete current steps seen in the energy session of 2026-11-06.

Supply current is logged at about 9 Hz (read-only queries) while the host does things that change nothing on
the board's firmware: open / close the VCP, attach the debugger without reset (register read), hardware reset.
Output: cubeai/bench/diag_power_<ts>.csv (t, i_mA, event marks) and a per-phase summary.

Usage (from mcu/):  P5_SUPPLY=VIN .venv/Scripts/python stage4_power_diag.py COM9 [n_resets]
"""
from __future__ import annotations

import csv
import re
import subprocess
import sys
import time

import numpy as np
import serial

import stage4_bench as sb
import stage4_energy as e
from p5common import CUBEPROG

LOG = e.LOG
REGS = {
    "RCC_GCR": 0x580244A0,      # bit 2 BOOT_C1, bit 3 BOOT_C2
    "RCC_RSR": 0x580244D0,      # reset flags
    "DBGMCU_CR": 0x5C001004,    # debug clock enables, sleep/stop debug bits
    "PWR_CR3": 0x5802480C,      # supply configuration
    "PWR_D3CR": 0x58024818,     # VOS
    "RCC_CR": 0x58024400,       # oscillators, D1CKRDY / D2CKRDY
    "FLASH_OPTSR_CUR": 0x5200201C,
}


def hotplug_read() -> dict:
    """Debugger attach without reset: reads registers, leaves the firmware running."""
    out = {}
    for name, addr in REGS.items():
        res = subprocess.run([CUBEPROG, "-c", "port=SWD", "freq=4000", "mode=HOTPLUG", "-r32", hex(addr), "4"],
                             capture_output=True, text=True)
        m = re.search(r"^0x[0-9A-Fa-f]+\s*:\s*([0-9A-Fa-f]{8})", res.stdout, re.M)
        out[name] = m.group(1) if m else "read failed"
    return out


def start_idle_firmware():
    """After a reset the firmware waits for 'G' over the VCP; send it, close the VCP, firmware goes to WFI."""
    with serial.Serial(sb.vcp(), sb.BAUD, timeout=0.5) as ser:
        t0 = time.time()
        buf = b""
        while time.time() - t0 < 10:
            buf += ser.read(256)
            if b"#INIT" in buf:
                ser.write(b"G")
                break
        time.sleep(0.5)


def main(port: str, n_resets: int):
    o = e.Owon(port)
    print(e.check_supply(o), flush=True)
    lg = e.Logger(o)
    lg.start()
    marks = []

    def phase(name: str, seconds: float, act=None):
        t0 = time.time()
        res = act() if act else None
        marks.append((name, t0))
        print(f"{time.strftime('%H:%M:%S')} {name}" + (f" -> {res}" if res else ""), flush=True)
        time.sleep(seconds)
        return res

    phase("as_left", 15)
    ser = serial.Serial(sb.vcp(), sb.BAUD, timeout=0.5)
    phase("vcp_open", 15)
    ser.close()
    phase("vcp_closed", 15)
    phase("hotplug_regs", 15, hotplug_read)
    for k in range(n_resets):
        phase(f"reset{k}_wait", 0, lambda: (sb.prog("-hardRst"), None)[1])
        time.sleep(1.5)
        phase(f"reset{k}_idle", 20, start_idle_firmware)
    phase("regs_after_resets", 15, hotplug_read)
    a = lg.stop()
    o.close()

    ts = time.strftime("%Y%m%d_%H%M%S")
    with (LOG / f"diag_power_{ts}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["# marks"] + [f"{n}={t:.3f}" for n, t in marks])
        w.writerow(["t_query_s", "v_V", "i_mA"])
        for t0, _, v, i in a:
            w.writerow([f"{t0:.3f}", f"{v:.3f}", f"{i * 1e3:.1f}"])
    ends = [t for _, t in marks[1:]] + [a[-1, 0]]
    print("\nphase                 I mean  min  max (mA, skipping the first 3 s of each phase)")
    for (n, t0), t1 in zip(marks, ends):
        s = (a[:, 0] >= t0 + 3) & (a[:, 0] < t1)
        if s.any():
            i = a[s, 3] * 1e3
            print(f"{n:20s} {i.mean():7.1f} {i.min():5.0f} {i.max():5.0f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6)
