"""Phase 5 / stage 4: do the MCU registers differ between the 'light' and 'heavy' current states after reset?

WFI blocks a hot-plug debugger attach ("Unable to get core ID"), so registers are read during the firmware's
30 s host-wait phase, when the core is polling the UART and nothing has been started yet. After each hardware reset:
current over 2 s -> state; hot-plug read of RCC, PWR, DBGMCU, GPIO modes, in one programmer call. Differences between
states are printed. RCC_GCR.BOOT_C2 and the D2 clock-ready flag tell whether the Cortex-M4 domain was started.
Output: cubeai/bench/diag_regs_<ts>.json. Usage: P5_SUPPLY=VIN .venv/Scripts/python stage4_power_regs.py COM9 [max_resets]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time

import numpy as np

import stage4_bench as sb
import stage4_energy as e
from p5common import CUBEPROG

BLOCKS = {"RCC": (0x58024400, 0x180), "PWR": (0x58024800, 0x40), "DBGMCU": (0x5C001000, 0x10),
          "GPIOA": (0x58020000, 0x10), "GPIOB": (0x58020400, 0x10), "GPIOC": (0x58020800, 0x10),
          "GPIOG": (0x58021800, 0x10)}
NAMED = {0x580244A0: "RCC_GCR", 0x580244D0: "RCC_RSR", 0x58024400: "RCC_CR", 0x5C001004: "DBGMCU_CR"}


def read_regs() -> dict:
    cmd = [CUBEPROG, "-c", "port=SWD", "freq=4000", "mode=HOTPLUG"]
    for addr, size in BLOCKS.values():
        cmd += ["-r32", hex(addr), hex(size)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    regs = {}
    for ln in res.stdout.splitlines():
        m = re.match(r"^0x([0-9A-Fa-f]{8})\s*:\s*((?:[0-9A-Fa-f]{8}\s*)+)$", ln.strip())
        if m:
            base = int(m.group(1), 16)
            for j, w in enumerate(m.group(2).split()):
                regs[base + 4 * j] = int(w, 16)
    return regs


def main(port: str, max_resets: int):
    o = e.Owon(port)
    seen: dict[int, list] = {}
    all_i = []
    for k in range(max_resets):
        sb.prog("-hardRst")
        time.sleep(3.0)
        i = np.mean([float(o.q("MEAS:CURR?")) * 1e3 for _ in range(12)])
        state = int(round((i - 135.0) / 40.0))           # wait phase: about 135 mA in the light state
        regs = read_regs()
        all_i.append(round(float(i), 1))
        d2 = (regs.get(0x58024400, 0) >> 15) & 1
        gcr = regs.get(0x580244A0, -1)
        print(f"reset {k}: I = {i:.1f} mA, state +{state * 40} mA, {len(regs)} regs, RCC_GCR=0x{gcr:08X} "
              f"(BOOT_C1={(gcr >> 2) & 1} BOOT_C2={(gcr >> 3) & 1}), RCC_CR.D2CKRDY={d2}, "
              f"DBGMCU_CR=0x{regs.get(0x5C001004, -1):08X}", flush=True)
        if regs:
            seen.setdefault(state, []).append(regs)
        if len(seen.get(0, [])) >= 2 and len(seen.get(1, [])) >= 2:
            break
    o.close()
    print("wait-phase currents:", all_i)
    ts = time.strftime("%Y%m%d_%H%M%S")
    (e.LOG / f"diag_regs_{ts}.json").write_text(json.dumps(
        {str(s): [{hex(a): f"0x{v:08X}" for a, v in r.items()} for r in rs] for s, rs in seen.items()}, indent=0), encoding="utf-8")
    if 0 in seen and len(seen) > 1:
        base = seen[0]
        stable = {a for a in base[0] if all(r.get(a) == base[0][a] for r in base)}
        for s, rs in seen.items():
            if s == 0:
                continue
            diffs = sorted(a for a in stable if any(r.get(a) != base[0][a] for r in rs))
            print(f"state +{s * 40} mA vs light: {len(diffs)} registers differ (of {len(stable)} stable in the light state)")
            for a in diffs:
                print(f"  {NAMED.get(a, hex(a))}: light 0x{base[0][a]:08X} / heavy " + " ".join(f"0x{r.get(a, 0):08X}" for r in rs))
    else:
        print("did not capture both states")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 25)
