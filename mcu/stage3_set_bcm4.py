"""Phase 5: set option byte BCM4 = 0 (Cortex-M4 boot disabled) on the STM32H755 by direct FLASH register writes.

Why direct: STM32CubeProgrammer v2.23 `-ob BCM4=0` answers "Option Byte: bcm4, does not exist" for this chip
(Data_Base/STM32_Prog_DB_0x450.xml has the BCM4 field only for configurations 1,4,5; the H755 is detected as
configuration 0).

Sequence = HAL v1.11.6 HAL_FLASH_OB_Unlock + MODIFY_REG(OPTSR_PRG, BCM4) + HAL_FLASH_OB_Launch:
  OPTKEYR <- 0x08192A3B, 0x4C5D6E7F; OPTSR_PRG <- OPTSR_CUR & ~BCM4; OPTCR <- OPTSTART.
Addresses / bits from cmsis-device-h7 v1.10.7 stm32h755xx.h (FLASH_R_BASE 0x52002000).
Nothing is written unless every pre-check passes. Every read and write is logged.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time

from p5common import CUBEPROG, MCU

FLASH = 0x52002000
OPTKEYR, OPTCR, OPTSR_CUR, OPTSR_PRG, SR1, SR2 = FLASH + 0x08, FLASH + 0x18, FLASH + 0x1C, FLASH + 0x20, FLASH + 0x10, FLASH + 0x110
KEY1, KEY2 = 0x08192A3B, 0x4C5D6E7F
BCM4 = 1 << 22
OPTSTART, OPTLOCK, OPT_BUSY, OPTCHANGEERR = 1 << 1, 1 << 0, 1 << 0, 1 << 30
SR_BUSY = 0xF                                    # BSY | WBNE | QW | CRC_BUSY
LOG = MCU / "cubeai" / f"ob_bcm4_write_{time.strftime('%Y%m%d_%H%M%S')}.log"
log_lines: list[str] = []


def log(s: str) -> None:
    print(s, flush=True)
    log_lines.append(s)
    LOG.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def cli(*args: str) -> str:
    cmd = [CUBEPROG, "-c", "port=SWD", "mode=UR", *args]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout + res.stderr
    log("$ " + " ".join(cmd[1:]) + f"   (rc={res.returncode})")
    if res.returncode != 0 or re.search(r"\bError\b", out):
        log(out[-2000:])
        raise SystemExit("CubeProgrammer error, stopped")
    return out


def read_regs() -> dict:
    out = cli("-r32", hex(FLASH), "0x50", "-r32", hex(SR2), "4")
    words = {}
    for m in re.finditer(r"^0x([0-9A-Fa-f]{8})\s*:\s*((?:[0-9A-Fa-f]{8}\s*)+)$", out, re.M):
        base = int(m.group(1), 16)
        for i, w in enumerate(m.group(2).split()):
            words[base + 4 * i] = int(w, 16)
    need = (OPTCR, OPTSR_CUR, OPTSR_PRG, SR1, SR2)
    missing = [hex(a) for a in need if a not in words]
    if missing:
        raise SystemExit(f"could not parse registers {missing}")
    r = {k: words[a] for k, a in (("OPTCR", OPTCR), ("OPTSR_CUR", OPTSR_CUR), ("OPTSR_PRG", OPTSR_PRG),
                                   ("SR1", SR1), ("SR2", SR2))}
    log("  " + "  ".join(f"{k}=0x{v:08X}" for k, v in r.items()))
    return r


log("== read before")
r = read_regs()
cur = r["OPTSR_CUR"]
new = cur & ~BCM4
checks = {
    "OPTSR_PRG == OPTSR_CUR": r["OPTSR_PRG"] == cur,
    "RDP == 0xAA (level 0)": (cur >> 8) & 0xFF == 0xAA,
    "BCM4 == 1 now": bool(cur & BCM4),
    "OPT_BUSY == 0": not (cur & OPT_BUSY),
    "OPTCHANGEERR == 0": not (cur & OPTCHANGEERR),
    "FLASH SR1/SR2 idle": not (r["SR1"] & SR_BUSY) and not (r["SR2"] & SR_BUSY),
    "OPTCR locked (== 0x1)": r["OPTCR"] == OPTLOCK,
    "new differs only in bit 22": (cur ^ new) == BCM4,
    "new RDP still 0xAA": (new >> 8) & 0xFF == 0xAA,
}
for k, ok in checks.items():
    log(f"  {'PASS' if ok else 'FAIL'}  {k}")
if not all(checks.values()):
    raise SystemExit("pre-check failed, nothing written")
log(f"== write OPTSR_PRG 0x{cur:08X} -> 0x{new:08X}, then OPTSTART")
# -nv: OPTKEYR is write-only (reads 0), so CubeProgrammer's read-back verify of -w32 would report a failure
# and stop the sequence after KEY1 (seen 2026-11-05 18:2x; state unchanged after that, re-locked by reset).
cli("-w32", hex(OPTKEYR), f"0x{KEY1:08X}", "-nv", "-w32", hex(OPTKEYR), f"0x{KEY2:08X}", "-nv",
    "-w32", hex(OPTSR_PRG), f"0x{new:08X}", "-nv", "-w32", hex(OPTCR), f"0x{OPTSTART:08X}", "-nv")
time.sleep(1.0)
log("== read after OPTSTART")
r2 = read_regs()
log("== hardware reset, read again")
cli("-hardRst")
time.sleep(1.0)
r3 = read_regs()
ok = (r2["OPTSR_CUR"] == new and r3["OPTSR_CUR"] == new and not (r3["OPTSR_CUR"] & (OPT_BUSY | OPTCHANGEERR)))
log(f"{'PASS' if ok else 'FAIL'}  OPTSR_CUR after = 0x{r3['OPTSR_CUR']:08X} (expected 0x{new:08X}); "
    f"BCM4={(r3['OPTSR_CUR'] >> 22) & 1}, RDP=0x{(r3['OPTSR_CUR'] >> 8) & 0xFF:02X}")
sys.exit(0 if ok else 1)
