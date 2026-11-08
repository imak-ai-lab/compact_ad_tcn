"""Phase 5 / stage 3b: build the Cube.AI aiValidation firmware (NUCLEO-H755ZI-Q, Cortex-M7) per artifact.

Usage (from mcu/):  .venv/Scripts/python firmware/validation/build_validation.py [tr047_fp32 ...]
Default: all six artifacts (TR047/TL082/TR138 x fp32/int8).

Per artifact:
  1. `stedgeai generate` with the SAME options as stage 2/3a (p5common.CUBEAI_OPTS), c-name "network"
     -> firmware/validation/gen/<tag>/
  2. arm-none-eabi-gcc 14.2: Core + App glue + generated network + aiValidation/Misc sources of
     ST Edge AI 10.2 + HAL v1.11.6 + cmsis-device-h7 v1.10.7,
     linked with NetworkRuntime1020_CM7_GCC.a (Lib/GCC/STM32H7)
     -> firmware/validation/build/<tag>/validation_<tag>.{elf,bin,hex,map}
  3. section sizes -> mcu/firmware_validation_build.csv, with checks:
       flash image <= 2 MiB                    PASS/FAIL (needs BCM4 = 0 when > 1 MiB)
       .ai_pool == Cube.AI activations bytes   PASS/FAIL (vs cubeai_summary.csv, stage 2)
       DTCM use <= 128 KiB, AXI use <= 512 KiB  PASS/FAIL (the linker also refuses overflow)
No board is needed for this step.
"""
from __future__ import annotations

import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MCU = HERE.parent.parent
sys.path.insert(0, str(MCU))
from p5common import CUBEAI_OPTS, MODELS, STEDGEAI, append_csv, cubeai_env, host_tag, now, verdict  # noqa: E402

GCC_BIN = Path(r"C:\ST\arm-gnu-14.2\bin")
AI = Path(STEDGEAI).parents[2] / "Middlewares" / "ST" / "AI"
TP = HERE.parent / "third_party"
HAL = TP / "stm32h7xx-hal-driver"
DEV = TP / "cmsis-device-h7"
CORE = TP / "CMSIS_5" / "CMSIS" / "Core" / "Include"
LIB_DIR = AI / "Lib" / "GCC" / os.environ.get("P5_RTLIB", "STM32H7")   # or ARMCortexM7
LIB = "NetworkRuntime1020_CM7_GCC.a"
OUT = MCU / ("firmware_validation_build_trial.csv" if os.environ.get("P5_TRIAL") else "firmware_validation_build.csv")

CPU = ["-mcpu=cortex-m7", "-mthumb", "-mfpu=fpv5-d16", "-mfloat-abi=hard"]
OPT = ["-Os", "-g3"]                      # as in ST's CM55_Validation Makefile
BAUD = int(os.environ.get("P5_BAUD", "460800"))
DEFS = ["-DSTM32H755xx", "-DCORE_CM7", "-DUSE_HAL_DRIVER", "-DUSE_PWR_DIRECT_SMPS_SUPPLY", "-DUSE_NUCLEO_H755ZI_Q",
        f"-DVALIDATION_BAUDRATE={BAUD}U"]
HAL_SRC = ["", "_cortex", "_rcc", "_rcc_ex", "_gpio", "_pwr", "_pwr_ex", "_flash", "_flash_ex", "_uart",
           "_uart_ex", "_dma", "_dma_ex", "_mdma", "_exti", "_hsem"]
AI_SRC = ["Validation/Src/" + f for f in ("aiValidation.c", "aiPbMgr.c", "aiPbIO.c", "aiPbMemRWServices.c",
                                          "pb_common.c", "pb_decode.c", "pb_encode.c", "stm32msg.pb.c")] + \
         ["Misc/Src/" + f for f in ("aiTestHelper.c", "aiTestUtility.c", "ai_device_adaptor.c", "lc_print.c")]
FLASH_SECTIONS = {".isr_vector", ".text", ".rodata", ".ARM.extab", ".ARM", ".preinit_array", ".init_array",
                  ".fini_array", ".data"}


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def run(cmd, **kw):
    res = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if res.returncode != 0:
        raise RuntimeError(f"failed ({res.returncode}): {' '.join(map(str, cmd))[:300]}\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}")
    return res


def stage2_activations() -> dict:
    act = {}
    with (MCU / "cubeai_summary.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            act[r["artifact"].removesuffix(".onnx").lower()] = int(r["activations_rw_B"])
    return act


def build(tag: str, act_ref: dict) -> list[dict]:
    code, art = tag.split("_")
    code = code.upper()
    gen, bld = HERE / "gen" / tag, HERE / "build" / tag
    for d in (gen, bld):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    run([STEDGEAI, "generate", "--model", MCU / "models" / f"{code}_{art}.onnx", *CUBEAI_OPTS[art], "--name", "network",
         "--workspace", gen / "ws", "--output", gen], env=cubeai_env())
    shutil.rmtree(gen / "ws", ignore_errors=True)

    src = [HERE / "Core" / "Src" / f for f in ("main.c", "stm32h7xx_it.c", "stm32h7xx_hal_msp.c")]
    src += [HERE / "App" / "app_x-cube-ai.c"]
    src += [gen / f for f in ("network.c", "network_data.c", "network_data_params.c")]
    src += [DEV / "Source" / "Templates" / "system_stm32h7xx_dualcore_bootcm7_cm4gated.c",
            DEV / "Source" / "Templates" / "gcc" / "startup_stm32h755xx.s"]
    src += [HAL / "Src" / f"stm32h7xx_hal{s}.c" for s in HAL_SRC]
    src += [AI / s for s in AI_SRC]
    inc = [HERE / "Core" / "Inc", HERE / "App", gen, HAL / "Inc", HAL / "Inc" / "Legacy", DEV / "Include", CORE,
           AI / "Inc", AI / "Validation" / "Inc", AI / "Misc" / "Inc"]
    elf = bld / f"validation_{tag}.elf"
    ld = HERE / "STM32H755ZITx_CM7.ld"
    cmd = [GCC_BIN / "arm-none-eabi-gcc", *CPU, *OPT, *DEFS, "-std=c11", "-Wall", "-ffunction-sections",
           "-fdata-sections", *[f"-I{p}" for p in inc], *src,
           "-specs=nano.specs", "-specs=nosys.specs", f"-T{ld}", f"-L{LIB_DIR}", f"-l:{LIB}", "-lc", "-lm",
           "-Wl,--wrap=malloc", "-Wl,--wrap=free", "-Wl,--gc-sections", f"-Wl,-Map={bld / ('validation_' + tag + '.map')}",
           "-Wl,--print-memory-usage", "-o", elf]
    res = run(cmd)
    (bld / "build.log").write_text(" ".join(map(str, cmd)) + "\n\n" + res.stdout + res.stderr, encoding="utf-8")
    run([GCC_BIN / "arm-none-eabi-objcopy", "-O", "binary", "-S", elf, elf.with_suffix(".bin")])
    run([GCC_BIN / "arm-none-eabi-objcopy", "-O", "ihex", elf, elf.with_suffix(".hex")])

    sizes = {}
    for line in run([GCC_BIN / "arm-none-eabi-size", "-A", elf]).stdout.splitlines():
        m = re.match(r"^(\.\S+)\s+(\d+)\s+(\d+)", line)
        if m:
            sizes[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    flash = sum(s for k, (s, _) in sizes.items() if k in FLASH_SECTIONS)
    dtcm = sum(s for k, (s, a) in sizes.items() if 0x20000000 <= a < 0x20020000)
    axi = sum(s for k, (s, a) in sizes.items() if 0x24000000 <= a < 0x24080000)
    pool = sizes.get(".ai_pool", (0, 0))[0]
    ref_act = act_ref.get(f"{code.lower()}_{art}")
    warn = len(re.findall(r"warning:", res.stderr))
    gcc_ver = run([GCC_BIN / "arm-none-eabi-gcc", "--version"]).stdout.splitlines()[0]
    base = dict(date=now(), host=host_tag(), gcc=gcc_ver, stedgeai_options=" ".join(CUBEAI_OPTS[art]),
                runtime_lib=f"{LIB_DIR.name}/{LIB} md5={md5(LIB_DIR / LIB)}", cflags=" ".join(CPU + OPT + DEFS),
                code=code, artifact=f"{code}_{art}.onnx", firmware=elf.name, bin_md5=md5(elf.with_suffix(".bin")),
                flash_B=flash, text_B=sizes.get(".text", (0, 0))[0], rodata_B=sizes.get(".rodata", (0, 0))[0],
                dtcm_B=dtcm, axi_B=axi, ai_pool_B=pool, fits_bank1=flash <= 1024 * 1024, gcc_warnings=warn)
    rows = [
        {**base, "check": "flash_le_2MiB", "value": flash, "reference": 2 * 1024 * 1024, "status": verdict(flash <= 2 * 1024 * 1024)},
        {**base, "check": "ai_pool_eq_stage2_activations", "value": pool, "reference": ref_act,
         "status": verdict(ref_act is not None and abs(pool - ref_act) < 32)},
        {**base, "check": "dtcm_le_128KiB", "value": dtcm, "reference": 128 * 1024, "status": verdict(dtcm <= 128 * 1024)},
        {**base, "check": "axi_le_512KiB", "value": axi, "reference": 512 * 1024, "status": verdict(axi <= 512 * 1024)},
    ]
    print(f"{tag}: flash={flash} ({flash / 1024:.1f} KiB, bank1={'yes' if flash <= 1 << 20 else 'NO'}) "
          f"dtcm={dtcm} axi={axi} pool={pool} (stage2 {ref_act}) warnings={warn}", flush=True)
    return rows


tags = sys.argv[1:] or [f"{c.lower()}_{a}" for c in MODELS for a in ("fp32", "int8")]
act_ref = stage2_activations()
rows = []
for t in tags:
    rows += build(t, act_ref)
append_csv(OUT, rows, fresh=len(tags) == 6)
for r in rows:
    print(f"{r['status']} {r['artifact']:16} {r['check']}: {r['value']} ref={r['reference']}")
if any(r["status"] == "FAIL" for r in rows):
    raise SystemExit(1)
