"""Phase 5 / stage 5b: build the feature-extraction firmware and its host reference.

Usage (from mcu/):  .venv/Scripts/python firmware/features/build_features.py [features_v0 features_v1]

Stored data (generated, gen/features_data.{h,c}):
  K = 8 real raw blocks of 3 x 3248 samples: 4 from the normal and 4 from the damaged test file of
  data/phase5_raw.npz (rng 2029), float32, bit-exact; wavelet filters from pywt; the preset scaler of the repo.
Host reference -> data/stage5_blocks.npz (x [K,3,N], y_v0, y_v1 [K,120,27] scaled float32), computed on the SAME float32
samples with the repo's own functions in float64:
  y_v0 = process_bearing_signals_to_features(block), the repo function applied to the block;
  y_v1 = the same steps with the whole-file factor sqrt(2 ln 256000) in the threshold.
Build: arm-none-eabi-gcc 14.2, CMSIS-DSP from CMSIS_5 5.9.0, HAL/startup/linker script as stage 3/4.
Optimisation -O3 with loop unrolling for this firmware (signal-processing code a user would compile for speed; the
model kernels of stage 4 are precompiled by ST). Sizes -> mcu/firmware_features_build.csv.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pywt

HERE = Path(__file__).resolve().parent
MCU = HERE.parent.parent
VAL = HERE.parent / "validation"
sys.path.insert(0, str(MCU))
from p5common import CKPT_DIR, append_csv, host_tag, now, verdict  # noqa: E402
from industrial_ad.datasets.PU.features import (CLASSIC_27D_FEATURE_STATS, build_pu_feature_pipeline,  # noqa: E402
                                                extract_features_vectorized, process_bearing_signals_to_features)

GCC_BIN = Path(r"C:\ST\arm-gnu-14.2\bin")
TP = HERE.parent / "third_party"
HAL = TP / "stm32h7xx-hal-driver"
DEV = TP / "cmsis-device-h7"
CORE = TP / "CMSIS_5" / "CMSIS" / "Core" / "Include"
DSP = TP / "CMSIS_5" / "CMSIS" / "DSP"
OUT = MCU / "firmware_features_build.csv"
GEN = HERE / "gen"

T, WIN, HOP, TRIM, K_PER_FILE, RNG, N_REF, BAUD = 120, 64, 16, 640, 4, 2029, 256000, 460800
N = WIN + (T - 1) * HOP + 2 * TRIM
N_TIMED = 4000
SPECS = [("sym8", 6), ("sym8", 6), ("db4", 4)]
CPU = ["-mcpu=cortex-m7", "-mthumb", "-mfpu=fpv5-d16", "-mfloat-abi=hard"]
OPT = ["-O3", "-g3"]
DEFS = ["-DSTM32H755xx", "-DCORE_CM7", "-DUSE_HAL_DRIVER", "-DUSE_PWR_DIRECT_SMPS_SUPPLY", "-DUSE_NUCLEO_H755ZI_Q",
        f"-DVALIDATION_BAUDRATE={BAUD}U", f"-DBENCH_N={N_TIMED}U", "-DARM_MATH_LOOPUNROLL"]
HAL_SRC = ["", "_cortex", "_rcc", "_rcc_ex", "_gpio", "_pwr", "_pwr_ex", "_flash", "_flash_ex", "_uart",
           "_uart_ex", "_dma", "_dma_ex", "_mdma", "_exti", "_hsem"]
DSP_SRC = ["TransformFunctions/arm_rfft_fast_f32.c", "TransformFunctions/arm_cfft_f32.c",
           "TransformFunctions/arm_cfft_radix8_f32.c", "TransformFunctions/arm_bitreversal2.c",
           "CommonTables/arm_common_tables.c", "CommonTables/arm_const_structs.c",
           "BasicMathFunctions/arm_dot_prod_f32.c"]
FLASH_SECTIONS = {".isr_vector", ".text", ".rodata", ".ARM.extab", ".ARM", ".preinit_array", ".init_array",
                  ".fini_array", ".data"}
ALGO_RAM_SYMS = ["g_block", "g_coef", "g_ext", "g_rows", "g_fft_in", "g_fft_out", "g_wext", "g_a1", "g_fft"]

selection = json.loads((CKPT_DIR / "selection.json").read_text())
cfg = json.loads((CKPT_DIR / f"TR047_s{selection['TR047']}.config.json").read_text())
pipe = build_pu_feature_pipeline(cfg["dataset"]["params"]["feature_pipeline"])
FS = int(pipe.sampling_rate)
assert FS == 40000 and int(pipe.drop_edges) == TRIM
MEAN = np.asarray(CLASSIC_27D_FEATURE_STATS["mean"], dtype=np.float32)
STD = np.asarray(CLASSIC_27D_FEATURE_STATS["std"], dtype=np.float32)
THR_V0, THR_V1 = float(np.sqrt(2 * np.log(N))), float(np.sqrt(2 * np.log(N_REF)))


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def run(cmd, **kw):
    res = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if res.returncode != 0:
        raise RuntimeError(f"failed ({res.returncode}): {' '.join(map(str, cmd))[:300]}\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}")
    return res


def scaled(f: np.ndarray) -> np.ndarray:
    return ((f.T.astype(np.float32) - MEAN) / STD).astype(np.float32)


def ref_with_factor(block: np.ndarray, k: float) -> np.ndarray:
    """features.py steps on one block, threshold factor k instead of sqrt(2 ln len(x))."""
    out = []
    for ch, (w, lv) in enumerate(SPECS):
        c = pywt.wavedec(block[ch], w, mode="per", level=lv)
        d = c[-1]
        thr = np.median(np.abs(d - np.median(d))) / 0.6745 * k
        y = pywt.waverec([c[0]] + [pywt.threshold(q, value=thr, mode="soft") for q in c[1:]], w, mode="per")[: block.shape[1]]
        out.append(np.asarray(list(extract_features_vectorized(y[TRIM:-TRIM], WIN, HOP, FS).values())))
    return np.vstack(out)


def c_arr(name: str, v, ctype="float32_t") -> str:
    vals = ",".join(f"{float(x):.9e}f" for x in v)
    return f"const {ctype} {name}[{len(v)}] = {{{vals}}};\n"


def prepare() -> dict:
    d = np.load(MCU / "data" / "phase5_raw.npz")
    rng = np.random.default_rng(RNG)
    xs, src = [], []
    for kind in ("norm", "anom"):
        raw = d[f"raw_{kind}"]
        starts = np.sort(rng.integers(TRIM, raw.shape[1] - N - TRIM, K_PER_FILE))
        for s in starts:
            xs.append(raw[:, s:s + N].astype(np.float32))
            src.append((kind, int(s)))
    x = np.stack(xs)                                             # [K, 3, N] float32
    y0, y1 = [], []
    for b in x:
        b64 = b.astype(np.float64)
        f0, _ = process_bearing_signals_to_features(b64, WIN, HOP, FS, TRIM)
        same = ref_with_factor(b64, THR_V0)
        assert np.abs(same - f0).max() == 0.0, "reference with the block factor must equal the repo function"
        y0.append(scaled(f0))
        y1.append(scaled(ref_with_factor(b64, THR_V1)))
    y0, y1 = np.stack(y0), np.stack(y1)
    assert y0.shape == (len(x), T, 27)
    np.savez(MCU / "data" / "stage5_blocks.npz", x=x, y_v0=y0, y_v1=y1, source=np.array([f"{k}:{s}" for k, s in src]), rng=RNG)
    GEN.mkdir(exist_ok=True)
    h = ["/* generated by build_features.py */", "#ifndef FEATURES_DATA_H", "#define FEATURES_DATA_H", '#include "arm_math.h"',
         f"#define BENCH_K {len(x)}U", f"#define BLOCK_N {N}U", f"#define THR_K_V0 {THR_V0:.9e}f", f"#define THR_K_V1 {THR_V1:.9e}f",
         f"extern const uint32_t bench_blocks_u32[{x.size}];"]
    c = ['/* generated by build_features.py: filters from pywt ' + pywt.__version__ + ', scaler from features.py, raw blocks ' +
         str(src) + ' */', '#include "features_data.h"']
    for name in ("sym8", "db4"):
        w = pywt.Wavelet(name)
        F = w.dec_len
        lo, hi = np.array(w.rec_lo), np.array(w.rec_hi)
        arrs = {"dlr": np.array(w.dec_lo)[::-1], "dhr": np.array(w.dec_hi)[::-1], "rle": lo[0::2][::-1], "rlo": lo[1::2][::-1],
                "rhe": hi[0::2][::-1], "rho": hi[1::2][::-1]}
        for k, v in arrs.items():
            h.append(f"extern const float32_t {name}_{k}[{len(v)}];")
            c.append(c_arr(f"{name}_{k}", v))
        assert F in (8, 16)
    h += ["extern const float32_t scaler_mean[27];", "extern const float32_t scaler_std[27];", "#endif"]
    c.append(c_arr("scaler_mean", MEAN))
    c.append(c_arr("scaler_std", STD))
    u = x.reshape(-1).view(np.uint32)
    c.append(f"__attribute__((aligned(32))) const uint32_t bench_blocks_u32[{x.size}] = {{")
    for s in range(0, len(u), 12):
        c.append(",".join(f"0x{v:08x}" for v in u[s:s + 12]) + ",")
    c.append("};")
    (GEN / "features_data.h").write_text("\n".join(h) + "\n", encoding="ascii")
    (GEN / "features_data.c").write_text("\n".join(c) + "\n", encoding="ascii")
    print(f"{len(x)} blocks {x.shape}, crc32={zlib.crc32(x.tobytes()):08x}, sources {src}; "
          f"V0 vs V1 reference max abs diff {np.abs(y0 - y1).max():.3f} (scaled)", flush=True)
    return dict(blocks_B=x.size * 4)


def build(tag: str, info: dict) -> list[dict]:
    variant = {"features_v0": 0, "features_v1": 1}[tag]
    bld = HERE / "build" / tag
    shutil.rmtree(bld, ignore_errors=True)
    bld.mkdir(parents=True)
    src = [HERE / "main.c", GEN / "features_data.c", VAL / "Core" / "Src" / "stm32h7xx_it.c",
           VAL / "Core" / "Src" / "stm32h7xx_hal_msp.c"]
    src += [DSP / "Source" / s for s in DSP_SRC]
    src += [DEV / "Source" / "Templates" / "system_stm32h7xx_dualcore_bootcm7_cm4gated.c",
            DEV / "Source" / "Templates" / "gcc" / "startup_stm32h755xx.s"]
    src += [HAL / "Src" / f"stm32h7xx_hal{s}.c" for s in HAL_SRC]
    inc = [VAL / "Core" / "Inc", GEN, HAL / "Inc", HAL / "Inc" / "Legacy", DEV / "Include", CORE, DSP / "Include",
           DSP / "PrivateInclude"]
    elf = bld / f"bench_{tag}.elf"
    defs = DEFS + [f"-DBENCH_TAG={tag}", f"-DBENCH_VARIANT={variant}"]
    cmd = [GCC_BIN / "arm-none-eabi-gcc", *CPU, *OPT, *defs, "-std=c11", "-Wall", "-Wdouble-promotion", "-ffunction-sections",
           "-fdata-sections", *[f"-I{p}" for p in inc], *src, "-specs=nano.specs", "-specs=nosys.specs",
           f"-T{VAL / 'STM32H755ZITx_CM7.ld'}", "-lc", "-lm", "-Wl,--gc-sections",
           f"-Wl,-Map={bld / ('bench_' + tag + '.map')}", "-Wl,--print-memory-usage", "-o", elf]
    res = run(cmd)
    (bld / "build.log").write_text(" ".join(map(str, cmd)) + "\n\n" + res.stdout + res.stderr, encoding="utf-8")
    run([GCC_BIN / "arm-none-eabi-objcopy", "-O", "binary", "-S", elf, elf.with_suffix(".bin")])
    sizes = {}
    for line in run([GCC_BIN / "arm-none-eabi-size", "-A", elf]).stdout.splitlines():
        m = re.match(r"^(\.\S+)\s+(\d+)\s+(\d+)", line)
        if m:
            sizes[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    flash = sum(s for k, (s, _) in sizes.items() if k in FLASH_SECTIONS)
    syms = {}
    for line in run([GCC_BIN / "arm-none-eabi-nm", "-S", "--size-sort", "-t", "d", elf]).stdout.splitlines():
        p = line.split()
        if len(p) == 4:
            syms[p[3]] = int(p[1])
    algo_ram = {k: syms.get(k, 0) for k in ALGO_RAM_SYMS}
    missing = [k for k, v in algo_ram.items() if v == 0]
    warn = re.findall(r"^.*main\.c.*warning:.*$", res.stderr, re.M)
    base = dict(date=now(), host=host_tag(), tag=tag, firmware=elf.name, bin_md5=md5(elf.with_suffix(".bin")),
                main_c_md5=md5(HERE / "main.c"), cflags=" ".join(CPU + OPT + defs), flash_B=flash, blocks_B=info["blocks_B"],
                flash_without_blocks_B=flash - info["blocks_B"], ram_raw_block_B=algo_ram["g_block"],
                ram_work_B=sum(algo_ram.values()) - algo_ram["g_block"], main_c_warnings=len(warn))
    rows = [{**base, "check": "flash_le_2MiB", "value": flash, "reference": 2 * 1024 * 1024, "status": verdict(flash <= 2 * 1024 * 1024)},
            {**base, "check": "algo_symbols_found", "value": len(ALGO_RAM_SYMS) - len(missing), "reference": len(ALGO_RAM_SYMS),
             "status": verdict(not missing)}]
    print(f"{tag}: flash {flash / 1024:.1f} KiB (blocks {info['blocks_B'] / 1024:.1f}, rest {(flash - info['blocks_B']) / 1024:.1f}); "
          f"RAM raw block {algo_ram['g_block'] / 1024:.1f} KiB, work {(sum(algo_ram.values()) - algo_ram['g_block']) / 1024:.1f} KiB; "
          f"main.c warnings {len(warn)}", flush=True)
    for w in warn[:12]:
        print("  ", w[-140:])
    return rows


if __name__ == "__main__":
    tags = sys.argv[1:] or ["features_v0", "features_v1"]
    info = prepare()
    rows = []
    for t in tags:
        rows += build(t, info)
    append_csv(OUT, rows, fresh=len(tags) == 2)
    for r in rows:
        print(r["status"], r["firmware"], r["check"], r["value"], r["reference"])
    if any(r["status"] == "FAIL" for r in rows):
        raise SystemExit(1)
