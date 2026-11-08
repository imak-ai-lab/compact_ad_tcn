"""Phase 5 / stage 5c: feature extraction on the Cortex-M7: output check, cycles per block, RAM, energy per block.

Usage (from mcu/):  P5_SUPPLY=VIN .venv/Scripts/python stage5_features.py COM9 [features_v1 features_v0]

Input block: 3 x 3248 samples (1968 under the 120 rows + 640 per trimmed edge, as features.py), 50.75 ms at 64 kHz.
Reference: data/stage5_blocks.npz, the repo's functions in float64 on the same float32 samples (build_features.py).
Metric, as in stages 1 and 3: |device - host| / (|host| + 1) in scaled feature units, tol 1e-3 PASS/FAIL. The plain
|device - host| / |host| is reported as INFO: it has no meaning for scaled features that cross zero.
Latency: DWT around the whole block (scaler included, flash-to-RAM copy of the test block excluded), 4000 blocks.
Clock check against the host wall clock as stage 4. Energy, board level at the chosen supply input, as stage 4c:
(P_run - P_idle) * t_run / N, with P_idle from the stage-4 idle firmware logs; a guarded light boot (see stage4_energy.py).
Rows -> features.csv; per-block cycles -> features_cycles.csv; UART and OWON logs -> cubeai/bench/.
"""
from __future__ import annotations

import csv
import re
import sys
import time
import zlib

import numpy as np

from p5common import MCU, append_csv, host_tag, now, verdict
import stage4_bench as sb
import stage4_energy as e

FW = MCU / "firmware" / "features" / "build"
OUT = MCU / "features.csv"
OUT_CYC = MCU / "features_cycles.csv"
LOG = e.LOG
TOL_REL, TOL_CLK = 1e-3, 0.01
T_BLOCK_GUESS = 0.05
FEATS = ["Mean", "Variance", "P2P", "LineInt", "Skew", "SpecEnergy", "SpecSpread", "WavD2", "WavD1"]


def build_row() -> dict:
    with (MCU / "firmware_features_build.csv").open(encoding="utf-8") as fh:
        return {r["tag"]: r for r in csv.DictReader(fh)}


def run_tag(o: e.Owon, tag: str, builds: dict, p_idle: float, idle_note: str) -> tuple[list[dict], list[dict]]:
    w = np.load(MCU / "data" / "stage5_blocks.npz")
    x, ref = w["x"], w["y_v0" if tag == "features_v0" else "y_v1"]
    K = len(x)
    binf = FW / tag / f"bench_{tag}.bin"
    flash_log = sb.prog("-w", binf, "0x08000000", "-v", "-hardRst")
    (LOG / f"flash_{tag}.log").write_text(flash_log, encoding="utf-8")
    discarded = e.light_boot(o, f"run_{tag}")
    lg = e.Logger(o)
    lg.start()
    marks = {}
    try:
        res = sb.session(tag, sb.vcp(), K, T_BLOCK_GUESS, LOG / f"uart_{tag}.log")
        marks = {"RUN_START": res[5], "RUN_END": res[6]}
    finally:
        a = lg.stop()
        e.save_raw(f"run_{tag}", a, marks)
        e.save_meta(f"run_{tag}", dict(supply=e.SUPPLY, poll_errors=lg.errors, firmware=binf.name, saved=now(),
                                       discarded_resets=discarded))
    hdr, y, crc_in, crc_out_ok, cyc_check, t_start, t_end, run_end, cyc, kk, done = res
    sysclk = int(hdr["sysclk"])
    y = y.reshape(K, 120, 27)
    crc_in_ok = sum(int(c == f"{zlib.crc32(x[k].tobytes()):08x}") for k, c in enumerate(crc_in))

    r_floor = np.abs(y - ref) / (np.abs(ref) + 1.0)
    r_plain = np.abs(y - ref) / np.maximum(np.abs(ref), 1e-9)
    worst = np.unravel_index(int(r_floor.argmax()), r_floor.shape)
    worst_name = f"ch{worst[2] // 9} {FEATS[worst[2] % 9]}"
    per_feat = r_floor.reshape(-1, 27).max(axis=0)
    ms = cyc / sysclk * 1e3
    host_wall, fw_loop = t_end - t_start, int(run_end["loop_ms"]) / 1e3
    clk_dev = abs(host_wall - fw_loop) / fw_loop
    stage = {k: int(done[k]) / len(cyc) / sysclk * 1e3 for k in
             ("denoise_ch0", "features_ch0", "denoise_ch1", "features_ch1", "denoise_ch2", "features_ch2")}
    denoise_ms = stage["denoise_ch0"] + stage["denoise_ch1"] + stage["denoise_ch2"]
    feat_ms = stage["features_ch0"] + stage["features_ch1"] + stage["features_ch2"]

    seg = e.window(a, t_start + e.EDGE_S, t_end - e.EDGE_S)
    pre = e.window(a, t_start - 1.8, t_start - 0.2)
    st, st_pre = e.stats(seg), e.stats(pre)
    t_run = float(cyc.sum()) / sysclk
    e_mj = (st["p"] - p_idle) * t_run / len(cyc) * 1e3
    e_loc = (st["p"] - st_pre["p"]) * t_run / len(cyc) * 1e3
    b = builds[tag]
    base = dict(date=now(), host=host_tag(), firmware=binf.name, bin_md5=b["bin_md5"], cflags=b["cflags"],
                conditions=f"CM7 {sysclk / 1e6:.0f} MHz VOS1 direct SMPS, I/D-cache on, data in DTCM, CM4 off (BCM4=0); {e.SUPPLY_DESC}, ST-LINK USB connected",
                block="3 x 3248 float32 samples -> [120][27] scaled", fs_hz=40000, threshold="block MAD, " +
                ("sqrt(2 ln 3248) (repo function on the block)" if tag == "features_v0" else "sqrt(2 ln 256000) (whole-file factor)"),
                n_blocks=len(cyc), k_distinct=K, discarded_resets=discarded)
    rows = [
        {**base, "check": "windows_crc_device_eq_host", "value": hdr["windows_crc"], "reference": f"{zlib.crc32(x.tobytes()):08x}",
         "tol": 0, "status": verdict(hdr["windows_crc"] == f"{zlib.crc32(x.tobytes()):08x}"), "note": "stored blocks exact"},
        {**base, "check": "per_block_crc_in_ok", "value": crc_in_ok, "reference": K, "tol": 0, "status": verdict(crc_in_ok == K), "note": ""},
        {**base, "check": "per_block_crc_out_ok", "value": crc_out_ok, "reference": K, "tol": 0, "status": verdict(crc_out_ok == K),
         "note": "UART transfer intact"},
        {**base, "check": "max_rel_diff_device_vs_host", "value": f"{r_floor.max():.3e}", "reference": 0, "tol": TOL_REL,
         "status": verdict(r_floor.max() < TOL_REL),
         "note": f"|a-b|/(|b|+1), scaled units, {r_floor.size} values; p99.9 {np.quantile(r_floor, .999):.3e}; worst {worst_name}; "
                 f"max abs {np.abs(y - ref).max():.3e}"},
        {**base, "check": "plain_rel_diff_device_vs_host", "value": f"{np.quantile(r_plain, .999):.3e}", "reference": "", "tol": "",
         "status": "INFO", "note": f"|a-b|/|b|: p99.9 shown, median {np.median(r_plain):.3e}, max {r_plain.max():.3e} (features crossing zero)"},
        {**base, "check": "worst_rel_diff_by_feature", "value": " ".join(f"{v:.1e}" for v in per_feat), "reference": "", "tol": "",
         "status": "INFO", "note": "27 values: ch0 9, ch1 9, ch2 9, order " + ",".join(FEATS)},
        {**base, "check": "ms_per_block", "value": f"{ms.mean():.4f}", "reference": "", "tol": "", "status": "MEASURED",
         "note": f"std {ms.std(ddof=1):.4f} min {ms.min():.4f} max {ms.max():.4f} ms; check-pass mean {np.mean(cyc_check) / sysclk * 1e3:.4f} ms"},
        {**base, "check": "ms_denoise_per_block", "value": f"{denoise_ms:.4f}", "reference": "", "tol": "", "status": "MEASURED",
         "note": f"ch0 {stage['denoise_ch0']:.3f}, ch1 {stage['denoise_ch1']:.3f}, ch2 {stage['denoise_ch2']:.3f} ms"},
        {**base, "check": "ms_features_per_block", "value": f"{feat_ms:.4f}", "reference": "", "tol": "", "status": "MEASURED",
         "note": f"ch0 {stage['features_ch0']:.3f}, ch1 {stage['features_ch1']:.3f}, ch2 {stage['features_ch2']:.3f} ms"},
        {**base, "check": "clock_host_wall_vs_fw_loop", "value": f"{clk_dev:.4f}", "reference": f"{fw_loop:.3f}", "tol": TOL_CLK,
         "status": verdict(clk_dev <= TOL_CLK), "note": f"host wall {host_wall:.3f} s; fw loop {fw_loop:.3f} s (includes the test-block copy); sum DWT {t_run:.3f} s"},
        {**base, "check": "ram_raw_block_KiB", "value": f"{int(b['ram_raw_block_B']) / 1024:.1f}", "reference": "", "tol": "", "status": "MEASURED",
         "note": "the 3-channel raw block; a device holds it anyway (DMA buffer); denoising works in place in it"},
        {**base, "check": "ram_work_KiB", "value": f"{int(b['ram_work_B']) / 1024:.1f}", "reference": "", "tol": "", "status": "MEASURED",
         "note": "coefficients, FIR extension, output rows, FFT, from the symbol sizes in the ELF"},
        {**base, "check": "stack_used_B", "value": int(done["stack_used"]), "reference": "", "tol": "", "status": "MEASURED",
         "note": "high-water mark of the whole firmware: painted at start, scanned after the run"},
        {**base, "check": "flash_algo_plus_firmware_KiB", "value": f"{int(b['flash_without_blocks_B']) / 1024:.1f}", "reference": "", "tol": "", "status": "MEASURED",
         "note": "image minus the stored test blocks: algorithm, CMSIS-DSP, HAL, UART, start-up"},
        {**base, "check": "energy_per_block_mJ_board", "value": f"{e_mj:.4f}", "reference": "", "tol": "", "status": "MEASURED",
         "note": f"(P_run - P_idle) * t_run / N: P_run {st['p'] * 1e3:.2f} mW over {st['t']:.1f} s ({st['n']} samples), P_idle {p_idle * 1e3:.2f} mW ({idle_note})"},
        {**base, "check": "energy_per_block_mJ_board_local_baseline", "value": f"{e_loc:.4f}", "reference": "", "tol": "", "status": "INFO",
         "note": f"baseline = 2 s WFI step of this firmware, {st_pre['p'] * 1e3:.2f} mW ({st_pre['n']} samples)"},
        {**base, "check": "run_window_ge_60s", "value": f"{st['t']:.1f}", "reference": "", "tol": e.MIN_RUN_S,
         "status": verdict(st["t"] >= e.MIN_RUN_S), "note": "s"},
    ]
    cyc_rows = [dict(date=base["date"], firmware=binf.name, i=i, k=int(kk[i]), cycles=int(cyc[i]), sysclk_hz=sysclk) for i in range(len(cyc))]
    print(f"{tag}: {ms.mean():.3f} ± {ms.std(ddof=1):.3f} ms/block (denoise {denoise_ms:.3f}, features {feat_ms:.3f}); max rel {r_floor.max():.2e} ({worst_name}); "
          f"E {e_mj:.3f} mJ; stack {done['stack_used']} B; clock dev {clk_dev:.4f}", flush=True)
    return rows, cyc_rows


def main(port: str, tags: list[str]):
    ob = sb.option_bytes()
    if ob["BCM4"] != 0:
        raise SystemExit(f"BCM4 = {ob['BCM4']} (must be 0). Nothing flashed.")
    if e.SUPPLY not in e.SUPPLIES:
        raise SystemExit("set P5_SUPPLY=EXT or P5_SUPPLY=VIN")
    o = e.Owon(port)
    st = e.check_supply(o)
    print("supply:", e.SUPPLY, st, flush=True)
    if not st["ok"]:
        raise SystemExit("supply out of range or output off: nothing flashed")
    idle = []
    for lab in ("pre", "post"):
        marks, a, _ = e.load_owon(f"idle_{lab}")
        idle.append(e.stats(e.window(a, marks["IDLE_START"] + 5.0, marks["IDLE_START"] + 5.0 + e.IDLE_S))["p"])
    p_idle = float(np.mean(idle))
    idle_note = f"stage-4 idle firmware, pre {idle[0] * 1e3:.2f} / post {idle[1] * 1e3:.2f} mW"
    builds = build_row()
    rows, cycs = [], []
    for t in tags:
        r, c = run_tag(o, t, builds, p_idle, idle_note)
        rows += r
        cycs += c
    o.close()
    append_csv(OUT, rows, fresh=len(tags) == 2)
    append_csv(OUT_CYC, cycs, fresh=len(tags) == 2)
    for r in rows:
        print(f"{r['status']:8} {r['firmware'][:20]:20} {r['check']}: {r['value']}  {r['note'][:110]}")
    if any(r["status"] == "FAIL" for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2:] or ["features_v1", "features_v0"])
