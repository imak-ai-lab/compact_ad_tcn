"""Phase 5 / stage 4e: one firmware, one measured decision: raw block -> features -> inference -> (target features) -> anomaly score.

Usage (from mcu/):
    P5_SUPPLY=VIN .venv/Scripts/python stage4e_e2e.py run COM9 [segment ...]    # default: warm idle_pre, six artifacts, idle_post
    P5_SUPPLY=VIN .venv/Scripts/python stage4e_e2e.py analyze                   # e2e.csv from the saved session files, no hardware

Run: the guarded light boot and the OWON logger of stage4_energy.py, the UART session of stage4_bench.py. Per artifact: session arrays -> cubeai/bench/e2e_<tag>.npz, supply
log -> cubeai/bench/owon_e2e_<tag>.csv. The supply stays read-only.

Checks of analyze -> e2e.csv (metric |a-b|/(|b|+1), scaled feature units, as stages 1, 3, 5):
  stored segments CRC; features on the device vs host reference (input rows, target rows), tol 1e-3;
  network output vs ORT fed with the device's own input rows: fp32 tol 1e-4, int8 INFO (mixed int8 against the ORT file);
  score on the device vs the score recomputed on the host from the device's output and target, rel tol 1e-3: all six, a test of the C scoring;
  clock against the host wall clock tol 1 %; run window >= 60 s; WFI step vs P_idle within 15 mW; two energy baselines within 3 %.
Measured: ms per decision, per stage; energy per decision, board level; RAM: static sections of the image without the cycle buffer, plus measured stack and heap.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import time
import zlib

import numpy as np
import onnxruntime as ort
import serial

from p5common import MCU, MODELS, append_csv, host_tag, now, verdict
import stage4_bench as sb
import stage4_energy as e

FW = MCU / "firmware" / "e2e" / "build"
LOG = e.LOG
OUT = MCU / "e2e.csv"
OUT_CYC = MCU / "e2e_cycles.csv"
TOL_FEAT, TOL_ORT, TOL_SCORE, TOL_CLK = 1e-3, 1e-4, 1e-3, 0.01
ALL_TAGS = [f"{c.lower()}_{a}" for c in MODELS for a in ("fp32", "int8")]
STAGES = ["den_in", "fea_in", "trans", "infer", "den_tgt", "fea_tgt", "score"]


def rel(a, b):
    return float((np.abs(a - b) / (np.abs(b) + 1.0)).max())


def run_segment(o: e.Owon, tag: str):
    binf = FW / tag / f"bench_e2e_{tag}.bin"
    sb.prog("-w", binf, "0x08000000", "-v", "-hardRst")
    for attempt in (1, 2):
        if attempt > 1:
            sb.prog("-hardRst")
        discarded = e.light_boot(o, f"e2e_{tag}")
        lg = e.Logger(o)
        lg.start()
        marks = {}
        try:
            res = sb.session(f"e2e_{tag}", sb.vcp(), 8, 1.0, LOG / f"uart_e2e_{tag}.log")
            marks = {"RUN_START": res[5], "RUN_END": res[6]}
            hdr, y, crc_in, crc_out_ok, cyc_check, t_start, t_end, run_end, cyc, kk, done = res
            np.savez(LOG / f"e2e_{tag}.npz", y=y, cyc=cyc, kk=kk, cyc_check=np.asarray(cyc_check), crc_in=np.asarray(crc_in), crc_out_ok=crc_out_ok, t_start=t_start, t_end=t_end,
                     meta=json.dumps(dict(hdr=hdr, run_end=run_end, done=done, attempt=attempt, discarded_resets=discarded, poll_errors=lg.errors, saved=now())))
            print(f"{tag}: {cyc.mean() / int(hdr['sysclk']) * 1e3:.2f} ms/decision, stack {done['stack_used']} heap {done['heap_used']}", flush=True)
            return
        except (serial.SerialException, TimeoutError, OSError, RuntimeError) as ex:
            print(f"{tag}: session {attempt} failed: {str(ex)[:140]}", flush=True)
            if attempt == 2:
                raise
        finally:
            a = lg.stop()
            e.save_raw(f"e2e_{tag}" if marks else f"e2e_{tag}_failed{attempt}", a, marks)


def cmd_run(port: str, segs: list[str]):
    ob = sb.option_bytes()
    if ob["BCM4"] != 0:
        raise SystemExit(f"BCM4 = {ob['BCM4']} (must be 0). Nothing flashed.")
    if e.SUPPLY not in e.SUPPLIES:
        raise SystemExit("set P5_SUPPLY=EXT or P5_SUPPLY=VIN")
    o = e.Owon(port)
    st = e.check_supply(o)
    print("supply:", e.SUPPLY, st, flush=True)
    if not st["ok"]:
        raise SystemExit("supply out of range or output off: nothing flashed (set it at the front panel)")
    for s in segs:
        if s == "warm":
            old = LOG / "owon_warm.csv"
            if old.exists():
                shutil.copy2(old, LOG / "owon_warm_stage4.csv")
            e.warm_segment(o)
        elif s.startswith("idle_"):
            e.idle_segment(o, "e2e_" + s.removeprefix("idle_"))
        else:
            run_segment(o, s)
        print(f"{s}: done {time.strftime('%H:%M:%S')}", flush=True)
    print("supply at end:", e.check_supply(o), flush=True)
    o.close()


def previous_parts(name: str) -> dict:
    with (MCU / "latency_dwt.csv").open(encoding="utf-8") as fh:
        lat = [float(r["value"]) for r in csv.DictReader(fh) if r["artifact"] == name and r["check"] == "latency_ms_mean"][0]
    with (MCU / "features.csv").open(encoding="utf-8") as fh:
        fea = [float(r["value"]) for r in csv.DictReader(fh) if r["firmware"] == "bench_features_v1.bin" and r["check"] == "ms_per_block"][0]
    return dict(lat=lat, fea=fea)


def cmd_analyze(tags: list[str]):
    refs = np.load(MCU / "data" / "stage4e_refs.npz")
    with (MCU / "firmware_e2e_build.csv").open(encoding="utf-8") as fh:
        builds = {r["tag"]: r for r in csv.DictReader(fh)}
    idle = []
    for lab in ("e2e_pre", "e2e_post"):
        if (LOG / f"owon_idle_{lab}.csv").exists():
            marks, a, _ = e.load_owon(f"idle_{lab}")
            idle.append(e.stats(e.window(a, marks["IDLE_START"] + 5.0, marks["IDLE_START"] + 5.0 + e.IDLE_S))["p"])
        else:
            print(f"WARNING: no idle_{lab} log (only acceptable for a trial)", flush=True)
    if len(idle) == 1:
        idle.append(idle[0])
    p_idle = float(np.mean(idle))
    drift = abs(idle[0] - idle[1]) * 1e3
    rows, cyc_rows = [], []
    for tag in tags:
        code, art = tag.split("_")
        code = code.upper()
        name = f"{code}_{art}.onnx"
        fc = code == "TL082"
        z = np.load(LOG / f"e2e_{tag}.npz")
        meta = json.loads(str(z["meta"]))
        hdr, done, run_end = meta["hdr"], meta["done"], meta["run_end"]
        sysclk = int(hdr["sysclk"])
        x = refs["x_tl" if fc else "x_tr"]
        K = len(x)
        y = z["y"]
        t_len = 240 if fc else 120
        n_in, n_tgt = 120 * 27, (240 * 27 if fc else 0)
        rows_in = y[:, :n_in].reshape(K, 120, 27)
        rows_tgt = y[:, n_in:n_in + n_tgt].reshape(K, 240, 27) if fc else rows_in
        out = y[:, n_in + n_tgt:n_in + n_tgt + 27 * t_len].reshape(K, 27, t_len)
        score = y[:, -1].astype(np.float64)
        assert y.shape[1] == n_in + n_tgt + 27 * t_len + 1, y.shape

        crc_in_ok = sum(int(c == f"{zlib.crc32(x[k].tobytes()):08x}") for k, c in enumerate(z["crc_in"]))
        r_in = rel(rows_in, refs["y_in_tl" if fc else "y_in_tr"])
        r_tgt = rel(rows_tgt, refs["y_tgt_tl"]) if fc else float("nan")
        sess = ort.InferenceSession(str(MCU / "models" / name), providers=["CPUExecutionProvider"])
        nin = sess.get_inputs()[0].name
        y_o = np.concatenate([sess.run(None, {nin: np.ascontiguousarray(rows_in[k].T[None])})[0] for k in range(K)])
        r_ort = rel(out, y_o)
        inv = np.load(MCU / "models" / f"{code}_{art}_cmodel_scorer.npz")["inv_cov"].astype(np.float64)
        err = np.abs(out.astype(np.float64) - rows_tgt.transpose(0, 2, 1).astype(np.float64)).mean(axis=2)
        s_host = np.einsum("ni,ij,nj->n", err, inv, err)
        r_score = float((np.abs(score - s_host) / np.abs(s_host)).max())

        cyc = z["cyc"]
        ms = cyc / sysclk * 1e3
        t_run = float(cyc.sum()) / sysclk
        st_ms = {s: int(done[s]) / len(cyc) / sysclk * 1e3 for s in STAGES}
        wall, fw_loop = float(z["t_end"]) - float(z["t_start"]), int(run_end["loop_ms"]) / 1e3
        clk_dev = abs(wall - fw_loop) / fw_loop
        marks, a, _ = e.load_owon(f"e2e_{tag}")
        seg = e.stats(e.window(a, marks["RUN_START"] + e.EDGE_S, marks["RUN_END"] - e.EDGE_S))
        pre = e.stats(e.window(a, marks["RUN_START"] - 1.8, marks["RUN_START"] - 0.2))
        e_mj = (seg["p"] - p_idle) * t_run / len(cyc) * 1e3
        e_loc = (seg["p"] - pre["p"]) * t_run / len(cyc) * 1e3
        b = builds[tag]
        stack, heap = int(done["stack_used"]), int(done["heap_used"])
        peak = int(b["static_ram_B"]) + stack + heap
        prev = previous_parts(name)
        fea_ms = st_ms["den_in"] + st_ms["fea_in"]
        fea_all = fea_ms + st_ms["den_tgt"] + st_ms["fea_tgt"]
        base = dict(date=now(), host=host_tag(), code=code, artifact=name, seed=b["seed"], firmware=b["firmware"], bin_md5=b["bin_md5"], n=len(cyc), k_distinct=K, attempt=meta["attempt"],
                    discarded_resets=meta["discarded_resets"], poll_errors=meta["poll_errors"],
                    conditions=f"CM7 {sysclk / 1e6:.0f} MHz VOS1 direct SMPS, I/D-cache on, CM4 off; {e.SUPPLY_DESC}, ST-LINK USB connected")
        R = rows.append
        R({**base, "check": "windows_crc_device_eq_host", "value": hdr["windows_crc"], "reference": f"{zlib.crc32(x.tobytes()):08x}", "tol": 0, "status": verdict(hdr["windows_crc"] == f"{zlib.crc32(x.tobytes()):08x}"), "note": ""})
        R({**base, "check": "per_segment_crc_in_ok", "value": crc_in_ok, "reference": K, "tol": 0, "status": verdict(crc_in_ok == K), "note": ""})
        R({**base, "check": "per_segment_crc_out_ok", "value": int(z["crc_out_ok"]), "reference": K, "tol": 0, "status": verdict(int(z["crc_out_ok"]) == K), "note": "UART transfer intact"})
        R({**base, "check": "features_in_vs_host", "value": f"{r_in:.3e}", "reference": 0, "tol": TOL_FEAT, "status": verdict(r_in < TOL_FEAT), "note": "|a-b|/(|b|+1), scaled"})
        if fc:
            R({**base, "check": "features_target_vs_host", "value": f"{r_tgt:.3e}", "reference": 0, "tol": TOL_FEAT, "status": verdict(r_tgt < TOL_FEAT), "note": "|a-b|/(|b|+1), scaled"})
        R({**base, "check": "output_vs_ort_on_device_features", "value": f"{r_ort:.3e}", "reference": 0 if art == "fp32" else "", "tol": TOL_ORT if art == "fp32" else "",
           "status": verdict(r_ort < TOL_ORT) if art == "fp32" else "INFO", "note": "" if art == "fp32" else "target is mixed int8/fp32, ORT file all-int8"})
        R({**base, "check": "score_vs_host_from_device_output", "value": f"{r_score:.3e}", "reference": 0, "tol": TOL_SCORE, "status": verdict(r_score < TOL_SCORE), "note": "rel diff, C scoring against the formula in float64"})
        R({**base, "check": "ms_per_decision", "value": f"{ms.mean():.5f}", "reference": "", "tol": "", "status": "MEASURED",
           "note": f"std {ms.std(ddof=1):.5f} min {ms.min():.5f} max {ms.max():.5f} ms; check-pass mean {z['cyc_check'].mean() / sysclk * 1e3:.3f} ms"})
        R({**base, "check": "ms_features_all", "value": f"{fea_all:.5f}", "reference": "", "tol": "", "status": "MEASURED",
           "note": f"input: denoise {st_ms['den_in']:.3f}, rows {st_ms['fea_in']:.3f}" + (f"; target: denoise {st_ms['den_tgt']:.3f}, rows {st_ms['fea_tgt']:.3f}" if fc else "") + f"; stage-5 firmware, DTCM data: {prev['fea']:.3f} per input block"})
        R({**base, "check": "ms_inference", "value": f"{st_ms['infer']:.5f}", "reference": f"{prev['lat']:.3f}", "tol": "", "status": "MEASURED", "note": "reference = stage 4 firmware; caches shared with the features here"})
        R({**base, "check": "ms_transpose_and_score", "value": f"{st_ms['trans'] + st_ms['score']:.5f}", "reference": "", "tol": "", "status": "MEASURED", "note": f"transpose {st_ms['trans']:.3f}, score {st_ms['score']:.3f}"})
        R({**base, "check": "ratio_measured_vs_sum_of_earlier_parts", "value": f"{ms.mean() / (prev['lat'] + prev['fea']):.3f}", "reference": f"{prev['lat'] + prev['fea']:.2f}", "tol": "", "status": "INFO",
           "note": "stage 4 inference + stage 5 input features" + ("; the sum has no target features" if fc else "")})
        R({**base, "check": "clock_host_wall_vs_fw_loop", "value": f"{clk_dev:.4f}", "reference": f"{fw_loop:.3f}", "tol": TOL_CLK, "status": verdict(clk_dev <= TOL_CLK), "note": f"host wall {wall:.3f} s; fw loop {fw_loop:.3f} s; sum DWT {t_run:.3f} s"})
        R({**base, "check": "peak_ram_KiB", "value": f"{peak / 1024:.1f}", "reference": "", "tol": "", "status": "MEASURED",
           "note": f"static {int(b['static_ram_B']) / 1024:.1f} KiB (.data + .bss + AXI pool, without the cycle buffer) + stack {stack} B + heap {heap} B; activations {int(b['activations_B']) / 1024:.1f}, feature buffers {int(b['feature_buffers_B']) / 1024:.1f} KiB"})
        R({**base, "check": "flash_without_stored_blocks_KiB", "value": f"{int(b['flash_without_blocks_B']) / 1024:.1f}", "reference": "", "tol": "", "status": "MEASURED", "note": "weights, runtime, features, HAL, UART"})
        R({**base, "check": "energy_per_decision_mJ_board", "value": f"{e_mj:.5f}", "reference": "", "tol": "", "status": "MEASURED", "note": f"P_run {seg['p'] * 1e3:.2f} mW over {seg['t']:.1f} s ({seg['n']} samples), P_idle {p_idle * 1e3:.2f} mW; dP {(seg['p'] - p_idle) * 1e3:.2f} mW"})
        R({**base, "check": "energy_local_baseline_mJ", "value": f"{e_loc:.5f}", "reference": "", "tol": "", "status": "INFO", "note": f"baseline = 2 s WFI step of this firmware, {pre['p'] * 1e3:.2f} mW ({pre['n']} samples)"})
        R({**base, "check": "energy_local_vs_idle_baseline", "value": f"{(e_loc - e_mj) / e_mj:+.4f}", "reference": "", "tol": 0.03, "status": verdict(abs(e_loc - e_mj) <= 0.03 * abs(e_mj)), "note": "the two baselines must agree"})
        R({**base, "check": "run_window_ge_60s", "value": f"{seg['t']:.1f}", "reference": "", "tol": e.MIN_RUN_S, "status": verdict(seg["t"] >= e.MIN_RUN_S), "note": "s"})
        R({**base, "check": "boot_state_wfi_step_vs_idle_mW", "value": f"{(pre['p'] - p_idle) * 1e3:.2f}", "reference": "", "tol": e.TOL_IDLE_DRIFT_MW, "status": verdict(abs(pre["p"] - p_idle) * 1e3 <= e.TOL_IDLE_DRIFT_MW), "note": f"discarded resets {meta['discarded_resets']}"})
        R({**base, "check": "idle_drift_mW", "value": f"{drift:.2f}", "reference": "", "tol": e.TOL_IDLE_DRIFT_MW, "status": verdict(drift <= e.TOL_IDLE_DRIFT_MW), "note": f"pre {idle[0] * 1e3:.2f} / post {idle[1] * 1e3:.2f} mW"})
        cyc_rows += [dict(firmware=b["firmware"], artifact=name, i=i, k=int(z["kk"][i]), cycles=int(cyc[i]), sysclk_hz=sysclk) for i in range(len(cyc))]
        print(f"{tag}: {ms.mean():.2f} ± {ms.std(ddof=1):.2f} ms; features {fea_all:.2f}, infer {st_ms['infer']:.2f} (stage 4 {prev['lat']:.2f}), score+transpose {st_ms['trans'] + st_ms['score']:.3f}; peak RAM {peak / 1024:.1f} KiB; E {e_mj:.2f} mJ ({e_loc:.2f} local); "
              f"rel: features {r_in:.1e}, output {r_ort:.1e}, score {r_score:.1e}", flush=True)
    append_csv(OUT, rows, fresh=True)
    append_csv(OUT_CYC, cyc_rows, fresh=True)
    fail = [r for r in rows if r["status"] == "FAIL"]
    for r in fail:
        print(f"FAIL {r['artifact']} {r['check']}: {r['value']} (tol {r['tol']}) {r['note'][:90]}")
    print(f"{len(rows)} rows, FAIL: {len(fail)}")
    return len(fail)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "analyze"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "run":
        cmd_run(sys.argv[2], sys.argv[3:] or ["warm", "idle_pre", *ALL_TAGS, "idle_post"])
    else:
        raise SystemExit(1 if cmd_analyze(sys.argv[2:] or ALL_TAGS) else 0)
