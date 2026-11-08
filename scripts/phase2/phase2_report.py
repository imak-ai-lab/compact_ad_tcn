#!/usr/bin/env python3
"""Phase 2 report: CWRU directional-consistency tables (runs on the JupyterServer, no GPU).

    python3.11 scripts/phase2/phase2_report.py                   # runs_phase2 -> phase2_report.md + CSVs
    python3.11 scripts/phase2/phase2_report.py --root /tmp/runs_p2/_smoke --out /tmp/p2rep

Per run (from summary.json + the score dumps):
  auc_window        window-level ROC-AUC on the fold's test split, best checkpoint (= summary test/roc_auc)
  auc_window_last   same from last.pt — model selection without any anomaly (PCA: = best)
  auc_recording     recording-level ROC-AUC: one score per recording = MEDIAN over its 150 windows
  auc_IR / auc_B / auc_OR   window-level AUC of the healthy test recording against one fault type
Per code: mean ± std over folds x seeds, n, and the mean of per-fold means.
Cross-dataset: PU reference (summary_table.csv, seeds 42-44) next to CWRU, family ranking (Spearman
over family-best codes of tier A), and the TL/TF matched pairs.

Protocol caveats are printed into the report header verbatim — they belong in the paper text.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_configs import CODE, PAIRS, SOTA_PU_REF, TIER_A_CODES  # noqa: E402

FAMILY_SHORT = {v: k for k, v in CODE.items()}


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based ROC-AUC (Mann-Whitney), ties get half credit. NaN if one class is missing."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load_runs(root: Path) -> pd.DataFrame:
    rows = []
    for summary_path in sorted(root.rglob("summary.json")):
        run_dir = summary_path.parent
        if "_smoke" in run_dir.parts and "_smoke" not in root.parts:
            continue
        config = json.loads((run_dir / "config.json").read_text())
        summary = json.loads(summary_path.read_text())
        p2 = config.get("phase2")
        if not p2:
            continue
        best = summary.get("best_metrics") or {}
        row = {
            "code": p2["code"], "family": p2["family"], "fold": p2["fold"], "seed": p2["seed"],
            "auc_window": best.get("test/roc_auc"), "val_auc": best.get("val/roc_auc"),
            "best_epoch": summary.get("best_epoch"), "params": summary.get("parameter_count"),
            "size_mib": (summary.get("parameter_count") or 0) * 4 / 2**20, "run_dir": str(run_dir),
        }
        for ckpt in ("best", "last"):
            path = run_dir / f"scores_test_{ckpt}.csv"
            if not path.exists():
                if ckpt == "last":
                    row["auc_window_last"] = row["auc_window"] if summary.get("model_name", "").lower() == "pca" else np.nan
                continue
            sc = pd.read_csv(path)
            if ckpt == "last":
                row["auc_window_last"] = auc(sc["is_anomaly"], sc["score"])
                continue
            row["auc_window_dump"] = auc(sc["is_anomaly"], sc["score"])  # must equal auc_window
            rec = sc.groupby(["file_id", "is_anomaly"], as_index=False)["score"].median()
            row["auc_recording"] = auc(rec["is_anomaly"], rec["score"])
            normal = sc[sc["is_anomaly"] == 0]
            for fault in ("IR", "B", "OR"):
                part = pd.concat([normal, sc[sc["fault"] == fault]])
                row[f"auc_{fault}"] = auc(part["is_anomaly"], part["score"])
            for size in sorted(sc["size_in"].dropna().unique()):
                part = pd.concat([normal, sc[sc["size_in"] == size]])
                row[f"auc_size{int(round(size * 1000)):03d}"] = auc(part["is_anomaly"], part["score"])
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in runs.columns if c.startswith("auc_") and c != "auc_window_dump"]
    out = []
    for code, g in runs.groupby("code"):
        row = {"code": code, "family": g["family"].iloc[0], "n_runs": len(g), "n_folds": g["fold"].nunique(),
               "n_seeds": g["seed"].nunique(), "size_mib_cwru": g["size_mib"].iloc[0]}
        for m in metrics:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std(ddof=1) if len(g) > 1 else np.nan
        fold_means = g.groupby("fold")["auc_window"].mean()
        row["auc_window_foldmean_min"] = fold_means.min()
        row["auc_window_foldmean_max"] = fold_means.max()
        out.append(row)
    return pd.DataFrame(out).sort_values(["family", "code"])


def pu_reference() -> pd.DataFrame:
    df = pd.read_csv(REPO / "summary_table.csv")
    df = df[df["seed"].isin([42, 43, 44])]
    agg = df.groupby(["family", "run_name"], as_index=False).agg(pu_auc=("best_metrics.test/roc_auc", "mean"),
                                                                 pu_auc_std=("best_metrics.test/roc_auc", "std"),
                                                                 pu_params=("parameter_count", "first"))
    agg["code"] = agg["family"].map(CODE) + agg["run_name"].str[:3]
    agg["size_mib_pu"] = agg["pu_params"] * 4 / 2**20
    ref = agg[["code", "pu_auc", "pu_auc_std", "size_mib_pu"]]
    sota = pd.DataFrame([{"code": c, "pu_auc": v[0], "pu_auc_std": v[1] if v[1] is not None else np.nan,
                          "size_mib_pu": np.nan} for c, v in SOTA_PU_REF.items()])
    return pd.concat([ref, sota], ignore_index=True)


def fmt(mean, std=None) -> str:
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "—"
    return f"{mean:.4f}" if std is None or np.isnan(std) else f"{mean:.4f} ± {std:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO / "runs_phase2"))
    parser.add_argument("--out", default=str(REPO))
    parser.add_argument("--tag", default="", help="suffix for output files, e.g. v1b -> phase2_report_v1b.md")
    parser.add_argument("--compare-root", default=None,
                        help="second run tree (e.g. runs_phase2 while --root is runs_phase2_v1b): adds a per-code delta table")
    args = parser.parse_args()
    tag = f"_{args.tag}" if args.tag else ""
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(root)
    if runs.empty:
        raise SystemExit(f"no phase-2 runs under {root}")
    if "auc_window_dump" in runs:
        drift = (runs["auc_window"] - runs["auc_window_dump"]).abs().max()
        assert drift < 1e-3 or np.isnan(drift), f"score dump disagrees with summary ROC-AUC by {drift}"
    summary = aggregate(runs).merge(pu_reference(), on="code", how="left")
    summary["delta_cwru_minus_pu"] = summary["auc_window_mean"] - summary["pu_auc"]
    runs.to_csv(out / f"phase2_runs{tag}.csv", index=False)
    summary.to_csv(out / f"phase2_summary{tag}.csv", index=False)

    lines = ["# Phase 2 — CWRU directional consistency (protocol lolo_v1)", "",
             f"Runs: {len(runs)} ({runs['code'].nunique()} codes, folds {sorted(int(f) for f in runs['fold'].unique())}, "
             f"seeds {sorted(int(s) for s in runs['seed'].unique())}). Source: `{root}`.", "",
             "Protocol (fixed before any run): split by recording; healthy class "
             "held out **by load** (leave-one-load-out, 4 folds — CWRU has a single healthy bearing, so bearing-wise "
             "hold-out of normals is impossible); faulty **bearings** disjoint between validation (0.014\") and test "
             "(0.007/0.021/0.028\"); 150 uniformly spaced windows per evaluated recording; scaler fitted on the fold's "
             "training frames only. Primary metric = window-level ROC-AUC on the held-out load; mean ± std over folds × seeds.", ""]

    # ---- main table
    lines += ["## Per-config results", "",
              "| code | family | size PU (MiB) | size CWRU (MiB) | PU ROC-AUC | CWRU ROC-AUC (window, best) | CWRU last-ckpt | CWRU recording-level (median) | fold-mean range | n |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(f"| {r['code']} | {r['family']} | {fmt(r['size_mib_pu'])} | {r['size_mib_cwru']:.3f} | "
                     f"{fmt(r['pu_auc'], r['pu_auc_std'])} | **{fmt(r['auc_window_mean'], r['auc_window_std'])}** | "
                     f"{fmt(r.get('auc_window_last_mean'), r.get('auc_window_last_std'))} | "
                     f"{fmt(r.get('auc_recording_mean'), r.get('auc_recording_std'))} | "
                     f"{fmt(r['auc_window_foldmean_min'])}–{fmt(r['auc_window_foldmean_max'])} | {int(r['n_runs'])} |")

    if any(c in summary["code"].values for c in SOTA_PU_REF):
        lines += ["", "SOTA rows (tier S): PU reference = phase-1 unified-scoring column (" +
                  "; ".join(f"{c}: {v[2]}" for c, v in SOTA_PU_REF.items() if c in summary["code"].values) +
                  "); trained on CWRU under the project's scoring with a fair checkpoint, USAD min-max range re-fitted on the fold's train frames."]

    # ---- per fault type
    fault_cols = [c for c in ("auc_IR_mean", "auc_B_mean", "auc_OR_mean") if c in summary]
    size_cols = sorted(c for c in summary.columns if c.startswith("auc_size") and c.endswith("_mean"))
    if fault_cols:
        lines += ["", "## Sensitivity by fault type / size (window-level, healthy test recording vs one class)", "",
                  "| code | " + " | ".join(c.replace("auc_", "").replace("_mean", "") for c in fault_cols + size_cols) + " |",
                  "|---|" + "---|" * (len(fault_cols) + len(size_cols))]
        for _, r in summary.iterrows():
            lines.append(f"| {r['code']} | " + " | ".join(fmt(r[c]) for c in fault_cols + size_cols) + " |")
        lines += ["", "Smith & Randall (2015) found rolling-element (B) faults and several 0.014\" records hard to "
                  "diagnose even with envelope analysis; a low B column is expected, not a pipeline defect."]

    # ---- family ranking
    tier_a = summary[summary["code"].isin(TIER_A_CODES)].dropna(subset=["pu_auc"])
    fam_best = tier_a.sort_values("pu_auc", ascending=False).groupby("family", as_index=False).first()
    if len(fam_best) >= 3:
        rho = fam_best[["pu_auc", "auc_window_mean"]].corr(method="spearman").iloc[0, 1]
        lines += ["", "## Family ranking transfer (tier A, best PU config per family)", "",
                  f"Spearman ρ between PU and CWRU family-best ROC-AUC over {len(fam_best)} families: **{rho:.3f}**", "",
                  "| family | code | PU | CWRU | PU rank | CWRU rank |", "|---|---|---|---|---|---|"]
        fam_best["pu_rank"] = fam_best["pu_auc"].rank(ascending=False).astype(int)
        fam_best["cwru_rank"] = fam_best["auc_window_mean"].rank(ascending=False).astype(int)
        for _, r in fam_best.sort_values("pu_rank").iterrows():
            lines.append(f"| {r['family']} | {r['code']} | {r['pu_auc']:.4f} | {fmt(r['auc_window_mean'], r['auc_window_std'])} | {r['pu_rank']} | {r['cwru_rank']} |")

    # ---- TL vs TF pairs
    by_code = summary.set_index("code")
    pair_rows = []
    for tl, tf, cell in PAIRS:
        if tl in by_code.index and tf in by_code.index:
            a, b = by_code.loc[tl], by_code.loc[tf]
            pair_rows.append(f"| {cell} | {tl} / {tf} | {a['size_mib_pu']:.3f} / {b['size_mib_pu']:.3f} | "
                             f"{a['size_mib_cwru']:.3f} / {b['size_mib_cwru']:.3f} | {a['pu_auc']:.4f} / {b['pu_auc']:.4f} ({a['pu_auc'] - b['pu_auc']:+.4f}) | "
                             f"{fmt(a['auc_window_mean'], a['auc_window_std'])} / {fmt(b['auc_window_mean'], b['auc_window_std'])} "
                             f"({a['auc_window_mean'] - b['auc_window_mean']:+.4f}) |")
    if pair_rows:
        lines += ["", "## TL vs TF matched-body pairs (contribution 2)", "",
                  "| cell | TL / TF | size PU | size CWRU | PU ROC-AUC (Δ TL−TF) | CWRU ROC-AUC (Δ TL−TF) |", "|---|---|---|---|---|---|", *pair_rows]

    # ---- variant comparison (v1b control vs v1)
    if args.compare_root:
        other = aggregate(load_runs(Path(args.compare_root)))
        both = summary.merge(other[["code", "auc_window_mean", "auc_window_std", "auc_B_mean"]], on="code",
                             suffixes=("", "_ref"))
        if not both.empty:
            lines += ["", f"## Variant comparison: `{root.name}` (this) vs `{Path(args.compare_root).name}` (ref)", "",
                      "| code | this ROC-AUC | ref ROC-AUC | Δ (this − ref) | this B | ref B |", "|---|---|---|---|---|---|"]
            for _, r in both.iterrows():
                lines.append(f"| {r['code']} | {fmt(r['auc_window_mean'], r['auc_window_std'])} | "
                             f"{fmt(r['auc_window_mean_ref'], r['auc_window_std_ref'])} | "
                             f"{r['auc_window_mean'] - r['auc_window_mean_ref']:+.4f} | {fmt(r.get('auc_B_mean'))} | {fmt(r['auc_B_mean_ref'])} |")
            lines += ["", "Read: |Δ| within one std → the acquisition-chain confound is excluded; a systematic drop → "
                      "the common-low-pass variant becomes the headline protocol."]

    lines += ["", "## Caveats to carry into §VI", "",
              "- The healthy test data of every fold is ONE recording (~5–10 s); the four folds together cover the four "
              "published healthy recordings of a single bearing. Fold-to-fold spread is the honest uncertainty, not seed spread.",
              "- Normal validation frames are the last 20 % of the training recordings (time-disjoint, not recording-disjoint); "
              "they only fit the scorer/threshold and select the checkpoint. Compare the *last-ckpt* column: selection without anomalies.",
              "- Recording-level AUC has 1 healthy vs 12 faulty recordings per fold — coarse by construction; it is reported, not headlined.",
              "- This is a directional-consistency check of configs tuned on PU, not a CWRU benchmark entry."]
    (out / f"phase2_report{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten: {out / f'phase2_report{tag}.md'}, phase2_summary{tag}.csv, phase2_runs{tag}.csv")


if __name__ == "__main__":
    main()
