"""
Phase 1: aggregate the SOTA baseline runs into the table that goes into §VI.

Reads runs_phase1/<family>/<run>/summary.json (+ optional rescore_mahalanobis.json)
and produces, per config: ROC-AUC mean+/-std over seeds under the author's own
score, the same under this project's Mahalanobis scorer, parameter count and
model size in MiB.

If runs_phase0/phase0_summary.csv exists, the paper's headline models are appended
as reference rows, so the size/quality contrast is visible in one table.

Usage (from repo root):
    python scripts/phase1/phase1_report.py

Outputs: runs_phase1/phase1_summary.csv, runs_phase1/phase1_report.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs_phase1"
PHASE0_SUMMARY = REPO / "runs_phase0" / "phase0_summary.csv"

MIB = 1024 * 1024


def collect() -> pd.DataFrame:
    rows = []
    for summary_path in sorted(RUNS.glob("*/*/summary.json")):
        summary = json.loads(summary_path.read_text())
        if summary.get("error"):
            continue
        config = json.loads((summary_path.parent / "config.json").read_text())
        best = summary.get("best_metrics", {})

        def _roc(name):
            path = summary_path.parent / name
            return json.loads(path.read_text())["metrics"].get("test/roc_auc") if path.exists() else None

        rescore_roc = _roc("rescore_mahalanobis.json")
        unified_roc = _roc("rescore_unified.json")

        phase1 = config.get("phase1", {})
        rows.append({
            "code": phase1.get("code", "?"),
            "family": summary_path.parent.parent.name,
            "tag": phase1.get("tag", ""),
            "seed": int(config["run"]["seed"]),
            "roc_auc_native": best.get("test/roc_auc"),
            "roc_auc_mahalanobis": rescore_roc,
            "roc_auc_unified": unified_roc,
            "val_roc_auc_native": best.get("val/roc_auc"),
            "parameter_count": summary.get("parameter_count"),
            "parameter_size_bytes": summary.get("parameter_size_bytes"),
            "best_epoch": summary.get("best_epoch"),
            "train_seconds": summary.get("total_time_seconds"),
        })
    if not rows:
        sys.exit(f"no phase-1 summaries under {RUNS} — did the shards run?")
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["code", "family", "tag"], as_index=False).agg(
        n_seeds=("seed", "count"),
        roc_native_mean=("roc_auc_native", "mean"),
        roc_native_std=("roc_auc_native", "std"),
        roc_maha_mean=("roc_auc_mahalanobis", "mean"),
        roc_maha_std=("roc_auc_mahalanobis", "std"),
        roc_unified_mean=("roc_auc_unified", "mean"),
        roc_unified_std=("roc_auc_unified", "std"),
        parameter_count=("parameter_count", "max"),
        size_bytes=("parameter_size_bytes", "max"),
        train_seconds_mean=("train_seconds", "mean"),
    )
    grouped["size_mib"] = (grouped["size_bytes"] / MIB).round(3)
    consistent = df.groupby(["code"])["parameter_count"].nunique().max()
    if consistent and consistent > 1:
        print("WARNING: parameter_count differs across seeds of the same config", file=sys.stderr)
    for column in ["roc_native_mean", "roc_native_std", "roc_maha_mean", "roc_maha_std",
                   "roc_unified_mean", "roc_unified_std"]:
        grouped[column] = grouped[column].round(4)
    return grouped.sort_values(["family", "code"])


def paper_reference_rows() -> pd.DataFrame | None:
    if not PHASE0_SUMMARY.exists():
        return None
    phase0 = pd.read_csv(PHASE0_SUMMARY)
    phase0 = phase0[phase0["status"] != "MISSING"]
    best = phase0.loc[phase0.groupby("family")["roc_mean_5seed"].idxmax()]
    return best[["code", "family", "run_name", "roc_mean_5seed", "roc_std_5seed"]].sort_values(
        "roc_mean_5seed", ascending=False
    )


def main() -> None:
    df = collect()
    table = aggregate(df)
    df.to_csv(RUNS / "phase1_runs.csv", index=False)
    table.to_csv(RUNS / "phase1_summary.csv", index=False)

    report = ["# Phase 1 — published baselines under the paper protocol\n"]
    report.append(
        "Same PU split, same 120x27 feature windows, same optimizer/schedule/checkpoint rule as the "
        "paper's own families. Architecture, training objective and anomaly score are the authors'.\n"
    )
    report.append(
        "**ROC-AUC (test), mean +/- std over seeds.** `native` = the author's own score; "
        "`mahalanobis` = our scorer on the method's own error map; `unified` = our full "
        "scoring pipeline (|x_hat - x| averaged over time -> 27 channel features -> "
        "Mahalanobis), identical for every model. The unified column is the like-for-like "
        "comparison: our headline numbers also carry a large scorer contribution "
        "(TR138 0.9431 with Mahalanobis vs 0.8501 with plain averaging), so a table that "
        "pairs our model+our scorer against their model+their scorer overstates the "
        "architectural gap.\n")
    report.append(table.to_markdown(index=False))

    reference = paper_reference_rows()
    if reference is not None:
        report.append("\n## Paper's own families (phase-0 5-seed reference)\n")
        report.append(reference.to_markdown(index=False))
        # Rank baselines by the UNIFIED column, not the native one. The native score
        # is each method's own and is not comparable across methods (that is the whole
        # phase-1 finding); the paper's comparison is like-for-like on unified.
        # Two headline numbers, because the paper's claim is two-layered:
        # the best baseline inside the compact budget, and the best at any budget.
        BUDGET_MIB = 1.0
        ranked = table.dropna(subset=["roc_unified_mean"])
        best_sota = ranked.loc[ranked["roc_unified_mean"].idxmax()]
        in_budget = ranked[ranked["size_mib"] <= BUDGET_MIB]
        best_paper = reference.iloc[0]
        if not in_budget.empty:
            best_budget = in_budget.loc[in_budget["roc_unified_mean"].idxmax()]
            report.append(
                f"\n- best baseline within {BUDGET_MIB:g} MiB: **{best_budget['code']} {best_budget['family']}** — "
                f"unified ROC-AUC {best_budget['roc_unified_mean']:.4f} +/- {best_budget['roc_unified_std']:.4f}, "
                f"{best_budget['size_mib']:.3f} MiB"
            )
        report.append(
            f"- best baseline at any budget: **{best_sota['code']} {best_sota['family']}** — "
            f"unified ROC-AUC {best_sota['roc_unified_mean']:.4f} +/- {best_sota['roc_unified_std']:.4f}, "
            f"{best_sota['size_mib']:.3f} MiB"
        )
        report.append(
            f"- best paper model: **{best_paper['code']} {best_paper['family']}** — "
            f"ROC-AUC {best_paper['roc_mean_5seed']:.4f}"
        )

    incomplete = table[table["n_seeds"] < 3]
    if not incomplete.empty:
        report.append(f"\n> WARNING: {len(incomplete)} config(s) have fewer than 3 seeds: "
                      f"{', '.join(incomplete['code'])}")
    # A config trained under the project's own scoring (TA006/TA011/US006/AT004) has no
    # author error map, so its `mahalanobis` cell is empty BY DESIGN and its `native`
    # cell already IS the unified number. Do not nag about those.
    missing_maha = table[table["roc_maha_mean"].isna() & table["roc_unified_mean"].notna()
                         & (table["roc_native_mean"] != table["roc_unified_mean"])]
    if not missing_maha.empty:
        report.append(f"\n> NOTE: Mahalanobis column empty for {', '.join(missing_maha['code'])} "
                      f"— run phase1_rescore.py.")

    (RUNS / "phase1_report.md").write_text("\n".join(report))
    print("\n".join(report))
    print(f"\nwritten: {RUNS/'phase1_summary.csv'}, {RUNS/'phase1_runs.csv'}, {RUNS/'phase1_report.md'}")


if __name__ == "__main__":
    main()
