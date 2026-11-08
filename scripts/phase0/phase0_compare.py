"""
Phase 0: compare A100 repro results against the paper numbers.

Reads runs_phase0/<family>/<run_name>/summary.json produced by phase0_repro.py,
aggregates, and checks against phase0_manifest.json:

  1. CONFIG IDENTITY: parameter_count must match the paper EXACTLY per config
     (if it doesn't, the run trained a different architecture — hard fail).
  2. REPRO (seeds 42-44): |mean ROC-AUC (repro) - mean ROC-AUC (paper)| per
     config: PASS <= 0.010, WARN <= 0.020, FAIL above.
  3. 5-SEED STATS (seeds 42-46): mean +/- std per config, ready for the updated
     headline tables in the manuscript.
  4. RANKING: family ordering by best ROC-AUC, paper vs repro.

Usage (from repo root, after all shards finished):
    python scripts/phase0/phase0_compare.py

Outputs: runs_phase0/phase0_summary.csv, runs_phase0/phase0_report.md
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).resolve().parent / "phase0_manifest.json"
RUNS = REPO / "runs_phase0"

PASS_TOL = 0.010
WARN_TOL = 0.020

REPRO_SEEDS = {42, 43, 44}


def collect_runs() -> pd.DataFrame:
    rows = []
    for summary_path in sorted(RUNS.glob("*/*/summary.json")):
        s = json.loads(summary_path.read_text())
        if s.get("error"):
            continue
        best = s.get("best_metrics", {})
        config = json.loads((summary_path.parent / "config.json").read_text())
        name = s["run_name"]
        rows.append(
            {
                "family": summary_path.parent.parent.name,
                "run_name_seeded": name,
                "seed": int(config["run"]["seed"]),
                "roc_auc": best.get("test/roc_auc"),
                "val_roc_auc": best.get("val/roc_auc"),
                "train_loss": best.get("train/loss"),
                "parameter_count": s.get("parameter_count"),
            }
        )
    if not rows:
        raise SystemExit(f"no summaries found under {RUNS} — did the shards run?")
    df = pd.DataFrame(rows)
    df["run_name"] = df["run_name_seeded"].str.replace(r"^(\d{3})-s\d+-", r"\1-", regex=True)
    return df


def main() -> None:
    manifest = {(m["family"], m["run_name"]): m for m in json.loads(MANIFEST.read_text())}
    df = collect_runs()

    report = ["# Phase 0 — A100 reproducibility report\n"]
    table_rows = []
    n_pass = n_warn = n_fail = 0

    for (family, run_name), m in sorted(manifest.items()):
        g = df[(df["family"] == family) & (df["run_name"] == run_name)]
        if g.empty:
            table_rows.append({"code": m["code"], "family": family, "run_name": run_name,
                               "status": "MISSING"})
            n_fail += 1
            continue

        # 1. config identity
        pc = g["parameter_count"].dropna().unique()
        identity_ok = len(pc) == 1 and int(pc[0]) == m["paper_parameter_count"]

        # 2. repro on original seeds
        repro = g[g["seed"].isin(REPRO_SEEDS)]
        repro_mean = repro["roc_auc"].mean()
        delta = repro_mean - m["paper_roc_auc_mean"]

        # 3. 5-seed stats
        all_mean, all_std = g["roc_auc"].mean(), g["roc_auc"].std()

        if not identity_ok:
            status = "FAIL(param_count)"
            n_fail += 1
        elif abs(delta) <= PASS_TOL:
            status = "PASS"
            n_pass += 1
        elif abs(delta) <= WARN_TOL:
            status = "WARN"
            n_warn += 1
        else:
            status = "FAIL"
            n_fail += 1

        table_rows.append(
            {
                "code": m["code"],
                "family": family,
                "run_name": run_name,
                "paper_roc_mean_s42_44": round(m["paper_roc_auc_mean"], 4),
                "repro_roc_mean_s42_44": round(float(repro_mean), 4),
                "delta": round(float(delta), 4),
                "roc_mean_5seed": round(float(all_mean), 4),
                "roc_std_5seed": round(float(all_std), 4),
                "n_runs_found": len(g),
                "param_count_ok": identity_ok,
                "status": status,
            }
        )

    out = pd.DataFrame(table_rows)
    out.to_csv(RUNS / "phase0_summary.csv", index=False)

    report.append(f"**Verdict: {n_pass} PASS / {n_warn} WARN / {n_fail} FAIL** "
                  f"(tolerance: PASS <= {PASS_TOL}, WARN <= {WARN_TOL} on mean ROC-AUC, seeds 42-44)\n")
    report.append(out.to_markdown(index=False))

    # 4. family ranking, paper vs repro (best config per family)
    report.append("\n## Family ranking (best headline config per family)\n")
    rank = out[out["status"] != "MISSING"].copy()
    paper_rank = rank.loc[rank.groupby("family")["paper_roc_mean_s42_44"].idxmax()]
    paper_rank = paper_rank.sort_values("paper_roc_mean_s42_44", ascending=False)["family"].tolist()
    repro_rank = rank.loc[rank.groupby("family")["repro_roc_mean_s42_44"].idxmax()]
    repro_rank = repro_rank.sort_values("repro_roc_mean_s42_44", ascending=False)["family"].tolist()
    report.append(f"- paper: {' > '.join(paper_rank)}")
    report.append(f"- repro: {' > '.join(repro_rank)}")
    report.append(f"- ranking preserved: **{paper_rank == repro_rank}**\n")

    env_path = RUNS / "env_report.json"
    if env_path.exists():
        report.append("## Environment\n```json\n" + env_path.read_text() + "\n```")

    (RUNS / "phase0_report.md").write_text("\n".join(report))
    print("\n".join(report))
    print(f"\nwritten: {RUNS/'phase0_summary.csv'}, {RUNS/'phase0_report.md'}")


if __name__ == "__main__":
    main()
