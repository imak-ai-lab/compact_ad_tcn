"""
Phase 0: select headline configs for the A100 reproducibility run.

Selection mirrors the paper's headline tables (single source of truth:
scripts/generate_paper_tables.py):
  A. Table I  — best representative per family at size <= 1 MiB (+ best MF overall,
                since MF has no configs under 1 MiB).
  B. Table II — best float32 model at each memory budget (0.05 .. 2.0 MiB).
  C. Contribution 2 — TL/TF matched-body pairs: the best TL per (K, H) cell
                (K in {1, 2}, H in {16, 120}) plus its matched TF; TL082 is
                guaranteed into the set.

Output: phase0_manifest.json with, per selected config:
  family, run_name (seedless), why it was selected, and reference (paper) values
  aggregated over seeds 42-44: test ROC-AUC mean/std, parameter_count.

Usage (from repo root):
    python scripts/phase0/select_phase0_configs.py
"""
import json
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "summary_table.csv"
OUT = Path(__file__).resolve().parent / "phase0_manifest.json"

CODE = {
    "PCA-reconstruction": "PR",
    "MLP-reconstruction": "MR",
    "MLP-forecasting": "MF",
    "Conv-reconstruction": "CR",
    "TCN-reconstruction": "TR",
    "TCN-forecasting": "TF",
    "TCN-light-forecasting": "TL",
    "GRU-repeated-reconstruction": "GR",
    "GRU-seq2seq-reconstruction": "GS",
    "GRU-seq2seq-forecasting": "GF",
    "Transformer-reconstruction": "XR",
}

df = pd.read_csv(CSV)
agg = df.groupby(["family", "run_name"], as_index=False).agg(
    roc=("best_metrics.test/roc_auc", "mean"),
    roc_std=("best_metrics.test/roc_auc", "std"),
    n_seeds=("seed", "nunique"),
    params=("parameter_count", "first"),
)
agg["size_mib"] = agg["params"] * 4 / (1024 * 1024)
agg["idx"] = agg["run_name"].str.extract(r"^(\d{3})").astype(int)
agg["code"] = agg["family"].map(CODE) + agg["idx"].apply(lambda i: f"{i:03d}")

selected: dict[tuple, list[str]] = {}


def pick(row, reason):
    key = (row["family"], row["run_name"])
    selected.setdefault(key, []).append(reason)


# --- A. Table I: best per family at <= 1 MiB --------------------------------
under1 = agg[agg["size_mib"] <= 1.0]
for _, r in under1.loc[under1.groupby("family")["roc"].idxmax()].iterrows():
    pick(r, "TableI_best_per_family_le1MiB")
# MF has nothing under 1 MiB -> take best MF overall for family coverage
mf = agg[agg["family"] == "MLP-forecasting"]
if not mf.empty:
    pick(mf.loc[mf["roc"].idxmax()], "best_MF_overall(no_le1MiB_configs)")

# --- B. Table II: budget winners --------------------------------------------
for budget in [0.05, 0.10, 0.25, 0.50, 1.00, 2.00]:
    sub = agg[agg["size_mib"] <= budget]
    if sub.empty:
        continue
    pick(sub.loc[sub["roc"].idxmax()], f"TableII_budget_winner_le{budget:.2f}MiB")

# --- C. TL/TF matched pairs for contribution 2 ------------------------------
def parse_tl(s):
    m = re.search(r"h(\d+).*sep(\d+)-hid(\d+)-bl(\d+)-ker(\d+)-fs(\d+)-tb(\d+)-mix(\d+)", s)
    return tuple(int(x) for x in m.groups()) if m else None


def parse_tf(s):
    m = re.search(r"h(\d+).*sep(\d+)-hid(\d+)-bl(\d+)-ker(\d+)-fs(\d+)", s)
    return tuple(int(x) for x in m.groups()) if m else None


tl = agg[agg["family"] == "TCN-light-forecasting"].copy()
tf = agg[agg["family"] == "TCN-forecasting"].copy()
tl["parsed"] = tl["run_name"].apply(parse_tl)
tf["key"] = tf["run_name"].apply(parse_tf)
tf_by_key = {r["key"]: r for _, r in tf.iterrows() if r["key"]}

pair_cells = [(1, 16), (1, 120), (2, 16), (2, 120)]
for K, H in pair_cells:
    cell = tl[tl["parsed"].apply(lambda p: p is not None and p[6] == K and p[0] == H)]
    if cell.empty:
        continue
    best_tl = cell.loc[cell["roc"].idxmax()]
    pick(best_tl, f"C2_best_TL_cell_K{K}_H{H}")
    body = best_tl["parsed"][:6]
    if body in tf_by_key:
        pick(tf_by_key[body], f"C2_matched_TF_for_K{K}_H{H}")

# TL082 must be in (headline model of the paper)
tl082 = agg[(agg["code"] == "TL082")]
if not tl082.empty:
    r = tl082.iloc[0]
    pick(r, "headline_TL082")
    body = parse_tl(r["run_name"])[:6]
    if body in tf_by_key:
        pick(tf_by_key[body], "headline_TL082_matched_TF")

# --- write manifest ----------------------------------------------------------
manifest = []
for (family, run_name), reasons in sorted(selected.items()):
    r = agg[(agg["family"] == family) & (agg["run_name"] == run_name)].iloc[0]
    seeds_present = sorted(
        df[(df["family"] == family) & (df["run_name"] == run_name)]["seed"].tolist()
    )
    manifest.append(
        {
            "family": family,
            "run_name": run_name,
            "code": r["code"],
            "reasons": reasons,
            "paper_seeds": seeds_present,
            "paper_roc_auc_mean": round(float(r["roc"]), 6),
            "paper_roc_auc_std": round(float(r["roc_std"]), 6),
            "paper_parameter_count": int(r["params"]),
            "paper_size_mib": round(float(r["size_mib"]), 6),
        }
    )

OUT.write_text(json.dumps(manifest, indent=2))
n_gpu = sum(1 for m in manifest if m["family"] != "PCA-reconstruction")
print(f"selected {len(manifest)} configs -> {OUT}")
print(f"  GPU-trained: {n_gpu} configs x 5 seeds = {n_gpu * 5} runs")
for m in manifest:
    print(
        f"  {m['code']:>6} {m['family']:<28} {m['run_name']:<44} "
        f"roc={m['paper_roc_auc_mean']:.4f}±{m['paper_roc_auc_std']:.4f} "
        f"size={m['paper_size_mib']:.3f}MiB  [{'; '.join(m['reasons'])}]"
    )
