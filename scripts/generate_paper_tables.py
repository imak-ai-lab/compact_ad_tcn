"""
Generate Tables I–IV for the paper from summary_table.csv and summary_table_quant.csv.

Outputs four Markdown tables to stdout:
  - Table I:   Best representative per class at memory budget ≤1 MiB.
  - Table II:  Best float32 model across memory budgets (≤0.05 .. ≤2 MiB).
  - Table III: Paired TF vs TL analysis (per H × K cell), 27 matching TCN bodies each.
  - Table IV:  Paired PTQ qint8 effect per class.

Also prints the Spearman ρ(training MSE, test ROC-AUC) breakdown (§VI.D)
and the per-class config counts at each budget slice (closes round-2 M5).

Usage (from repo root):
    python scripts/generate_paper_tables.py

Reproducibility note: this script is the single source of truth for the
numbers cited in §VI of the manuscript.
"""
import re
import numpy as np
import pandas as pd
from scipy import stats

# === Load aggregated per-config data ==========================================

df = pd.read_csv("summary_table.csv")
dq = pd.read_csv("summary_table_quant.csv")

agg = df.groupby(["family", "run_name"], as_index=False).agg(
    roc=("best_metrics.test/roc_auc", "mean"),
    roc_std=("best_metrics.test/roc_auc", "std"),
    val_roc=("best_metrics.val/roc_auc", "mean"),
    f1=("best_metrics.test/f1", "mean"),
    train_loss=("best_metrics.train/loss", "mean"),
    params=("parameter_count", "first"),
    lat=("benchmark.latency_mean_seconds", "mean"),
)
agg["size_mib"] = agg["params"] * 4 / (1024 * 1024)
agg["lat_ms"] = agg["lat"] * 1000
agg["idx"] = agg["run_name"].str.extract(r"^(\d{3})").astype(int)

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
agg["code"] = agg["family"].map(CODE) + agg["idx"].apply(lambda i: f"{i:03d}")

# === TABLE I — best per class at ≤1 MiB =======================================

print("## TABLE I — best representative per class at size ≤ 1 MiB (float32)\n")
under1 = agg[agg["size_mib"] <= 1.0]
best_per_class = under1.loc[under1.groupby("family")["roc"].idxmax()].sort_values(
    "roc", ascending=False
)
print(
    "| Code | Family | ROC-AUC (test) | Size [MiB] | Latency [ms] | # configs ≤1 MiB |"
)
print("|------|--------|---------------:|-----------:|-------------:|----:|")
n_per = under1.groupby("family").size().to_dict()
for _, r in best_per_class.iterrows():
    print(
        f"| {r['code']} | {r['family']} | {r['roc']:.3f} ± {r['roc_std']:.3f} | "
        f"{r['size_mib']:.3f} | {r['lat_ms']:.2f} | {n_per.get(r['family'], 0)} |"
    )
# MF: report absence explicitly
mf_min = agg[agg["family"] == "MLP-forecasting"]["size_mib"].min()
print(
    f"| MF —  | MLP-forecasting | — | min size {mf_min:.2f} MiB > 1 MiB | — | 0 |"
)

# === TABLE II — best float32 across budgets ===================================

print("\n## TABLE II — best float32 model at memory budget ≤ B\n")
print("| Budget B [MiB] | Code | ROC-AUC (test) | Size [MiB] | Latency [ms] | "
      "# configs total |")
print("|---:|---|---:|---:|---:|---:|")
for budget in [0.05, 0.10, 0.25, 0.50, 1.00, 2.00]:
    sub = agg[agg["size_mib"] <= budget]
    if len(sub) == 0:
        continue
    b = sub.loc[sub["roc"].idxmax()]
    print(
        f"| ≤{budget:.2f} | {b['code']} | {b['roc']:.3f} ± {b['roc_std']:.3f} | "
        f"{b['size_mib']:.4f} | {b['lat_ms']:.2f} | {len(sub)} |"
    )

# Per-class counts at each budget slice (round-2 M5 fairness fix)
print("\n### TABLE II.b — # configs per class at each budget slice (closes M5)\n")
budgets = [0.05, 0.10, 0.25, 0.50, 1.00, 2.00]
classes = ["PR", "MR", "MF", "CR", "TR", "TF", "TL", "GR", "GS", "GF", "XR"]
hdr = "| Class |" + "|".join(f" ≤{b:.2f} " for b in budgets) + "|"
sep = "|---|" + "|".join("---:" for _ in budgets) + "|"
print(hdr); print(sep)
for cls in classes:
    fam = [k for k, v in CODE.items() if v == cls][0]
    row = [str((agg[(agg["family"] == fam) & (agg["size_mib"] <= b)].shape[0]))
           for b in budgets]
    print("| " + cls + " | " + " | ".join(row) + " |")

# === TABLE III — TF vs TL paired ==============================================

def parse_tf(s):
    m = re.search(r"h(\d+).*sep(\d+)-hid(\d+)-bl(\d+)-ker(\d+)-fs(\d+)", s)
    return tuple(int(x) for x in m.groups()) if m else None

def parse_tl(s):
    m = re.search(r"h(\d+).*sep(\d+)-hid(\d+)-bl(\d+)-ker(\d+)-fs(\d+)-tb(\d+)-mix(\d+)", s)
    return tuple(int(x) for x in m.groups()) if m else None

tf_agg = agg[agg["family"] == "TCN-forecasting"].copy()
tl_agg = agg[agg["family"] == "TCN-light-forecasting"].copy()
tf_agg["key"] = tf_agg["run_name"].apply(parse_tf)
tl_agg["body_key"] = tl_agg["run_name"].apply(lambda s: parse_tl(s)[:6] if parse_tl(s) else None)
tl_agg["K"] = tl_agg["run_name"].apply(lambda s: parse_tl(s)[6] if parse_tl(s) else None)
tl_agg["H"] = tl_agg["run_name"].apply(lambda s: parse_tl(s)[0] if parse_tl(s) else None)

print("\n## TABLE III — paired TF vs TL across (K, H) cells\n")
print("| K | H | # pairs | TL wins | median ΔROC | median size TF/TL | "
      "median Δlat (ms) | Wilcoxon p |")
print("|---:|---:|---:|---:|---:|---:|---:|---:|")
all_diffs_K = {1: [], 2: []}
for K in (1, 2):
    for H in (16, 32, 64, 120):
        tls = tl_agg[(tl_agg["K"] == K) & (tl_agg["H"] == H)]
        tfl = {r["key"]: r for _, r in tf_agg.iterrows()
               if r["key"] and r["key"][0] == H}
        droc, sizr, latd = [], [], []
        for _, t in tls.iterrows():
            bk = t["body_key"]
            if bk in tfl:
                tr = tfl[bk]
                droc.append(t["roc"] - tr["roc"])
                sizr.append(tr["params"] / t["params"])
                latd.append((t["lat_ms"] - tr["lat_ms"]))
        if not droc:
            continue
        p = stats.wilcoxon(droc).pvalue if np.std(droc) > 0 else float("nan")
        wins = sum(1 for d in droc if d >= 0)
        all_diffs_K[K].extend(droc)
        print(
            f"| {K} | {H} | {len(droc)} | {wins} | {np.median(droc):+.4f} | "
            f"{np.median(sizr):.2f}× | {np.median(latd):+.3f} | {p:.1e} |"
        )

print("\n### TABLE III aggregate (across all H per K)\n")
print("| K | # pairs | TL wins | median ΔROC | Wilcoxon p |")
print("|---:|---:|---:|---:|---:|")
for K in (1, 2):
    d = all_diffs_K[K]
    p = stats.wilcoxon(d).pvalue
    wins = sum(1 for x in d if x >= 0)
    print(f"| {K} | {len(d)} | {wins} | {np.median(d):+.4f} | {p:.1e} |")

# === TABLE IV — PTQ qint8 paired effect =======================================

aggq = dq.groupby(["family", "run_name"], as_index=False).agg(
    roc=("best_metrics.test/roc_auc", "mean"),
    size_bytes=("parameter_size_bytes", "first"),
    lat=("benchmark.latency_mean_seconds", "mean"),
)
aggq["size_mib_q"] = aggq["size_bytes"] / (1024 * 1024)
aggq["lat_ms_q"] = aggq["lat"] * 1000
aggq["family_fp"] = aggq["family"].str.replace("-quant", "", regex=False)
pairs = pd.merge(
    agg[["family", "run_name", "roc", "size_mib", "lat_ms"]].rename(
        columns={"roc": "roc_fp", "size_mib": "size_mib_fp", "lat_ms": "lat_ms_fp"}
    ),
    aggq[["family_fp", "run_name", "roc", "size_mib_q", "lat_ms_q"]].rename(
        columns={"roc": "roc_q"}
    ),
    left_on=["family", "run_name"],
    right_on=["family_fp", "run_name"],
)
print("\n## TABLE IV — PTQ qint8 paired effect per class\n")
print("| Code | Family | # pairs | median ΔROC | median size FP/Q | median lat FP/Q |")
print("|---|---|---:|---:|---:|---:|")
rows = []
for fam in pairs["family"].unique():
    g = pairs[pairs["family"] == fam]
    droc = (g["roc_q"] - g["roc_fp"]).values
    sx = (g["size_mib_fp"] / g["size_mib_q"]).values
    lx = (g["lat_ms_fp"] / g["lat_ms_q"]).values
    rows.append((np.median(droc), CODE[fam], fam, len(g), sx, lx, droc))
for med_d, code, fam, n, sx, lx, droc in sorted(rows, reverse=True):
    print(
        f"| {code} | {fam} | {n} | {np.median(droc):+.4f} | "
        f"{np.median(sx):.2f}× | {np.median(lx):.2f}× |"
    )
glob_med = np.median((pairs["roc_q"] - pairs["roc_fp"]).values)
print(f"\n**Global median ΔROC over {len(pairs)} pairs: {glob_med:+.4f}**")

# === §VI.D — Spearman breakdown ===============================================

print("\n## §VI.D — Spearman ρ(training MSE, test ROC-AUC)\n")
print("| Family / H | n | Spearman ρ |")
print("|---|---:|---:|")

recon = ["MLP-reconstruction", "Conv-reconstruction", "TCN-reconstruction",
         "GRU-repeated-reconstruction", "GRU-seq2seq-reconstruction",
         "Transformer-reconstruction"]
agg_rc = agg[agg["family"].isin(recon) & agg["train_loss"].notna()]
pe, _ = stats.pearsonr(agg_rc["train_loss"], agg_rc["roc"])
sp, _ = stats.spearmanr(agg_rc["train_loss"], agg_rc["roc"])
print(f"| **All reconstruction (pooled)** | {len(agg_rc)} | Pearson {pe:+.3f}, Spearman {sp:+.3f} |")
for fam in recon:
    s = agg_rc[agg_rc["family"] == fam]
    if len(s) < 5:
        continue
    rho, _ = stats.spearmanr(s["train_loss"], s["roc"])
    print(f"| {CODE[fam]} ({fam}) | {len(s)} | {rho:+.3f} |")

# Forecasting by H
forecast = ["TCN-forecasting", "TCN-light-forecasting",
            "MLP-forecasting", "GRU-seq2seq-forecasting"]
df_fc = agg[agg["family"].isin(forecast)].copy()

def get_H(s):
    m = re.search(r"(?:^|-)h(?:or)?(\d+)", s)
    return int(m.group(1)) if m else None

df_fc["H"] = df_fc["run_name"].apply(get_H)
for H in sorted(df_fc["H"].dropna().unique()):
    s = df_fc[df_fc["H"] == H]
    if len(s) < 5:
        continue
    rho, _ = stats.spearmanr(s["train_loss"], s["roc"])
    print(f"| Forecasting H={H} | {len(s)} | {rho:+.3f} |")

# Val ROC vs Test ROC sanity check
pe, _ = stats.pearsonr(agg["val_roc"], agg["roc"])
sp, _ = stats.spearmanr(agg["val_roc"], agg["roc"])
mad = (agg["val_roc"] - agg["roc"]).abs().median()
print(f"\n**Validation ROC-AUC vs Test ROC-AUC (n={len(agg)}): Pearson={pe:.3f}, "
      f"Spearman={sp:.3f}, median |Δ|={mad:.4f}**")
