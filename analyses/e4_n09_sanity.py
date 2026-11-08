"""E4 sanity: is the N09 collapse real, or an error in the labels or in the dump?

  1. labels: which bearings count as normal and damaged in the N09 dump, against the same bearings in N15;
  2. feature shift: scaled features of normal test windows, N09 against N15, per channel and feature, in scaler std;
  3. a method with no model at all: distance of normal and damaged N09 files from the N09 normal val mean, in the same features.
     If damage is visible in the features, a simple distance separates it, and the collapse belongs to the models.
Output: e4_n09_sanity.csv next to this script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
MCU = HERE.parent / "mcu"
sys.path.insert(0, str(MCU))
from p5common import SPLITS, SPLITS_META, append_csv, host_tag, now, verdict  # noqa: E402

N09 = np.load(MCU / "data" / "sr006_bundle" / "phase5_n09_features.npz")
M09 = json.loads((MCU / "data" / "sr006_bundle" / "phase5_n09_features.meta.json").read_text())
N15 = np.load(SPLITS)
M15 = json.loads(SPLITS_META.read_text())
FEATS = ["Mean", "Variance", "P2P", "LineInt", "Skew", "SpecEnergy", "SpecSpread", "WavD2", "WavD1"]
CH = ["cur1", "cur2", "vib"]
rows = []
base = dict(date=now(), host=host_tag())

# 1. labels
b9 = {(e["bearing"], e["is_anomaly"]) for s in M09["splits"].values() for e in s}
b15 = {(e["bearing"], e["is_anomaly"]) for s in M15["splits"].values() for e in s}
rows.append({**base, "check": "bearing_labels_n09_equal_n15", "value": len(b9 ^ b15), "reference": 0, "status": verdict(b9 == b15), "note": f"{len(b9)} bearings in N09; K0* normal, KA* damaged"})
wrong = [b for b, a in b9 if a != b.startswith("KA")]
rows.append({**base, "check": "normal_iff_K0_damaged_iff_KA", "value": len(wrong), "reference": 0, "status": verdict(not wrong), "note": ""})


def stack(z, meta, split, anom):
    return np.concatenate([z[e["key"]] for e in meta["splits"][split] if e["is_anomaly"] == anom])


# 2. shift of the normal class
n15, n09 = stack(N15, M15, "test", False), stack(N09, M09, "test", False)
shift = (n09.mean(0) - n15.mean(0)) / n15.std(0)
order = np.argsort(-np.abs(shift))
rows.append({**base, "check": "normal_feature_mean_shift_median_abs", "value": f"{np.median(np.abs(shift)):.3f}", "reference": "", "status": "MEASURED", "note": "N09 minus N15 normal test, in std of N15 normal test, over 27 features"})
for i in order[:5]:
    rows.append({**base, "check": "normal_feature_mean_shift_top", "value": f"{shift[i]:+.2f}", "reference": "", "status": "MEASURED", "note": f"{CH[i // 9]} {FEATS[i % 9]}"})

# 3. no model: per-row and per-file standardized distance from normal val of the same condition
for name, z, meta in (("N15", N15, M15), ("N09", N09, M09)):
    v = stack(z, meta, "val", False)
    mu, sd = v.mean(0), v.std(0) + 1e-9
    s, lab = [], []
    for e in meta["splits"]["test"]:
        d = np.sqrt((((z[e["key"]] - mu) / sd) ** 2).mean(1))
        s.append(d)
        lab.append(np.full(len(d), int(e["is_anomaly"])))
    s, lab = np.concatenate(s), np.concatenate(lab)
    roc = float(roc_auc_score(lab, s))
    rows.append({**base, "check": f"roc_no_model_distance_{name}", "value": f"{roc:.4f}", "reference": "", "status": "MEASURED", "note": "feature rows, z-distance from normal val of the same condition; no training, a sanity baseline, not a method of the paper"})

append_csv(HERE / "e4_n09_sanity.csv", rows, fresh=True)
for r in rows:
    print(f"{r['status']:8} {r['check']}: {r['value']}  {r['note'][:90]}")
if any(r["status"] == "FAIL" for r in rows):
    raise SystemExit(1)
