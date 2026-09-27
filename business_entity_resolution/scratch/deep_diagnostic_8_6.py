import json
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

# Load reports
with open("output/turn7_validation_report.json", "r") as f:
    r5k = json.load(f)

with open("output/turn8_5_100k/turn7_validation_report.json", "r") as f:
    r100k = json.load(f)

print("="*80)
print("TURN 8.6 DIAGNOSTIC DEEP-DIVE")
print("="*80)

# Load 5k and 100k cached datasets
print("Loading cached 5k and 100k datasets...")
with open("cache/benchmark_sample_s1_5000_dist_5000.pkl", "rb") as f:
    d5k = pickle.load(f)

with open("cache/benchmark_sample_s1_100000_dist_5000.pkl", "rb") as f:
    d100k = pickle.load(f)

gt5k = d5k["gt"]
gt100k = d100k["gt"]

# Ground truth maps
gt_map_5k = {}
for _, r in gt5k.iterrows():
    s1_id = str(r["source1_entity_id"])
    m_str = str(r.get("matched_entity_ids", "")).strip()
    m_set = {x.strip() for x in m_str.split(",") if x.strip()}
    gt_map_5k[s1_id] = m_set

gt_map_100k = {}
for _, r in gt100k.iterrows():
    s1_id = str(r["source1_entity_id"])
    m_str = str(r.get("matched_entity_ids", "")).strip()
    m_set = {x.strip() for x in m_str.split(",") if x.strip()}
    gt_map_100k[s1_id] = m_set

# Let's inspect the validation errors and candidate diagnostics
print("\n--- 1. Candidate Recall & Miss Analysis ---")
print(f"5k Validation Total True Matches:   {r5k['error_analysis']['total_true_matches']}")
print(f"5k Blocking Misses:                 {r5k['error_analysis']['blocking_misses_count']} ({r5k['error_analysis']['blocking_miss_rate_pct']:.2f}%)")
print(f"5k Classifier Misses:               {r5k['error_analysis']['classifier_misses_count']} ({r5k['error_analysis']['classifier_miss_rate_pct']:.2f}%)")
print(f"5k False Positives:                 {r5k['error_analysis']['false_positives']}")

print(f"\n100k Validation Total True Matches: {r100k['error_analysis']['total_true_matches']}")
print(f"100k Blocking Misses:               {r100k['error_analysis']['blocking_misses_count']} ({r100k['error_analysis']['blocking_miss_rate_pct']:.2f}%)")
print(f"100k Classifier Misses:             {r100k['error_analysis']['classifier_misses_count']} ({r100k['error_analysis']['classifier_miss_rate_pct']:.2f}%)")
print(f"100k False Positives:               {r100k['error_analysis']['false_positives']}")

print("\n--- 2. Precision & Recall Shift ---")
print(f"Pair Precision:   5k={r5k['validation_metrics']['pair_precision']*100:.2f}% -> 100k={r100k['validation_metrics']['pair_precision']*100:.2f}% (Drop: {r5k['validation_metrics']['pair_precision']*100 - r100k['validation_metrics']['pair_precision']*100:.2f}%)")
print(f"Pair Recall:      5k={r5k['validation_metrics']['pair_recall']*100:.2f}% -> 100k={r100k['validation_metrics']['pair_recall']*100:.2f}% (Drop: {r5k['validation_metrics']['pair_recall']*100 - r100k['validation_metrics']['pair_recall']*100:.2f}%)")
print(f"Entity Precision: 5k={r5k['validation_metrics']['mean_entity_precision']:.4f} -> 100k={r100k['validation_metrics']['mean_entity_precision']:.4f} (Drop: {r5k['validation_metrics']['mean_entity_precision'] - r100k['validation_metrics']['mean_entity_precision']:.4f})")
print(f"Entity Recall:    5k={r5k['validation_metrics']['mean_entity_recall']:.4f} -> 100k={r100k['validation_metrics']['mean_entity_recall']:.4f} (Drop: {r5k['validation_metrics']['mean_entity_recall'] - r100k['validation_metrics']['mean_entity_recall']:.4f})")
print(f"Macro F0.5:       5k={r5k['validation_metrics']['macro_f05']:.4f} -> 100k={r100k['validation_metrics']['macro_f05']:.4f} (Drop: {r5k['validation_metrics']['macro_f05'] - r100k['validation_metrics']['macro_f05']:.4f})")

# Let's inspect the threshold curves side-by-side
print("\n--- 3. Threshold Calibration & F0.5 Sensitivity ---")
for t in ["0.8", "0.82", "0.84", "0.86", "0.88", "0.9", "0.92", "0.94", "0.96", "0.98"]:
    f5 = r5k["threshold_sweep"].get(t, 0)
    f100 = r100k["threshold_sweep"].get(t, 0)
    print(f"Threshold {t}: 5k F0.5 = {f5:.4f} | 100k F0.5 = {f100:.4f} | Delta = {f100 - f5:+.4f}")

